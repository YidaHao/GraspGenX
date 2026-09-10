#include "kernel_operator.h"
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Sync() {
    SetFlag<event>(0);
    WaitFlag<event>(0);
}

extern "C" __global__ __aicore__ void subm_conv3d(
    GM_ADDR features, GM_ADDR neighbors, GM_ADDR weight, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    TPipe pipe;
    TBuf<TPosition::VECCALC> gatherBuf, resultBuf;
    TBuf<TPosition::A1> a1Buf;
    TBuf<TPosition::B1> b1Buf;
    TBuf<TPosition::A2> a2Buf;
    TBuf<TPosition::B2> b2Buf;
    TBuf<TPosition::CO1> c1Buf;
    TBuf<TPosition::CO2> c2Buf;
    pipe.InitBuffer(gatherBuf, 256 * sizeof(half));
    pipe.InitBuffer(resultBuf, 256 * sizeof(half));
    pipe.InitBuffer(a1Buf, 256 * sizeof(half));
    pipe.InitBuffer(b1Buf, 256 * sizeof(half));
    pipe.InitBuffer(a2Buf, 256 * sizeof(half));
    pipe.InitBuffer(b2Buf, 256 * sizeof(half));
    pipe.InitBuffer(c1Buf, 256 * sizeof(float));
    pipe.InitBuffer(c2Buf, 256 * sizeof(float));
    auto gather = gatherBuf.Get<half>();
    auto result = resultBuf.Get<half>();
    auto a1 = a1Buf.Get<half>();
    auto b1 = b1Buf.Get<half>();
    auto a2 = a2Buf.Get<half>();
    auto b2 = b2Buf.Get<half>();
    auto c1 = c1Buf.Get<float>();
    auto c2 = c2Buf.Get<float>();
    GlobalTensor<half> x, w, out;
    GlobalTensor<int32_t> map;
    x.SetGlobalBuffer((__gm__ half*)features);
    w.SetGlobalBuffer((__gm__ half*)weight);
    out.SetGlobalBuffer((__gm__ half*)output);
    map.SetGlobalBuffer((__gm__ int32_t*)neighbors);

    const uint32_t outputTiles = t.cout / 16;
    for (uint32_t job = GetBlockIdx(); job < t.jobs; job += GetBlockNum()) {
        const uint32_t first = job / outputTiles * 16;
        const uint32_t channel = job % outputTiles * 16;
        const uint16_t rows = (t.points - first < 16) ? t.points - first : 16;
        bool firstProduct = true;
        for (uint32_t k = 0; k < t.volume; ++k) {
            int32_t idx[16];
            bool any = false;
            for (uint32_t r = 0; r < rows; ++r) {
                idx[r] = map.GetValue((first + r) * t.width + k);
                any |= idx[r] >= 0 && idx[r] < t.points;
            }
            if (!any) continue;
            for (uint32_t ci = 0; ci < t.cin; ci += 16) {
                Duplicate(gather, half(0), 256);
                Sync<HardEvent::V_MTE2>();
                for (uint32_t r = 0; r < rows; ++r) {
                    if (idx[r] >= 0 && idx[r] < t.points)
                        DataCopy(gather[r * 16], x[idx[r] * t.cin + ci], 16);
                }
                Sync<HardEvent::MTE2_MTE3>();
                DataCopy(a1, gather, 256);
                Sync<HardEvent::MTE3_MTE1>();
                DataCopy(b1, w[k * t.cin * t.cout + ci * t.cout + channel],
                         {16, 1, uint16_t(t.cout / 16 - 1), 0});
                Sync<HardEvent::MTE2_MTE1>();
                LoadData2DParams load;
                load.repeatTimes = 1;
                load.srcStride = 1;
                LoadData(a2, a1, load);
                load.ifTranspose = true;
                LoadData(b2, b1, load);
                Sync<HardEvent::MTE1_M>();
                // 310P Cube: 16x16 FP16 tiles, accumulated in L0C FP32.
                MmadParams mm;
                mm.m = 16;
                mm.n = 16;
                mm.k = 16;
                mm.isBias = !firstProduct;
                mm.cmatrixInitVal = firstProduct;
                Mmad(c1, a2, b2, mm);
                Sync<HardEvent::M_S>();
                firstProduct = false;
            }
        }
        if (firstProduct) {
            Duplicate(result, half(0), 256);
        } else {
            Sync<HardEvent::M_V>();
            DataCopyEnhancedParams enhanced;
            enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
            DataCopy(c2, c1, {1, 1, 0, 0}, enhanced);
            PipeBarrier<PIPE_V>();
            // dav_m200 FP32 -> FP16 implements CAST_NONE, not CAST_RINT.
            Cast(result, c2, RoundMode::CAST_NONE, 256);
        }
        Sync<HardEvent::V_MTE3>();
        // Each core owns its output tile, so no scatter atomics are required.
        DataCopy(out[first * t.cout + channel], result,
                 {rows, 1, 0, uint16_t(t.cout / 16 - 1)});
        Sync<HardEvent::MTE3_S>();
    }
}
