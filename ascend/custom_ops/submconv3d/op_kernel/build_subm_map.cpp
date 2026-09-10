#include "kernel_operator.h"
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Sync() {
    SetFlag<event>(0);
    WaitFlag<event>(0);
}

extern "C" __global__ __aicore__ void build_subm_map(
    GM_ADDR indices, GM_ADDR neighbors, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    TPipe pipe;
    TBuf<TPosition::VECCALC> coordsBuf, rowBuf;
    pipe.InitBuffer(coordsBuf, 4096 * 4 * sizeof(int32_t));
    pipe.InitBuffer(rowBuf, 128 * sizeof(int32_t));
    auto coords = coordsBuf.Get<int32_t>();
    auto row = rowBuf.Get<int32_t>();
    GlobalTensor<int32_t> input, output;
    input.SetGlobalBuffer((__gm__ int32_t*)indices);
    output.SetGlobalBuffer((__gm__ int32_t*)neighbors);

    // 310P DMA uses 32-byte blocks. Never over-read the final odd coordinate row.
    const uint32_t aligned = t.points / 2 * 8;
    if (aligned) DataCopy(coords, input, aligned);
    Sync<HardEvent::MTE2_S>();
    for (uint32_t i = aligned; i < t.points * 4; ++i) coords.SetValue(i, input.GetValue(i));
    const int64_t radius = t.kernel / 2;
    for (uint32_t i = GetBlockIdx(); i < t.points; i += GetBlockNum()) {
        Duplicate(row, int32_t(-1), t.width);
        Sync<HardEvent::V_S>();
        const int32_t batch = coords.GetValue(i * 4);
        const int64_t x = coords.GetValue(i * 4 + 1);
        const int64_t y = coords.GetValue(i * 4 + 2);
        const int64_t z = coords.GetValue(i * 4 + 3);
        // Correctness-first O(N^2) map builder; no host roundtrip or hash collision.
        for (uint32_t j = 0; j < t.points; ++j) {
            if (coords.GetValue(j * 4) != batch) continue;
            const int64_t dx = int64_t(coords.GetValue(j * 4 + 1)) - x + radius;
            const int64_t dy = int64_t(coords.GetValue(j * 4 + 2)) - y + radius;
            const int64_t dz = int64_t(coords.GetValue(j * 4 + 3)) - z + radius;
            if (dx >= 0 && dx < t.kernel && dy >= 0 && dy < t.kernel && dz >= 0 && dz < t.kernel)
                row.SetValue((dx * t.kernel + dy) * t.kernel + dz, j);
        }
        Sync<HardEvent::S_MTE3>();
        DataCopy(output[i * t.width], row, t.width);
        Sync<HardEvent::MTE3_S>();
    }
}
