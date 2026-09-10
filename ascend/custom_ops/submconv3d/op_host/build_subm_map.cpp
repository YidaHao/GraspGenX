#include "build_subm_map_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* ctx) {
    const auto& shape = ctx->GetInputShape(0)->GetStorageShape();
    const auto k = *ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 4 ||
        shape.GetDim(0) < 1 || shape.GetDim(0) > 4096 ||
        (k != 1 && k != 3 && k != 5)) return ge::GRAPH_FAILED;
    const auto& out = ctx->GetOutputShape(0)->GetStorageShape();
    if (out.GetDimNum() != 2 || out.GetDim(0) != shape.GetDim(0) ||
        out.GetDim(1) != (k * k * k + 7) / 8 * 8) return ge::GRAPH_FAILED;
    BuildSubmMapTilingData data;
    data.set_points(shape.GetDim(0));
    data.set_kernel(k);
    data.set_width((k * k * k + 7) / 8 * 8);
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    ctx->SetBlockDim(std::min<uint32_t>(shape.GetDim(0), platform.GetCoreNumAic()));
    ctx->SetTilingKey(0);
    ctx->GetWorkspaceSizes(1)[0] = 0;
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class BuildSubmMap : public OpDef {
public:
    explicit BuildSubmMap(const char* name) : OpDef(name) {
        Input("indices").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("neighbors").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Attr("kernel_size").AttrType(REQUIRED).Int();
        SetInferShape([](gert::InferShapeContext* ctx) {
            const auto k = *ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
            const auto* in = ctx->GetInputShape(0);
            if (in->GetDimNum() != 2 || (k != 1 && k != 3 && k != 5)) return ge::GRAPH_FAILED;
            auto* out = ctx->GetOutputShape(0);
            out->SetDimNum(2);
            out->SetDim(0, in->GetDim(0));
            out->SetDim(1, (k * k * k + 7) / 8 * 8);
            return ge::GRAPH_SUCCESS;
        });
        SetInferDataType([](gert::InferDataTypeContext* ctx) {
            ctx->SetOutputDataType(0, ge::DT_INT32);
            return ge::GRAPH_SUCCESS;
        });
        AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
    }
};
OP_ADD(BuildSubmMap);
}
