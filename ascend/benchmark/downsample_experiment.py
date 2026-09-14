"""Benchmark-only pooling ablations. The production model/reference are untouched."""

import math
from functools import partial

import torch
import torch.nn.functional as F

from ascend.benchmark import validate_ptv3  # noqa: F401 - install the repository import shims
from graspgenx.models.ptv3.ptv3_vanilla import VanillaPoint, segment_csr_vanilla

VARIANTS = ("reference", "drop_coords", "fp16", "drop_coords_fp16")


@torch.inference_mode()
def prepare(model):
    prepared = []
    for stage in model.enc:
        if "down" not in stage._modules:
            continue
        down = stage.down
        if down.training or down.reduce != "max" or "forward" in down.__dict__:
            raise RuntimeError("Experiment requires an unmodified eval/max pooling module")
        if down.norm is not None and not (isinstance(down.norm, torch.nn.BatchNorm1d)
                                          and down.norm.track_running_stats):
            raise RuntimeError("Only fixed-stat BatchNorm1d is supported")
        if down.act is not None and not isinstance(down.act, torch.nn.GELU):
            raise RuntimeError("Only the current GELU activation is supported")
        state = {"weight": down.proj.weight.to("npu:0", torch.float16),
                 "bias": down.proj.bias.to("npu:0", torch.float16) if down.proj.bias is not None else None}
        if down.norm is not None:
            for name in ("weight", "bias", "running_mean", "running_var"):
                value = getattr(down.norm, name)
                state[f"bn_{name}"] = value.to("npu:0", torch.float16) if value is not None else None
        prepared.append((down, down.forward, state))
    return prepared


def select(prepared, variant, diagnostics=None):
    if variant not in VARIANTS:
        raise ValueError(variant)
    for index, (down, original, state) in enumerate(prepared):
        if variant == "reference":
            down.forward = original
        else:
            down.forward = partial(
                forward, down, remove_coords="drop_coords" in variant,
                npu_state=state if "fp16" in variant else None,
                diagnostics=diagnostics, stage=index + 1)


def restore(prepared):
    for down, _, _ in prepared:
        if "forward" in down.__dict__:
            del down.forward


@torch.inference_mode()
def forward(source, point, *, remove_coords, npu_state=None, diagnostics=None, stage=None):
    if source.training or source.reduce != "max":
        raise RuntimeError("Only eval/max pooling is supported")
    if remove_coords and source.traceable:
        raise RuntimeError("Removing centroids is limited to non-traceable embedding-only experiments")
    if point.feat.device.type != "cpu" or point.feat.dtype != torch.float32:
        raise RuntimeError("Experiment boundary requires CPU FP32 features")
    depth = (math.ceil(source.stride) - 1).bit_length()
    if depth > point.serialized_depth:
        depth = 0
    code = point.serialized_code >> (depth * 3)
    _, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
    _, indices = torch.sort(cluster)
    ptr = torch.cat((counts.new_zeros(1), torch.cumsum(counts, dim=0)))
    heads = indices[ptr[:-1]]

    padded_rows = None
    if npu_state is None:
        projected = source.proj(point.feat)
        features = segment_csr_vanilla(projected[indices], ptr, reduce=source.reduce)
    else:
        projected = F.linear(point.feat.to("npu:0", torch.float16),
                             npu_state["weight"], npu_state["bias"])
        # scatter_reduce_ falls back to CPU on this Torch-NPU. Repeating a valid
        # last row pads nonempty groups without changing inference-time max.
        width = int(counts.max())
        positions = ptr[:-1, None] + torch.arange(width).minimum(counts[:, None] - 1)
        grouped = indices[positions].to("npu:0", torch.int32)
        padded_rows = grouped.numel()
        features = projected.index_select(0, grouped.flatten()).reshape(
            len(counts), width, source.out_channels).amax(1)
        if features.dtype != torch.float16 or features.device.type != "npu":
            raise RuntimeError("FP16 feature reduction left the intended NPU tensor path")

    coordinates = None if remove_coords else segment_csr_vanilla(point.coord[indices], ptr, reduce="mean")
    code = code[:, heads]
    order = torch.argsort(code)
    inverse = torch.zeros_like(order).scatter_(
        1, order, torch.arange(code.shape[1]).repeat(code.shape[0], 1))
    if source.shuffle_orders:
        permutation = torch.randperm(code.shape[0])
        code, order, inverse = code[permutation], order[permutation], inverse[permutation]
    result = VanillaPoint(
        feat=features, grid_coord=point.grid_coord[heads] >> depth, batch=point.batch[heads],
        serialized_code=code, serialized_order=order, serialized_inverse=inverse,
        serialized_depth=point.serialized_depth - depth)
    if not remove_coords:
        result.coord = coordinates
    for key in ("condition", "context"):
        if key in point:
            result[key] = point[key]
    if source.traceable:
        result.pooling_inverse, result.pooling_parent = cluster, point
    if source.norm is not None:
        if npu_state is None:
            result.feat = source.norm(result.feat)
        else:
            result.feat = F.batch_norm(
                result.feat, npu_state["bn_running_mean"], npu_state["bn_running_var"],
                npu_state["bn_weight"], npu_state["bn_bias"], training=False, eps=source.norm.eps)
    if source.act is not None:
        result.feat = source.act(result.feat)
    if npu_state is not None:
        if result.feat.dtype != torch.float16 or result.feat.device.type != "npu":
            raise RuntimeError("FP16 normalization/activation left the intended NPU tensor path")
        result.feat = result.feat.to("cpu", torch.float32)
    if diagnostics is not None:
        diagnostics.append({"stage": stage, "input_points": len(point.batch), "output_points": len(counts),
                            "max_cluster": int(counts.max()), "padded_rows": padded_rows,
                            "padding_ratio": padded_rows / len(point.batch) if padded_rows is not None else None,
                            "coordinate_removed": remove_coords, "npu_fp16": npu_state is not None})
    return result
