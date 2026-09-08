"""Inference-only PTV3: CPU geometry/CPE/FFN, mandatory NPU FP16 attention.

Load CANN's set_env.sh before use. Checkpoint keys and the point/output contract
match ptv3_vanilla. Construct, load_state_dict(strict=True), eval(), then forward;
do not move/cast the whole model, which intentionally has mixed placement.
No Flash, RPE, training or FP32 attention/upcast path is provided.
"""

import torch
import torch_npu  # noqa: F401

from .ptv3_vanilla import (
    PointTransformerV3Vanilla,
    VanillaPoint,  # re-export for the stage profiler
    VanillaSerializedAttention,
    offset2bincount,
    segment_csr_vanilla,  # re-export for the stage profiler
)

NPU_DEVICE = "npu:0"
NPU_JIT_COMPILE = False


class AscendSerializedAttention(VanillaSerializedAttention):
    """Reuse CPU padding/index construction, but compute attention only in half."""

    def __init__(self, source):
        if source.enable_rpe or source.enable_flash:
            raise ValueError("Ascend attention does not support RPE or Flash")
        super().__init__(
            channels=source.channels,
            num_heads=source.num_heads,
            patch_size=source.patch_size_max,
            qkv_bias=source.qkv.bias is not None,
            qk_scale=source.scale,
            order_index=source.order_index,
            upcast_attention=False,
            upcast_softmax=False,
        )
        self.load_state_dict(source.state_dict(), strict=True)
        self.to(device=NPU_DEVICE, dtype=torch.float16)

    def prepare_indices(self, point):
        count = point.feat.shape[0]
        single_batch = point.offset.numel() == 1
        self.patch_size = min(
            count if single_batch else offset2bincount(point.offset).min().item(),
            self.patch_size_max,
        )
        # For a single cloud with complete patches, pad/unpad are identities.
        # Use the current point's views, not a cache shared across point clouds.
        if single_batch and self.patch_size > 0 and count % self.patch_size == 0:
            return (
                point.serialized_order[self.order_index],
                point.serialized_inverse[self.order_index],
            )
        pad, unpad, _ = self.get_padding_and_inverse(point)
        return (
            point.serialized_order[self.order_index][pad],
            unpad[point.serialized_inverse[self.order_index]],
        )

    def forward(self, point):
        if self.training:
            raise RuntimeError("Ascend PTV3 is inference-only; call eval() first")
        if point.feat.device.type != "cpu" or point.feat.dtype != torch.float32:
            raise RuntimeError("The surrounding PTV3 encoder must remain CPU FP32")
        for parameter in self.parameters():
            if parameter.device.type != "npu" or parameter.dtype != torch.float16:
                raise RuntimeError("Attention parameters must remain NPU FP16")

        order, inverse = self.prepare_indices(point)
        h, k, c = self.num_heads, self.patch_size, self.channels

        device = self.qkv.weight.device
        features = point.feat.to(device=device, dtype=torch.float16)
        order = order.to(device=device, dtype=torch.int32)
        inverse = inverse.to(device=device, dtype=torch.int32)
        qkv = self.qkv(features).index_select(0, order)
        q, key, value = (
            qkv.reshape(-1, k, 3, h, c // h)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        q, key, value = q.contiguous(), key.contiguous(), value.contiguous()
        scores = (q * self.scale) @ key.transpose(-2, -1)
        probabilities = self.softmax(scores)
        features = (probabilities @ value).transpose(1, 2).reshape(-1, c)
        features = self.proj(features.index_select(0, inverse))
        point.feat = features.to(device="cpu", dtype=torch.float32)
        return point


class PointTransformerV3Ascend(PointTransformerV3Vanilla):
    """Same encoder/checkpoint layout as vanilla, with FP16 attention by default."""

    def __init__(self, **kwargs):
        for option in ("enable_flash", "enable_rpe", "upcast_attention", "upcast_softmax"):
            if kwargs.get(option, False):
                raise ValueError(f"Ascend PTV3 does not support {option}=True")
            kwargs[option] = False
        torch.npu.set_device(NPU_DEVICE)
        torch.npu.set_compile_mode(jit_compile=NPU_JIT_COMPILE)
        super().__init__(**kwargs)
        for stage in self.enc:
            for block in stage.children():
                if hasattr(block, "attn"):
                    block.attn = AscendSerializedAttention(block.attn)
