"""Inference-only PTV3: CPU geometry/CPE, continuous NPU FP16 transformer tail.

Load CANN's set_env.sh before use. Checkpoint keys and the point/output contract
match ptv3_vanilla. Construct, load_state_dict(strict=True), eval(), then forward;
do not move/cast the whole model, which intentionally has mixed placement.
No Flash, RPE, training or FP32 feature/upcast path is provided.
PointTransformerV3AttentionOnly is the previous CPU-FFN control.
"""

import torch
import torch_npu  # noqa: F401

from .ptv3_vanilla import (
    PointTransformerV3Vanilla,
    VanillaPoint,  # re-export for the stage profiler
    VanillaPointModule,
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
        if point.feat.device.type != "cpu" or point.feat.dtype != torch.float32:
            raise RuntimeError("The attention-only surrounding encoder must be CPU FP32")
        order, inverse = self.prepare_indices(point)
        features = point.feat.to(device=self.qkv.weight.device, dtype=torch.float16)
        point.feat = self.forward_features(features, order, inverse).to(
            device="cpu", dtype=torch.float32
        )
        return point

    def forward_features(self, features, order, inverse):
        if self.training:
            raise RuntimeError("Ascend PTV3 is inference-only; call eval() first")
        if features.device.type != "npu" or features.dtype != torch.float16:
            raise RuntimeError("Attention features must remain NPU FP16")
        for parameter in self.parameters():
            if parameter.device.type != "npu" or parameter.dtype != torch.float16:
                raise RuntimeError("Attention parameters must remain NPU FP16")

        h, k, c = self.num_heads, self.patch_size, self.channels

        device = self.qkv.weight.device
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
        return features


class PointTransformerV3AttentionOnly(PointTransformerV3Vanilla):
    """Previous attention-only implementation, retained as the experiment control."""

    def __init__(self, **kwargs):
        for option in ("enable_flash", "enable_rpe", "upcast_attention", "upcast_softmax"):
            if kwargs.get(option, False):
                raise ValueError(f"Ascend PTV3 does not support {option}=True")
            kwargs[option] = False
        torch.npu.set_device(NPU_DEVICE)
        torch.npu.set_compile_mode(jit_compile=NPU_JIT_COMPILE)
        super().__init__(**kwargs)
        self.execution_config = {
            "dense_resident": False,
            "attention_dtype": "fp16",
            "jit_compile": NPU_JIT_COMPILE,
        }
        for stage in self.enc:
            for block in stage.children():
                if hasattr(block, "attn"):
                    block.attn = AscendSerializedAttention(block.attn)


class AscendFFN(torch.nn.Module):
    """FP16 FFN retaining the original checkpoint keys."""

    def __init__(self, source):
        super().__init__()
        self.fc1 = source.fc1.to(device=NPU_DEVICE, dtype=torch.float16)
        self.fc2 = source.fc2.to(device=NPU_DEVICE, dtype=torch.float16)
        self.act = source.act

    def forward(self, features):
        if (
            self.training
            or features.dtype != torch.float16
            or features.device.type != "npu"
        ):
            raise RuntimeError("FFN requires eval mode and NPU FP16 features")
        return self.fc2(self.act(self.fc1(features)))


class AscendBlock(VanillaPointModule):
    """CPU CPE followed by one H2D/D2H pair around the entire FP16 dense tail."""

    def __init__(self, source):
        super().__init__()
        if not source.pre_norm:
            raise ValueError("The resident Ascend block requires pre_norm=True")
        self.channels = source.channels
        self.pre_norm = source.pre_norm
        for name, module in source.named_children():
            self.add_module(name, module)
        self.norm1.to(device=NPU_DEVICE, dtype=torch.float16)
        self.norm2.to(device=NPU_DEVICE, dtype=torch.float16)
        self.mlp = AscendFFN(source.mlp)

    def forward_attention(self, point):
        if self.training:
            raise RuntimeError("Ascend PTV3 is inference-only; call eval() first")
        if point.feat.device.type != "cpu" or point.feat.dtype != torch.float32:
            raise RuntimeError("CPE must produce CPU FP32 features")
        order, inverse = self.attn.prepare_indices(point)
        residual = point.feat.to(device=NPU_DEVICE, dtype=torch.float16)
        normalized = torch_npu.npu_layer_norm_eval(
            residual, self.norm1.normalized_shape,
            self.norm1.weight, self.norm1.bias, self.norm1.eps,
        )
        point.feat = self.attn.forward_features(normalized, order, inverse)
        point["_attention_residual"] = residual
        return point

    def forward_ffn(self, point):
        residual = point.pop("_attention_residual")
        residual = residual + point.feat
        normalized = torch_npu.npu_layer_norm_eval(
            residual, self.norm2.normalized_shape,
            self.norm2.weight, self.norm2.bias, self.norm2.eps,
        )
        output = self.mlp(normalized)
        if output.dtype != torch.float16 or residual.dtype != torch.float16:
            raise RuntimeError("FFN output and residual must remain FP16")
        point.feat = (residual + output).to(device="cpu", dtype=torch.float32)
        return point

    def forward(self, point):
        cpe = self.cpe_conv(point.feat, point.grid_coord, point.batch)
        point.feat = point.feat + self.cpe_norm(self.cpe_linear(cpe))
        point = self.forward_attention(point)
        return self.forward_ffn(point)


class PointTransformerV3Ascend(PointTransformerV3AttentionOnly):
    """Default experiment: resident FP16 attention, residual/norm and FFN."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_config.update(
            dense_resident=True,
            norm_residual_ffn_dtype="fp16",
        )
        for stage in self.enc:
            for name, block in list(stage.named_children()):
                if hasattr(block, "attn"):
                    setattr(stage, name, AscendBlock(block))
