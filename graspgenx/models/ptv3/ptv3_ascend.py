"""Inference-only PTV3: CPU geometry/CPE, continuous NPU FP16 transformer tail.

Load CANN's set_env.sh before use. Checkpoint keys and the point/output contract
match ptv3_vanilla. Construct, load_state_dict(strict=True), eval(), then forward;
do not move/cast the whole model, which intentionally has mixed placement.
No Flash, RPE, training or FP32 feature/upcast path is provided. LayerNorm's
unused mean/rstd statistics can be FP32. PointTransformerV3AttentionOnly is the
previous CPU-FFN control; fusion switches below affect only new resident models.
"""

import torch
import torch_npu  # noqa: F401

from .ptv3_vanilla import (
    HashSparseConv3d,
    PointTransformerV3Vanilla,
    VanillaPoint,  # re-export for the stage profiler
    VanillaPointModule,
    VanillaSerializedAttention,
    offset2bincount,
    segment_csr_vanilla,  # re-export for the stage profiler
)

NPU_DEVICE = "npu:0"
NPU_JIT_COMPILE = False
FUSE_ADD_LAYER_NORM = True
FUSE_FFN = True


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
    """FP16 FFN with the original checkpoint keys and optional CANN FFN fusion."""

    def __init__(self, source):
        super().__init__()
        self.fc1 = source.fc1.to(device=NPU_DEVICE, dtype=torch.float16)
        self.fc2 = source.fc2.to(device=NPU_DEVICE, dtype=torch.float16)
        self.act = source.act
        self.use_fused = FUSE_FFN
        self.register_buffer("weight1", None, persistent=False)
        self.register_buffer("weight2", None, persistent=False)
        self.register_load_state_dict_post_hook(self._pack_weights)
        self._pack_weights(self, None)

    @staticmethod
    @torch.no_grad()
    def _pack_weights(module, _incompatible):
        # FFN expects [C, 4C] and [4C, C], unlike nn.Linear's transposed layout.
        # Repack after checkpoint reloads; never transpose weights per request.
        if module.use_fused:
            module.weight1 = module.fc1.weight.T.contiguous()
            module.weight2 = module.fc2.weight.T.contiguous()

    def forward(self, features):
        if (
            self.training
            or features.dtype != torch.float16
            or features.device.type != "npu"
        ):
            raise RuntimeError("FFN requires eval mode and NPU FP16 features")
        if self.use_fused:
            return torch_npu.npu_ffn(
                features, self.weight1, self.weight2, "gelu",
                bias1=self.fc1.bias, bias2=self.fc2.bias, inner_precise=1,
            )
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
        self.cpe_conv = CachedCPEConv(source.cpe_conv)
        self.norm1.to(device=NPU_DEVICE, dtype=torch.float16)
        self.norm2.to(device=NPU_DEVICE, dtype=torch.float16)
        self.mlp = AscendFFN(source.mlp)
        self.fuse_add_norm = FUSE_ADD_LAYER_NORM

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
        if self.fuse_add_norm:
            # y and residual_sum are FP16. FP32 mean/rstd are unused statistics,
            # not an upcast feature path between attention and FFN.
            normalized, _, _, residual = torch_npu.npu_add_layer_norm(
                residual, point.feat, self.norm2.weight, self.norm2.bias,
                epsilon=self.norm2.eps, additional_output=True,
            )
        else:
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

    def forward_cpe(self, point):
        cpe = self.cpe_conv(point.feat, point.grid_coord, point.batch, point)
        point.feat = point.feat + self.cpe_norm(self.cpe_linear(cpe))
        return point

    def forward(self, point):
        point = self.forward_cpe(point)
        point = self.forward_attention(point)
        return self.forward_ffn(point)


class PointTransformerV3Ascend(PointTransformerV3AttentionOnly):
    """Default experiment: resident FP16 attention, residual/norm and FFN."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_config.update(
            dense_resident=True,
            norm_residual_ffn_dtype="fp16",
            fuse_add_layer_norm=FUSE_ADD_LAYER_NORM,
            fuse_ffn=FUSE_FFN,
            ffn_inner_precise=1 if FUSE_FFN else None,
            cpe_map_cache="per_point_per_forward",
            cpe_compute="cpu_fp32",
            cpe_post_ops="cpu_fp32",
        )
        for stage in self.enc:
            for name, block in list(stage.named_children()):
                if hasattr(block, "attn"):
                    setattr(stage, name, AscendBlock(block))


class CachedCPEConv(HashSparseConv3d):
    """CPU CPE with a point-local map shared by blocks with identical geometry."""

    def __init__(self, source):
        # Reuse parameters/buffers: wrapping must not change keys or consume RNG.
        torch.nn.Module.__init__(self)
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer("offsets", source.offsets)

    def _get_neighbor_map(self, grid_coord, batch, point):
        n, volume = grid_coord.shape[0], self.offsets.shape[0]
        cache_key = f"_cpe_hash_map_{self.kernel_size}"
        if cache_key not in point:
            # Preserve vanilla's sort/searchsorted tie-breaking, including duplicate
            # voxels and hash collisions. Geometry stays fixed within this stage.
            sorted_keys, sort_idx = self._hash(batch, grid_coord).sort()
            neighbor_coords = grid_coord.unsqueeze(1) + self.offsets.unsqueeze(0)
            neighbor_batch = batch.unsqueeze(1).expand(-1, volume)
            keys = self._hash(neighbor_batch.reshape(-1), neighbor_coords.reshape(-1, 3))
            positions = torch.searchsorted(sorted_keys, keys).clamp(max=n - 1)
            point[cache_key] = {
                "indices": sort_idx[positions],
                "found": sorted_keys[positions] == keys,
            }
        return point[cache_key]

    def forward(self, feat, grid_coord, batch, point):
        if self.training or feat.device.type != "cpu" or feat.dtype != torch.float32:
            raise RuntimeError("Cached CPE requires eval mode and CPU FP32 features")
        cache = self._get_neighbor_map(grid_coord, batch, point)
        neighbors = feat[cache["indices"]] * cache["found"].unsqueeze(-1)
        output = torch.einsum(
            "nki,kio->no",
            neighbors.view(feat.shape[0], self.offsets.shape[0], self.in_channels),
            self.weight,
        )
        return output if self.bias is None else output + self.bias


class SubMCPEConv(CachedCPEConv):
    """Optional NPU compute; platform support is declared by the custom OPP."""

    def __init__(self, source):
        super().__init__(source)
        self._supported_shape = self.kernel_size in (1, 3, 5) and all(
            16 <= c <= 512 and c % 16 == 0
            for c in (self.in_channels, self.out_channels)
        )
        self.register_buffer("npu_weight", None, persistent=False)
        if self._supported_shape:
            from ascend.custom_ops.submconv3d import submconv3d  # noqa: F401
        self.register_load_state_dict_post_hook(self._pack_weights)
        self._pack_weights(self, None)

    @staticmethod
    @torch.no_grad()
    def _pack_weights(module, _incompatible):
        if module._supported_shape:
            module.npu_weight = module.weight.to(NPU_DEVICE, torch.float16).contiguous()

    def forward(self, feat, grid_coord, batch, point):
        if not self._supported_shape or not 1 <= feat.shape[0] <= 4096:
            return super().forward(feat, grid_coord, batch, point)
        if self.training or feat.device.type != "cpu" or feat.dtype != torch.float32:
            raise RuntimeError("SubM CPE requires eval mode and CPU FP32 features")
        cache = self._get_neighbor_map(grid_coord, batch, point)
        n, volume = feat.shape[0], self.offsets.shape[0]
        if "npu" not in cache:
            packed = torch.full((n, (volume + 7) // 8 * 8), -1, dtype=torch.int32)
            packed[:, :volume] = cache["indices"].view(n, volume)
            packed[:, :volume].masked_fill_(~cache["found"].view(n, volume), -1)
            cache["npu"] = packed.to(NPU_DEVICE)
        output = torch.ops.graspgenx_subm.subm_conv3d(
            feat.to(NPU_DEVICE, torch.float16), cache["npu"], self.npu_weight,
        ).to(device="cpu", dtype=torch.float32)
        return output if self.bias is None else output + self.bias


class PointTransformerV3Subm(PointTransformerV3Ascend):
    """CPU geometry/map/post-ops, optional SubM compute, resident FP16 dense tail."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_config["cpe_compute"] = "npu_fp16_supported_shapes"
        for stage in self.enc:
            for block in stage.children():
                if hasattr(block, "attn"):
                    block.cpe_conv = SubMCPEConv(block.cpe_conv)
