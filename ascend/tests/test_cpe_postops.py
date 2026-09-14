"""CPE post-op units: python -B ascend/tests/test_cpe_postops.py -v.

Default execution checks CPU oracles only, without importing torch_npu or custom
ops. ASCEND_TEST_NPU=1 opts into real, already-built SubM/CANN operators; missing
runtime artifacts are errors after opting in. Source the existing environment
and set repository PYTHONPATH first. Nothing is built, compiled or installed.

Fresh random vanilla blocks are attention-wrapped before AscendBlock construction.
The local control is the old SubMCPEConv(CPU FP32) plus CPU linear/norm/residual,
not a benchmark import or a new model implementation. The folded oracle composes
CPU FP32 convolution/projection parameters before one FP16 quantization. Packing
and integer metadata stay exact; floating results require cosine >= 0.9999, with
other errors report-only. Invalid shapes/non-finite outputs fail. These synthetic
units do not replace frozen full-encoder validation.

Copy counts are Python dispatch structure, NOT DMA/kernel counts or latency
measurements. Integer metadata copies and unused FP32 LayerNorm statistics are
allowed. Map/cache/serialization coverage belongs to the existing suites.
"""

from contextlib import ExitStack
from functools import cache
import importlib
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch._dynamo  # Initialize before temporary dependency shims.
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode


TEST_NPU = os.environ.get("ASCEND_TEST_NPU") == "1"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PACKAGE = "graspgenx.models.ptv3"
SEED = 2703
COSINE_GATE = 0.9999
PACKED_PARAMETERS = {
    "cpe_conv.npu_weight": "cpe_conv.weight",
    "cpe_conv.npu_bias": "cpe_conv.bias",
    "cpe_linear_weight_npu": "cpe_linear.weight",
    "cpe_linear_bias_npu": "cpe_linear.bias",
    "cpe_norm_weight_npu": "cpe_norm.weight",
    "cpe_norm_bias_npu": "cpe_norm.bias",
}
PROJECTED_BUFFERS = ("cpe_projected_weight_npu", "cpe_projected_bias_npu")


@cache
def sources():
    # Eval-only dependency shims, like the validator, without package asset setup
    # or importing that executable validator (which imports Ascend at top level).
    class PointDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        __setattr__ = dict.__setitem__

    npu = importlib.import_module("torch_npu") if TEST_NPU else SimpleNamespace()
    shims = {
        "addict": SimpleNamespace(Dict=PointDict, __spec__=ModuleSpec("addict", None)),
        "timm": SimpleNamespace(__spec__=ModuleSpec("timm", None)),
        "timm.models.layers": SimpleNamespace(DropPath=torch.nn.Identity),
        "torch_npu": npu,
    }
    with patch.dict(sys.modules, shims):
        vanilla = SimpleNamespace(**runpy.run_path(str(
            REPO_ROOT / "graspgenx/models/ptv3/ptv3_vanilla.py"),
            run_name=f"{MODEL_PACKAGE}.ptv3_vanilla"))
        sys.modules[f"{MODEL_PACKAGE}.ptv3_vanilla"] = vanilla
        ascend = SimpleNamespace(**runpy.run_path(str(
            REPO_ROOT / "graspgenx/models/ptv3/ptv3_ascend.py"),
            run_name=f"{MODEL_PACKAGE}.ptv3_ascend"))
    return SimpleNamespace(vanilla=vanilla, ascend=ascend, npu=npu)


def make_point(vanilla, n=17, channels=32):
    grid = torch.zeros(n, 3, dtype=torch.int64)
    grid[:, 0] = torch.arange(n) - n // 2
    batch = torch.zeros(n, dtype=torch.int64)
    if n == 17:
        # Negative coordinates, different features at duplicate voxels, and the
        # same coordinates in separate contiguous batches. No serializer needed.
        grid[:8] = torch.tensor([
            [-1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0],
            [-2, 1, 0], [2, 1, 0], [0, -1, 1], [1, 1, -1],
        ])
        grid[8:16] = grid[:8]
        batch[8:] = 1
    point = vanilla.VanillaPoint(
        feat=torch.randn(n, channels), grid_coord=grid, batch=batch,
    )
    # An explicit valid order keeps the N=1/all-zero/depth-zero case independent
    # of vanilla Hilbert's depth-zero limitation. CPE never consumes these codes.
    order = torch.cat([torch.where(batch == b)[0].flip(0) for b in batch.unique()])
    point.serialized_order = order.unsqueeze(0)
    point.serialized_inverse = order.argsort().unsqueeze(0)
    point.serialized_depth = 0 if n == 1 else int(grid.abs().max()).bit_length()
    return point


def cpu_cpe_control(block, point):
    """Old route; on NPU tests only convolution has a device round trip."""
    cpe = block.cpe_conv(point.feat, point.grid_coord, point.batch, point)
    point.feat = point.feat + block.cpe_norm(block.cpe_linear(cpe))
    return point


class FloatCopies(TorchDispatchMode):
    """Observe floating CPU/NPU copy boundaries, including accidental packs."""

    def __init__(self):
        super().__init__()
        self.copies = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if str(func) in ("aten._to_copy.default", "aten.to.device", "aten.to.dtype_layout",
                         "aten.to.dtype", "aten.to.other"):
            source = args[0]
            if source.is_floating_point() and {
                source.device.type, output.device.type,
            } == {"cpu", "npu"}:
                self.copies.append((source.device.type, source.dtype,
                                    output.device.type, output.dtype, tuple(source.shape)))
        return output


class _CPEChecks(unittest.TestCase):
    def setUp(self):
        self.sources = sources()
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state())
        torch.random.default_generator.manual_seed(SEED)
        context = torch.no_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def source_block(self, channels=32):
        block = self.sources.vanilla.VanillaBlock(
            channels=channels, num_heads=2 if channels >= 32 else 1,
            patch_size=16, drop_path=0, upcast_attention=False, upcast_softmax=False,
        ).eval()
        # Nonzero biases and nonidentity gamma expose stale/missing packed values.
        block.cpe_conv.bias.normal_(0, 0.03)
        block.cpe_norm.weight.uniform_(0.8, 1.2)
        block.cpe_norm.bias.normal_(0, 0.03)
        return block

    def check_tensor(self, value, shape, device, dtype):
        self.assertEqual(tuple(value.shape), tuple(shape))
        self.assertEqual(value.device, torch.device(device))
        self.assertEqual(value.dtype, dtype)
        self.assertTrue(torch.isfinite(value).all().item(), "non-finite output")

    def check_copies(self, trace, shape, round_trips):
        expected = [
            ("cpu", torch.float32, "npu", torch.float16, tuple(shape)),
            ("npu", torch.float16, "cpu", torch.float32, tuple(shape)),
        ] * round_trips
        self.assertEqual(trace.copies, expected)

    def compare(self, actual, expected, label):
        self.check_tensor(actual, expected.shape, expected.device, expected.dtype)
        self.assertTrue(torch.isfinite(expected).all().item(), "non-finite oracle")
        a, b = actual.cpu().double().flatten(), expected.cpu().double().flatten()
        self.assertGreater(a.norm().item(), 0, "fixture output has no cosine")
        self.assertGreater(b.norm().item(), 0, "fixture oracle has no cosine")
        cosine = F.cosine_similarity(a, b, dim=0).item()
        print(json.dumps({"unit": "cpe_postops", "case": label, "cosine": cosine,
                          "cosine_gate": COSINE_GATE,
                          "max_abs_report_only": (a - b).abs().max().item()}), flush=True)
        self.assertGreaterEqual(cosine, COSINE_GATE, label)


class CPEPostOpsCPUOracleTests(_CPEChecks):
    def test_local_control_matches_original_cpu_cpe_formula(self):
        source = self.source_block()
        control = SimpleNamespace(
            cpe_conv=self.sources.ascend.CachedCPEConv(source.cpe_conv).eval(),
            cpe_linear=source.cpe_linear, cpe_norm=source.cpe_norm,
        )
        for n in (1, 17):
            with self.subTest(N=n):
                point = make_point(self.sources.vanilla, n)
                features = point.feat.clone()
                expected = features + source.cpe_norm(source.cpe_linear(
                    source.cpe_conv(features, point.grid_coord, point.batch)))
                with FloatCopies() as trace:
                    self.assertIs(cpu_cpe_control(control, point), point)
                self.check_tensor(point.feat, features.shape, "cpu", torch.float32)
                self.compare(point.feat, expected, f"cpu_control_{n}")
                self.assertEqual(trace.copies, [])


@unittest.skipUnless(TEST_NPU, "requires built SubM/CANN runtime; set ASCEND_TEST_NPU=1")
class CPEPostOpsNPUTests(_CPEChecks):
    @classmethod
    def setUpClass(cls):
        path = patch.object(sys, "path", [str(REPO_ROOT), *sys.path])
        path.start()
        cls.addClassCleanup(path.stop)
        runtime = sources()
        # Opt-in must fail, not silently skip, when the prebuilt bridge is absent.
        importlib.import_module("ascend.custom_ops.submconv3d.submconv3d")
        torch.npu.set_device(runtime.ascend.NPU_DEVICE)
        cls.addClassCleanup(torch.npu.set_compile_mode,
                            jit_compile=not torch.npu.is_jit_compile_false())
        torch.npu.set_compile_mode(jit_compile=runtime.ascend.NPU_JIT_COMPILE)

    def wrap(self, source):
        source.attn = self.sources.ascend.AscendSerializedAttention(source.attn).eval()
        return self.sources.ascend.AscendBlock(source).eval()

    def check_packed(self, block):
        buffers = dict(block.named_buffers())
        for name, parameter in PACKED_PARAMETERS.items():
            with self.subTest(buffer=name):
                self.assertIn(name, buffers)
                packed, original = buffers[name], block.get_parameter(parameter)
                self.check_tensor(packed, original.shape, self.sources.ascend.NPU_DEVICE,
                                  torch.float16)
                self.assertEqual(original.device.type, "cpu")
                self.assertEqual(original.dtype, torch.float32)
                self.assertTrue(packed.is_contiguous())
                self.assertFalse(packed.requires_grad)
                self.assertTrue(torch.equal(packed.cpu(), original.half()))
                owner, _, local_name = name.rpartition(".")
                self.assertIn(local_name, block.get_submodule(owner)._non_persistent_buffers_set)
                self.assertNotIn(name, block.state_dict())
        # Derive from checkpoint parameters, never the old half packs or the
        # candidate's packing hook. Exact equality checks the single quantization.
        weight = block.cpe_conv.weight @ block.cpe_linear.weight.T
        bias = F.linear(block.cpe_conv.bias, block.cpe_linear.weight, block.cpe_linear.bias)
        for name, expected in zip(PROJECTED_BUFFERS, (weight, bias)):
            with self.subTest(buffer=name):
                self.check_tensor(expected, expected.shape, "cpu", torch.float32)
                packed = block.get_buffer(name)
                self.assertIs(buffers[name], packed)
                self.assertNotIn(name, dict(block.named_parameters()))
                self.assertIn(name, block._non_persistent_buffers_set)
                self.assertNotIn(name, block.state_dict())
                self.check_tensor(packed, expected.shape, self.sources.ascend.NPU_DEVICE, torch.float16)
                self.assertTrue(packed.is_contiguous())
                self.assertFalse(packed.requires_grad)
                self.assertTrue(torch.equal(packed.cpu(), expected.half()), name)

    def test_construction_checkpoint_keys_cpu_parameters_rng_and_strict_reload(self):
        source = self.source_block()
        checkpoint = {key: value.clone() for key, value in source.state_dict().items()}
        originals = {name: source.get_parameter(name) for name in PACKED_PARAMETERS.values()}
        # Attention wrapping constructs fresh Linear modules and consumes RNG;
        # only subsequent resident/CPE packing is required to preserve CPU RNG.
        source.attn = self.sources.ascend.AscendSerializedAttention(source.attn).eval()
        before = torch.random.get_rng_state().clone()
        block = self.sources.ascend.AscendBlock(source).eval()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(set(block.state_dict()), set(checkpoint))
        self.check_packed(block)
        for name, original in originals.items():
            self.assertIs(block.get_parameter(name), original)
            self.assertTrue(torch.equal(original, checkpoint[name]), name)

        previous = {name: block.get_buffer(name).cpu().clone()
                    for name in (*PACKED_PARAMETERS, *PROJECTED_BUFFERS)}
        for name in originals:
            checkpoint[name].mul_(0.75).add_(0.0625)
        before = torch.random.get_rng_state().clone()
        incompatible = block.load_state_dict(checkpoint, strict=True)
        self.assertFalse(incompatible.missing_keys or incompatible.unexpected_keys)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(set(block.state_dict()), set(checkpoint))
        self.check_packed(block)
        for name, original in originals.items():
            self.assertIs(block.get_parameter(name), original)
            self.assertTrue(torch.equal(original, checkpoint[name]), name)
        for name, old in previous.items():
            self.assertFalse(torch.equal(block.get_buffer(name).cpu(), old), name)

        # A strict reload must retain normal missing/unexpected-key diagnostics.
        with self.assertRaises(RuntimeError):
            block.load_state_dict({k: v for k, v in checkpoint.items()
                                   if k != "cpe_linear.weight"}, strict=True)
        with self.assertRaises(RuntimeError):
            block.load_state_dict({**checkpoint, "unexpected": torch.zeros(1)}, strict=True)
        for name in PROJECTED_BUFFERS:
            with self.subTest(unexpected_buffer=name), self.assertRaises(RuntimeError):
                block.load_state_dict({**checkpoint, name: block.get_buffer(name)}, strict=True)

    def test_supported_cpe_matches_explicit_half_ops_and_uses_half_parameters(self):
        block = self.wrap(self.source_block())
        npu = self.sources.npu
        for n in (1, 15, 16, 17, 33, 4096):
            with self.subTest(N=n):
                point = make_point(self.sources.vanilla, n)
                features = point.feat.clone()
                metadata = {key: value.clone() for key, value in point.items()
                            if isinstance(value, torch.Tensor) and key != "feat"}
                half = features.to(self.sources.ascend.NPU_DEVICE, torch.float16)
                neighbors = block.cpe_conv._get_npu_map(point.grid_coord, point.batch, point)
                weight = (block.cpe_conv.weight @ block.cpe_linear.weight.T).to(
                    half.device, torch.float16)
                bias = F.linear(block.cpe_conv.bias, block.cpe_linear.weight,
                                block.cpe_linear.bias).to(half.device, torch.float16)
                convolution = torch.ops.graspgenx_subm.subm_conv3d(
                    half, neighbors, weight) + bias
                expected = half + npu.npu_layer_norm_eval(
                    convolution, block.cpe_norm.normalized_shape,
                    block.cpe_norm_weight_npu, block.cpe_norm_bias_npu, block.cpe_norm.eps)

                with patch.object(block.cpe_linear, "forward", side_effect=AssertionError(
                    "supported CPE used CPU linear")), patch.object(
                    block.cpe_norm, "forward", side_effect=AssertionError("supported CPE used CPU norm")), \
                        patch.object(F, "linear", side_effect=AssertionError(
                            "folded CPE performed a separate linear")) as linear_op, \
                        patch.object(block.cpe_conv, "forward", wraps=block.cpe_conv.forward) as conv_op, \
                        patch.object(torch.ops.graspgenx_subm, "subm_conv3d",
                                     wraps=torch.ops.graspgenx_subm.subm_conv3d) as kernel, \
                        patch.object(npu, "npu_layer_norm_eval", wraps=npu.npu_layer_norm_eval) as norm_op:
                    self.assertIs(block.forward_cpe(point), point)
                self.check_tensor(point.feat, features.shape, self.sources.ascend.NPU_DEVICE,
                                  torch.float16)
                self.compare(point.feat, expected, f"folded_half_ops_{n}")
                linear_op.assert_not_called()
                conv_op.assert_called_once()
                self.assertIs(conv_op.call_args.kwargs["projected_weight"], block.cpe_projected_weight_npu)
                self.assertIs(conv_op.call_args.kwargs["projected_bias"], block.cpe_projected_bias_npu)
                kernel.assert_called_once()
                self.assertIs(kernel.call_args.args[1], neighbors)
                self.assertIs(kernel.call_args.args[2], block.cpe_projected_weight_npu)
                norm_op.assert_called_once()
                for value in (kernel.call_args.args[0], kernel.call_args.args[2],
                              conv_op.call_args.kwargs["projected_bias"]):
                    self.assertEqual(value.dtype, torch.float16)
                    self.assertEqual(value.device, half.device)
                for index in (0, 2, 3):
                    value = norm_op.call_args.args[index]
                    self.assertEqual(value.dtype, torch.float16)
                    self.assertEqual(value.device, half.device)
                for key, value in metadata.items():
                    self.assertEqual(point[key].device.type, "cpu", key)
                    self.assertEqual(point[key].dtype, value.dtype, key)
                    self.assertTrue(torch.equal(point[key], value), key)

                if n in (1, 17):
                    control = self.sources.vanilla.VanillaPoint(dict(point, feat=features))
                    cpu_cpe_control(block, control)
                    self.check_tensor(control.feat, features.shape, "cpu", torch.float32)
                    self.compare(point.feat.cpu().float(), control.feat,
                                 f"folded_vs_SubMCPEConv_CPU32_plus_CPU_postops_{n}")

    def test_raw_convolution_preserves_cpu_control_bias_and_accepts_npu_half(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                source = self.sources.vanilla.HashSparseConv3d(16, 16, 3, bias=bias).eval()
                if bias:
                    source.bias.normal_(0, 0.03)
                saved = {key: value.clone() for key, value in source.state_dict().items()}
                conv = self.sources.ascend.SubMCPEConv(source).eval()
                point = make_point(self.sources.vanilla, channels=16)
                half = point.feat.to(self.sources.ascend.NPU_DEVICE, torch.float16)
                neighbors = conv._get_npu_map(point.grid_coord, point.batch, point)
                raw = torch.ops.graspgenx_subm.subm_conv3d(half, neighbors, conv.npu_weight)
                expected_cpu = raw.to(device="cpu", dtype=torch.float32)
                expected_half = raw
                if bias:
                    expected_cpu = expected_cpu + source.bias
                    expected_half = expected_half + conv.npu_bias
                else:
                    self.assertIsNone(conv.npu_bias)
                for features, expected, device, dtype in (
                    (point.feat, expected_cpu, "cpu", torch.float32),
                    (half, expected_half, self.sources.ascend.NPU_DEVICE, torch.float16),
                ):
                    actual = conv(features, point.grid_coord, point.batch, point)
                    self.check_tensor(actual, point.feat.shape, device, dtype)
                    self.compare(actual, expected, f"raw_conv_{bias}_{device}")
                projected_weight = (source.weight * 0.625).to(half.device, torch.float16)
                projected = torch.ops.graspgenx_subm.subm_conv3d(half, neighbors, projected_weight)
                for projected_bias in (None, torch.full((16,), 0.125, device=half.device, dtype=half.dtype)):
                    expected = projected if projected_bias is None else projected + projected_bias
                    actual = conv(half, point.grid_coord, point.batch, point,
                                  projected_weight=projected_weight, projected_bias=projected_bias)
                    self.compare(actual, expected, f"projected_conv_{bias}_{projected_bias is not None}")
                # Projected arguments must not change the raw CPU control or packs.
                self.compare(conv(point.feat, point.grid_coord, point.batch, point), expected_cpu,
                             f"raw_cpu_after_projected_{bias}")
                self.assertTrue(torch.equal(conv.npu_weight.cpu(), source.weight.half()))
                self.assertIs(conv.weight, source.weight)
                self.assertIs(conv.bias, source.bias)
                self.assertEqual(set(conv.state_dict()), set(saved))
                for name, value in saved.items():
                    self.assertTrue(torch.equal(conv.state_dict()[name], value), name)

    def test_cached_cpu_control_replacement_supports_strict_reload(self):
        block = self.wrap(self.source_block())
        state = {name: value.clone() for name, value in block.state_dict().items()}
        block.cpe_conv = self.sources.ascend.CachedCPEConv(block.cpe_conv).eval()
        block.load_state_dict(state, strict=True)
        point = make_point(self.sources.vanilla)
        original = point.feat.clone()
        expected = original + block.cpe_norm(block.cpe_linear(block.cpe_conv(
            original, point.grid_coord, point.batch, {})))
        result = block.forward_cpe(point).feat
        self.check_tensor(result, original.shape, "cpu", torch.float32)
        self.compare(result, expected, "reloaded_cached_cpu_control")
        self.assertEqual(set(block.state_dict()), set(state))

    def test_cpe_attention_ffn_route_and_attention_cpu_entry(self):
        block = self.wrap(self.source_block())
        for n in (1, 17, 32):
            with self.subTest(N=n):
                point = make_point(self.sources.vanilla, n)
                shape = point.feat.shape
                block.forward_cpe(point)
                self.check_tensor(point.feat, shape, self.sources.ascend.NPU_DEVICE, torch.float16)
                cpe_features = point.feat
                with torch.inference_mode(), FloatCopies() as trace:
                    self.assertIs(block.forward_attention(point), point)
                self.assertEqual(trace.copies, [], "resident attention copied features")
                self.assertIs(point["_attention_residual"], cpe_features)
                self.check_tensor(point.feat, shape, self.sources.ascend.NPU_DEVICE, torch.float16)
                self.assertIs(block.forward_ffn(point), point)
                self.check_tensor(point.feat, shape, "cpu", torch.float32)
                self.assertNotIn("_attention_residual", point)

        point = make_point(self.sources.vanilla)
        with torch.inference_mode(), FloatCopies() as trace:
            block.forward_attention(point)
            block.forward_ffn(point)
        self.check_copies(trace, (17, 32), round_trips=1)
        self.check_tensor(point.feat, (17, 32), "cpu", torch.float32)

    def test_block_copy_counts_against_local_control_and_no_forward_repacking(self):
        block = self.wrap(self.source_block())
        packed = {name: block.get_buffer(name) for name in (*PACKED_PARAMETERS, *PROJECTED_BUFFERS)}
        originals = {name: block.get_parameter(name).clone() for name in PACKED_PARAMETERS.values()}
        features = torch.randn(17, 32)
        for repeat in range(2):
            for control in (False, True):
                with self.subTest(repeat=repeat, cpu_postops=control):
                    point = make_point(self.sources.vanilla)
                    point.feat = features.clone() + repeat * 0.125
                    with torch.inference_mode(), FloatCopies() as trace:
                        if control:
                            cpu_cpe_control(block, point)
                            block.forward_attention(point)
                            block.forward_ffn(point)
                        else:
                            self.assertIs(block(point), point)
                    self.check_copies(trace, features.shape, round_trips=2 if control else 1)
                    self.check_tensor(point.feat, features.shape, "cpu", torch.float32)
                    for name, value in packed.items():
                        self.assertIs(block.get_buffer(name), value, "per-forward repack: " + name)
                    for name, value in originals.items():
                        self.assertTrue(torch.equal(block.get_parameter(name), value), name)

    def test_unsupported_channels_and_n4097_use_original_cpu_postops(self):
        for channels, n in ((8, 17), (16, 4097)):
            with self.subTest(C=channels, N=n):
                source = self.source_block(channels)
                block = self.wrap(source)
                point = make_point(self.sources.vanilla, n, channels)
                features = point.feat.clone()
                expected = features + source.cpe_norm(source.cpe_linear(
                    source.cpe_conv(features, point.grid_coord, point.batch)))
                if channels == 8:
                    self.assertFalse(block.cpe_conv._supported_shape)
                    self.assertIsNone(block.cpe_conv.npu_weight)
                    self.assertIsNone(block.cpe_conv.npu_bias)
                with ExitStack() as stack:
                    guards = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
                        "CPU fallback reached " + name))) for owner, name in (
                            (block.cpe_conv, "_get_npu_map"),
                            (torch.ops.graspgenx_subm, "subm_conv3d"),
                            (self.sources.npu, "npu_layer_norm_eval"),
                        )]
                    linear = stack.enter_context(patch.object(
                        block.cpe_linear, "forward", wraps=block.cpe_linear.forward))
                    norm = stack.enter_context(patch.object(
                        block.cpe_norm, "forward", wraps=block.cpe_norm.forward))
                    with FloatCopies() as trace:
                        self.assertIs(block.forward_cpe(point), point)
                    self.assertEqual(trace.copies, [])
                    linear.assert_called_once()
                    norm.assert_called_once()
                    for guard in guards:
                        guard.assert_not_called()
                self.check_tensor(point.feat, features.shape, "cpu", torch.float32)
                self.compare(point.feat, expected, f"cpu_shape_fallback_{channels}_{n}")
                self.assertIs(block.cpe_linear, source.cpe_linear)
                self.assertIs(block.cpe_norm, source.cpe_norm)
                self.assertNotIn("_cpe_npu_map_3", point)
                half = features.to(self.sources.ascend.NPU_DEVICE, torch.float16)
                with self.assertRaises(RuntimeError):
                    block.cpe_conv(half, point.grid_coord, point.batch, point)

    def test_invalid_dtypes_wrong_device_and_training_fail_before_compute(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        block = self.wrap(self.source_block())
        point = make_point(self.sources.vanilla)
        cpu = point.feat
        half = cpu.to(self.sources.ascend.NPU_DEVICE, torch.float16)
        wrong_device = torch.device("npu", 1 if half.device.index == 0 else 0)
        # Test the device guard without allocating on/requiring a second NPU.
        wrong = FakeTensor(FakeTensorMode(), torch.empty(cpu.shape, device="meta", dtype=torch.float16),
                           wrong_device)
        invalid = (cpu.half(), cpu.double(), half.float(), wrong)
        with ExitStack() as stack:
            for owner, name in ((block.cpe_conv, "_get_npu_map"),
                                (self.sources.ascend.CachedCPEConv, "forward"),
                                (block.attn, "prepare_indices"),
                                (self.sources.npu, "npu_layer_norm_eval")):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
                    "invalid input reached " + name)))
            for features in invalid:
                for entry in ("conv", "cpe", "attention"):
                    with self.subTest(entry=entry, device=str(features.device), dtype=features.dtype):
                        point.feat = features
                        with self.assertRaises(RuntimeError):
                            if entry == "conv":
                                block.cpe_conv(features, point.grid_coord, point.batch, point)
                            elif entry == "cpe":
                                block.forward_cpe(point)
                            else:
                                block.forward_attention(point)
            # Only attention and raw SubMCPEConv accept an already-resident
            # tensor; the block's CPE entry still follows the CPU stage boundary.
            point.feat = half
            with self.assertRaises(RuntimeError):
                block.forward_cpe(point)
            with self.assertRaisesRegex(RuntimeError, "Projected CPE weights require NPU FP16"):
                block.cpe_conv(cpu, point.grid_coord, point.batch, point,
                               projected_weight=block.cpe_projected_weight_npu,
                               projected_bias=block.cpe_projected_bias_npu)
            block.train()
            point.feat = cpu
            for forward in (block.forward_cpe, block.forward_attention, block.forward):
                with self.subTest(training=forward.__name__), self.assertRaises(RuntimeError):
                    forward(point)
            for features in (cpu, half):
                with self.subTest(training="conv", device=features.device), self.assertRaises(RuntimeError):
                    block.cpe_conv(features, point.grid_coord, point.batch, point)

    def test_custom_convolution_and_layernorm_errors_propagate_without_cpu_retry(self):
        block = self.wrap(self.source_block())
        for owner, name in ((torch.ops.graspgenx_subm, "subm_conv3d"),
                            (self.sources.npu, "npu_layer_norm_eval")):
            with self.subTest(op=name), ExitStack() as stack:
                point = make_point(self.sources.vanilla)
                sentinel = RuntimeError("sentinel " + name)
                guards = [stack.enter_context(patch.object(module, "forward", side_effect=AssertionError(
                    "NPU error retried CPU CPE"))) for module in (
                        self.sources.ascend.CachedCPEConv, block.cpe_linear, block.cpe_norm)]
                op = stack.enter_context(patch.object(owner, name, side_effect=sentinel))
                with self.assertRaises(RuntimeError) as raised:
                    block.forward_cpe(point)
                self.assertIs(raised.exception, sentinel)
                op.assert_called_once()
                for guard in guards:
                    guard.assert_not_called()

    def test_tiny_model_wiring_cpu_stem_and_output_without_golden_accuracy_claim(self):
        config = dict(in_channels=3, output_dim=32, order=("z",), stride=(),
                      enc_depths=(1,), enc_channels=(32,), enc_num_head=(2,),
                      enc_patch_size=(16,), drop_path=0, shuffle_orders=False,
                      upcast_attention=False, upcast_softmax=False)
        reference = self.sources.vanilla.PointTransformerV3Vanilla(**config).eval()
        checkpoint = reference.state_dict()
        model = self.sources.ascend.PointTransformerV3Ascend(**config).eval()
        self.assertEqual(set(model.state_dict()), set(checkpoint))
        model.load_state_dict(checkpoint, strict=True)
        block = model.enc.enc0.block0
        self.assertIsInstance(block, self.sources.ascend.AscendBlock)
        self.assertIsInstance(block.attn, self.sources.ascend.AscendSerializedAttention)
        self.check_packed(block)
        self.assertIsInstance(model.embedding.conv, self.sources.vanilla.HashSparseConv3d)
        self.assertEqual(model.embedding.conv.kernel_size, 5)
        self.assertEqual(model.embedding.conv.in_channels, 3)
        self.assertEqual(model.embedding.conv.weight.device.type, "cpu")
        self.assertEqual(model.embedding.conv.weight.dtype, torch.float32)
        grid = torch.zeros(17, 3, dtype=torch.int64)
        grid[:, 0] = torch.arange(17)
        data = dict(feat=torch.randn(17, 3), grid_coord=grid, offset=torch.tensor([17]))
        before = {key: value.clone() for key, value in data.items()}
        self.check_tensor(model(data), (1, 32), "cpu", torch.float32)
        self.assertEqual(set(model.state_dict()), set(checkpoint))
        self.assertEqual(set(data), set(before))
        for key, value in before.items():
            self.assertTrue(torch.equal(data[key], value), key)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
