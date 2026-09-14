"""Regression units for performance paths, not latency benchmarks.

From the repository root:
    source ascend/env.sh
    ASCEND_TEST_NPU=1 python -B ascend/tests/test_performance.py -v

Without ASCEND_TEST_NPU=1, only CPU tests run, using the existing CPE suite's
temporary import shims. Opt-in requires the real, already-built SubM/CANN runtime;
missing artifacts or runtime failures are errors, not skips. Nothing is built or
installed. Small synthetic fixtures do not replace frozen full-encoder validation.
Integer metadata and packing/gather invariants are exact. Numerical agreement is
gated only by cosine >= 0.9999, with valid finite shapes required; absolute errors
are report-only. Attention construction intentionally consumes CPU RNG, unlike
the CPU pooling/stem wrappers and checkpoint reloads.
"""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode


# Support both direct script execution and unittest discovery without importing
# the validator or collecting the helper module's test classes into this suite.
with patch.object(sys, "path", [str(Path(__file__).resolve().parent), *sys.path]):
    from test_cpe_postops import REPO_ROOT, TEST_NPU, make_point, sources


COSINE_GATE = 0.9999
SEED = 3703
METADATA = (
    "grid_coord", "batch", "offset", "serialized_code",
    "serialized_order", "serialized_inverse",
)


def pooling_point(vanilla, dtype=torch.int64):
    grid = torch.tensor([
        [0, 0, 0], [1, 0, 1], [1, 0, 1], [2, 1, 0],
        [3, 1, 1], [6, 2, 3], [7, 2, 3], [4, 7, 5],
    ], dtype=dtype)
    grid = torch.cat((grid, grid[:7], grid[:5]))
    point = vanilla.VanillaPoint(
        feat=torch.randn(20, 4), grid_coord=grid,
        coord=grid.float() * 0.25 + torch.rand(20, 3) * 0.125,
        offset=torch.tensor([8, 15, 20]),
        condition=("unit",), context=torch.randn(3, 4),
    )
    point.serialization(order=("z", "z-trans", "hilbert", "hilbert-trans"))
    return point


def attention_point(vanilla, counts):
    point = vanilla.VanillaPoint(
        feat=torch.randn(sum(counts), 32), offset=torch.tensor(counts).cumsum(0),
    )
    rows = torch.arange(sum(counts)).split(counts)
    point.serialized_order = torch.stack((
        torch.cat([row.flip(0) for row in rows]),
        torch.cat([row.roll(1) for row in rows]),
    ))
    point.serialized_inverse = point.serialized_order.argsort(dim=1)
    return point


class _StemTrace(TorchDispatchMode):
    """Observe the real gather and flattened GEMM, including the missing mask."""

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.indices = []
        self.operands = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.index.Tensor and args[0] is self.features:
            self.indices.append(args[1][0])
        if func == torch.ops.aten.mm.default:
            self.operands.append(args[:2])
        return func(*args, **(kwargs or {}))


class _PoolTrace(TorchDispatchMode):
    """Python API spies cannot see reductions inside the scripted helper."""

    def __init__(self):
        super().__init__()
        self.reductions = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.segment_reduce.default:
            self.reductions.append((args[0], args[1], kwargs["offsets"]))
        return func(*args, **(kwargs or {}))


class _PerformanceChecks(unittest.TestCase):
    def setUp(self):
        self.sources = sources()
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state())
        torch.random.default_generator.manual_seed(SEED)
        context = torch.no_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def exact(self, actual, expected, label=""):
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertEqual(actual.dtype, expected.dtype, label)
        self.assertEqual(actual.device.type, "cpu", label)
        self.assertTrue(torch.equal(actual, expected), label)

    def valid(self, value, shape, device="cpu", dtype=torch.float32):
        self.assertEqual(tuple(value.shape), tuple(shape))
        self.assertEqual(value.device, torch.device(device))
        self.assertEqual(value.dtype, dtype)
        self.assertTrue(torch.isfinite(value).all().item(), "non-finite output")

    def compare(self, actual, expected, label):
        self.valid(actual, expected.shape, expected.device, expected.dtype)
        self.assertTrue(torch.isfinite(expected).all().item(), "non-finite oracle")
        a, b = actual.cpu().double().flatten(), expected.cpu().double().flatten()
        self.assertGreater(a.norm().item(), 0, "fixture output has no cosine")
        self.assertGreater(b.norm().item(), 0, "fixture oracle has no cosine")
        cosine = F.cosine_similarity(a, b, dim=0).item()
        print(json.dumps({"unit": "performance_paths", "case": label,
                          "cosine": cosine, "cosine_gate": COSINE_GATE,
                          "max_abs_report_only": (a - b).abs().max().item()}), flush=True)
        self.assertGreaterEqual(cosine, COSINE_GATE, label)

    def metadata(self, actual, expected):
        self.assertEqual(actual.serialized_depth, expected.serialized_depth)
        self.assertIs(type(actual.serialized_depth), int)
        for key in METADATA:
            self.exact(actual[key], expected[key], key)
        positions = torch.arange(len(actual.batch)).expand_as(actual.serialized_order)
        self.exact(actual.serialized_order.gather(1, actual.serialized_inverse), positions)
        self.exact(actual.serialized_inverse.gather(1, actual.serialized_order), positions)

    def source_attention(self, bias=True, num_heads=4):
        source = self.sources.vanilla.VanillaSerializedAttention(
            channels=32, num_heads=num_heads, patch_size=16, qkv_bias=bias,
            qk_scale=0.37, upcast_attention=False, upcast_softmax=False,
        ).eval()
        if bias:
            source.qkv.bias.uniform_(0.05, 0.25)
        return source

    def check_pool_packed(self, pooling, fused):
        expected = (None, None)
        if fused:
            norm = pooling.norm
            scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
            bias = pooling.proj.bias if pooling.proj.bias is not None else 0
            expected = (pooling.proj.weight * scale[:, None],
                        (bias - norm.running_mean) * scale + norm.bias)
        for name, value in zip(("fused_proj_weight", "fused_proj_bias"), expected):
            self.assertIn(name, pooling._non_persistent_buffers_set)
            self.assertNotIn(name, pooling.state_dict())
            self.assertNotIn(name, dict(pooling.named_parameters()))
            packed = getattr(pooling, name)
            if value is None:
                self.assertIsNone(packed)
                self.assertNotIn(name, dict(pooling.named_buffers()))
            else:
                self.assertIs(dict(pooling.named_buffers())[name], packed)
                self.valid(packed, value.shape)
                self.assertFalse(packed.requires_grad)
                self.assertTrue(packed.is_contiguous())
                self.exact(packed, value, name)

    def explicit_indices(self, attention, point):
        # Force the unchanged padding/permutation path on a separate point, even
        # for single patches. Never use candidate prepare_indices as the oracle.
        oracle = self.sources.vanilla.VanillaPoint(dict(point))
        counts = torch.diff(point.offset, prepend=point.offset.new_zeros(1))
        attention.patch_size = min(int(counts.min()), attention.patch_size_max)
        pad, unpad, _ = self.sources.vanilla.VanillaSerializedAttention.get_padding_and_inverse(
            attention, oracle)
        return (point.serialized_order[attention.order_index][pad],
                unpad[point.serialized_inverse[attention.order_index]])


class PerformanceCPUTests(_PerformanceChecks):
    def test_pooling_all_reducers_exact_multibatch_duplicate_metadata(self):
        scripted = self.sources.ascend._pool_tensors
        self.assertIsInstance(scripted, torch.jit.ScriptFunction)
        for dtype in (torch.int32, torch.int64):
            point = pooling_point(self.sources.vanilla, dtype)
            for reducer in ("sum", "mean", "min", "max"):
                with self.subTest(dtype=dtype, reducer=reducer):
                    source = self.sources.vanilla.VanillaSerializedPooling(
                        4, 6, reduce=reducer, shuffle_orders=False).eval()
                    candidate = self.sources.ascend.AscendSerializedPooling(source).eval()
                    expected = source(point)
                    helper = Mock(wraps=scripted)
                    with _PoolTrace() as trace, patch.dict(candidate.forward.__globals__,
                                                          {"_pool_tensors": helper}):
                        actual = candidate(point)
                    helper.assert_called_once()
                    for passed, original in zip(helper.call_args.args[:7], (
                            point.feat, point.coord, point.grid_coord, point.batch,
                            point.serialized_code, source.proj.weight, source.proj.bias)):
                        self.assertIs(passed, original)
                    self.assertEqual(helper.call_args.args[7:], (1, reducer, False))
                    self.assertEqual(len(trace.reductions), 2)
                    self.assertEqual([call[1] for call in trace.reductions], [reducer, "mean"])
                    cluster = expected.pooling_inverse
                    indices = cluster.argsort()
                    offsets = F.pad(cluster.bincount().cumsum(0), (1, 0))
                    for data, _, actual_offsets in trace.reductions:
                        self.valid(data, data.shape)
                        self.exact(actual_offsets, offsets, "scripted CSR offsets")
                    self.compare(trace.reductions[0][0], source.proj(point.feat)[indices],
                                 f"scripted_projection_{reducer}_{dtype}")
                    self.exact(trace.reductions[1][0], point.coord[indices], "coordinate gather")
                    self.metadata(actual, expected)
                    self.exact(actual.pooling_inverse, expected.pooling_inverse)
                    self.assertIs(actual.pooling_parent, point)
                    self.compare(actual.feat, expected.feat, f"pool_{reducer}_{dtype}")
                    self.compare(actual.coord, expected.coord, f"coord_{reducer}_{dtype}")
                    self.assertEqual(actual.offset.numel(), 3)
                    self.assertLess(len(actual.feat), len(point.feat))

    def test_pooling_shuffle_rng_traceability_context_and_depth_fallback(self):
        point = pooling_point(self.sources.vanilla)
        point["_cpe_normalized"] = object()
        before = {key: value.clone() for key, value in point.items()
                  if isinstance(value, torch.Tensor)}
        keys = set(point)
        for stride, reducer in ((1, "mean"), (2, "mean"), (16, "mean"),
                                (2, "max"), (2, "min")):
            for traceable in (False, True):
                for shuffle in (False, True):
                    with self.subTest(stride=stride, reducer=reducer, trace=traceable, shuffle=shuffle):
                        source = self.sources.vanilla.VanillaSerializedPooling(
                            4, 6, stride=stride, reduce=reducer, traceable=traceable,
                            shuffle_orders=shuffle, norm_layer=torch.nn.BatchNorm1d,
                            act_layer=torch.nn.GELU).eval()
                        source.norm.weight.uniform_(0.7, 1.3)
                        source.norm.running_mean.normal_(0, 0.2)
                        candidate = self.sources.ascend.AscendSerializedPooling(source).eval()
                        rng = torch.random.get_rng_state()
                        expected = source(point)
                        expected_rng = torch.random.get_rng_state()
                        torch.random.set_rng_state(rng)
                        actual = candidate(point)
                        self.exact(torch.random.get_rng_state(), expected_rng, "shuffle RNG")
                        self.assertEqual(torch.equal(rng, expected_rng), not shuffle)
                        self.metadata(actual, expected)
                        self.compare(actual.feat, expected.feat,
                                     f"pool_{reducer}_s{stride}_{traceable}_{shuffle}")
                        self.compare(actual.coord, expected.coord,
                                     f"coord_{reducer}_s{stride}_{traceable}_{shuffle}")
                        self.assertEqual(actual.serialized_depth,
                                         point.serialized_depth - (1 if stride == 2 else 0))
                        for key in ("context", "condition"):
                            self.assertIs(actual[key], point[key])
                        self.assertNotIn("_cpe_normalized", actual)
                        if traceable:
                            self.assertIs(actual.pooling_parent, point)
                            self.exact(actual.pooling_inverse, expected.pooling_inverse)
                        else:
                            self.assertNotIn("pooling_parent", actual)
                            self.assertNotIn("pooling_inverse", actual)
                        self.assertEqual(set(point), keys)
                        for key, value in before.items():
                            self.exact(point[key], value, f"input {key}")

    def test_pooling_wrap_and_strict_reload_preserve_keys_parameters_and_rng(self):
        source = self.sources.vanilla.VanillaSerializedPooling(
            4, 6, norm_layer=torch.nn.BatchNorm1d, act_layer=torch.nn.GELU).eval()
        checkpoint = {key: value.clone() for key, value in source.state_dict().items()}
        parameters = dict(source.named_parameters())
        rng = torch.random.get_rng_state()
        candidate = self.sources.ascend.AscendSerializedPooling(source).eval()
        self.exact(torch.random.get_rng_state(), rng, "wrapping RNG")
        self.check_pool_packed(candidate, True)
        previous = candidate.fused_proj_weight.clone(), candidate.fused_proj_bias.clone()
        for name in ("proj", "norm", "act"):
            self.assertIs(getattr(candidate, name), getattr(source, name))
        self.assertEqual(set(candidate.state_dict()), set(checkpoint))
        for key, value in checkpoint.items():
            self.exact(candidate.state_dict()[key], value, key)
            value.add_(0.125 if value.is_floating_point() else 1)
        incompatible = candidate.load_state_dict(checkpoint, strict=True)
        self.assertFalse(incompatible.missing_keys or incompatible.unexpected_keys)
        self.exact(torch.random.get_rng_state(), rng, "reload RNG")
        self.assertEqual(set(candidate.state_dict()), set(checkpoint))
        self.check_pool_packed(candidate, True)
        for name, old in zip(("fused_proj_weight", "fused_proj_bias"), previous):
            self.assertFalse(torch.equal(getattr(candidate, name), old), name)
        for name, parameter in parameters.items():
            self.assertIs(candidate.get_parameter(name), parameter)
            self.assertEqual(parameter.device.type, "cpu")
            self.assertEqual(parameter.dtype, torch.float32)
        for key, value in checkpoint.items():
            self.exact(candidate.state_dict()[key], value, key)
        point = pooling_point(self.sources.vanilla)
        for negative in (True, False):
            checkpoint["norm.weight"][0] = -0.5 if negative else 0.75
            rng = torch.random.get_rng_state()
            incompatible = candidate.load_state_dict(checkpoint, strict=True)
            self.assertFalse(incompatible.missing_keys or incompatible.unexpected_keys)
            self.exact(torch.random.get_rng_state(), rng, "BN fold reload RNG")
            self.assertEqual(set(candidate.state_dict()), set(checkpoint))
            self.check_pool_packed(candidate, not negative)
            expected = source(point)
            expected_rng = torch.random.get_rng_state()
            torch.random.set_rng_state(rng)
            actual = candidate(point)
            self.exact(torch.random.get_rng_state(), expected_rng, "reload forward RNG")
            self.metadata(actual, expected)
            self.compare(actual.feat, expected.feat, f"pool_reload_negative_{negative}")
            for name, parameter in parameters.items():
                self.assertIs(candidate.get_parameter(name), parameter)
                self.exact(parameter, checkpoint[name], name)
        for invalid in ({k: v for k, v in checkpoint.items() if k != "proj.weight"},
                        {**checkpoint, "unexpected": torch.zeros(1)},
                        {**checkpoint, "fused_proj_weight": candidate.fused_proj_weight}):
            with self.assertRaises(RuntimeError):
                candidate.load_state_dict(invalid, strict=True)

    def test_pooling_bn_fold_positive_zero_gamma_and_fallbacks(self):
        cases = (
            ("positive_max", "max", torch.nn.BatchNorm1d, True),
            ("positive_min", "min", torch.nn.BatchNorm1d, True),
            ("zero_gamma", "max", torch.nn.BatchNorm1d, True),
            ("negative_max", "max", torch.nn.BatchNorm1d, False),
            ("negative_min", "min", torch.nn.BatchNorm1d, False),
            ("sum", "sum", torch.nn.BatchNorm1d, False),
            ("mean", "mean", torch.nn.BatchNorm1d, False),
            ("nonaffine", "max", lambda c: torch.nn.BatchNorm1d(c, affine=False), False),
            ("stateless", "max", lambda c: torch.nn.BatchNorm1d(c, track_running_stats=False), False),
            ("layernorm", "max", torch.nn.LayerNorm, False),
            ("no_norm", "max", None, False),
        )
        point = pooling_point(self.sources.vanilla)
        for name, reducer, norm, fused in cases:
            with self.subTest(case=name):
                source = self.sources.vanilla.VanillaSerializedPooling(
                    4, 6, reduce=reducer, shuffle_orders=False,
                    norm_layer=norm, act_layer=torch.nn.GELU).eval()
                if isinstance(source.norm, torch.nn.BatchNorm1d):
                    if source.norm.affine:
                        source.norm.weight.uniform_(0.4, 1.7)
                        source.norm.bias.uniform_(-0.3, 0.2)
                        if name.startswith("negative"):
                            source.norm.weight[0] = -0.75
                        elif name == "zero_gamma":
                            source.norm.weight[0] = 0
                            source.proj.register_parameter("bias", None)
                    if source.norm.track_running_stats:
                        source.norm.running_mean.uniform_(-0.4, 0.5)
                        source.norm.running_var.uniform_(0.3, 1.5)
                saved = {key: value.clone() for key, value in source.state_dict().items()}
                candidate = self.sources.ascend.AscendSerializedPooling(source).eval()
                self.check_pool_packed(candidate, fused)
                expected = source(point)
                helper = Mock(wraps=self.sources.ascend._pool_tensors)
                with ExitStack() as stack:
                    stack.enter_context(patch.dict(candidate.forward.__globals__, {"_pool_tensors": helper}))
                    norm_call = None if norm is None else stack.enter_context(patch.object(
                        candidate.norm, "forward", wraps=candidate.norm.forward))
                    actual = candidate(point)
                helper.assert_called_once()
                self.assertIs(helper.call_args.args[5],
                              candidate.fused_proj_weight if fused else source.proj.weight)
                self.assertIs(helper.call_args.args[6],
                              candidate.fused_proj_bias if fused else source.proj.bias)
                if norm_call is not None:
                    self.assertEqual(norm_call.call_count, 0 if fused else 1)
                self.metadata(actual, expected)
                self.compare(actual.feat, expected.feat, f"bn_fold_{name}")
                self.compare(actual.coord, expected.coord, f"bn_coord_{name}")
                self.assertEqual(set(candidate.state_dict()), set(saved))
                for key, value in saved.items():
                    self.exact(candidate.state_dict()[key], value, key)

    def test_pooling_training_bypasses_fold_and_preserves_bn_stats_and_gradients(self):
        for mode in ("pooling", "pooling_only", "norm_only", "eval_grad"):
            with self.subTest(training=mode):
                source = self.sources.vanilla.VanillaSerializedPooling(
                    4, 6, shuffle_orders=False, norm_layer=torch.nn.BatchNorm1d,
                    act_layer=torch.nn.GELU).eval()
                # A pre-BN bias in training has zero analytical gradient, so its
                # roundoff residue is not a meaningful cosine fixture.
                if mode != "eval_grad":
                    source.proj.register_parameter("bias", None)
                source.norm.running_mean.uniform_(-0.4, 0.5)
                source.norm.running_var.uniform_(0.3, 1.5)
                reference = deepcopy(source)
                candidate = self.sources.ascend.AscendSerializedPooling(source).eval()
                self.check_pool_packed(candidate, True)
                for pooling in (candidate, reference):
                    if mode != "eval_grad":
                        (pooling.norm if mode == "norm_only" else pooling).train()
                    if mode == "pooling_only":
                        pooling.norm.eval()
                    if mode == "eval_grad":
                        self.assertFalse(pooling.training)
                        self.assertFalse(pooling.norm.training)
                point = pooling_point(self.sources.vanilla)
                point.feat.requires_grad_()
                oracle = self.sources.vanilla.VanillaPoint(dict(
                    point, feat=point.feat.detach().clone().requires_grad_()))
                helper = Mock(wraps=self.sources.ascend._pool_tensors)
                with torch.enable_grad(), patch.dict(candidate.forward.__globals__, {"_pool_tensors": helper}), \
                        patch.object(candidate.norm, "forward", wraps=candidate.norm.forward) as norm:
                    expected, actual = reference(oracle), candidate(point)
                    self.assertTrue(actual.feat.requires_grad)
                    expected.feat.square().sum().backward()
                    actual.feat.square().sum().backward()
                norm.assert_called_once()
                helper.assert_called_once()
                self.assertIs(helper.call_args.args[5], candidate.proj.weight)
                self.assertIs(helper.call_args.args[6], candidate.proj.bias)
                self.metadata(actual, expected)
                self.compare(actual.feat, expected.feat, f"pool_train_{mode}")
                self.compare(point.feat.grad, oracle.feat.grad, f"pool_input_grad_{mode}")
                for name, parameter in candidate.named_parameters():
                    self.compare(parameter.grad, reference.get_parameter(name).grad, f"{mode}_{name}_grad")
                for name in ("running_mean", "running_var"):
                    self.compare(getattr(candidate.norm, name), getattr(reference.norm, name), f"{mode}_{name}")
                self.exact(candidate.norm.num_batches_tracked, reference.norm.num_batches_tracked)
                self.assertEqual(candidate.norm.num_batches_tracked.item(),
                                  0 if mode in ("pooling_only", "eval_grad") else 1)
                self.check_pool_packed(candidate, False)
                for pooling in (candidate, reference):
                    for parameter in pooling.parameters():
                        parameter.add_(parameter.grad, alpha=-0.001)
                    pooling.eval()
                with torch.inference_mode():
                    expected, actual = reference(oracle), candidate(point)
                self.metadata(actual, expected)
                self.compare(actual.feat, expected.feat, f"pool_after_update_{mode}")
                self.check_pool_packed(candidate, False)
                candidate.load_state_dict(reference.state_dict(), strict=True)
                self.check_pool_packed(candidate, True)

    def test_stem_integer_hashes_maps_collisions_duplicates_and_flattened_mm(self):
        fixtures = {
            "duplicates": [[0, 0, 0, 0]] * 17 + [[0, 1, 0, 0], [1, 0, 0, 0]],
            "collisions": [[0, 19349669, -73856093, 0], [0, 0, 0, 0],
                           [0, 19349668, -73856093, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
            "crossbatch": [[73856093, -334214467, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
            "int64_wrap": [[0, 2**62, 0, 0], [0, 2**62 + 1, 0, 0], [0, -(2**62), 1, 0]],
        }
        for bias in (False, True):
            source = self.sources.vanilla.HashSparseConv3d(3, 16, 5, bias=bias).eval()
            if bias:
                source.bias.normal_(0, 0.03)
            candidate = self.sources.ascend.AscendEmbeddingConv(source).eval()
            for name, rows in fixtures.items():
                for dtype in (torch.int32, torch.int64):
                    if name == "int64_wrap" and dtype == torch.int32:
                        continue
                    rows_tensor = torch.tensor(rows, dtype=dtype)
                    features = torch.randn(len(rows), 3)
                    for reverse in (False, True):
                        with self.subTest(case=name, dtype=dtype, bias=bias, reverse=reverse):
                            permutation = torch.arange(len(rows))
                            if reverse:
                                permutation = permutation.flip(0)
                            grid, batch = rows_tensor[permutation, 1:], rows_tensor[permutation, 0]
                            feat = features[permutation]
                            volume = len(source.offsets)
                            sorted_keys, sorted_rows = source._hash(batch, grid).sort()
                            queries = source._hash(
                                batch[:, None].expand(-1, volume).reshape(-1),
                                (grid[:, None] + source.offsets[None]).reshape(-1, 3))
                            positions = torch.searchsorted(sorted_keys, queries).clamp(max=len(feat) - 1)
                            indices = sorted_rows[positions]
                            found = sorted_keys[positions] == queries
                            expected_neighbors = feat[indices] * found[:, None]
                            expected = source(feat, grid, batch)
                            with _StemTrace(feat) as trace, patch.object(
                                    torch, "searchsorted", wraps=torch.searchsorted) as search:
                                actual = candidate(feat, grid, batch)
                            search.assert_called_once()
                            self.exact(search.call_args.args[0], sorted_keys, "sorted hashes")
                            self.exact(search.call_args.args[1], queries, "additive queries")
                            self.assertEqual(len(trace.indices), 1)
                            self.exact(trace.indices[0], indices, "actual gather representatives")
                            self.assertEqual(len(trace.operands), 1)
                            self.exact(trace.operands[0][0], expected_neighbors.reshape(len(feat), -1),
                                       "gather/missing-mask packing")
                            self.exact(trace.operands[0][1], source.weight.reshape(-1, 16),
                                       "flattened weight packing")
                            self.compare(actual, expected, f"stem_{name}_{dtype}_{bias}_{reverse}")

    def test_stem_fractional_float_grid_uses_reference_fallback(self):
        source = self.sources.vanilla.HashSparseConv3d(3, 16, 5).eval()
        candidate = self.sources.ascend.AscendEmbeddingConv(source).eval()
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                grid = torch.tensor([[-1.25, 0.25, 0], [-0.25, 0.25, 0],
                                     [0.75, -0.75, 0], [0.75, -0.75, 0]], dtype=dtype)
                batch, features = torch.tensor([0, 0, 0, 1]), torch.randn(4, 3)
                explicit = source._hash(
                    batch[:, None].expand(-1, 125).reshape(-1),
                    (grid[:, None] + source.offsets[None]).reshape(-1, 3))
                additive = (source._hash(batch, grid)[:, None] + candidate.offset_keys).reshape(-1)
                self.assertFalse(torch.equal(explicit, additive), "fixture must expose truncation")
                expected = source(features, grid, batch)
                with patch.object(self.sources.vanilla.HashSparseConv3d, "forward", autospec=True,
                                  side_effect=self.sources.vanilla.HashSparseConv3d.forward) as fallback:
                    actual = candidate(features, grid, batch)
                fallback.assert_called_once()
                self.assertIs(fallback.call_args.args[0], candidate)
                self.assertIs(fallback.call_args.args[2], grid)
                self.compare(actual, expected, f"stem_float_fallback_{dtype}")

    def test_stem_offsets_reload_derived_keys_and_original_cpu_parameter_identity(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                source = self.sources.vanilla.HashSparseConv3d(3, 16, 5, bias=bias).eval()
                reference = deepcopy(source)
                parameters = dict(source.named_parameters())
                original = {key: value.clone() for key, value in source.state_dict().items()}
                rng = torch.random.get_rng_state()
                candidate = self.sources.ascend.AscendEmbeddingConv(source).eval()
                self.exact(torch.random.get_rng_state(), rng, "stem wrapping RNG")
                self.assertIs(candidate.offsets, source.offsets)
                self.assertEqual(set(candidate.state_dict()), set(original))
                for key, value in original.items():
                    self.exact(candidate.state_dict()[key], value, key)
                self.assertIn("offset_keys", candidate._non_persistent_buffers_set)
                old_keys = candidate.offset_keys.clone()
                reference.offsets.copy_(reference.offsets.roll(1, 0))
                reference.weight.mul_(0.75).add_(0.0625)
                if bias:
                    reference.bias.add_(0.125)
                for _ in range(2):
                    incompatible = candidate.load_state_dict(reference.state_dict(), strict=True)
                    self.assertFalse(incompatible.missing_keys or incompatible.unexpected_keys)
                    self.exact(torch.random.get_rng_state(), rng, "stem reload RNG")
                    self.assertEqual(set(candidate.state_dict()), set(original))
                    self.assertIs(candidate.offsets, source.offsets)
                    self.exact(candidate.offsets, reference.offsets)
                    expected_keys = source._hash(torch.zeros(125, dtype=torch.long), reference.offsets)
                    self.exact(candidate.offset_keys, expected_keys, "repacked offsets")
                    self.assertFalse(torch.equal(candidate.offset_keys, old_keys))
                    for name, parameter in parameters.items():
                        self.assertIs(candidate.get_parameter(name), parameter)
                        self.assertEqual(parameter.dtype, torch.float32)
                        self.exact(parameter, reference.get_parameter(name), name)
                grid = torch.arange(18).reshape(6, 3) % 3
                features, batch = torch.randn(6, 3), torch.tensor([0, 0, 0, 1, 1, 1])
                self.compare(candidate(features, grid, batch), reference(features, grid, batch),
                             f"stem_reloaded_offsets_{bias}")
                self.assertEqual(set(candidate.state_dict()), set(original))
                checkpoint = reference.state_dict()
                for invalid in ({k: v for k, v in checkpoint.items() if k != "offsets"},
                                {**checkpoint, "offset_keys": expected_keys}):
                    with self.assertRaises(RuntimeError):
                        candidate.load_state_dict(invalid, strict=True)

    def test_singlecloud_batch_factory_preserves_reference_and_caller_data(self):
        vanilla = self.sources.vanilla
        model = SimpleNamespace(order=("z",), shuffle_orders=False)  # Genuine CPU serialization.
        grid, features = torch.arange(18).reshape(6, 3) % 7, torch.randn(6, 3)
        cases = {
            "single": (dict(grid_coord=grid, offset=torch.tensor([6])), 0),
            "quantize": (dict(coord=grid.float() * 0.25, grid_size=0.25, offset=torch.tensor([6])), 0),
            "explicit_batch": (dict(grid_coord=grid, batch=torch.zeros(6, dtype=torch.long),
                                    offset=torch.tensor([6])), 0),
            "int32_offset": (dict(grid_coord=grid, offset=torch.tensor([6], dtype=torch.int32)), 1),
            "multibatch": (dict(grid_coord=grid, offset=torch.tensor([2, 6])), 1),
        }
        for name, (data, conversions) in cases.items():
            with self.subTest(case=name):
                data["feat"] = features
                before = deepcopy(data)
                expected = vanilla.VanillaPoint(data)
                expected.serialization(order=model.order)
                convert = Mock(wraps=vanilla.offset2batch)
                rng = torch.random.get_rng_state()
                with patch.dict(vanilla.VanillaPoint.__init__.__globals__, {"offset2batch": convert}):
                    actual = self.sources.ascend.PointTransformerV3Ascend.serialize_point(model, data)
                self.assertEqual(convert.call_count, conversions)
                self.exact(torch.random.get_rng_state(), rng)
                self.metadata(actual, expected)
                self.assertIs(actual.feat, features)
                self.assertIsNot(actual, data)
                self.assertEqual(set(data), set(before))
                for key, value in before.items():
                    if isinstance(value, torch.Tensor):
                        self.exact(data[key], value, key)
                    else:
                        self.assertEqual(data[key], value)

    def test_shared_pool_features_is_used_by_forward_for_single_and_multiple_clouds(self):
        production = self.sources.ascend.PointTransformerV3Ascend
        model = SimpleNamespace(order=("z",), shuffle_orders=False,
                                embedding=Mock(side_effect=lambda point: point),
                                enc=Mock(side_effect=lambda point: point),
                                projection=Mock(side_effect=lambda features: features))
        model.serialize_point = MethodType(production.serialize_point, model)
        model.pool_features = Mock(side_effect=production.pool_features)
        for counts in ((7,), (2, 5, 7)):
            with self.subTest(counts=counts):
                n = sum(counts)
                data = dict(feat=torch.randn(n, 4), grid_coord=torch.arange(n * 3).reshape(n, 3),
                            offset=torch.tensor(counts).cumsum(0))
                before = deepcopy(data)
                expected = torch.stack([part.mean(0) for part in data["feat"].split(counts)])
                with patch.object(torch, "segment_reduce", wraps=torch.segment_reduce) as native:
                    actual = production.forward(model, data)
                native.assert_called_once()
                self.assertEqual(native.call_args.args[1], "mean")
                self.exact(native.call_args.kwargs["offsets"], F.pad(data["offset"], (1, 0)))
                model.pool_features.assert_called_once()
                point = model.pool_features.call_args.args[0]
                self.assertIs(model.embedding.call_args.args[0], point)
                self.assertIs(model.enc.call_args.args[0], point)
                self.assertIs(model.projection.call_args.args[0], actual)
                self.compare(actual, expected, f"shared_pool_{counts}")
                self.assertEqual(set(data), set(before))
                for key, value in before.items():
                    self.exact(data[key], value, key)
                model.pool_features.reset_mock()

    def test_attention_index_routes_match_explicit_oracle_without_npu(self):
        attention = self.source_attention()
        prepare = MethodType(self.sources.ascend.AscendSerializedAttention.prepare_indices, attention)
        for counts in ((1,), (7,), (16,), (32,), (17,), (4, 4), (5, 11), (16, 17), (1, 7), (32,)):
            for order_index in (0, 1):
                with self.subTest(counts=counts, order=order_index):
                    point = attention_point(self.sources.vanilla, counts)
                    attention.order_index = order_index
                    expected_order, expected_inverse = self.explicit_indices(attention, point)
                    before = dict(point)
                    with patch.object(attention, "get_padding_and_inverse",
                                      wraps=attention.get_padding_and_inverse) as padding:
                        order, inverse = prepare(point)
                    self.assertEqual(attention.patch_size, min(min(counts), 16))
                    fast = len(counts) == 1 and (counts[0] <= 16 or counts[0] % 16 == 0)
                    self.assertEqual(padding.call_count, 0 if fast else 1)
                    if len(counts) == 1 and counts[0] <= 16:
                        self.assertIsNone(order)
                        self.assertIsNone(inverse)
                    else:
                        self.exact(order, expected_order, "attention order")
                        self.exact(inverse, expected_inverse, "attention inverse")
                        self.exact(order[inverse], torch.arange(sum(counts)), "unpadding rows")
                        batch_patches = point.batch[order].reshape(-1, attention.patch_size)
                        self.exact(batch_patches, batch_patches[:, :1].expand_as(batch_patches),
                                   "patches must not mix batches")
                        if fast:
                            self.assertEqual(order.data_ptr(), point.serialized_order[order_index].data_ptr())
                            self.assertEqual(inverse.data_ptr(), point.serialized_inverse[order_index].data_ptr())
                    if fast:
                        self.assertEqual(set(point), set(before), "fast route created a padding cache")
                    for key, value in before.items():
                        self.assertIs(point[key], value, key)


@unittest.skipUnless(TEST_NPU, "requires real SubM/CANN runtime; set ASCEND_TEST_NPU=1")
class PerformanceNPUTests(_PerformanceChecks):
    @classmethod
    def setUpClass(cls):
        path = patch.object(sys, "path", [str(REPO_ROOT), *sys.path])
        path.start()
        cls.addClassCleanup(path.stop)
        runtime = sources()
        torch.npu.set_device(runtime.ascend.NPU_DEVICE)
        cls.addClassCleanup(torch.npu.set_compile_mode,
                            jit_compile=not torch.npu.is_jit_compile_false())
        torch.npu.set_compile_mode(jit_compile=runtime.ascend.NPU_JIT_COMPILE)

    def attention_oracle(self, attention, features, order, inverse):
        # Old explicit permutation and post-linear Q scaling, independent of the
        # candidate's packed weights and its single-patch bypass.
        order = order.to(features.device, torch.int32)
        inverse = inverse.to(features.device, torch.int32)
        qkv = F.linear(features, attention.qkv.weight, attention.qkv.bias).index_select(0, order)
        q, key, value = qkv.reshape(
            -1, attention.patch_size, 3, attention.num_heads,
            attention.channels // attention.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q.contiguous() * attention.scale) @ key.contiguous().transpose(-2, -1)
        output = (scores.softmax(-1) @ value.contiguous()).transpose(1, 2).reshape(-1, attention.channels)
        return F.linear(output.index_select(0, inverse), attention.proj.weight, attention.proj.bias)

    def test_singlepatch_attention_matches_explicit_permutation_and_cpu_entry(self):
        attention = self.sources.ascend.AscendSerializedAttention(self.source_attention()).eval()
        for n in (1, 7, 16):
            for order_index in (0, 1):
                with self.subTest(n=n, order=order_index):
                    point = attention_point(self.sources.vanilla, (n,))
                    attention.order_index = order_index
                    explicit = self.explicit_indices(attention, point)
                    features = point.feat.to(self.sources.ascend.NPU_DEVICE, torch.float16)
                    expected = self.attention_oracle(attention, features, *explicit)
                    with patch.object(attention, "get_padding_and_inverse",
                                      side_effect=AssertionError("single patch built padding")):
                        order, inverse = attention.prepare_indices(point)
                        self.assertIsNone(order)
                        self.assertIsNone(inverse)
                        actual = attention.forward_features(features, order, inverse)
                        self.compare(actual, expected, f"singlepatch_{n}_{order_index}")
                        self.assertIs(attention(point), point)
                    self.compare(point.feat, expected.cpu().float(), f"cpu_entry_{n}_{order_index}")
                    self.assertNotIn("pad", point)

    def test_multipatch_padded_and_multibatch_attention_matches_explicit_oracle(self):
        attention = self.sources.ascend.AscendSerializedAttention(self.source_attention(num_heads=2)).eval()
        for counts in ((32,), (17,), (4, 4), (5, 11), (16, 17), (1, 7)):
            for order_index in (0, 1):
                with self.subTest(counts=counts, order=order_index):
                    point = attention_point(self.sources.vanilla, counts)
                    attention.order_index = order_index
                    explicit = self.explicit_indices(attention, point)
                    order, inverse = attention.prepare_indices(point)
                    self.exact(order, explicit[0])
                    self.exact(inverse, explicit[1])
                    features = point.feat.to(self.sources.ascend.NPU_DEVICE, torch.float16)
                    expected = self.attention_oracle(attention, features, *explicit)
                    qkv = F.linear(features, attention.scaled_qkv_weight, attention.scaled_qkv_bias)
                    packed = qkv.index_select(0, order.to(features.device, torch.int32)).reshape(
                        -1, attention.patch_size, 3, attention.channels)
                    npu = self.sources.npu
                    with patch.object(npu, "npu_prompt_flash_attention",
                                      wraps=npu.npu_prompt_flash_attention) as pfa:
                        actual = attention.forward_features(features, order, inverse)
                    pfa.assert_called_once()
                    self.assertEqual(pfa.call_args.kwargs, dict(
                        num_heads=attention.num_heads, input_layout="BSH", scale_value=1.0,
                        pre_tokens=2147483647, next_tokens=2147483647))
                    for tensor, oracle in zip(pfa.call_args.args, packed.unbind(2)):
                        self.valid(tensor, oracle.shape, features.device, torch.float16)
                        self.assertTrue(tensor.is_contiguous())
                        self.exact(tensor.cpu(), oracle.cpu(), "PFA BSH packing")
                    self.compare(actual, expected, f"attention_{counts}_{order_index}")

    def test_attention_unsupported_pfa_shapes_use_half_matmul_softmax(self):
        for heads, patch_size, n in ((4, 7, 14), (2, 1, 129)):
            with self.subTest(head_dim=32 // heads, patches=n // patch_size):
                attention = self.sources.ascend.AscendSerializedAttention(
                    self.source_attention(num_heads=heads)).eval()
                # Explicit patches/permutations isolate each PFA shape guard
                # from prepare_indices and the single-cloud patch bypass.
                attention.patch_size = patch_size
                order = torch.arange(n).roll(3)
                inverse = order.argsort()
                self.exact(order[inverse], torch.arange(n), "manual attention inverse")
                features = torch.randn(n, 32).to(self.sources.ascend.NPU_DEVICE, torch.float16)
                expected = self.attention_oracle(attention, features, order, inverse)
                with patch.object(self.sources.npu, "npu_prompt_flash_attention",
                                  side_effect=AssertionError("unsupported shape reached PFA")) as pfa, \
                        patch.object(attention.softmax, "forward", wraps=attention.softmax.forward) as softmax:
                    actual = attention.forward_features(features, order, inverse)
                pfa.assert_not_called()
                softmax.assert_called_once()
                self.valid(softmax.call_args.args[0], (n // patch_size, heads, patch_size, patch_size),
                           self.sources.ascend.NPU_DEVICE, torch.float16)
                self.valid(actual, (n, 32), self.sources.ascend.NPU_DEVICE, torch.float16)
                self.compare(actual, expected, f"attention_fallback_d{32 // heads}_patches{n // patch_size}")

    def test_scaled_qkv_half_buffers_nonpersistent_strict_reload_and_no_repacking(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                source = self.source_attention(bias)
                checkpoint = {key: value.clone() for key, value in source.state_dict().items()}
                attention = self.sources.ascend.AscendSerializedAttention(source).eval()
                parameters = dict(attention.named_parameters())
                previous = None
                for reload in (False, True):
                    if reload:
                        for value in checkpoint.values():
                            value.mul_(0.5).add_(0.125)
                        rng = torch.random.get_rng_state()
                        incompatible = attention.load_state_dict(checkpoint, strict=True)
                        self.assertFalse(incompatible.missing_keys or incompatible.unexpected_keys)
                        self.exact(torch.random.get_rng_state(), rng, "attention reload RNG")
                    self.assertEqual(set(attention.state_dict()), set(checkpoint))
                    for name, parameter in parameters.items():
                        self.assertIs(attention.get_parameter(name), parameter)
                        self.valid(parameter, checkpoint[name].shape,
                                   self.sources.ascend.NPU_DEVICE, torch.float16)
                        self.exact(parameter.cpu(), checkpoint[name].half(), name)
                    for kind in ("weight", "bias"):
                        name = f"scaled_qkv_{kind}"
                        self.assertIn(name, attention._non_persistent_buffers_set)
                        self.assertNotIn(name, attention.state_dict())
                        packed, original = getattr(attention, name), getattr(attention.qkv, kind)
                        if original is None:
                            self.assertIsNone(packed)
                            self.assertNotIn(name, dict(attention.named_buffers()))
                            continue
                        self.assertIs(dict(attention.named_buffers())[name], packed)
                        self.assertNotIn(name, dict(attention.named_parameters()))
                        expected = original.clone()
                        expected[:attention.channels].mul_(attention.scale)
                        self.valid(packed, original.shape, self.sources.ascend.NPU_DEVICE, torch.float16)
                        self.assertTrue(packed.is_contiguous())
                        self.assertFalse(packed.requires_grad)
                        self.assertNotEqual(packed.data_ptr(), original.data_ptr())
                        self.exact(packed.cpu(), expected.cpu(), name)
                        if previous is not None:
                            self.assertFalse(torch.equal(packed.cpu(), previous[name]))
                    previous = {name: getattr(attention, name).cpu().clone()
                                for name in ("scaled_qkv_weight", "scaled_qkv_bias")
                                if getattr(attention, name) is not None}
                    packed_before = {name: getattr(attention, name) for name in previous}
                    point = attention_point(self.sources.vanilla, (7,))
                    explicit = self.explicit_indices(attention, point)
                    half = point.feat.to(self.sources.ascend.NPU_DEVICE, torch.float16)
                    expected = self.attention_oracle(attention, half, *explicit)
                    actual = attention.forward_features(half, *attention.prepare_indices(point))
                    self.compare(actual, expected, f"qkv_reload_{bias}_{reload}")
                    for name, value in packed_before.items():
                        self.assertIs(getattr(attention, name), value, "forward repacked QKV")
                    self.assertEqual(set(attention.state_dict()), set(checkpoint))
                for invalid in ({k: v for k, v in checkpoint.items() if k != "qkv.weight"},
                                {**checkpoint, "scaled_qkv_weight": attention.scaled_qkv_weight}):
                    with self.assertRaises(RuntimeError):
                        attention.load_state_dict(invalid, strict=True)

    def test_attention_preserves_training_device_dtype_and_unsupported_option_rejection(self):
        source = self.source_attention()
        for option in ("enable_rpe", "enable_flash"):
            with self.subTest(option=option), patch.object(source, option, True):
                with self.assertRaises(ValueError):
                    self.sources.ascend.AscendSerializedAttention(source)
        attention = self.sources.ascend.AscendSerializedAttention(source).eval()
        cpu = torch.randn(7, 32)
        half = cpu.to(self.sources.ascend.NPU_DEVICE, torch.float16)
        invalid = (cpu, cpu.half(), half.float(), torch.empty_like(cpu, device="meta", dtype=torch.float16))
        with patch.object(F, "linear", side_effect=AssertionError("invalid attention reached linear")):
            attention.train()
            with self.assertRaisesRegex(RuntimeError, "inference-only"):
                attention.forward_features(half, None, None)
            attention.eval()
            for features in invalid:
                with self.subTest(device=features.device, dtype=features.dtype):
                    with self.assertRaisesRegex(RuntimeError, "NPU FP16"):
                        attention.forward_features(features, None, None)
            for owner in (attention.qkv, attention.proj):
                for name in ("weight", "bias"):
                    original = getattr(owner, name)
                    for invalid_parameter in (original.cpu(), original.float()):
                        with self.subTest(parameter=name, device=invalid_parameter.device,
                                          dtype=invalid_parameter.dtype), patch.object(
                                owner, name, torch.nn.Parameter(invalid_parameter)):
                            with self.assertRaisesRegex(RuntimeError, "parameters must remain NPU FP16"):
                                attention.forward_features(half, None, None)
        with patch.object(attention, "prepare_indices", side_effect=AssertionError("invalid CPU entry")):
            for features in (cpu.half(), cpu.double(), half):
                with self.assertRaisesRegex(RuntimeError, "CPU FP32"):
                    attention(self.sources.vanilla.VanillaPoint(feat=features, offset=torch.tensor([7])))

    def test_fused_cpe_norm1_cache_is_consumed_once_and_fresh_points_recompute(self):
        source = self.sources.vanilla.VanillaBlock(
            channels=32, num_heads=2, patch_size=16, drop_path=0,
            upcast_attention=False, upcast_softmax=False).eval()
        source.norm1.weight.uniform_(0.8, 1.2)
        source.norm1.bias.normal_(0, 0.03)
        source.attn = self.sources.ascend.AscendSerializedAttention(source.attn).eval()
        block = self.sources.ascend.AscendBlock(source).eval()
        npu = self.sources.npu
        cached = []
        for repeat in range(2):
            point = make_point(self.sources.vanilla, n=7)
            point.grid_coord.mul_(repeat + 1)
            self.assertNotIn("_cpe_normalized", point)
            block.forward_cpe(point)
            normalized, residual = point["_cpe_normalized"], point.feat
            self.valid(normalized, (7, 32), self.sources.ascend.NPU_DEVICE, torch.float16)
            expected = npu.npu_layer_norm_eval(
                residual, block.norm1.normalized_shape, block.norm1.weight, block.norm1.bias, block.norm1.eps)
            self.compare(normalized, expected, f"fused_norm1_cache_{repeat}")
            cached.append(normalized)
            with ExitStack() as stack:
                norm = stack.enter_context(patch.object(npu, "npu_layer_norm_eval",
                                                       side_effect=AssertionError("cached norm1 recomputed")))
                attention = stack.enter_context(patch.object(block.attn, "forward_features",
                                                            wraps=block.attn.forward_features))
                self.assertIs(block.forward_attention(point), point)
            norm.assert_not_called()
            self.assertIs(attention.call_args.args[0], normalized)
            self.assertIs(point["_attention_residual"], residual)
            self.assertNotIn("_cpe_normalized", point)
            block.forward_ffn(point)
            self.valid(point.feat, (7, 32))
            self.assertNotIn("_cpe_normalized", point)
            self.assertNotIn("_attention_residual", point)
        self.assertIsNot(cached[0], cached[1])
        # The same block's standalone attention entry must still compute norm1.
        point = make_point(self.sources.vanilla, n=7)
        with patch.object(npu, "npu_layer_norm_eval", wraps=npu.npu_layer_norm_eval) as norm:
            block.forward_attention(point)
        norm.assert_called_once()
        self.assertNotIn("_cpe_normalized", point)
        block.forward_ffn(point)
        self.valid(point.feat, (7, 32))
        self.assertNotIn("_attention_residual", point)

    def test_serialization_prefetch_reuses_first_stage_map_and_refreshes_each_forward(self):
        model = self.sources.ascend.PointTransformerV3Ascend(
            in_channels=3, output_dim=32, stride=(), enc_depths=(2,), enc_channels=(32,),
            enc_num_head=(2,), enc_patch_size=(16,), drop_path=0, shuffle_orders=False,
            upcast_attention=False, upcast_softmax=False).eval()
        conv = model.enc.enc0.block0.cpe_conv
        self.assertIsInstance(conv, self.sources.ascend.SubMCPEConv)
        rows = torch.arange(1024)
        grid = torch.stack((rows % 32, rows // 32, torch.zeros_like(rows)), dim=1)
        grid[1] = grid[0]  # Duplicate representatives must remain reference-exact.
        data = dict(feat=torch.randn(1024, 3), grid_coord=grid, offset=torch.tensor([1024]))
        points, maps = [], []
        embedding = model.embedding.forward

        def observe_embedding(point):
            self.assertEqual(builder.call_count, len(points) + 1, "prefetch must precede the stem")
            self.assertIn("_cpe_npu_map_3", point)
            points.append(point)
            maps.append(point["_cpe_npu_map_3"])
            return embedding(point)

        saved = []
        with patch.object(torch.ops.graspgenx_subm, "build_subm_map",
                          wraps=torch.ops.graspgenx_subm.build_subm_map) as builder, \
                patch.object(torch.ops.graspgenx_subm, "subm_conv3d",
                             wraps=torch.ops.graspgenx_subm.subm_conv3d) as kernel, \
                patch.object(model.embedding, "forward", side_effect=observe_embedding):
            for repeat in range(3):
                if repeat == 2:
                    data["grid_coord"].mul_(3)  # Same allocation, different neighborhoods.
                before = deepcopy(data)
                expected = self.sources.vanilla.VanillaPoint(data)
                expected.serialization(order=model.order)
                self.valid(model(data), (1, 32))
                point, cache = points[-1], maps[-1]
                self.metadata(point, expected)
                self.assertEqual(builder.call_count, repeat + 1, "blocks rebuilt the prefetched map")
                self.assertEqual(kernel.call_count, (repeat + 1) * 2)
                for call in kernel.call_args_list[-2:]:
                    self.assertIs(call.args[1], cache, "first-stage blocks must share the prefetched map")
                self.assertIs(conv._get_npu_map(point.grid_coord, point.batch, point), cache)
                keys, indices = conv._hash(point.batch, point.grid_coord).sort()
                queries = conv._hash(
                    point.batch[:, None].expand(-1, 27).reshape(-1),
                    (point.grid_coord[:, None] + conv.offsets[None]).reshape(-1, 3))
                positions = torch.searchsorted(keys, queries).clamp(max=len(grid) - 1)
                packed = torch.full((1024, 32), -1, dtype=torch.int32)
                packed[:, :27] = indices[positions].view(-1, 27)
                packed[:, :27].masked_fill_((keys[positions] != queries).view(-1, 27), -1)
                self.assertEqual(cache.device, torch.device(self.sources.ascend.NPU_DEVICE))
                self.assertTrue(cache.is_contiguous())
                self.exact(cache.cpu(), packed, "prefetched rows/missing/padding")
                saved.append(cache.cpu().clone())
                self.assertNotIn("_cpe_hash_map_3", point)
                self.assertNotIn("_cpe_normalized", point)
                self.assertNotIn("_attention_residual", point)
                self.assertEqual(set(data), set(before), "cache leaked to caller")
                for key, value in before.items():
                    self.exact(data[key], value, key)
        self.assertEqual(len({id(point) for point in points}), 3)
        self.assertEqual(len({cache.data_ptr() for cache in maps}), 3)
        self.exact(saved[0], saved[1], "identical geometry, fresh maps")
        self.assertFalse(torch.equal(saved[0], saved[2]), "changed geometry reused a stale map")
        for cache, snapshot in zip(maps, saved):
            self.exact(cache.cpu(), snapshot, "later forwards mutated an earlier map")

    def test_serialization_prefetch_skips_geometry_stub_and_unsupported_routes(self):
        orders = ("z", "z-trans", "hilbert", "hilbert-trans")
        for name, n in (("geometry_only", 1024), ("missing_stage", 1024),
                        ("unsupported_conv", 1024), ("below_threshold", 1023),
                        ("above_limit", 4097), ("noncanonical", 1024),
                        ("negative_grid", 1024), ("float_grid", 1024)):
            with self.subTest(case=name):
                prefetch = Mock(side_effect=AssertionError("unsupported route prefetched CPE"))
                conv = SimpleNamespace(_supported_shape=name != "unsupported_conv", _get_npu_map=prefetch)
                model = SimpleNamespace(order=("z",) if name == "noncanonical" else orders,
                                        shuffle_orders=False)
                if name != "geometry_only":
                    model.enc = SimpleNamespace() if name == "missing_stage" else SimpleNamespace(
                        enc0=SimpleNamespace(block0=SimpleNamespace(cpe_conv=conv)))
                grid = torch.arange(n * 3).reshape(n, 3) % 32
                if name == "negative_grid":
                    grid[0, 0] = -1
                elif name == "float_grid":
                    grid = grid.float() + 0.25
                data = dict(grid_coord=grid, offset=torch.tensor([n]))
                before = deepcopy(data)
                expected = self.sources.vanilla.VanillaPoint(data)
                expected.serialization(order=model.order)
                actual = self.sources.ascend.PointTransformerV3Ascend.serialize_point(model, data)
                prefetch.assert_not_called()
                self.metadata(actual, expected)
                self.assertFalse(any(key.startswith("_cpe_") for key in actual))
                self.assertEqual(set(data), set(before))
                for key, value in before.items():
                    self.exact(data[key], value, key)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
