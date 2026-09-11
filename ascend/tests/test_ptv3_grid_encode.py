"""Serialization integration: python -B ascend/tests/test_ptv3_grid_encode.py -v.

Default tests are CPU-only fallback/dispatch checks. ASCEND_TEST_NPU=1 also
executes the canonical branch with the already-built GridEncode package. Source
its environment and set repository PYTHONPATH first; missing artifacts are errors
after opting in. Nothing is built, compiled, benchmarked or installed here.

The unchanged reference and complete candidate/profiler sources are loaded with
temporary import shims, without package setup hooks or a real torch_npu import.
Only the opted-in tests import the real GridEncode runtime. Metadata tests bind
the production method to a SimpleNamespace, never construct a full encoder, and
need no CPE package/weights. Integer metadata comparisons are exact. Full-pipeline
accuracy, checkpoint compatibility and performance belong to device validation.
"""

from copy import deepcopy
from functools import cache
import importlib
from importlib.machinery import ModuleSpec
import os
from pathlib import Path
import runpy
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch._dynamo  # Load before temporary sys.modules isolation.


TEST_NPU = os.environ.get("ASCEND_TEST_NPU") == "1"
SEED = 1703
ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PACKAGE = "graspgenx.models.ptv3"
GRID_MODULE = "ascend.custom_ops.grid_encode.grid_encode"
TENSOR_METADATA = (
    "grid_coord", "batch", "offset", "serialized_code",
    "serialized_order", "serialized_inverse",
)


@cache
def sources():
    # These shims only support metadata and nn.Identity dispatch, not model eval.
    # Keep the real functions/decorators; do not extract/reimplement their bodies.
    class PointDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        __setattr__ = dict.__setitem__

    shims = {
        "addict": SimpleNamespace(Dict=PointDict, __spec__=ModuleSpec("addict", None)),
        "timm": SimpleNamespace(__spec__=ModuleSpec("timm", None)),
        "timm.models.layers": SimpleNamespace(DropPath=torch.nn.Identity),
        "torch_npu": SimpleNamespace(),
    }
    with patch.dict(sys.modules, shims), patch.dict(os.environ):
        vanilla = SimpleNamespace(**runpy.run_path(str(
            REPO_ROOT / "graspgenx/models/ptv3/ptv3_vanilla.py"),
            run_name=f"{MODEL_PACKAGE}.ptv3_vanilla"))
        sys.modules[f"{MODEL_PACKAGE}.ptv3_vanilla"] = vanilla
        ascend = SimpleNamespace(**runpy.run_path(str(
            REPO_ROOT / "graspgenx/models/ptv3/ptv3_ascend.py"),
            run_name=f"{MODEL_PACKAGE}.ptv3_ascend"))
        sys.modules[f"{MODEL_PACKAGE}.ptv3_ascend"] = ascend
        profiler = SimpleNamespace(**runpy.run_path(str(
            REPO_ROOT / "ascend/tools/profile_ptv3_stages.py")))
    return SimpleNamespace(vanilla=vanilla, ascend=ascend, profiler=profiler)


def grid_fixture(dtype=torch.int32):
    # Within-batch duplicates and repeated coordinates across batch boundaries.
    return torch.tensor([
        [0, 0, 0], [1, 0, 3], [2, 5, 1], [1, 0, 3],
        [7, 2, 4], [0, 0, 0], [6, 3, 5], [7, 2, 4],
        [4, 1, 2], [2, 5, 1], [3, 6, 0], [4, 1, 2],
    ], dtype=dtype)


class _SerializationChecks(unittest.TestCase):
    def setUp(self):
        self.sources = sources()
        self.model = SimpleNamespace(order=ORDERS, shuffle_orders=False)
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state())

    def reference(self, data):
        point = self.sources.vanilla.VanillaPoint(data)
        point.serialization(order=self.model.order, shuffle_orders=self.model.shuffle_orders)
        return point

    def serialize(self, data):
        return self.sources.ascend.PointTransformerV3Ascend.serialize_point(self.model, data)

    def check_tensor(self, actual, expected, label):
        self.assertEqual(actual.device.type, "cpu", label)
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertEqual(actual.dtype, expected.dtype, label)
        self.assertTrue(torch.equal(actual, expected), label)

    def check_input(self, data, before):
        self.assertEqual(set(data), set(before), "metadata/cache leaked into caller data")
        for key, expected in before.items():
            if isinstance(expected, torch.Tensor):
                self.check_tensor(data[key], expected, f"input {key}")
            else:
                self.assertEqual(data[key], expected, key)

    def check_metadata(self, actual, expected):
        self.assertIsInstance(actual, self.sources.vanilla.VanillaPoint)
        self.assertIs(type(actual.serialized_depth), int)
        self.assertEqual(actual.serialized_depth, expected.serialized_depth)
        for key in TENSOR_METADATA:
            self.check_tensor(actual[key], expected[key], key)
        for key in ("serialized_code", "serialized_order", "serialized_inverse"):
            self.assertEqual(actual[key].dtype, torch.int64, key)
        positions = torch.arange(len(actual.batch)).repeat(len(self.model.order), 1)
        self.assertTrue(torch.equal(
            actual.serialized_order.gather(1, actual.serialized_inverse), positions))
        self.assertTrue(torch.equal(
            actual.serialized_inverse.gather(1, actual.serialized_order), positions))

    def check_serialization(self, data):
        before = deepcopy(data)
        torch.manual_seed(SEED)
        expected = self.reference(data)
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(SEED)
        actual = self.serialize(data)
        actual_rng = torch.random.get_rng_state()
        self.assertTrue(torch.equal(actual_rng, expected_rng), "CPU RNG consumption changed")
        self.assertIsNot(actual, data)
        self.check_metadata(actual, expected)
        self.check_input(data, before)
        return actual

    def check_reference_error(self, data, error_type=Exception):
        before = deepcopy(data)
        with self.assertRaises(error_type) as expected:
            self.reference(data)
        with self.assertRaises(type(expected.exception)) as actual:
            self.serialize(data)
        self.assertIs(type(actual.exception), type(expected.exception))
        self.assertEqual(str(actual.exception), str(expected.exception))
        self.check_input(data, before)


class PTV3GridEncodeCPUTests(_SerializationChecks):
    def setUp(self):
        super().setUp()
        self.grid_op = Mock(side_effect=AssertionError("CPU fallback reached GridEncode"))
        modules = patch.dict(sys.modules, {
            GRID_MODULE: SimpleNamespace(grid_encode=self.grid_op),
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.addCleanup(self.grid_op.assert_not_called)

    def test_noncanonical_orders_fall_back_without_grid_encode(self):
        grid = grid_fixture()
        for order in (("z",), ("hilbert", "z-trans"), ORDERS[::-1], ORDERS + ("z",)):
            for shuffle in (False, True):
                with self.subTest(order=order, shuffle=shuffle):
                    self.model.order, self.model.shuffle_orders = order, shuffle
                    self.check_serialization(dict(grid_coord=grid, offset=torch.tensor([4, 8, 12])))
                    self.check_serialization(dict(coord=grid.float() * 0.25 - 1,
                                                  grid_size=0.25, offset=torch.tensor([4, 8, 12])))

    def test_n4097_depth_one_falls_back_without_grid_encode(self):
        grid = (torch.arange(4097 * 3).reshape(4097, 3) % 2).int()
        point = self.check_serialization(dict(grid_coord=grid, offset=torch.tensor([1025, 4097])))
        self.assertEqual(point.serialized_depth, 1)

    def test_negative_grids_preserve_cpu_serialization(self):
        for minimum in (-1, -(2**31) - 1):
            with self.subTest(minimum=minimum):
                grid = grid_fixture(torch.int64)
                grid[0, 0] = minimum
                self.check_serialization(dict(grid_coord=grid, offset=torch.tensor([4, 8, 12])))

    def test_unsupported_grid_dtypes_preserve_cpu_serialization(self):
        # Vanilla explicitly converts coordinates to long inside both encoders;
        # floating grids are valid there and must not be rejected or overwritten.
        for dtype in (torch.int8, torch.uint8, torch.int16, torch.bool,
                      torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                grid = grid_fixture(dtype)
                if grid.is_floating_point():
                    grid = grid + 0.25
                point = self.check_serialization(dict(
                    grid_coord=grid, offset=torch.tensor([4, 8, 12])))
                self.assertEqual(point.grid_coord.dtype, dtype)

    def test_depth_zero_and_empty_inputs_preserve_reference_errors(self):
        for n in (0, 3):
            for quantize in (False, True):
                with self.subTest(N=n, quantize=quantize):
                    data = dict(offset=torch.tensor([n]))
                    if quantize:
                        data.update(coord=torch.zeros(n, 3), grid_size=0.25)
                    else:
                        data["grid_coord"] = torch.zeros(n, 3, dtype=torch.int32)
                    self.check_reference_error(data)

    def test_depth_above_sixteen_preserves_reference_error(self):
        for maximum in (65536, 2**31 + 1):
            with self.subTest(maximum=maximum):
                grid = grid_fixture(torch.int64)
                grid[0, 0] = maximum
                self.check_reference_error(dict(
                    grid_coord=grid, offset=torch.tensor([12])), AssertionError)

    def test_current_model_class_and_profiler_default(self):
        ascend, profiler = self.sources.ascend, self.sources.profiler
        self.assertIs(profiler.PointTransformerV3, ascend.PointTransformerV3Ascend)
        for name in ("GridEncodeModel", "PointTransformerV3GridEncode",
                     "PointTransformerV3Subm"):
            self.assertFalse(hasattr(ascend, name), name)

    def test_profiler_helper_calls_model_serializer_exactly_once(self):
        profiler = self.sources.profiler
        result, data = object(), {"sentinel_input": object()}
        model = SimpleNamespace(serialize_point=Mock(return_value=result))
        with patch.dict(profiler.serialize_point.__globals__, {
            "VanillaPoint": Mock(side_effect=AssertionError("profiler bypassed model serializer")),
        }):
            self.assertIs(profiler.serialize_point(model, data), result)
        model.serialize_point.assert_called_once_with(data)
        self.assertIs(model.serialize_point.call_args.args[0], data)

    def test_partitioned_profiler_uses_model_serializer(self):
        profiler = self.sources.profiler
        sentinel = RuntimeError("sentinel serialization stage")
        model = SimpleNamespace(serialize_point=Mock(side_effect=sentinel))
        data = {}
        # Run the real stage dispatcher, stopping before embedding or any CPE.
        with patch.dict(profiler.synchronize.__globals__, {"synchronize": lambda: None}):
            with self.assertRaises(profiler.StageFailure) as raised:
                profiler.run_partitioned(model, data)
        self.assertEqual(raised.exception.stage, "1.serialization")
        self.assertIs(raised.exception.cause, sentinel)
        model.serialize_point.assert_called_once_with(data)

    def test_forward_dispatch_reuses_serialized_point_and_manual_pooling(self):
        production = self.sources.ascend.PointTransformerV3Ascend
        model = torch.nn.Module()
        model.order, model.shuffle_orders = ("z",), False  # Genuine CPU fallback.
        model.embedding = torch.nn.Identity()
        model.enc = torch.nn.Identity()
        model.projection = torch.nn.Identity()
        model.forward = MethodType(production.forward, model)
        bound_serialize = MethodType(production.serialize_point, model)
        points = []

        def serialize(data):
            point = bound_serialize(data)
            points.append(point)
            point["_test_point_cache"] = object()
            return point

        model.serialize_point = Mock(side_effect=serialize)
        data = dict(grid_coord=grid_fixture()[:6], feat=torch.arange(18).float().reshape(6, 3),
                    offset=torch.tensor([2, 6]))
        with patch.object(model.embedding, "forward", wraps=model.embedding.forward) as embedding, \
                patch.object(model.enc, "forward", wraps=model.enc.forward) as encoder:
            for _ in range(2):
                before = deepcopy(data)
                expected = torch.stack((data["feat"][:2].mean(0), data["feat"][2:].mean(0)))
                self.check_tensor(model(data), expected, "manual mean pooling")
                self.check_input(data, before)
                self.assertIs(embedding.call_args.args[0], points[-1])
                self.assertIs(encoder.call_args.args[0], points[-1])
                data["feat"].add_(6)
        self.assertEqual(model.serialize_point.call_count, 2)
        self.assertTrue(all(call.args[0] is data for call in model.serialize_point.call_args_list))
        self.assertIsNot(points[0], points[1])
        self.assertIsNot(points[0]["_test_point_cache"], points[1]["_test_point_cache"])


@unittest.skipUnless(TEST_NPU, "requires built GridEncode package; set ASCEND_TEST_NPU=1")
class PTV3GridEncodeNPUTests(_SerializationChecks):
    @classmethod
    def setUpClass(cls):
        # No model constructor, validation import, SubM package or CPE execution.
        with patch.object(sys, "path", [str(REPO_ROOT), *sys.path]):
            cls.grid_module = importlib.import_module(GRID_MODULE)

    def setUp(self):
        super().setUp()
        torch.npu.set_device(self.sources.ascend.NPU_DEVICE)
        self.addCleanup(torch.npu.set_compile_mode,
                        jit_compile=not torch.npu.is_jit_compile_false())
        torch.npu.set_compile_mode(jit_compile=False)

    def serialize(self, data):
        with patch.object(self.grid_module, "grid_encode", wraps=self.grid_module.grid_encode) as op, \
                patch.object(self.sources.vanilla.VanillaPoint, "serialization",
                             side_effect=AssertionError("canonical branch used CPU serialization")):
            point = super().serialize(data)
        op.assert_called_once()
        grid, depth = op.call_args.args
        self.assertEqual(depth, point.serialized_depth)
        self.assertEqual(grid.device, torch.device(self.sources.ascend.NPU_DEVICE))
        self.assertEqual(grid.dtype, torch.int32)
        self.assertTrue(grid.is_contiguous())
        self.check_tensor(grid.cpu(), point.grid_coord.int(), "GridEncode input")
        return point

    def test_canonical_coordinate_quantization_matches_reference(self):
        grid = grid_fixture()
        coord = grid.float() * 0.25 + torch.tensor([-1.0, 0.5, -2.0])
        coord += (torch.arange(len(grid)) % 3)[:, None] * 0.0625
        point = self.check_serialization(dict(
            coord=coord, grid_size=0.25, offset=torch.tensor([4, 8, 12])))
        self.check_tensor(point.grid_coord, grid, "quantized grid")

    def test_canonical_integer_grids_batches_permutations_and_duplicates(self):
        permutation = torch.tensor([7, 2, 10, 0, 11, 4, 1, 8, 5, 9, 3, 6])
        batch = torch.arange(12) // 4
        for dtype in (torch.int32, torch.int64):
            grid = grid_fixture(dtype)
            cases = {
                "offset": dict(grid_coord=grid, offset=torch.tensor([4, 8, 12])),
                "batch": dict(grid_coord=grid, batch=batch),
                "permuted": dict(grid_coord=grid[permutation], batch=batch[permutation]),
                "depth16": dict(grid_coord=grid * 8192, batch=batch),
                "singleton": dict(grid_coord=grid[1:2], offset=torch.tensor([1])),
            }
            for name, data in cases.items():
                with self.subTest(dtype=dtype, case=name):
                    # Supplied grids take priority over even contradictory coords.
                    data.update(coord=torch.zeros_like(data["grid_coord"], dtype=torch.float32),
                                grid_size=0.25)
                    self.check_serialization(data)

    def test_shuffle_matches_reference_and_rng_state(self):
        self.model.shuffle_orders = True
        data = dict(grid_coord=grid_fixture(), offset=torch.tensor([4, 8, 12]))
        shuffled = self.check_serialization(data)
        self.model.shuffle_orders = False
        plain = self.reference(data)
        self.assertFalse(torch.equal(shuffled.serialized_code, plain.serialized_code),
                         "fixture seed must exercise a nonidentity shuffle")

    def test_repeated_changed_and_fresh_coordinates_do_not_share_caches(self):
        for key in ("grid_coord", "coord"):
            with self.subTest(input=key):
                base = grid_fixture() if key == "grid_coord" else grid_fixture().float() * 0.25 - 1
                data = {key: base.clone(), "grid_size": 0.25, "offset": torch.tensor([4, 8, 12])}
                first = self.check_serialization(data)
                saved = {name: first[name].clone() for name in TENSOR_METADATA[3:]}
                first["_test_point_cache"] = object()
                repeated = self.check_serialization(data)
                data[key].copy_(base.flip(0))  # Same allocation, shape and depth.
                changed = self.check_serialization(data)
                fresh_data = {key: base.roll(1, 0), "grid_size": 0.25, "offset": data["offset"].clone()}
                fresh = self.check_serialization(fresh_data)
                data[key].copy_(base)
                restored = self.check_serialization(data)
                self.assertFalse(torch.equal(first.serialized_code, changed.serialized_code))
                self.assertFalse(torch.equal(first.serialized_code, fresh.serialized_code))
                self.assertTrue(torch.equal(first.serialized_code, restored.serialized_code))
                points = (first, repeated, changed, fresh, restored)
                self.assertEqual(len({id(point) for point in points}), len(points))
                for point in points[1:]:
                    self.assertNotIn("_test_point_cache", point)
                for name, value in saved.items():
                    self.check_tensor(first[name], value, f"earlier result {name}")
                    self.assertEqual(len({point[name].data_ptr() for point in points}), len(points), name)

    def test_grid_encode_runtime_error_propagates_without_cpu_retry(self):
        sentinel = RuntimeError("sentinel GridEncode runtime failure")
        data = dict(grid_coord=grid_fixture(), offset=torch.tensor([12]))
        before = deepcopy(data)
        with patch.object(self.grid_module, "grid_encode", side_effect=sentinel) as op, \
                patch.object(self.sources.vanilla.VanillaPoint, "serialization",
                             side_effect=AssertionError("NPU error retried on CPU")) as cpu:
            with self.assertRaises(RuntimeError) as raised:
                super().serialize(data)
        self.assertIs(raised.exception, sentinel)
        op.assert_called_once()
        cpu.assert_not_called()
        self.check_input(data, before)

    def test_raw_grid_encode_contiguous_storage_offset_view(self):
        # Keep this focused check here rather than editing the main-owned raw suite.
        grid = grid_fixture()
        backing = torch.cat((torch.full((1, 3), 7, dtype=torch.int32), grid)).to(
            self.sources.ascend.NPU_DEVICE)
        view = backing[1:]
        self.assertTrue(view.is_contiguous())
        self.assertEqual(view.storage_offset(), 3)  # Non-aligned scalar-load path.
        before = backing.cpu().clone()
        expected = self.reference(dict(grid_coord=grid, batch=torch.zeros(len(grid), dtype=torch.long)))
        actual = self.grid_module.grid_encode(view, expected.serialized_depth)
        self.assertEqual(actual.device, view.device)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(actual.shape, (len(grid), len(ORDERS)))
        self.assertTrue(actual.is_contiguous())
        self.check_tensor(actual.cpu(), expected.serialized_code.T.contiguous(), "offset-view codes")
        self.check_tensor(backing.cpu(), before, "offset-view backing storage")


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
