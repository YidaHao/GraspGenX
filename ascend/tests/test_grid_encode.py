"""Grid metadata tests: python -B ascend/tests/test_grid_encode.py -v.

Default execution is CPU oracle analysis only. ASCEND_TEST_NPU=1 opts into the
already-built grid_encode package, eager/fake/meta/TorchAir and OPP checks.
Source that package's environment and set repository PYTHONPATH beforehand.
Nothing is built or installed here; missing runtime artifacts are errors, not
skips, once opted in. Only three torch.compile calls are used by this suite.

The raw op returns PURE spatial int64 [N, 4], in the order below. Coordinate
contents in [0, 2**depth) are a caller precondition, not a device readback check.
Batch packing, argsort/inverse and order shuffling stay on CPU. All comparisons
are exact integer comparisons; cosine and floating-point tolerances do not apply.
"""

from contextlib import nullcontext
from functools import cache
import itertools
import json
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


TEST_NPU = os.environ.get("ASCEND_TEST_NPU") == "1"
NPU_DEVICE = "npu:0"
CPU_THREADS = 4
SEED = 1703
ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
POINT_COUNTS = (1, 15, 16, 17, 63, 64, 65, 2048, 3500, 4096)
DEPTHS = tuple(range(1, 17))
BULK_N = 4096
JIT_CASES = ((1, 1), (17, 9), (4096, 16))
FULLGRAPH_CASES = ((1, 1), (17, 9), (4096, 16))
REPO_ROOT = Path(__file__).resolve().parents[2]
OP_ROOT = REPO_ROOT / "ascend/custom_ops/grid_encode"
REFERENCE_PATH = REPO_ROOT / "graspgenx/models/ptv3/ptv3_vanilla.py"


@cache
def reference():
    # Load the unchanged source without package setup hooks or Ascend imports.
    # These temporary dependency shims serve only encode/VanillaPoint metadata;
    # no model or DropPath is constructed, and sys.modules is restored afterward.
    class ReferenceDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    with patch.dict(sys.modules, {
        "addict": SimpleNamespace(Dict=ReferenceDict),
        "timm.models.layers": SimpleNamespace(DropPath=torch.nn.Identity),
    }):
        module = runpy.run_path(str(REFERENCE_PATH))
    return SimpleNamespace(encode=module["encode"], Point=module["VanillaPoint"])


def reference_codes(grid, depth, batch=None):
    assert grid.device.type == "cpu", "the oracle must not execute on NPU"
    if batch is None:
        # Also prevents vanilla Hilbert's squeeze from losing N=1's row axis.
        batch = torch.zeros(len(grid), dtype=torch.int64)
    assert batch.device.type == "cpu"
    return torch.stack([
        reference().encode(grid, batch=batch, depth=depth, order=order)
        for order in ORDERS
    ], dim=1)


def coordinates(n, depth):
    maximum = (1 << depth) - 1
    generator = torch.Generator().manual_seed(SEED)
    grid = torch.randint(0, maximum + 1, (n, 3), generator=generator, dtype=torch.int32)
    corners = list(itertools.product((0, maximum), repeat=3))
    anchors = corners.copy()
    # Hilbert's endpoint-neighbor orientation changes with depth.
    for corner in corners:
        for axis in range(3):
            neighbor = list(corner)
            neighbor[axis] = maximum - 1 if corner[axis] else 1
            anchors.append(neighbor)
    for bit in range(depth):
        for value in ((1 << bit) - 1, 1 << bit, min(maximum, (1 << bit) + 1)):
            anchors.extend(itertools.permutations((value, 0, maximum)))
    count = min(n, len(anchors))
    grid[:count] = torch.tensor(anchors[:count], dtype=torch.int32)
    if n == 1:
        grid[0] = torch.tensor((maximum, 0, 1), dtype=torch.int32)
    else:
        grid[-1] = grid[0]
    return grid


class _GridChecks(unittest.TestCase):
    def check_cpu_metadata(self, spatial, grid, depth, shuffle=False):
        self.assertEqual(spatial.device.type, "cpu")
        self.assertEqual(spatial.dtype, torch.int64)
        batch = torch.arange(len(grid), dtype=torch.int64) % 3
        code = (spatial.T | (batch << (3 * depth))).contiguous()
        order = torch.argsort(code)  # Keep vanilla's default, non-stable tie order.
        positions = torch.arange(len(grid)).repeat(len(ORDERS), 1)
        inverse = torch.zeros_like(order).scatter_(1, order, positions)
        point = reference().Point(grid_coord=grid, batch=batch)
        if shuffle:
            permutation = torch.randperm(len(ORDERS), generator=torch.Generator().manual_seed(SEED))
            self.assertFalse(torch.equal(permutation, torch.arange(len(ORDERS))))
            code, order, inverse = (value[permutation] for value in (code, order, inverse))
            with patch.object(torch, "randperm", return_value=permutation) as randperm:
                point.serialization(order=ORDERS, depth=depth, shuffle_orders=True)
            randperm.assert_called_once_with(len(ORDERS))
        else:
            point.serialization(order=ORDERS, depth=depth, shuffle_orders=False)
        for name, actual in (("code", code), ("order", order), ("inverse", inverse)):
            expected = point[f"serialized_{name}"]
            self.assertEqual(actual.device.type, "cpu")
            self.assertEqual(actual.dtype, torch.int64)
            self.assertTrue(torch.equal(actual, expected), f"CPU serialized_{name}")
        self.assertTrue(torch.equal(order.gather(1, inverse), positions))
        self.assertTrue(torch.equal(inverse.gather(1, order), positions))

    def check_codes(self, actual, grid, depth, device):
        self.assertEqual(actual.device, device)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(actual.shape, (len(grid), len(ORDERS)))
        self.assertTrue(actual.is_contiguous())
        result = actual.cpu()
        expected = reference_codes(grid, depth)
        for column, name in enumerate(ORDERS):
            self.assertTrue(torch.equal(result[:, column], expected[:, column]), name)
        self.check_cpu_metadata(result, grid, depth)
        return result


class GridEncodeReferenceTests(_GridChecks):
    def test_reference_all_depths_bounds_and_transitions(self):
        for depth in DEPTHS:
            with self.subTest(depth=depth):
                grid = coordinates(BULK_N, depth)
                self.assertEqual(grid.dtype, torch.int32)
                self.assertTrue(grid.is_contiguous())
                self.assertGreaterEqual(grid.min().item(), 0)
                self.assertLess(grid.max().item(), 1 << depth)
                codes = reference_codes(grid, depth)
                limit = 1 << (3 * depth)
                self.assertEqual(codes.shape, (BULK_N, 4))
                self.assertEqual(codes.dtype, torch.int64)
                self.assertTrue(torch.equal(codes[0], torch.zeros(4, dtype=torch.int64)))
                self.assertGreaterEqual(codes.min().item(), 0)
                for column in range(4):
                    self.assertEqual(codes[:, column].max().item(), limit - 1)
                    self.assertTrue((codes[:, column] == limit - 2).any().item())
                for bit in range(depth):
                    for value in ((1 << bit) - 1, 1 << bit, min((1 << depth) - 1, (1 << bit) + 1)):
                        self.assertTrue((grid == value).any(dim=0).all().item())
                # Packing reaches the signed-int64 ceiling without any float cast.
                batch = torch.full((BULK_N,), (1 << (63 - 3 * depth)) - 1, dtype=torch.int64)
                packed = codes | (batch[:, None] << (3 * depth))
                self.assertTrue(torch.equal(packed, reference_codes(grid, depth, batch)))
                self.assertEqual(packed.max().item(), (1 << 63) - 1)
                self.assertTrue(torch.equal(packed >> (3 * depth), batch[:, None].expand(-1, 4)))

    def test_reference_cpu_packing_sort_inverse_and_shuffle(self):
        for n in POINT_COUNTS:
            with self.subTest(N=n):
                grid = coordinates(n, 16)
                spatial = reference_codes(grid, 16)
                self.check_cpu_metadata(spatial, grid, 16)
                self.check_cpu_metadata(spatial, grid, 16, shuffle=True)
                permutation = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
                self.assertTrue(torch.equal(reference_codes(grid[permutation], 16), spatial[permutation]))
                if n > 1:
                    self.assertTrue(torch.equal(spatial[0], spatial[-1]))

    def test_reference_singleton_batch_preserves_row_axis(self):
        grid = torch.tensor([[65535, 0, 1]], dtype=torch.int32)
        for name in ORDERS:
            with self.subTest(order=name):
                unbatched = reference().encode(grid, depth=16, order=name)
                self.assertEqual(unbatched.shape, () if name.startswith("hilbert") else (1,))
                batched = reference().encode(grid, batch=torch.zeros(1, dtype=torch.int64),
                                             depth=16, order=name)
                self.assertEqual(batched.shape, (1,))
                self.assertEqual(batched.item(), unbatched.item())
        self.assertEqual(reference_codes(grid, 16).shape, (1, 4))

    def test_reference_unsupported_depth_and_empty_failures(self):
        # These are existing reference failures, not supported raw-op outputs.
        zero = torch.zeros(1, 3, dtype=torch.int32)
        batch = torch.zeros(1, dtype=torch.int64)
        for name in ("hilbert", "hilbert-trans"):
            with self.subTest(order=name):
                with self.assertRaisesRegex(RuntimeError, "shape"):
                    reference().encode(zero, batch=batch, depth=0, order=name)
                with self.assertRaisesRegex(ValueError, "63-bit budget"):
                    reference().encode(zero, batch=batch, depth=22, order=name)
        for name in ("z", "z-trans"):
            with self.subTest(order=name), self.assertRaisesRegex(IndexError, "out of bounds"):
                reference().encode(torch.tensor([[65536, 0, 0]], dtype=torch.int32),
                                   batch=batch, depth=17, order=name)
        point = reference().Point(grid_coord=zero, batch=batch)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            point.serialization(order=ORDERS)  # max=0 infers depth=0, not depth=1.
        empty = reference().Point(grid_coord=torch.empty(0, 3, dtype=torch.int32),
                                  batch=torch.empty(0, dtype=torch.int64))
        with self.assertRaisesRegex(RuntimeError, "max\\(\\)"):
            empty.serialization(order=ORDERS)


@unittest.skipUnless(TEST_NPU, "requires built grid_encode package; set ASCEND_TEST_NPU=1")
class GridEncodeRuntimeTests(_GridChecks):
    @classmethod
    def setUpClass(cls):
        from ascend.custom_ops.grid_encode.grid_encode import grid_encode
        import torchair

        cls.ops = (grid_encode, torch.ops.graspgenx_grid.grid_encode)
        cls.torchair = torchair

    def setUp(self):
        torch.npu.set_device(NPU_DEVICE)
        previous_jit = not torch.npu.is_jit_compile_false()
        self.addCleanup(torch.npu.set_compile_mode, jit_compile=previous_jit)
        torch.npu.set_compile_mode(jit_compile=False)

    @torch.no_grad()
    def test_eager_jit_false_all_depths_and_point_counts(self):
        # Avoid an N x depth compilation matrix: one N for all depth attributes,
        # then only depth=16 for the remaining alignment/tail and workload sizes.
        cases = [(BULK_N, depth) for depth in DEPTHS]
        cases += [(n, 16) for n in POINT_COUNTS if n != BULK_N]
        for n, depth in cases:
            grid = coordinates(n, depth)
            device_grid = grid.to(NPU_DEVICE)
            for op in self.ops:
                with self.subTest(N=n, depth=depth, op=str(op)):
                    self.check_codes(op(device_grid, depth), grid, depth, device_grid.device)

    @torch.no_grad()
    def test_eager_both_jit_modes(self):
        for jit in (False, True):
            torch.npu.set_compile_mode(jit_compile=jit)
            for n, depth in JIT_CASES:
                grid = coordinates(n, depth)
                device_grid = grid.to(NPU_DEVICE)
                for op in self.ops:
                    with self.subTest(jit=jit, N=n, depth=depth, op=str(op)):
                        self.check_codes(op(device_grid, depth), grid, depth, device_grid.device)

    @torch.no_grad()
    def test_eager_permutations_duplicates_and_changed_storage(self):
        depth = 16
        base = coordinates(65, depth)
        permutation = torch.randperm(len(base), generator=torch.Generator().manual_seed(SEED))
        versions = (base, base ^ 0x55AA, base[permutation], torch.full_like(base, 65535))
        device_grid = base.to(NPU_DEVICE)
        previous = None
        for version, grid in enumerate(versions):
            old_grid = device_grid  # Keep old allocation alive when testing new storage.
            if version == 2:
                device_grid = grid.to(NPU_DEVICE)
                self.assertNotEqual(device_grid.data_ptr(), old_grid.data_ptr())
            else:
                device_grid.copy_(grid)
                self.assertEqual(device_grid.data_ptr(), old_grid.data_ptr())
            for op in self.ops:
                with self.subTest(version=version, op=str(op)):
                    result = self.check_codes(op(device_grid, depth), grid, depth, device_grid.device)
                    self.check_codes(op(device_grid, depth), grid, depth, device_grid.device)
            if previous is not None:
                self.assertFalse(torch.equal(previous, result), "fixture must change codes")
            self.assertTrue(torch.equal(device_grid.cpu(), grid), "input must remain unchanged")
            self.check_cpu_metadata(result, grid, depth, shuffle=True)
            previous = result

    def test_meta_and_fake_output_without_npu_storage(self):
        for kind in ("meta", "fake"):
            with FakeTensorMode() if kind == "fake" else nullcontext():
                device = NPU_DEVICE if kind == "fake" else "meta"
                for n, depth in FULLGRAPH_CASES:
                    grid = torch.empty(n, 3, dtype=torch.int32, device=device)
                    for op in self.ops:
                        with self.subTest(kind=kind, N=n, depth=depth, op=str(op)):
                            result = op(grid, depth)
                            self.assertEqual(result.shape, (n, 4))
                            self.assertEqual(result.dtype, torch.int64)
                            self.assertEqual(result.device, grid.device)
                            self.assertTrue(result.is_contiguous())
                            self.assertFalse(result.requires_grad)
                            self.assertEqual(result.untyped_storage().device.type, "meta")
                            if kind == "fake":
                                self.assertIsInstance(result, FakeTensor)
                            else:
                                self.assertTrue(result.is_meta)

    def test_invalid_contracts_eager_meta_fake(self):
        for kind in ("npu", "meta", "fake"):
            with FakeTensorMode() if kind == "fake" else nullcontext():
                device = NPU_DEVICE if kind in ("npu", "fake") else "meta"
                grid = torch.zeros(17, 3, dtype=torch.int32, device=device)
                cases = [("CPU", torch.zeros(17, 3, dtype=torch.int32), 4)]
                for dtype in (torch.int64, torch.float32, torch.float16, torch.bool):
                    cases.append((str(dtype), torch.zeros(17, 3, dtype=dtype, device=device), 4))
                for shape in ((), (51,), (1, 17, 3), (17, 0), (17, 2), (17, 4), (0, 3), (4097, 3)):
                    cases.append((str(shape), torch.zeros(shape, dtype=torch.int32, device=device), 4))
                for label, bad in (
                    ("transpose", torch.zeros(3, 17, dtype=torch.int32, device=device).T),
                    ("row stride", torch.zeros(34, 3, dtype=torch.int32, device=device)[::2]),
                    ("column stride", torch.zeros(17, 6, dtype=torch.int32, device=device)[:, ::2]),
                    ("broadcast", torch.zeros(1, 3, dtype=torch.int32, device=device).expand(17, 3)),
                ):
                    self.assertFalse(bad.is_contiguous())
                    cases.append((label, bad, 4))
                for depth in (-1, 0, 17, 1.5, "4", None, []):
                    cases.append((f"depth={depth!r}", grid, depth))
                # Deliberately no out-of-range coordinate CONTENT cases: the raw
                # API has a domain precondition, not a host scan or fallback.
                for op in self.ops:
                    for label, bad, depth in cases:
                        with self.subTest(kind=kind, op=str(op), invalid=label):
                            with self.assertRaises((RuntimeError, TypeError, ValueError)):
                                op(bad, depth)

    @torch.no_grad()
    def test_torchair_fullgraph_and_changed_tensor_contents(self):
        grid_encode = self.ops[0]
        for n, depth in FULLGRAPH_CASES:
            with self.subTest(N=n, depth=depth):
                torch._dynamo.reset()
                with tempfile.TemporaryDirectory(prefix="grid-encode-ge-") as directory:
                    config = self.torchair.CompilerConfig()
                    config.debug.graph_dump.type = "txt"
                    config.debug.graph_dump._path = directory
                    real_backend = self.torchair.get_npu_backend(compiler_config=config)
                    captured = []

                    def backend(graph, example_inputs):
                        captured.append({str(node.target).removesuffix(".default")
                                         for node in graph.graph.nodes if node.op == "call_function"})
                        return real_backend(graph, example_inputs)

                    def forward(grid):
                        return grid_encode(grid, depth)

                    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
                    base = coordinates(n, depth)
                    versions = (base, base ^ ((1 << depth) - 1), base.flip(0))
                    device_grid = base.to(NPU_DEVICE)
                    previous = None
                    for version, grid in enumerate(versions):
                        old_grid = device_grid
                        if version == 2:
                            device_grid = grid.to(NPU_DEVICE)
                            self.assertNotEqual(device_grid.data_ptr(), old_grid.data_ptr())
                        else:
                            device_grid.copy_(grid)
                            self.assertEqual(device_grid.data_ptr(), old_grid.data_ptr())
                        self.check_codes(forward(device_grid), grid, depth, device_grid.device)
                        result = self.check_codes(compiled(device_grid), grid, depth, device_grid.device)
                        self.check_codes(compiled(device_grid), grid, depth, device_grid.device)
                        if previous is not None:
                            self.assertFalse(torch.equal(previous, result), "changed input reused output")
                        self.assertTrue(torch.equal(device_grid.cpu(), grid))
                        previous = result
                    self.assertEqual(len(captured), 1, "same shape must not recompile on storage changes")
                    self.assertEqual(captured[0], {"graspgenx_grid.grid_encode"})
                    paths = list(Path(directory).glob("dynamo_optimized_graph_*.txt"))
                    self.assertEqual(len(paths), 1, "require real GE, not a passthrough FX callback")
                    text = paths[0].read_text()
                    types = re.findall(r'^\s*type:\s*"([^"]+)"', text, re.MULTILINE)
                    self.assertEqual(types.count("GridEncode"), 1)
                    self.assertFalse(any(marker in op.lower() for op in types
                                         for marker in ("fallback", "pyfunc", "python", "callback")))
                    self.assertRegex(text, rf'key: "depth"\s+value\s*\{{\s*i: {depth}\s')


@unittest.skipUnless(TEST_NPU, "requires built grid_encode OPP; set ASCEND_TEST_NPU=1")
class GridEncodeArtifactTests(unittest.TestCase):
    def test_platform_registration_artifacts(self):
        tbe = OP_ROOT / "opp/vendors/graspgenx_grid/op_impl/ai_core/tbe"
        self.assertTrue((OP_ROOT / "build/torch_bridge.so").is_file())
        for directory in (tbe / "config", tbe / "kernel/config"):
            self.assertEqual({p.name for p in directory.iterdir() if p.is_dir()}, {"ascend310p"})
        info = json.loads((tbe / "config/ascend310p/aic-ascend310p-ops-info.json").read_text())
        binaries = json.loads((tbe / "kernel/config/ascend310p/binary_info_config.json").read_text())
        self.assertEqual(set(info), {"GridEncode"})
        self.assertEqual(set(binaries), {"GridEncode"})
        op = info["GridEncode"]
        for field in ("opFile", "opInterface"):
            self.assertEqual(op[field]["value"], "grid_encode")
        self.assertEqual({key for key in op if re.fullmatch(r"input\d+", key)}, {"input0"})
        self.assertEqual({key for key in op if re.fullmatch(r"output\d+", key)}, {"output0"})
        for field, dtype in (("input0", "int32"), ("output0", "int64")):
            self.assertEqual(op[field]["dtype"], dtype)
            self.assertEqual(op[field]["format"], "ND")
            self.assertEqual(op[field]["paramType"], "required")
        self.assertEqual(op["attr"]["list"], "depth")
        self.assertEqual(op["attr_depth"]["type"], "int")
        self.assertEqual(op["attr_depth"]["paramType"], "required")
        autogen = OP_ROOT / "build/cmake_opp/autogen"
        api = (autogen / "aclnn_grid_encode.cpp").read_text()
        support = re.search(r"socSupportList\[\]\s*=\s*\{([^}]*)\}", api)
        self.assertIsNotNone(support)
        self.assertEqual(re.findall(r"SOC_VERSION_\w+", support.group(1)), ["SOC_VERSION_ASCEND310P"])
        header = (autogen / "aclnn_grid_encode.h").read_text()
        signature = re.search(r"aclnnGridEncodeGetWorkspaceSize\s*\(([^;]+)\);", header)
        self.assertIsNotNone(signature)
        self.assertEqual(len(re.findall(r"const aclTensor\s*\*", signature.group(1))), 2)
        self.assertRegex(signature.group(1), r"int64_t\s+depth\b")
        self.assertRegex(header, r"aclnnGridEncode\s*\(")
        self.assertTrue(binaries["GridEncode"]["binaryList"])
        for binary in binaries["GridEncode"]["binaryList"]:
            for field in ("binPath", "jsonPath"):
                self.assertEqual(Path(binary[field]).parts[0], "ascend310p")
                self.assertTrue((tbe / "kernel" / binary[field]).is_file())
            metadata = json.loads((tbe / "kernel" / binary["jsonPath"]).read_text())
            self.assertEqual(metadata["coreType"], "AiCore")
        print(json.dumps({"registered_ops": ["GridEncode"], "soc_support": ["ascend310p"],
                          "scope": "new OPP artifacts only; no unsupported-device execution"}), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(CPU_THREADS)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
