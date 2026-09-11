"""Builder-only tests. Run only after main coordinates regeneration/build/device use.

No build, installation, benchmark, or CPU execution fallback is performed here.
All maps must match exactly, including padding. CPU code is only the test oracle.
"""

from contextlib import nullcontext
import itertools
from pathlib import Path
import re
import sys
import tempfile
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ascend.custom_ops.submconv3d.submconv3d import build_subm_map
import torchair


def reference_hash(indices):
    indices = indices.long()
    return (indices[..., 0] * 334214467 + indices[..., 1] * 73856093
            + indices[..., 2] * 19349669 + indices[..., 3] * 83492791)


def representatives(indices):
    # Exactly the original CPU sort, including its unspecified order of ties.
    keys, order = reference_hash(indices).sort()
    first = torch.ones(keys.numel(), dtype=torch.bool)
    first[1:] = keys[1:] != keys[:-1]
    return keys[first], order[first].int(), keys, order


def hash_map(indices, kernel_size, keys, rows):
    radius = kernel_size // 2
    offsets = torch.tensor([(0, *d) for d in itertools.product(
        range(-radius, radius + 1), repeat=3)], dtype=torch.int64)
    queries = reference_hash(indices.long()[:, None, :] + offsets[None, :, :])
    positions = torch.searchsorted(keys, queries).clamp(max=keys.numel() - 1)
    result = torch.full((len(indices), ((kernel_size**3 + 7) // 8) * 8),
                        -1, dtype=torch.int32)
    result[:, :kernel_size**3] = torch.where(keys[positions] == queries,
                                            rows[positions], -1).int()
    return result


def coordinate_map(indices, kernel_size):
    points = [tuple(p) for p in indices.tolist()]
    lookup = {p: i for i, p in enumerate(points)}
    assert len(lookup) == len(points), "coordinate-only API requires unique rows"
    radius = kernel_size // 2
    offsets = list(itertools.product(range(-radius, radius + 1), repeat=3))
    padding = [-1] * (((kernel_size**3 + 7) // 8) * 8 - kernel_size**3)
    return torch.tensor([
        [lookup.get((b, x + dx, y + dy, z + dz), -1) for dx, dy, dz in offsets] + padding
        for b, x, y, z in points
    ], dtype=torch.int32)


def coordinates(n):
    lo, hi = -(2**31), 2**31 - 1
    anchors = [
        (lo, lo, lo, lo), (lo, lo + 1, lo, lo),
        (hi, hi, hi, hi), (hi, hi - 1, hi, hi),
        (-3, -5, -2, -1), (-3, -4, -2, -1),
        (7, lo, 0, hi), (7, hi, 0, hi),
        (0, 0, 0, 0), (0, 19349669, -73856093, 0),
        (0, 0, 0, 1), (0, 19349669, -73856093, 2),
        (73856093, -334214467, 0, 0), (1, 0, 0, 0),
    ]
    # Distinct input coordinates can have identical reference hashes. The z=2
    # anchor also collides with a queried coordinate that is not in the input.
    points = anchors[:n] + [(2, j // 64, (j // 8) % 8, j % 8)
                           for j in range(max(0, n - len(anchors)))]
    return torch.tensor(points, dtype=torch.int32)


class BuildSubmMapTests(unittest.TestCase):
    def check_map(self, actual, expected):
        self.assertEqual(actual.device.type, "npu")
        self.assertEqual(actual.dtype, torch.int32)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual.cpu(), expected), "neighbor map mismatch")

    def test_standalone_extremes_tails_and_large_n(self):
        for n in (1, 17, 127, 128, 129, 257, 4095, 4096):
            cpu = coordinates(n)
            indices = cpu.to("npu")
            for k in (1, 3, 5):
                with self.subTest(N=n, K=k):
                    expected = coordinate_map(cpu, k)
                    self.check_map(build_subm_map(indices, k), expected)
                    if n == 17:
                        self.check_map(torch.ops.graspgenx_subm.build_subm_map(indices, k), expected)

    def test_private_table_probe_chain(self):
        # Up to 57 distinct coordinates per z plane have the same hash/bucket.
        # Adjacent planes still need to find their own exact-coordinate row.
        cpu = torch.tensor([(0, (j % 57 - 28) * 19349669,
                             -(j % 57 - 28) * 73856093, j // 57)
                            for j in range(257)], dtype=torch.int32)
        for k in (3, 5):
            with self.subTest(K=k):
                self.check_map(build_subm_map(cpu.to("npu"), k), coordinate_map(cpu, k))

    def test_reference_sort_duplicates_and_hash_collisions(self):
        for n in (1, 17, 129, 4096):
            cpu = coordinates(n)
            if n > 1:
                cpu[-1] = cpu[8]
            keys, rows, original_keys, original_order = representatives(cpu)
            indices, npu_keys, npu_rows = (x.to("npu") for x in (cpu, keys, rows))
            for k in (1, 3, 5):
                with self.subTest(N=n, M=len(keys), K=k):
                    expected = hash_map(cpu, k, original_keys, original_order)
                    self.assertTrue(torch.equal(hash_map(cpu, k, keys, rows), expected))
                    self.check_map(build_subm_map(indices, k, npu_keys, npu_rows), expected)
                    if k != 1 and n > 1:
                        column = k**3 // 2 + 1
                        self.assertEqual(expected[10, column].item(), 11,
                                         "query collision must return a non-coordinate match")

    def test_table_lengths_tails_and_authoritative_k1(self):
        for m in (1, 3, 4, 7, 8, 9, 17, 4096):
            cpu = torch.zeros(min(m + 1, 4096), 4, dtype=torch.int32)
            cpu[:m, 1] = torch.arange(m, dtype=torch.int32)
            keys, rows, _, _ = representatives(cpu)
            self.assertEqual(len(keys), m)
            source = m if m < 4096 else 0
            rows[0] = source  # Prefer the duplicate's non-minimum row when present.
            indices, npu_keys, npu_rows = (x.to("npu") for x in (cpu, keys, rows))
            for k in (1, 3, 5):
                with self.subTest(M=m, K=k):
                    expected = hash_map(cpu, k, keys, rows)
                    self.assertEqual(expected[0, k**3 // 2].item(), source)
                    self.check_map(build_subm_map(indices, k, npu_keys, npu_rows), expected)

    def test_meta_and_fake_shapes(self):
        for kind in ("meta", "fake"):
            with FakeTensorMode() if kind == "fake" else nullcontext():
                device = "npu:0" if kind == "fake" else "meta"
                for n in (1, 17, 4096):
                    for m in (None, 1, n):
                        indices = torch.empty(n, 4, dtype=torch.int32, device=device)
                        keys = None if m is None else torch.empty(m, dtype=torch.int64, device=device)
                        rows = None if m is None else torch.empty(m, dtype=torch.int32, device=device)
                        for k in (1, 3, 5):
                            for op in (build_subm_map, torch.ops.graspgenx_subm.build_subm_map):
                                result = op(indices, k, keys, rows)
                                self.assertEqual(result.shape, (n, ((k**3 + 7) // 8) * 8))
                                self.assertEqual(result.dtype, torch.int32)
                                self.assertEqual(result.device, indices.device)
                                self.assertTrue(result.is_contiguous())

    def test_optional_contracts_eager_meta_fake(self):
        for kind in ("npu", "meta", "fake"):
            with FakeTensorMode() if kind == "fake" else nullcontext():
                device = "npu:0" if kind == "fake" else kind
                indices = torch.empty(17, 4, dtype=torch.int32, device=device)
                keys = torch.empty(9, dtype=torch.int64, device=device)
                rows = torch.empty(9, dtype=torch.int32, device=device)
                cases = [(keys, None, "both present"), (None, rows, "both present")]
                for slot, dtype, name in ((0, torch.int64, "sorted_keys"),
                                           (1, torch.int32, "source_rows")):
                    wrong_dtype = torch.int32 if dtype == torch.int64 else torch.int64
                    for bad, message in (
                        (torch.empty(9, dtype=wrong_dtype, device=device), f"{name} has unsupported dtype"),
                        (torch.empty(9, dtype=torch.float32, device=device), f"{name} has unsupported dtype"),
                        (torch.empty(1, 9, dtype=dtype, device=device), f"{name} has unsupported rank"),
                        (torch.empty(18, dtype=dtype, device=device)[::2], f"{name} must be contiguous"),
                        (torch.empty(9, dtype=dtype, device="cpu"), "NPU tensor"),
                    ):
                        pair = [keys, rows]
                        pair[slot] = bad
                        cases.append((*pair, message))
                for m in (0, 18):
                    cases.append((torch.empty(m, dtype=torch.int64, device=device), rows, "M must be"))
                for m in (0, 8, 18):
                    cases.append((keys, torch.empty(m, dtype=torch.int32, device=device), "same M"))
                for op in (build_subm_map, torch.ops.graspgenx_subm.build_subm_map):
                    for bad_keys, bad_rows, message in cases:
                        with self.subTest(kind=kind, op=str(op), message=message):
                            with self.assertRaisesRegex(RuntimeError, message):
                                op(indices, 3, bad_keys, bad_rows)

    def test_cpu_and_cross_device_rejected(self):
        for op in (build_subm_map, torch.ops.graspgenx_subm.build_subm_map):
            cpu = coordinates(17)
            keys, rows, _, _ = representatives(cpu)
            with self.assertRaises(RuntimeError):
                op(cpu, 3, keys, rows)
            with self.assertRaises(RuntimeError):
                op(cpu, 3, keys.to("npu"), rows.to("npu"))
        with FakeTensorMode():
            indices = torch.empty(17, 4, dtype=torch.int32, device="npu:0")
            keys = torch.empty(9, dtype=torch.int64, device="npu:0")
            rows = torch.empty(9, dtype=torch.int32, device="npu:0")
            other_keys = torch.empty(9, dtype=torch.int64, device="npu:1")
            other_rows = torch.empty(9, dtype=torch.int32, device="npu:1")
            for op in (build_subm_map, torch.ops.graspgenx_subm.build_subm_map):
                for pair in ((other_keys, rows), (keys, other_rows)):
                    with self.assertRaisesRegex(RuntimeError, "same device"):
                        op(indices, 3, *pair)

    def test_jit_disabled_model_shapes(self):
        torch.npu.set_compile_mode(jit_compile=False)
        try:
            cpu = torch.zeros(64, 4, dtype=torch.int32)
            cpu[:, 1] = torch.arange(64)
            for table in (False, True):
                if table:
                    cpu[-1] = cpu[0]
                keys, rows, _, _ = representatives(cpu)
                actual = build_subm_map(cpu.npu(), 3,
                                        keys.npu() if table else None,
                                        rows.npu() if table else None)
                self.check_map(actual, hash_map(cpu, 3, keys, rows) if table else coordinate_map(cpu, 3))
        finally:
            torch.npu.set_compile_mode(jit_compile=True)

    def test_torchair_fullgraph_and_changed_storage(self):
        for table, n, k in ((False, 17, 1), (False, 17, 3), (False, 129, 5),
                             (True, 17, 1), (True, 17, 3), (True, 17, 5)):
            with self.subTest(table=table, N=n, K=k):
                torch._dynamo.reset()
                with tempfile.TemporaryDirectory(prefix="build-subm-map-ge-") as directory:
                    config = torchair.CompilerConfig()
                    config.debug.graph_dump.type = "txt"
                    config.debug.graph_dump._path = directory
                    real_backend = torchair.get_npu_backend(compiler_config=config)
                    captured = []

                    def backend(graph, example_inputs):
                        captured.append(str(graph.graph))
                        return real_backend(graph, example_inputs)

                    def forward(indices, keys=None, rows=None):
                        return build_subm_map(indices, k, keys, rows)

                    compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
                    base = coordinates(n)
                    if table:
                        base = base.clamp(-(2**30), 2**30)
                        base[-1] = base[8]
                    cpu_keys, cpu_rows, _, _ = representatives(base)
                    indices = base.to("npu")
                    keys = cpu_keys.to("npu") if table else None
                    rows = cpu_rows.to("npu") if table else None
                    previous = None
                    for version in range(3):
                        cpu = base.clone() if version < 2 else base.flip(0)
                        if table and version == 2:
                            # A uniform translation preserves M but changes every key.
                            cpu[:, 3] += 10
                        cpu_keys, cpu_rows, _, _ = representatives(cpu)
                        if table and version == 1:
                            position = torch.searchsorted(cpu_keys, torch.tensor(0)).item()
                            cpu_rows[position] = 8 if cpu_rows[position].item() != 8 else n - 1
                        indices.copy_(cpu)
                        if table:
                            keys.copy_(cpu_keys)
                            rows.copy_(cpu_rows)
                        expected = (hash_map(cpu, k, cpu_keys, cpu_rows) if table
                                    else coordinate_map(cpu, k))
                        self.check_map(forward(indices, keys, rows), expected)
                        self.check_map(compiled(indices, keys, rows), expected)
                        self.check_map(compiled(indices, keys, rows), expected)
                        if previous is not None and (table or (version == 2 and k != 1)):
                            self.assertFalse(torch.equal(previous, expected), "fixture must change the map")
                        previous = expected
                    self.assertEqual(len(captured), 1, "same-shaped storage updates must not recompile")
                    self.assertIn("graspgenx_subm.build_subm_map", captured[0])
                    paths = list(Path(directory).glob("dynamo_optimized_graph_*.txt"))
                    self.assertEqual(len(paths), 1, "require a real optimized GE graph")
                    text = paths[0].read_text()
                    types = re.findall(r'^\s*type:\s*"([^"]+)"', text, re.MULTILINE)
                    self.assertEqual(types.count("BuildSubmMap"), 1)
                    self.assertNotIn("SubmConv3d", types)
                    self.assertFalse(any("fallback" in t.lower() or "pyfunc" in t.lower() for t in types))

    def test_torchair_partial_table_rejected(self):
        indices = torch.empty(17, 4, dtype=torch.int32, device="npu")
        keys = torch.empty(9, dtype=torch.int64, device="npu")
        rows = torch.empty(9, dtype=torch.int32, device="npu")
        for op in (build_subm_map, torch.ops.graspgenx_subm.build_subm_map):
            for pair in ((keys, None), (None, rows)):
                torch._dynamo.reset()
                captured = []
                real_backend = torchair.get_npu_backend()

                def backend(graph, example_inputs):
                    captured.append(graph)
                    return real_backend(graph, example_inputs)

                def forward(indices, keys, rows):
                    return op(indices, 3, keys, rows)

                compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
                with self.assertRaises(RuntimeError):
                    compiled(indices, *pair)
                self.assertFalse(captured, "partial tables must fail before GE compilation")


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.npu.set_compile_mode(jit_compile=True)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
