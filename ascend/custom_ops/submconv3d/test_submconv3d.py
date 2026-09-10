"""Run after build/environment setup: python test_submconv3d.py [-v].

Set SUBM_SMOKE=1 to include N=2048/3500/4096. No packages are installed or built here.
Metrics go to stdout; latency includes Python/launch overhead, and the first
compiled call includes compilation. Only cosine >= 0.9999 is a numeric gate.
"""

import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from ascend.custom_ops.submconv3d.submconv3d import (
    SubMConv3d, build_subm_map, subm_conv3d,
)
import torchair


CASES = (
    (1, 16, 32, 1),
    (17, 32, 16, 1),
    (31, 16, 48, 3),
    (64, 48, 32, 3),
    (17, 32, 16, 5),
    (65, 16, 32, 5),
    (1, 512, 16, 3),
    (1, 16, 512, 5),
)
COSINE_GATE = 0.9999


def coordinates(n, seed):
    generator = torch.Generator().manual_seed(seed)
    side = max(3, math.ceil((2 * n) ** (1 / 3)))
    axis = range(-(side // 2), side - side // 2)
    candidates = list(itertools.product(range(2), axis, axis, axis))
    # Include identical xyz in different batches to catch cross-batch lookup.
    anchors = [(0, axis.start, axis.start, axis.start)]
    if n > 1:
        anchors.append((1, axis.start, axis.start, axis.start))
    candidates = [p for p in candidates if p not in anchors]
    order = torch.randperm(len(candidates), generator=generator).tolist()
    rows = anchors + [candidates[i] for i in order[: n - len(anchors)]]
    result = torch.tensor(rows, dtype=torch.int32)
    return result[torch.randperm(n, generator=generator)].contiguous()


def reference_map(indices, kernel_size):
    rows = [tuple(row) for row in indices.tolist()]
    lookup = {row: i for i, row in enumerate(rows)}
    assert len(lookup) == len(rows), "reference requires unique coordinates"
    radius = kernel_size // 2
    offsets = list(itertools.product(range(-radius, radius + 1), repeat=3))
    result = torch.full((len(rows), ((kernel_size**3 + 7) // 8) * 8), -1, dtype=torch.int32)
    for i, (batch, x, y, z) in enumerate(rows):
        for column, (dx, dy, dz) in enumerate(offsets):
            result[i, column] = lookup.get((batch, x + dx, y + dy, z + dz), -1)
    return result


def reference_conv(features, indices, weight, bias, kernel_size):
    """CPU FP32 coordinate lookup and accumulation, independent of device maps."""
    rows = [tuple(row) for row in indices.tolist()]
    lookup = {row: i for i, row in enumerate(rows)}
    assert len(lookup) == len(rows)
    features, weight = features.float(), weight.float()
    output = torch.zeros(len(rows), weight.shape[2], dtype=torch.float32)
    radius = kernel_size // 2
    for column, (dx, dy, dz) in enumerate(
        itertools.product(range(-radius, radius + 1), repeat=3)
    ):
        destinations, sources = [], []
        for i, (batch, x, y, z) in enumerate(rows):
            source = lookup.get((batch, x + dx, y + dy, z + dz))
            if source is not None:
                destinations.append(i)
                sources.append(source)
        if sources:
            output[destinations] += features[sources] @ weight[column]
    return output if bias is None else output + bias.float()


class SubMConvTests(unittest.TestCase):
    def test_platform_registration_artifacts(self):
        root = Path(__file__).resolve().parent
        tbe = root / "opp/vendors/graspgenx_subm/op_impl/ai_core/tbe"
        expected = {"BuildSubmMap": "build_subm_map", "SubmConv3d": "subm_conv3d"}
        for directory in (tbe / "config", tbe / "kernel/config"):
            self.assertEqual({p.name for p in directory.iterdir() if p.is_dir()}, {"ascend310p"})
        info = json.loads((tbe / "config/ascend310p/aic-ascend310p-ops-info.json").read_text())
        binaries = json.loads((tbe / "kernel/config/ascend310p/binary_info_config.json").read_text())
        self.assertEqual(set(info), set(expected))
        self.assertEqual(set(binaries), set(expected))
        for op, snake in expected.items():
            self.assertEqual(info[op]["opFile"]["value"], snake)
            self.assertEqual(info[op]["opInterface"]["value"], snake)
            api = (root / f"build/cmake_opp/autogen/aclnn_{snake}.cpp").read_text()
            support = re.search(r"socSupportList\[\]\s*=\s*\{([^}]*)\}", api)
            self.assertIsNotNone(support)
            self.assertEqual(re.findall(r"SOC_VERSION_\w+", support.group(1)), ["SOC_VERSION_ASCEND310P"])
            self.assertTrue(binaries[op]["binaryList"])
            for binary in binaries[op]["binaryList"]:
                self.assertEqual(Path(binary["binPath"]).parts[0], "ascend310p")
                self.assertTrue((tbe / "kernel" / binary["binPath"]).is_file())
                metadata = json.loads((tbe / "kernel" / binary["jsonPath"]).read_text())
                self.assertEqual(metadata["coreType"], "AiCore")
        print(json.dumps({"registered_ops": list(expected), "soc_support": ["ascend310p"],
                          "scope": "generated metadata and binaries; not an unsupported-device execution test"}), flush=True)

    def compare(self, label, actual, expected, **metrics):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape), label)
        self.assertEqual(actual.dtype, torch.float16, label)
        self.assertEqual(actual.device.type, "npu", label)
        self.assertTrue(actual.is_contiguous(), label)
        result = actual.detach().cpu().float()
        self.assertTrue(torch.isfinite(result).all().item(), label)
        self.assertTrue(torch.isfinite(expected).all().item(), label)
        a, b = result.flatten().double(), expected.flatten().double()
        self.assertGreater(a.norm().item(), 0, "cosine is undefined for a zero output")
        self.assertGreater(b.norm().item(), 0, "cosine is undefined for a zero reference")
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        error = result - expected
        print(json.dumps({
            "comparison": label, "cosine": cosine,
            "max_abs": error.abs().max().item(),
            "mean_abs": error.abs().mean().item(),
            "rmse": error.square().mean().sqrt().item(),
            "relative_l2": (error.norm() / expected.norm().clamp_min(1e-12)).item(),
            **metrics,
        }), flush=True)
        self.assertGreaterEqual(cosine, COSINE_GATE, label)
        return result

    def check_map(self, label, actual, expected):
        self.assertEqual(actual.device.type, "npu")
        self.assertEqual(actual.dtype, torch.int32)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertTrue(actual.is_contiguous())
        result = actual.cpu()
        mismatches = (result != expected).sum().item()
        print(json.dumps({"map": label, "shape": list(result.shape),
                          "mismatches": mismatches}), flush=True)
        self.assertEqual(mismatches, 0, label)

    def check_ge(self, directory, captured, kernel_size, cached):
        self.assertEqual(len(captured), 1, "must capture one full graph without recompiling")
        self.assertIn("graspgenx_subm.subm_conv3d", captured[0])
        self.assertEqual("graspgenx_subm.build_subm_map" in captured[0], not cached)
        paths = list(Path(directory).glob("dynamo_optimized_graph_*.txt"))
        self.assertEqual(len(paths), 1, "a real optimized GE graph must be dumped")
        text = paths[0].read_text()
        types = re.findall(r'^\s*type:\s*"([^"]+)"', text, re.MULTILINE)
        self.assertEqual(types.count("SubmConv3d"), 1)
        self.assertEqual(types.count("BuildSubmMap"), 0 if cached else 1)
        self.assertFalse(any("fallback" in t.lower() or "pyfunc" in t.lower() for t in types))
        if not cached:
            self.assertRegex(text, rf'key: "kernel_size"\s+value\s*\{{\s*i: {kernel_size}\s')
        print(json.dumps({"fullgraph": True, "cached_map": cached,
                          "backend_calls": len(captured), "ge_op_types": types}), flush=True)

    @torch.no_grad()
    def run_case(self, n, cin, cout, kernel_size):
        for cached in (False, True):
            torch._dynamo.reset()
            with tempfile.TemporaryDirectory(prefix="graspgenx_subm-ge-") as directory:
                config = torchair.CompilerConfig()
                config.debug.graph_dump.type = "txt"
                config.debug.graph_dump._path = directory
                real_backend = torchair.get_npu_backend(compiler_config=config)
                captured = []

                def backend(graph, example_inputs):
                    captured.append({str(node.target).removesuffix(".default")
                                     for node in graph.graph.nodes if node.op == "call_function"})
                    return real_backend(graph, example_inputs)

                def forward(features, indices, weight, bias, neighbor_map):
                    if neighbor_map is None:
                        neighbor_map = build_subm_map(indices, kernel_size)
                    output = subm_conv3d(features, indices, weight, bias,
                                         kernel_size, neighbor_map)
                    return neighbor_map, output

                compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
                indices_cpu = coordinates(n, 101)
                indices = indices_cpu.to("npu")
                fixed_map = build_subm_map(indices, kernel_size) if cached else None
                features = torch.empty(n, cin, dtype=torch.float16, device="npu")
                weight = torch.empty(kernel_size**3, cin, cout, dtype=torch.float16, device="npu")
                bias = torch.empty(cout, dtype=torch.float16, device="npu")
                previous = None
                for version in range(3):
                    generator = torch.Generator().manual_seed(1001 + version)
                    features_cpu = torch.randn(n, cin, generator=generator).half()
                    weight_cpu = (torch.randn(kernel_size**3, cin, cout, generator=generator)
                                  / math.sqrt(kernel_size**3 * cin)).half()
                    bias_cpu = (torch.randn(cout, generator=generator) * 0.1).half()
                    if not cached:
                        indices_cpu = coordinates(n, 101 + version)
                        indices.copy_(indices_cpu)
                    # Change storage contents, not only Python objects, on repeated calls.
                    features.copy_(features_cpu)
                    weight.copy_(weight_cpu)
                    bias.copy_(bias_cpu)
                    expected_map = reference_map(indices_cpu, kernel_size)
                    expected = reference_conv(features_cpu, indices_cpu, weight_cpu,
                                              bias_cpu, kernel_size)
                    data = dict(N=n, Cin=cin, Cout=cout, K=kernel_size,
                                cached_map=cached, version=version)
                    results = {}
                    for name, function in (("eager", forward), ("torchair", compiled)):
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        actual_map, actual = function(features, indices, weight, bias, fixed_map)
                        torch.npu.synchronize()
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        self.check_map(name, actual_map, expected_map)
                        results[name] = self.compare(name, actual, expected, **data,
                                                     latency_ms=elapsed_ms,
                                                      cold_call=version == 0,
                                                      includes_graph_compile=name == "torchair" and version == 0)
                    self.compare("eager_vs_torchair", actual, results["eager"], **data)
                    if previous is not None:
                        self.assertFalse(torch.equal(results["torchair"], previous),
                                         "changed inputs must not reuse cached outputs")
                    previous = results["torchair"].clone()
                    _, repeat = compiled(features, indices, weight, bias, fixed_map)
                    repeated = self.compare("repeat", repeat, expected, **data)
                    print(json.dumps({"repeat_max_abs": (repeated - previous).abs().max().item(),
                                      **data}), flush=True)
                self.check_ge(directory, captured, kernel_size, cached)

    def test_eager_and_torchair_fullgraph(self):
        for case in CASES:
            with self.subTest(case=case):
                self.run_case(*case)

    @unittest.skipUnless(os.environ.get("SUBM_SMOKE") == "1", "optional large-N smoke")
    def test_smoke_large(self):
        for n in (2048, 3500, 4096):
            with self.subTest(N=n):
                self.run_case(n, 16, 32, 3)

    @torch.no_grad()
    def test_offset_order_batches_and_missing_neighbors(self):
        # Deliberately asymmetric: +/- directions and x/y/z cannot be swapped.
        indices_cpu = torch.tensor([[0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0],
                                    [0, 0, 0, 2], [1, 0, 0, 0]], dtype=torch.int32)
        features_cpu = torch.arange(1, 81).reshape(5, 16).half() / 80
        for kernel_size in (1, 3, 5):
            weight_cpu = torch.arange(1, kernel_size**3 + 1).half()[:, None, None]
            weight_cpu = (weight_cpu.expand(-1, 16, 32) / (kernel_size**3 * 16)).contiguous()
            indices, features, weight = (t.to("npu") for t in (indices_cpu, features_cpu, weight_cpu))
            neighbors = build_subm_map(indices, kernel_size)
            self.check_map("asymmetric", neighbors, reference_map(indices_cpu, kernel_size))
            output = torch.ops.graspgenx_subm.subm_conv3d(features, neighbors, weight)
            expected = reference_conv(features_cpu, indices_cpu, weight_cpu, None, kernel_size)
            self.compare("asymmetric_raw_op", output, expected, K=kernel_size)
            self.compare("uncached_python_api", subm_conv3d(features, indices, weight,
                         kernel_size=kernel_size), expected, K=kernel_size)

    def test_module(self):
        indices_cpu = coordinates(17, 31)
        features_cpu = torch.randn(17, 16).half()
        features, indices = features_cpu.to("npu"), indices_cpu.to("npu")
        features.requires_grad_()
        for use_bias in (False, True):
            layer = SubMConv3d(16, 32, 3, bias=use_bias).to("npu").eval()
            with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "forward only"):
                layer(features, indices)
            with torch.no_grad():
                expected = reference_conv(features_cpu, indices_cpu, layer.weight.cpu(),
                                          None if layer.bias is None else layer.bias.cpu(), 3)
                output = layer(features, indices)
                self.assertFalse(output.requires_grad)
                self.compare("module", output, expected, bias=use_bias)
                neighbors = build_subm_map(indices, 3)
                self.compare("module_cached_map", layer(features, indices, neighbors),
                             expected, bias=use_bias)
                compiled = torch.compile(layer, backend=torchair.get_npu_backend(),
                                         fullgraph=True, dynamic=False)
                self.compare("module_torchair", compiled(features, indices),
                             expected, bias=use_bias)

    def test_fake_and_meta_shapes(self):
        for mode in (None, FakeTensorMode()):
            context = torch.no_grad() if mode is None else mode
            with context:
                device = "meta" if mode is None else "npu:0"
                for n in (1, 17, 4096):
                    for kernel_size in (1, 3, 5):
                        indices = torch.empty(n, 4, dtype=torch.int32, device=device)
                        features = torch.empty(n, 16, dtype=torch.float16, device=device)
                        weight = torch.empty(kernel_size**3, 16, 32, dtype=torch.float16, device=device)
                        neighbors = torch.ops.graspgenx_subm.build_subm_map(indices, kernel_size)
                        output = torch.ops.graspgenx_subm.subm_conv3d(features, neighbors, weight)
                        self.assertEqual(neighbors.shape, (n, ((kernel_size**3 + 7) // 8) * 8))
                        self.assertEqual(neighbors.dtype, torch.int32)
                        self.assertEqual(output.shape, (n, 32))
                        self.assertEqual(output.dtype, torch.float16)
                        self.assertEqual(output.device, features.device)
                        self.assertTrue(output.is_contiguous())
                        self.assertFalse(output.requires_grad)

    def contract_cases(self, device):
        def tensor(shape, dtype=torch.float16):
            return torch.empty(shape, device=device, dtype=dtype)

        indices = tensor((17, 4), torch.int32)
        features = tensor((17, 16))
        weight = tensor((27, 16, 32))
        neighbors = tensor((17, 32), torch.int32)
        map_op = torch.ops.graspgenx_subm.build_subm_map
        conv_op = torch.ops.graspgenx_subm.subm_conv3d
        # Each input has rank, dtype and contiguity checked through raw dispatcher ops.
        for bad, message in (
            (tensor((17, 4), torch.int64), "indices has unsupported dtype"),
            (tensor((68,), torch.int32), "indices has unsupported rank"),
            (tensor((17, 5), torch.int32), "indices must have shape"),
            (tensor((4, 17), torch.int32).t(), "indices must be contiguous"),
            (tensor((0, 4), torch.int32), "N must be"),
            (tensor((4097, 4), torch.int32), "N must be"),
        ):
            yield lambda bad=bad: map_op(bad, 3), message
        for kernel_size in (0, 2, 4, 7):
            yield lambda k=kernel_size: map_op(indices, k), "kernel_size must be"
        for bad, message in (
            (tensor((17, 16), torch.float32), "features has unsupported dtype"),
            (tensor((272,)), "features has unsupported rank"),
            (tensor((16, 17)).t(), "features must be contiguous"),
            (tensor((0, 16)), "N must be"),
            (tensor((4097, 16)), "N must be"),
            (tensor((17, 8)), "Cin must be"),
            (tensor((17, 17)), "Cin must be"),
            (tensor((17, 528)), "Cin must be"),
        ):
            yield lambda bad=bad: conv_op(bad, neighbors, weight), message
        for bad, message in (
            (tensor((27, 16, 32), torch.float32), "weight has unsupported dtype"),
            (tensor((16, 32)), "weight has unsupported rank"),
            (tensor((27, 32, 16)).transpose(1, 2), "weight must be contiguous"),
            (tensor((8, 16, 32)), "weight must have K"),
            (tensor((27, 32, 32)), "weight Cin must match"),
            (tensor((27, 16, 8)), "Cout must be"),
            (tensor((27, 16, 17)), "Cout must be"),
            (tensor((27, 16, 528)), "Cout must be"),
        ):
            yield lambda bad=bad: conv_op(features, neighbors, bad), message
        for bad, message in (
            (tensor((17, 32), torch.int64), "neighbors has unsupported dtype"),
            (tensor((544,), torch.int32), "neighbors has unsupported rank"),
            (tensor((32, 17), torch.int32).t(), "neighbors must be contiguous"),
            (tensor((16, 32), torch.int32), "neighbors must have shape"),
            (tensor((17, 27), torch.int32), "neighbors must have shape"),
            (tensor((17, 40), torch.int32), "neighbors must have shape"),
        ):
            yield lambda bad=bad: conv_op(features, bad, weight), message
        for target in ("features", "weight"):
            f = features.detach().requires_grad_(target == "features")
            w = weight.detach().requires_grad_(target == "weight")
            yield lambda f=f, w=w: conv_op(f, neighbors, w), "forward only"
        for bad, message in (
            (tensor((32,), torch.float32), "bias has unsupported dtype"),
            (tensor((1, 32)), "bias has unsupported rank"),
            (tensor((64,))[::2], "bias must be contiguous"),
            (tensor((16,)), "bias must have shape"),
            (tensor((32,)).requires_grad_(), "forward only"),
        ):
            yield lambda bad=bad: subm_conv3d(features, indices, weight, bad,
                                             neighbor_map=neighbors), message
        yield lambda: subm_conv3d(features, indices, weight, kernel_size=1,
                                  neighbor_map=neighbors), "weight K"
        yield lambda: subm_conv3d(features, indices[:16], weight,
                                  neighbor_map=neighbors), "same N"

    def test_negative_contracts_eager_meta_fake(self):
        for kind in ("npu", "meta", "fake"):
            context = FakeTensorMode() if kind == "fake" else torch.enable_grad()
            with context, torch.enable_grad():
                device = "npu:0" if kind == "fake" else kind
                count = 0
                for function, message in self.contract_cases(device):
                    with self.subTest(kind=kind, message=message):
                        with self.assertRaisesRegex(RuntimeError, message):
                            function()
                    count += 1
                print(json.dumps({"negative_contracts": kind, "cases": count}), flush=True)

    def test_cpu_and_mixed_device_rejected(self):
        for context in (torch.no_grad(), FakeTensorMode()):
            with context:
                indices = torch.empty(17, 4, dtype=torch.int32)
                features = torch.empty(17, 16, dtype=torch.float16)
                weight = torch.empty(27, 16, 32, dtype=torch.float16)
                neighbors = torch.empty(17, 32, dtype=torch.int32)
                with self.assertRaises(RuntimeError):
                    torch.ops.graspgenx_subm.build_subm_map(indices, 3)
                with self.assertRaises(RuntimeError):
                    torch.ops.graspgenx_subm.subm_conv3d(features, neighbors, weight)
                with self.assertRaisesRegex(RuntimeError, "NPU tensor"):
                    subm_conv3d(features, indices, weight, neighbor_map=neighbors)
                npu_features, npu_indices, npu_weight, npu_neighbors = (
                    t.to("npu") for t in (features, indices, weight, neighbors)
                )
                for args in ((features, npu_neighbors, npu_weight),
                             (npu_features, neighbors, npu_weight),
                             (npu_features, npu_neighbors, weight)):
                    with self.assertRaises(RuntimeError):
                        torch.ops.graspgenx_subm.subm_conv3d(*args)
                with self.assertRaises(RuntimeError):
                    subm_conv3d(npu_features, indices, npu_weight, neighbor_map=npu_neighbors)
                with self.assertRaises(RuntimeError):
                    subm_conv3d(npu_features, npu_indices, npu_weight,
                                torch.empty(32, dtype=torch.float16), neighbor_map=npu_neighbors)
        # A second fake NPU tests device-index checks without requiring two cards.
        with FakeTensorMode(), torch.no_grad():
            features = torch.empty(17, 16, dtype=torch.float16, device="npu:0")
            weight = torch.empty(27, 16, 32, dtype=torch.float16, device="npu:1")
            neighbors = torch.empty(17, 32, dtype=torch.int32, device="npu:0")
            with self.assertRaises(RuntimeError):
                torch.ops.graspgenx_subm.subm_conv3d(features, neighbors, weight)

    def test_invalid_module_arguments(self):
        for args in ((8, 16, 3), (16, 17, 3), (528, 16, 3), (16, 528, 3),
                     (16, 16, 2), (16, 16, 3.0)):
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                SubMConv3d(*args)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.npu.set_compile_mode(jit_compile=True)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
