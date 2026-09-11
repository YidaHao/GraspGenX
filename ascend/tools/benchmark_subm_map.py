#!/usr/bin/env python3
"""Builder-only benchmark: one manually selected OPP/bridge per fresh process.

No build, artifact switching, model calls, or CLI. Compare JSON from separate
old-scanner/new-builder runs; source hashes do not identify installed binaries.
"""

import hashlib
import itertools
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_TAG = "unique_coordinate"
N = (1, 17, 64, 128, 129, 256, 512, 2048, 3500, 4096)
K = (1, 3, 5)
WORKLOADS = ("densecube", "spreadgrid")
WARMUP, MEASURES, SEED, CPU_THREADS = 3, 20, 20260910, 4
MEASURE_UPLOAD = True
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "ascend/results"

import torch
import torch_npu

sys.path.insert(0, str(REPO_ROOT))
from ascend.custom_ops.submconv3d import submconv3d as custom


def fingerprint(path):
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def coordinates(n, workload):
    count = (n + 1) // 2
    side = 1
    while side**3 < count:
        side += 1
    cells = list(itertools.product(range(side), repeat=3))[:count]
    order = torch.randperm(count, generator=torch.Generator().manual_seed(SEED + n)).tolist()
    stride = 1 if workload == "densecube" else 4  # Spread has only center hits, even at K=5.
    # Same shuffled spatial cells in both batches, interleaved to balance scan row order.
    return torch.tensor([(i % 2, *(stride * (v - side // 2) for v in cells[order[i // 2]]))
                         for i in range(n)], dtype=torch.int32)


def oracle(indices, k):
    points = [tuple(p) for p in indices.tolist()]
    lookup = {p: i for i, p in enumerate(points)}
    if len(lookup) != len(points):
        raise ValueError("coordinates must be unique")
    offsets = list(itertools.product(range(-(k // 2), k // 2 + 1), repeat=3))
    padding = [-1] * (((k**3 + 7) // 8) * 8 - k**3)
    return torch.tensor([[lookup.get((b, x + dx, y + dy, z + dz), -1)
                          for dx, dy, dz in offsets] + padding
                         for b, x, y, z in points], dtype=torch.int32)


@torch.inference_mode()
def main():
    if sys.argv[1:] or not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_TAG):
        raise ValueError("No CLI arguments; RUN_TAG must be a safe nonempty name")
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)
    torch.manual_seed(SEED)
    source = Path(custom.__file__).resolve().parent
    roots = list(dict.fromkeys(Path(p).resolve() for p in
                              os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p))
    providers = []
    for root in roots:
        kernel = root / "op_impl/ai_core/tbe/kernel"
        config = kernel / "config/ascend310p/binary_info_config.json"
        if config.is_file():
            entry = json.loads(config.read_text()).get("BuildSubmMap")
            if entry and entry.get("binaryList"):
                providers.append((root, kernel, config, entry))
    if len(providers) != 1:
        raise RuntimeError("Require exactly one configured BuildSubmMap OPP provider")
    root, kernel, config, entry = providers[0]
    if root != (source / "opp/vendors/graspgenx_subm").resolve():
        raise RuntimeError("Bridge/source and configured OPP must belong to the same private package")
    artifacts = {config}
    for binary in entry["binaryList"]:
        artifacts.update(kernel / binary[key] for key in ("binPath", "jsonPath"))
    artifacts.update(root.glob("op_proto/lib/linux/*/*.so"))
    artifacts.update(root.glob("op_impl/ai_core/tbe/op_tiling/lib/linux/*/*.so"))
    report = {
        "tag": RUN_TAG, "pid": os.getpid(), "host": platform.node(),
        "config": {"N": N, "K": K, "workloads": WORKLOADS, "warmup": WARMUP,
                   "measures": MEASURES, "seed": SEED, "cpu_threads": CPU_THREADS,
                   "measure_upload": MEASURE_UPLOAD, "jit_compile": False},
        "scope": "sync host wall ms; resident builder or CPU int32 upload+builder; "
                 "warmup/cold start, oracle, output D2H and validation excluded",
        "gate": "every warmup/measured output: NPU int32, padded shape, finite, exact oracle",
        "p95": "nearest-rank", "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "environment": {key: os.environ.get(key) for key in
                        ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH",
                         "LD_LIBRARY_PATH", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_DEVICE_ID")},
        "schema": str(torch.ops.graspgenx_subm.build_subm_map.default._schema),
        "bridge": fingerprint(custom._LIBRARY), "configured_opp_provider": str(root),
        "builder_artifacts": [fingerprint(p) for p in sorted(artifacts)],
        "sources": [fingerprint(p) for p in (Path(__file__), source / "submconv3d.py",
                    source / "torch_bridge.cpp", source / "op_kernel/build_subm_map.cpp",
                    source / "op_host/build_subm_map.cpp", source / "op_host/build_subm_map_tiling.h")],
        "results": [],
    }
    destination = RESULT_ROOT / f"subm_map_{RUN_TAG}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "provenance.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    torch.npu.set_compile_mode(jit_compile=False)
    builder = torch.ops.graspgenx_subm.build_subm_map  # Never call the four-argument wrapper.
    for workload, n, k in itertools.product(WORKLOADS, N, K):
        row = {"workload": workload, "N": n, "K": k, "passed": False,
               "checked_outputs": 0, "latency": {}}
        try:
            cpu = coordinates(n, workload)
            expected = oracle(cpu, k)
            row["coordinates_sha256"] = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
            row["neighbor_hits"] = int((expected[:, :k**3] >= 0).sum())
            indices = cpu.to("npu")
            for mode in ("resident", "upload_builder") if MEASURE_UPLOAD else ("resident",):
                samples = []
                for step in range(WARMUP + MEASURES):
                    torch.npu.synchronize()
                    started = time.perf_counter_ns()
                    output = builder(indices if mode == "resident" else cpu.to("npu"), k)
                    torch.npu.synchronize()
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if output.device.type != "npu" or output.dtype != torch.int32 or output.shape != expected.shape:
                        raise ValueError("map device/dtype/shape mismatch")
                    actual = output.cpu()
                    if not bool(torch.isfinite(actual).all()) or not torch.equal(actual, expected):
                        raise ValueError(f"map mismatch (including padding): {mode} step={step}")
                    row["checked_outputs"] += 1
                    del output
                    if step >= WARMUP:
                        samples.append(elapsed)
                row["latency"][mode] = {"median_ms": statistics.median(samples),
                                        "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
                                        "max_ms": max(samples), "samples_ms": samples}
            row["passed"] = True
        except Exception as error:
            row["error"] = repr(error)
        report["results"].append(row)
        with (destination / f"{workload}_n{n}_k{k}.json").open("x") as stream:
            json.dump(row, stream, indent=2, allow_nan=False)
        timings = " ".join(f"{mode}={v['median_ms']:.3f}/{v['p95_ms']:.3f}/{v['max_ms']:.3f}ms"
                           for mode, v in row["latency"].items())
        print(f"{workload:10s} N={n:4d} K={k} {'PASS' if row['passed'] else 'FAIL'} "
              f"med/p95/max {timings} {row.get('error', '')}", flush=True)
        if not row["passed"]:
            break  # Do not keep launching kernels after a runtime/correctness failure.
    total = len(WORKLOADS) * len(N) * len(K)
    passed = sum(row["passed"] for row in report["results"])
    report["summary"] = {"total": total, "passed": passed,
                         "failed": len(report["results"]) - passed,
                         "skipped": total - len(report["results"])}
    with (destination / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"Summary {report['summary']} results={destination}", flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
