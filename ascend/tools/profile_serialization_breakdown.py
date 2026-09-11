#!/usr/bin/env python3
"""CPU serialization and transfer-budget diagnostics; no model/kernel changes.

Initial serialization is measured both untouched and split, in alternating order.
Pooling measurements contain geometry/index operations only, not feature/coordinate
aggregation. All metadata is checked exactly against the production reference.
After CPU workers exit, a separate probe copies grid input and two precomputed
Hilbert outputs. This is transfer-only timing, NOT an NPU encoding benchmark.
"""

from collections import defaultdict
from datetime import datetime
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "ascend/results"
POINT_COUNTS = (64, 2048, 3500)
ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
STRIDES = (2, 2, 2, 2)
PROCESS_RUNS, WARMUP_RUNS, MEASURED_RUNS = 3, 3, 20
CPU_THREADS = 16

sys.path.insert(0, str(REPO_ROOT))
from ascend.tools import validate_ptv3 as validation
from graspgenx.models.ptv3 import ptv3_vanilla as reference


def measure(parts, name, operation):
    start = time.perf_counter_ns()
    result = operation()
    parts[name] = (time.perf_counter_ns() - start) / 1e6
    return result


def inverse_order(code, order):
    return torch.zeros_like(order).scatter_(
        dim=1, index=order,
        src=torch.arange(code.shape[1], device=order.device).repeat(code.shape[0], 1))


def initial(data, parts=None):
    if parts is None:
        point = reference.VanillaPoint(data)
        point.serialization(order=ORDERS, shuffle_orders=False)
        return point
    point = measure(parts, "point_init", lambda: reference.VanillaPoint(data))
    minimum = measure(parts, "grid_min", lambda: point.coord.min(0)[0])
    grid = measure(parts, "grid_quantize", lambda: torch.div(
        point.coord - minimum, point.grid_size, rounding_mode="trunc").int())
    depth = measure(parts, "depth", lambda: int(grid.max()).bit_length())
    assert depth <= 16 and depth * 3 + len(point.offset).bit_length() <= 63
    codes = []
    for name in ORDERS:
        code = measure(parts, f"encode_{name}", lambda: reference.encode(
            grid, batch=None, depth=depth, order=name))
        code = measure(parts, f"batch_{name}", lambda: (point.batch.long() << depth * 3) | code)
        codes.append(code)
    code = measure(parts, "stack", lambda: torch.stack(codes))
    order = measure(parts, "argsort", lambda: torch.argsort(code))
    inverse = measure(parts, "inverse", lambda: inverse_order(code, order))
    point.update(grid_coord=grid, serialized_depth=depth, serialized_code=code,
                 serialized_order=order, serialized_inverse=inverse)
    return point


def pooling_geometry(point, stride, parts):
    depth = (math.ceil(stride) - 1).bit_length()
    if depth > point.serialized_depth:
        depth = 0
    code = measure(parts, "coarsen_code", lambda: point.serialized_code >> depth * 3)
    _, cluster, counts = measure(parts, "unique_clusters", lambda: torch.unique(
        code[0], sorted=True, return_inverse=True, return_counts=True))
    _, indices = measure(parts, "sort_clusters", lambda: torch.sort(cluster))

    def heads():
        ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        return indices[ptr[:-1]]

    head = measure(parts, "cluster_heads", heads)
    code = measure(parts, "select_codes", lambda: code[:, head])
    order = measure(parts, "argsort", lambda: torch.argsort(code))
    inverse = measure(parts, "inverse", lambda: inverse_order(code, order))
    grid, batch = measure(parts, "grid_batch_select", lambda: (
        point.grid_coord[head] >> depth, point.batch[head]))
    return measure(parts, "point_init", lambda: reference.VanillaPoint(
        grid_coord=grid, batch=batch, serialized_code=code, serialized_order=order,
        serialized_inverse=inverse, serialized_depth=point.serialized_depth - depth))


def check(actual, expected):
    assert actual.serialized_depth == expected.serialized_depth
    for name in ("grid_coord", "batch", "offset", "serialized_code",
                 "serialized_order", "serialized_inverse"):
        assert actual[name].device.type == "cpu", name
        assert actual[name].dtype == expected[name].dtype, name
        assert torch.equal(actual[name], expected[name]), name
    positions = torch.arange(actual.serialized_code.shape[1]).expand(len(ORDERS), -1)
    assert torch.equal(actual.serialized_inverse.gather(1, actual.serialized_order), positions)


@torch.inference_mode()
def run_round(index, destination):
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(0)
    report = {"round": index, "pid": os.getpid(), "cases": [], "completed": False}
    try:
        for count in POINT_COUNTS:
            with np.load(validation.BASELINE_DIR / f"reference_n{count}.npz") as frozen:
                data = validation.make_input(frozen)
            expected = initial(data)
            samples = defaultdict(list)
            for step in range(WARMUP_RUNS + MEASURED_RUNS):
                modes = ("full", "split") if (step + index) % 2 == 0 else ("split", "full")
                for mode in modes:
                    parts = {} if mode == "split" else None
                    start = time.perf_counter_ns()
                    actual = initial(data, parts)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    check(actual, expected)
                    assert "grid_coord" not in data, "must create fresh point metadata"
                    if step >= WARMUP_RUNS:
                        samples[mode].append(elapsed)
                        if parts is not None:
                            for name, value in parts.items():
                                samples[name].append(value)
                            samples["parts_sum"].append(sum(parts.values()))
            row = {"N": count, "depth": expected.serialized_depth,
                   "grid_max": int(expected.grid_coord.max()),
                   "max_code": int(expected.serialized_code.max()),
                   "fp32_code_roundtrip_changes": int((expected.serialized_code.float().long() != expected.serialized_code).sum()),
                   "initial": {k: validation.summarize_ms(v) for k, v in samples.items()},
                   "initial_samples_ms": dict(samples), "pooling": []}
            # Codes and pooling membership are independent of learned features.
            # A one-channel reference pooling verifies the exact metadata path.
            point = expected
            point.feat = torch.ones(count, 1)
            for stage, stride in enumerate(STRIDES, 1):
                pool = reference.VanillaSerializedPooling(
                    1, 1, stride=stride, norm_layer=None, act_layer=None,
                    shuffle_orders=False, traceable=False).eval()
                expected_pool = pool(point)
                samples = defaultdict(list)
                for step in range(WARMUP_RUNS + MEASURED_RUNS):
                    parts = {}
                    start = time.perf_counter_ns()
                    actual = pooling_geometry(point, stride, parts)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    check(actual, expected_pool)
                    if step >= WARMUP_RUNS:
                        samples["geometry_total"].append(elapsed)
                        samples["parts_sum"].append(sum(parts.values()))
                        for name, value in parts.items():
                            samples[name].append(value)
                row["pooling"].append({
                    "stage": stage, "N_in": len(point.batch), "N_out": len(expected_pool.batch),
                    "timings": {k: validation.summarize_ms(v) for k, v in samples.items()},
                    "samples_ms": dict(samples)})
                point = expected_pool
            report["cases"].append(row)
            print(f"round={index} N={count} depth={row['depth']} "
                  f"initial={row['initial']['full']['median_ms']:.3f} ms "
                  f"split={row['initial']['split']['median_ms']:.3f} ms metadata=EXACT", flush=True)
        report["completed"] = True
    finally:
        with (destination / f"round_{index}.json").open("x") as stream:
            json.dump(report, stream, indent=2)


@torch.inference_mode()
def transfer_probe():
    torch.set_num_threads(CPU_THREADS)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    rows = []
    for count in POINT_COUNTS:
        with np.load(validation.BASELINE_DIR / f"reference_n{count}.npz") as frozen:
            point = initial(validation.make_input(frozen))
        grid = point.grid_coord.contiguous()
        expected = point.serialized_code[2:].contiguous()
        codes_npu = expected.to("npu:0")
        samples = []
        for step in range(WARMUP_RUNS + MEASURED_RUNS):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            grid_npu = grid.to("npu:0")
            copied_codes = codes_npu.cpu()
            torch.npu.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert torch.equal(copied_codes, expected)
            assert torch.equal(grid_npu.cpu(), grid)
            if step >= WARMUP_RUNS:
                samples.append(elapsed)
        row = {"N": count, "grid_h2d_bytes": grid.numel() * grid.element_size(),
               "two_codes_d2h_bytes": expected.numel() * expected.element_size(),
               "timing": validation.summarize_ms(samples), "samples_ms": samples}
        rows.append(row)
        print("transfer-only", json.dumps(row), flush=True)
    return {"processes": 1, "scope": "grid H2D + precomputed two-code D2H; no encoding kernel/launch",
            "warmup": WARMUP_RUNS, "samples": MEASURED_RUNS, "cases": rows}


def main():
    defaults = inspect.signature(reference.PointTransformerV3Vanilla).parameters
    assert defaults["order"].default == ORDERS and defaults["stride"].default == STRIDES
    for name, digest in validation.EXPECTED_SHA256.items():
        assert validation.sha256_file(validation.BASELINE_DIR / name) == digest
    destination = RESULT_ROOT / f"serialization_breakdown_{datetime.now():%Y%m%d_%H%M%S_%f}"
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "environment": {"host": platform.node(), "torch": torch.__version__, "cpu_threads": CPU_THREADS},
        "contract": {"point_counts": POINT_COUNTS, "orders": ORDERS, "strides": STRIDES,
                     "processes": PROCESS_RUNS, "warmup": WARMUP_RUNS, "samples": MEASURED_RUNS,
                     "scope": "CPU geometry only; identical G/D input; per one encoder forward",
                     "initial": "fresh Point creation plus actual serialization, alternated with split replay",
                     "pooling": "geometry/index subset only; excludes Linear, feature/coordinate reductions, norm/activation",
                     "gate": "exact metadata versus untouched reference; no weights or implementation changes",
                     "aggregate": "median of process medians; sums of component medians are diagnostics, not total latency"},
        "sources": {str(p.relative_to(REPO_ROOT)): validation.sha256_file(p)
                    for p in (Path(__file__), Path(reference.__file__))},
        "rounds": [], "aggregate": [], "completed": False}
    print(f"Results: {destination}", flush=True)
    try:
        for index in range(PROCESS_RUNS):
            env = dict(os.environ, SERIALIZATION_PROFILE_ROUND=str(index),
                       SERIALIZATION_PROFILE_DEST=str(destination))
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve())],
                           env=env, cwd=REPO_ROOT, check=True, timeout=600)
            report["rounds"].append(json.loads((destination / f"round_{index}.json").read_text()))
        for count in POINT_COUNTS:
            rows = [r for run in report["rounds"] for r in run["cases"] if r["N"] == count]
            initial_ms = {key: statistics.median(r["initial"][key]["median_ms"] for r in rows)
                          for key in rows[0]["initial"]}
            pools = []
            for i in range(len(STRIDES)):
                stages = [r["pooling"][i] for r in rows]
                pools.append({"stage": i + 1, "N_in": stages[0]["N_in"], "N_out": stages[0]["N_out"],
                              "median_ms": {key: statistics.median(s["timings"][key]["median_ms"] for s in stages)
                                            for key in stages[0]["timings"]}})
            row = {"N": count, "depth": rows[0]["depth"], "grid_max": rows[0]["grid_max"],
                   "max_code": rows[0]["max_code"],
                   "fp32_code_roundtrip_changes": rows[0]["fp32_code_roundtrip_changes"],
                   "initial_median_ms": initial_ms, "pooling": pools}
            report["aggregate"].append(row)
            print(json.dumps(row), flush=True)
        report["transfer_probe"] = transfer_probe()
        report["completed"] = all(r["completed"] for r in report["rounds"])
    finally:
        with (destination / "summary.json").open("x") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    if "SERIALIZATION_PROFILE_ROUND" in os.environ:
        run_round(int(os.environ["SERIALIZATION_PROFILE_ROUND"]),
                  Path(os.environ["SERIALIZATION_PROFILE_DEST"]))
    else:
        main()
