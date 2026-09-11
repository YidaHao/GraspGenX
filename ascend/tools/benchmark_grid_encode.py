#!/usr/bin/env python3
"""Paired serialization-only ablation of the current Ascend encoder.

Run on P1 after sourcing the existing SubM and then GridEncode env.sh files:
    python3 -B ascend/tools/benchmark_grid_encode.py
Requires model.forward and profile.serialize_point to call model.serialize_point.
No builds, alternate CPE, rescaling, golden writes, or implementation CLI.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import MethodType

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "ascend/baselines/ptv3-cuda-fp32-eager"
RESULT_ROOT = REPO_ROOT / "ascend/results"
BENCHMARK_HOST, BENCHMARK_DEVICE = "192.168.7.101", "Ascend310P1"
CPU_THREADS, PROCESS_RUNS, WARMUP_RUNS, MEASURED_RUNS, REPEAT_CHECKS = 16, 3, 3, 20, 3
SERIALIZATION_RUNS, RAW_RUNS, PROFILE_RUNS, WORKER_TIMEOUT = 20, 20, 3, 900
POINT_COUNTS, ENCODERS = (64, 2048, 3500), ("generator", "discriminator")
VARIANTS = ("cpu", "grid_encode")
ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
METADATA = ("coord", "feat", "grid_coord", "batch", "offset", "serialized_code", "serialized_order", "serialized_inverse")


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def cpu_control(self, data):
    point = VanillaPoint(data)
    point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
    return point


def timed(operation):
    validation.synchronize()
    started = time.perf_counter_ns()
    output = operation()
    validation.synchronize()
    return output, (time.perf_counter_ns() - started) / 1e6


def metadata_equal(expected, actual):
    require(isinstance(actual, VanillaPoint), "serialize_point must return VanillaPoint")
    require(type(actual.serialized_depth) is int and actual.serialized_depth == expected.serialized_depth,
            "serialized_depth mismatch or non-host scalar")
    for key in METADATA:
        a, b = expected[key], actual[key]
        require(b.device.type == "cpu" and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a, b), f"serialize_point metadata mismatch: {key}")


def compare(expected, actual):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return {"passed": False, "shape": list(actual.shape), "finite": bool(np.isfinite(actual).all())}
    return validation.compare(expected, actual)


def provenance():
    configured = list(dict.fromkeys(Path(p).resolve() for p in
                      os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p))
    paths = {Path(__file__), Path(validation.__file__), Path(profiler.__file__)}
    private_providers = {}
    for module, vendor, names in (
        (sys.modules[grid_encode.__module__], "graspgenx_grid", ("GridEncode",)),
        (cpe_ops, "graspgenx_subm", ("BuildSubmMap", "SubmConv3d")),
    ):
        source = Path(module.__file__).resolve().parent
        providers = []
        for root in configured:
            config = root / "op_impl/ai_core/tbe/kernel/config/ascend310p/binary_info_config.json"
            if config.is_file():
                binaries = json.loads(config.read_text())
                if all(binaries.get(name, {}).get("binaryList") for name in names):
                    providers.append((root, config, binaries))
        require(len(providers) == 1, f"Require one configured {vendor} OPP provider")
        root, config, binaries = providers[0]
        require(root == (source / "opp/vendors" / vendor).resolve(), f"{vendor} private provider mismatch")
        private_providers[vendor] = str(root)
        paths.update((Path(module.__file__), source / "torch_bridge.cpp", module._LIBRARY, config))
        paths.update(p for p in source.glob("op_*/*") if p.is_file())
        paths.update(root.rglob("*.so"))
        paths.update(root.glob("op_impl/ai_core/tbe/config/ascend310p/*.json"))
        for name in names:
            for binary in binaries[name]["binaryList"]:
                paths.update(config.parents[2] / binary[k] for k in ("binPath", "jsonPath"))
    paths.update(REPO_ROOT / "graspgenx/models/ptv3" / f"ptv3_{name}.py" for name in ("vanilla", "ascend"))
    return {"private_providers": private_providers, "sha256": {str(p): validation.sha256_file(p) for p in sorted(paths)},
            "environment": {k: os.environ.get(k) for k in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
                            "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH", "ASCEND_RT_VISIBLE_DEVICES")},
            "host": platform.node(), "torch": torch.__version__, "torch_npu": torch_npu.__version__}


def raw_timings(point, row, arrays):
    grid, depth = point.grid_coord, point.serialized_depth
    require(grid.dtype == torch.int32 and grid.is_contiguous(), "Do not cast raw operator inputs")
    # Vanilla Hilbert squeezes N=1 to a scalar without this zero-batch oracle.
    batch = torch.zeros(len(grid), dtype=torch.int64) if len(grid) == 1 else None
    def cpu_four():
        return torch.stack([encode(grid, batch=batch, depth=depth, order=o) for o in ORDERS])
    expected, resident = cpu_four(), grid.to(NPU_DEVICE)
    samples = defaultdict(list)
    row.update(depth=depth, samples_ms=samples, exact=False, scope="once per N/process; shared G/D geometry")
    for step in range(WARMUP_RUNS + RAW_RUNS):
        cpu, cpu_ms = timed(cpu_four)
        uploaded, h2d_ms = timed(lambda: grid.to(NPU_DEVICE))
        raw, kernel_ms = timed(lambda: grid_encode(resident, depth))
        require(raw.shape == (len(grid), 4) and raw.dtype == torch.int64
                and raw.device.type == "npu" and raw.is_contiguous(), "Raw GridEncode must be ND N4 int64")
        host, d2h_ms = timed(raw.cpu)
        spatial, layout_ms = timed(lambda: host.T.contiguous())
        packed, batch_ms = timed(lambda: spatial | (point.batch << (3 * depth)))
        arrays["raw_n4"], arrays["cpu_four_4n"] = host.numpy().copy(), cpu.numpy().copy()
        require(torch.equal(spatial, expected) and torch.equal(cpu, expected), "Raw spatial code mismatch")
        require(torch.equal(packed, point.serialized_code), "CPU batch packing mismatch")
        require(torch.equal(uploaded.cpu(), grid), "Raw H2D changed input")
        if step >= WARMUP_RUNS:
            for key, value in zip(("cpu_four", "h2d", "resident_kernel_host_sync", "d2h", "layout", "batch_pack"),
                                  (cpu_ms, h2d_ms, kernel_ms, d2h_ms, layout_ms, batch_ms)):
                samples[key].append(value)
    row.update(exact=True, latency={k: validation.summarize_ms(v) for k, v in samples.items()})


def run_round(report, destination):
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)
    torch.manual_seed(0)
    report["provenance"] = provenance()
    report["baseline_sha256"] = {k: validation.sha256_file(BASELINE_DIR / k) for k in validation.EXPECTED_SHA256}
    require(report["baseline_sha256"] == validation.EXPECTED_SHA256, "Frozen baseline hashes differ")
    require(validation.COSINE_GATE == 0.9999, "Validation cosine gate must remain 0.9999")
    write_json(destination / f"round_{report['round']}.json", report)
    require(callable(getattr(PointTransformerV3Ascend, "serialize_point", None)), "Missing model.serialize_point integration")
    require("serialize_point" in PointTransformerV3Ascend.forward.__code__.co_names
            and "serialize_point" in profiler.serialize_point.__code__.co_names, "forward/profile must dispatch model.serialize_point")
    for encoder in ENCODERS:
        model = validation.make_model(encoder)  # Shared frozen-weight/eval/shuffle-off setup, once per encoder.
        require(type(model) is PointTransformerV3Ascend, "Validation import must select current Ascend")
        require(tuple(model.order) == ORDERS and not any(getattr(m, "shuffle_orders", False) for m in model.modules()),
                "All serialization orders must remain unshuffled")
        candidate = model.serialize_point
        methods = {"cpu": MethodType(cpu_control, model), "grid_encode": candidate}
        report.setdefault("execution_configs", {})[encoder] = dict(model.execution_config)
        try:
            for count in POINT_COUNTS:
                arrays = {}
                row = {"encoder": encoder, "point_count": count, "passed": False, "stage": "serialize_point",
                       "variants": {n: {"samples_ms": [], "entry_samples_ms": [], "checks": []} for n in VARIANTS}}
                report["results"].append(row)
                try:
                    with np.load(BASELINE_DIR / f"reference_n{count}.npz") as reference:
                        data, golden = validation.make_input(reference), reference[f"{encoder}_embedding"].copy()
                    expected = cpu_control(model, data)
                    for inputs in (data, dict(data, offset=torch.tensor([count // 2, count], dtype=torch.int64))):
                        control = cpu_control(model, inputs)
                        for name in VARIANTS:
                            model.serialize_point = methods[name]
                            metadata_equal(control, model.serialize_point(inputs))
                            metadata_equal(control, profiler.serialize_point(model, inputs))
                    row["metadata_exact"] = True
                    arrays["serialized_depth"] = np.asarray(expected.serialized_depth)
                    for k in METADATA:
                        arrays[k] = expected[k].numpy().copy()
                    if encoder == ENCODERS[0] and RAW_RUNS:
                        row["stage"] = "raw_grid_encode"
                        raw_timings(expected, row.setdefault("raw", {}), arrays)
                    row["stage"] = "full_encoder"
                    for phase, runs in (("warmup", WARMUP_RUNS), ("measured", MEASURED_RUNS), ("repeat", REPEAT_CHECKS)):
                        for step in range(runs):
                            for name in VARIANTS[::1 if (report["round"] + step) % 2 == 0 else -1]:
                                model.serialize_point = methods[name]
                                output, elapsed = timed(lambda model=model, data=data: model(data))
                                actual = output.detach().cpu().numpy().copy()  # Preserve old output before switching.
                                arrays[f"{name}_{phase}_{step}"] = actual
                                metrics = compare(golden, actual)
                                variant = row["variants"][name]
                                variant["checks"].append({"phase": phase, "step": step, **metrics})
                                if phase == "measured":
                                    variant["samples_ms"].append(elapsed)
                                require(metrics["passed"], f"CUDA golden failed: {name}/{phase}/{step}")
                    row["stage"] = "serialization_entry"
                    for step in range(SERIALIZATION_RUNS):
                        for name in VARIANTS[::1 if (report["round"] + step) % 2 == 0 else -1]:
                            model.serialize_point = methods[name]
                            point, elapsed = timed(lambda model=model, data=data: model.serialize_point(data))
                            row["variants"][name]["entry_samples_ms"].append(elapsed)
                            metadata_equal(expected, point)
                    for name, variant in row["variants"].items():
                        actual = arrays[f"{name}_measured_0"]
                        variant["latency"] = validation.summarize_ms(variant["samples_ms"])
                        if SERIALIZATION_RUNS:
                            variant["serialization_entry"] = validation.summarize_ms(variant["entry_samples_ms"])
                        variant["repeat_max_abs"] = max(float(np.abs(arrays[f"{name}_repeat_{i}"] - actual).max())
                                                        for i in range(REPEAT_CHECKS))
                        variant["vs_cpu_report_only"] = compare(arrays["cpu_measured_0"], actual)
                        if count == 2048 and PROFILE_RUNS:
                            row["stage"] = f"profile/{name}"
                            model.serialize_point = methods[name]
                            samples = defaultdict(list)
                            profile = variant["profile"] = {"samples_ms": samples, "checks": []}
                            for step in range(PROFILE_RUNS):
                                profiled = profiler.run_partitioned(model, data, samples)
                                arrays[f"{name}_profile_{step}"] = profiled.copy()
                                checks = {"cuda": compare(golden, profiled), "direct": compare(actual, profiled)}
                                profile["checks"].append(checks)
                                require(all(c["passed"] for c in checks.values()), f"Profile validation failed: {name}/{step}")
                            require(len(samples["1.serialization"]) == PROFILE_RUNS, "Missing serialization profile label")
                            ranking = sorted((k for k in samples if profiler.is_leaf_stage(k)),
                                             key=lambda k: statistics.median(samples[k]), reverse=True)
                            profile["ranking"] = [{"rank": i + 1, "stage": k, "category": profiler.stage_category(k),
                                                   **validation.summarize_ms(samples[k])} for i, k in enumerate(ranking)]
                    row["byte_equal_report_only"] = [arrays[f"cpu_measured_{i}"].dtype == arrays[f"grid_encode_measured_{i}"].dtype
                        and arrays[f"cpu_measured_{i}"].tobytes() == arrays[f"grid_encode_measured_{i}"].tobytes() for i in range(MEASURED_RUNS)]
                    row["passed"], row["stage"] = True, "complete"
                    print(f"p{report['round']} {encoder:13s} N={count:4d} PASS " + " ".join(
                        f"{n}={v['latency']['median_ms']:.3f}/{v['latency']['p95_ms']:.3f}/{v['latency']['max_ms']:.3f}ms"
                        for n, v in row["variants"].items()) + " (med/p95/max) "
                        f"mincos={min(c['cosine'] for v in row['variants'].values() for c in v['checks']):.8f}", flush=True)
                finally:
                    np.savez(destination / f"round_{report['round']}_{encoder}_n{count}.npz", **arrays)
                    write_json(destination / f"round_{report['round']}.json", report)
        finally:
            if "serialize_point" in model.__dict__:
                del model.serialize_point  # Restore the class method without a self-cycle.
        del methods, candidate, model
        torch.npu.empty_cache()
    require(provenance()["sha256"] == report["provenance"]["sha256"], "Runtime artifacts changed during worker")
    report["completed"] = True


def main():
    destination = RESULT_ROOT / f"grid_encode_compare_{datetime.now():%Y%m%d_%H%M%S_%f}"
    destination.mkdir(parents=True, exist_ok=False)
    report = {"contract": {"host": BENCHMARK_HOST, "device": BENCHMARK_DEVICE, "cpu_threads": CPU_THREADS,
              "processes": PROCESS_RUNS, "warmup": WARMUP_RUNS, "runs": MEASURED_RUNS, "repeats": REPEAT_CHECKS,
              "serialization_runs": SERIALIZATION_RUNS, "raw_runs": RAW_RUNS, "profile_runs": PROFILE_RUNS,
              "control": "eff0a76 current optimized CPE/dense, benchmark-only original CPU initial serialization",
              "candidate": "same resident model; default GridEncode pure spatial codes, CPU packing/argsort/inverse",
              "scope": "encoder-only AB/BA; all transfers included; no rescale; not a fully NPU encoder",
              "timing": "synchronized host wall ms, including resident kernel; NOT device-event time; raw parts not entry latency",
              "float_gate": "CUDA cosine >= 0.9999; shape/finite required for every output; drift/bytes report-only",
              "integer_gate": "exact raw N4 transpose and CPU metadata, including batch; shuffle off"},
              "rounds": [], "aggregate": [], "failures": [], "passed": False}
    report["contract"]["combined"] = "median of per-process G+D encoder-median sums; NOT request latency"
    print(f"Results: {destination}", flush=True)
    try:
        for index in range(PROCESS_RUNS):
            try:
                completed = subprocess.run([sys.executable, "-B", "-u", str(Path(__file__).resolve())], cwd=REPO_ROOT,
                    env=dict(os.environ, GRID_BENCH_ROUND=str(index), GRID_BENCH_DEST=str(destination)), timeout=WORKER_TIMEOUT)
            finally:
                path = destination / f"round_{index}.json"
                if path.is_file():
                    report["rounds"].append(json.loads(path.read_text()))
                write_json(destination / "summary.json", report)
            require(completed.returncode == 0 and len(report["rounds"]) == index + 1
                    and report["rounds"][-1].get("completed"), f"Worker {index} failed")
        require(all(len(r["results"]) == len(ENCODERS) * len(POINT_COUNTS) for r in report["rounds"]), "Incomplete cases")
        require(all(r["provenance"]["sha256"] == report["rounds"][0]["provenance"]["sha256"] for r in report["rounds"]),
                "Source/bridge/OPP changed between processes")
        print("Final: median of process medians (ms), candidate minus CPU control", flush=True)
        for count in POINT_COUNTS:
            for encoder in (*ENCODERS, "G+D sum"):
                values = {n: [sum(r["variants"][n]["latency"]["median_ms"] for r in run["results"]
                                  if r["point_count"] == count and (encoder == "G+D sum" or r["encoder"] == encoder))
                              for run in report["rounds"]] for n in VARIANTS}
                medians = {n: statistics.median(v) for n, v in values.items()}
                delta = medians["grid_encode"] - medians["cpu"]
                report["aggregate"].append({"encoder": encoder, "point_count": count, "process_medians_ms": values,
                    "median_of_process_medians_ms": medians, "change_ms": delta, "change_percent": 100 * delta / medians["cpu"]})
                print(f"{encoder:13s} N={count:4d} CPU={medians['cpu']:.3f} new={medians['grid_encode']:.3f} "
                      f"delta={delta:+.3f}ms ({100 * delta / medians['cpu']:+.2f}%)", flush=True)
        cases = [r for run in report["rounds"] for r in run["results"]]
        report["code_comparison_integrity"] = {"source_bridge_private_opp_unchanged": True,
            "metadata_exact": all(r["metadata_exact"] for r in cases),
            "raw_exact": all(r["raw"]["exact"] for r in cases if "raw" in r) if RAW_RUNS else "not measured",
            "embedding_bytes_equal_report_only": all(all(r["byte_equal_report_only"]) for r in cases)}
        report["passed"] = all(r["passed"] for run in report["rounds"] for r in run["results"])
        print("G+D sums separate encoder medians, NOT request latency; three processes, no significance claim.", flush=True)
    except Exception:
        report["failures"].append(traceback.format_exc())
        print(f"FAIL: {report['failures'][-1].splitlines()[-1]}", flush=True)
    finally:
        write_json(destination / "summary.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    require(not sys.argv[1:], "No CLI arguments; edit top configuration")
    if "GRID_BENCH_ROUND" not in os.environ:
        raise SystemExit(main())
    destination = Path(os.environ["GRID_BENCH_DEST"])
    report = {"round": int(os.environ["GRID_BENCH_ROUND"]), "pid": os.getpid(), "results": [], "failures": []}
    write_json(destination / f"round_{report['round']}.json", report)
    try:  # Imports stay in workers so missing integrations/artifacts are reported too.
        import numpy as np
        import torch
        import torch_npu
        sys.path.insert(0, str(REPO_ROOT))
        from ascend.tools import validate_ptv3 as validation, profile_ptv3_stages as profiler
        from graspgenx.models.ptv3.ptv3_ascend import PointTransformerV3Ascend, NPU_DEVICE
        from graspgenx.models.ptv3.ptv3_vanilla import VanillaPoint, encode
        from ascend.custom_ops.grid_encode.grid_encode import grid_encode
        from ascend.custom_ops.submconv3d import submconv3d as cpe_ops
        with torch.inference_mode():
            run_round(report, destination)
    except Exception:
        report["failures"].append(traceback.format_exc())
        print(f"p{report['round']} FAIL: {report['failures'][-1].splitlines()[-1]}", flush=True)
    finally:
        write_json(destination / f"round_{report['round']}.json", report)
    raise SystemExit(0 if report.get("completed") and not report["failures"] else 1)
