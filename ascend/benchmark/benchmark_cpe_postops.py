#!/usr/bin/env python3
"""Paired CPE post-ops/attention-boundary ablation, not an alternate encoder.

Source the existing SubM then GridEncode env.sh; run this file with python3 -B.
Only forward_cpe is rebound. No builds, golden writes, downsample experiments,
profiler.main, execution audits or coordinate probes. Importing cpu_cpe_control
does not import Torch or initialize an NPU. Edit configuration here, not via CLI.
"""

import inspect
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
PROFILE_RUNS, WORKER_TIMEOUT = 3, 900
POINT_COUNTS, ENCODERS, VARIANTS = (64, 2048, 3500), ("generator", "discriminator"), ("cpu", "npu")
# P3 private copies use their own paired CPU reference, not P1 historical files.
CONTROL_BASELINE_DIR = (RESULT_ROOT / "grid_encode_compare_20260911_135156_786032"
                        if BENCHMARK_DEVICE == "Ascend310P1" else None)
VARIANT_CONFIG = {
    "cpu": {"label": "old CPU FP32 CPE post-ops (75ee225 control)", "cpe_post_ops": "cpu_fp32",
            "attention_input": "cpu_fp32_then_h2d_fp16", "cpe_residual": "cpu_fp32"},
    "npu": {"label": "new NPU FP16 CPE post-ops + direct attention", "cpe_post_ops": "npu_fp16",
            "attention_input": "resident_npu_fp16_no_feature_copy", "cpe_residual": "npu_fp16"},
}


def cpu_cpe_control(self, point):
    cpe = self.cpe_conv(point.feat, point.grid_coord, point.batch, point)
    point.feat = point.feat + self.cpe_norm(self.cpe_linear(cpe))
    return point


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, arrays):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def compare(expected, actual, report_only=False):
    valid = expected.shape == actual.shape == (1, 512)
    finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
    metrics = validation.compare(expected, actual) if valid and finite else {
        "passed": False, "reference_shape": list(expected.shape), "shape": list(actual.shape), "finite": finite}
    if report_only:
        metrics.pop("passed", None)
        metrics["report_only"] = True
    return metrics


def provenance():
    configured = list(dict.fromkeys(Path(p).resolve() for p in
                      os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p))
    paths = {Path(__file__), Path(validation.__file__), Path(profiler.__file__)}
    paths.update(REPO_ROOT / "graspgenx/models/ptv3" / f"ptv3_{n}.py" for n in ("vanilla", "ascend"))
    providers = {}
    for module, vendor, names in ((grid_ops, "graspgenx_grid", ("GridEncode",)),
                                  (cpe_ops, "graspgenx_subm", ("BuildSubmMap", "SubmConv3d"))):
        source, matches = Path(module.__file__).resolve().parent, []
        for root in configured:
            config = root / "op_impl/ai_core/tbe/kernel/config/ascend310p/binary_info_config.json"
            if config.is_file():
                binaries = json.loads(config.read_text())
                if all(binaries.get(n, {}).get("binaryList") for n in names):
                    matches.append((root, config, binaries))
        require(len(matches) == 1, f"Require exactly one configured {vendor} provider")
        root, config, binaries = matches[0]
        require(root == (source / "opp/vendors" / vendor).resolve(), f"Private {vendor} provider mismatch")
        providers[vendor] = {"opp": str(root), "bridge": str(module._LIBRARY.resolve())}
        paths.update((Path(module.__file__), source / "torch_bridge.cpp", module._LIBRARY, config))
        paths.update(p for p in source.glob("op_*/*") if p.is_file())
        paths.update(root.rglob("*.so"))
        paths.update(root.glob("op_impl/ai_core/tbe/config/ascend310p/*.json"))
        for name in names:
            for binary in binaries[name]["binaryList"]:
                paths.update(config.parents[2] / binary[k] for k in ("binPath", "jsonPath"))
    if CONTROL_BASELINE_DIR is not None:
        paths.add(CONTROL_BASELINE_DIR / "round_0.json")
        paths.update(CONTROL_BASELINE_DIR / f"round_0_{e}_n{n}.npz" for e in ENCODERS for n in POINT_COUNTS)
    return {"private_providers": providers,
            "sha256": {str(p): validation.sha256_file(p) for p in sorted({p.resolve(strict=True) for p in paths})},
            "baseline_sha256": {k: validation.sha256_file(BASELINE_DIR / k) for k in validation.EXPECTED_SHA256},
            "environment": {k: os.environ.get(k) for k in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
                "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH", "ASCEND_RT_VISIBLE_DEVICES")},
            "host": platform.node(), "torch": torch.__version__, "torch_npu": torch_npu.__version__}


def run_round(report, destination):
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)
    torch.manual_seed(0)
    require(len(validation.EXPECTED_SHA256) == 5 and
            report["provenance_before"]["baseline_sha256"] == validation.EXPECTED_SHA256, "Frozen five assets differ")
    require(validation.BASELINE_DIR == BASELINE_DIR and validation.COSINE_GATE == 0.9999, "Validator contract differs")
    require(profiler.PointTransformerV3 is PointTransformerV3Ascend, "Profiler must import current Ascend")
    require("serialize_point" in PointTransformerV3Ascend.forward.__code__.co_names and
            "serialize_point" in profiler.serialize_point.__code__.co_names, "Direct/profile serialization must be shared")
    require({"linear", "npu_layer_norm_eval"} <= set(AscendBlock.forward_cpe.__code__.co_names),
            "NPU CPE post-ops integration missing; refuse a CPU/CPU comparison")
    report["method_sources"] = {"cpu": inspect.getsource(cpu_cpe_control),
        "npu": inspect.getsource(AscendBlock.forward_cpe), "attention": inspect.getsource(AscendBlock.forward_attention)}
    if CONTROL_BASELINE_DIR is not None:
        historical = json.loads((CONTROL_BASELINE_DIR / "round_0.json").read_text())
        require(historical.get("completed") and historical["baseline_sha256"] == validation.EXPECTED_SHA256,
                "Historical control is incomplete or uses different assets")
        report["historical_control"] = {"directory": str(CONTROL_BASELINE_DIR), "provenance": historical["provenance"]}
    for encoder in ENCODERS:
        model = validation.make_model(encoder)
        require(type(model) is PointTransformerV3Ascend, "Validator must import current Ascend")
        require(not any(getattr(m, "shuffle_orders", False) for m in model.modules()), "Shuffle must stay disabled")
        blocks = [m for m in model.modules() if isinstance(m, AscendBlock)]
        require(blocks and all("forward_cpe" not in b.__dict__ for b in blocks), "Unexpected block method override")
        methods = {"npu": [b.forward_cpe for b in blocks], "cpu": [MethodType(cpu_cpe_control, b) for b in blocks]}
        report.setdefault("execution_configs", {})[encoder] = dict(model.execution_config)
        report["method_sources"]["convolution"] = inspect.getsource(type(blocks[0].cpe_conv).forward)
        try:
            for count in POINT_COUNTS:
                arrays, sequence = {}, report["round"]
                output_path = destination / f"round_{report['round']}_{encoder}_n{count}.npz"
                row = {"encoder": encoder, "point_count": count, "passed": False, "stage": "input",
                       "output_path": str(output_path), "execution_order": [], "paired_checks_report_only": [],
                       "variants": {n: {"samples_ms": [], "checks": [],
                           "execution_config": dict(model.execution_config, **VARIANT_CONFIG[n])} for n in VARIANTS}}
                report["results"].append(row)
                try:
                    with np.load(BASELINE_DIR / f"reference_n{count}.npz") as reference:
                        data, golden = validation.make_input(reference), reference[f"{encoder}_embedding"].copy()
                        arrays.update(cuda_reference=golden, points=reference["points"].copy(), offset=reference["offset"].copy())
                    if CONTROL_BASELINE_DIR is not None:
                        with np.load(CONTROL_BASELINE_DIR / f"round_0_{encoder}_n{count}.npz") as historical:
                            for key in ("cpu_measured_0", "grid_encode_measured_0"):
                                arrays[f"historical_{key}"] = historical[key].copy()
                    phases = [("warmup", WARMUP_RUNS), ("measured", MEASURED_RUNS), ("repeat", REPEAT_CHECKS)]
                    if count == 2048:
                        phases.append(("profile", PROFILE_RUNS))
                    for phase, runs in phases:
                        for step in range(runs):
                            order = VARIANTS[::1 if sequence % 2 == 0 else -1]
                            sequence += 1
                            row["execution_order"].append({"phase": phase, "step": step, "order": order})
                            for name in order:
                                row["stage"] = f"{phase}/{name}/{step}"
                                for block, method in zip(blocks, methods[name]):
                                    block.forward_cpe = method
                                variant = row["variants"][name]
                                if phase == "profile":
                                    samples = variant.setdefault("profile", {"samples_ms": defaultdict(list)})["samples_ms"]
                                    actual = profiler.run_partitioned(model, data, samples).copy()
                                    elapsed = samples["0.total"][-1]
                                else:
                                    validation.synchronize()
                                    started = time.perf_counter_ns()
                                    output = model(data)
                                    validation.synchronize()
                                    elapsed = (time.perf_counter_ns() - started) / 1e6
                                    actual = output.detach().cpu().numpy().copy()
                                arrays[f"{name}_{phase}_{step}"] = actual
                                check = {"phase": phase, "step": step, "elapsed_ms": elapsed, "cuda": compare(golden, actual)}
                                variant["checks"].append(check)
                                if phase == "measured":
                                    variant["samples_ms"].append(elapsed)
                                require(check["cuda"]["passed"], f"CUDA gate failed: {row['stage']}")
                                if phase == "profile":
                                    check["direct"] = compare(arrays[f"{name}_measured_0"], actual)
                                    require(check["direct"]["passed"], f"Direct/profile gate failed: {row['stage']}")
                                if name == "cpu" and phase == "warmup" and step == 0 and CONTROL_BASELINE_DIR is not None:
                                    row["historical_checks"] = {k: compare(v, actual) for k, v in arrays.items() if k.startswith("historical_")}
                                    require(all(c["passed"] for c in row["historical_checks"].values()), "Historical CPU control differs")
                            row["paired_checks_report_only"].append({"phase": phase, "step": step,
                                **compare(arrays[f"cpu_{phase}_{step}"], arrays[f"npu_{phase}_{step}"], report_only=True)})
                            write_npz(output_path, arrays)
                            write_json(destination / f"round_{report['round']}.json", report)
                    for name, variant in row["variants"].items():
                        variant["latency"] = validation.summarize_ms(variant["samples_ms"])
                        variant["repeat_report_only"] = [compare(arrays[f"{name}_measured_0"], arrays[f"{name}_repeat_{i}"], True)
                                                         for i in range(REPEAT_CHECKS)]
                        if "profile" in variant:
                            variant["profile"]["stage_latency"] = {k: validation.summarize_ms(v)
                                for k, v in variant["profile"]["samples_ms"].items()}
                    arrays["cpu_reference"], arrays["npu_candidate"] = arrays["cpu_measured_0"], arrays["npu_measured_0"]
                    row["passed"], row["stage"] = True, "complete"
                    print(f"p{report['round']} {encoder} N={count} PASS " + " ".join(
                        f"{n}={v['latency']['median_ms']:.3f}/{v['latency']['p95_ms']:.3f}/{v['latency']['max_ms']:.3f}ms"
                        for n, v in row["variants"].items()) + " (median/p95/max)", flush=True)
                finally:
                    try:
                        write_npz(output_path, arrays)
                    finally:
                        write_json(destination / f"round_{report['round']}.json", report)
        finally:
            for block in blocks:
                if "forward_cpe" in block.__dict__:
                    del block.forward_cpe  # Restore class lookup, breaking bound-method self-cycles.
        del methods, blocks, block, method, model
        torch.npu.empty_cache()


def main():
    destination = RESULT_ROOT / f"cpe_postops_compare_{datetime.now():%Y%m%d_%H%M%S_%f}"
    destination.mkdir(parents=True, exist_ok=False)
    report = {"contract": {"host": BENCHMARK_HOST, "device": BENCHMARK_DEVICE, "cpu_threads": CPU_THREADS,
        "processes": PROCESS_RUNS, "warmup": WARMUP_RUNS, "runs": MEASURED_RUNS, "repeats": REPEAT_CHECKS,
        "profile_runs_n2048": PROFILE_RUNS, "worker_timeout_s": WORKER_TIMEOUT, "variants": VARIANT_CONFIG,
        "point_counts": POINT_COUNTS, "encoders": ENCODERS,
        "scope": "same resident model per encoder; ONLY forward_cpe rebound; AB/BA per run/process",
        "unchanged": "serialization, CPE maps/kernels, attention/dense operations, CPU FP32 FFN exit and downsampling",
        "timing": "synchronized full-encoder host wall ms including transfers; not a fully NPU encoder",
        "profile": "diagnostic synchronized stages and 0.total; partial sums are NOT full-encoder latency",
        "combined": "median of per-process G+D encoder-median sums; NOT request latency; no significance claim",
        "gate": "every warmup/measured/repeat/profile: CUDA cosine >= 0.9999, finite [1,512]; profile also vs direct",
        "report_only": "CPU/candidate errors, repeat drift and latency; no bit-exact requirement",
        "dtype_transfer_audit": "not run; separate profiler.ExecutionAudit diagnostic only",
        "historical_control": str(CONTROL_BASELINE_DIR) if CONTROL_BASELINE_DIR is not None else "not configured"},
        "rounds": [], "aggregate": [], "failures": [], "passed": False}
    print(f"Results: {destination}", flush=True)
    try:
        for index in range(PROCESS_RUNS):
            try:
                completed = subprocess.run([sys.executable, "-B", "-u", str(Path(__file__).resolve())], cwd=REPO_ROOT,
                    env=dict(os.environ, CPE_POSTOPS_ROUND=str(index), CPE_POSTOPS_DEST=str(destination),
                             OMP_NUM_THREADS=str(CPU_THREADS), MKL_NUM_THREADS=str(CPU_THREADS)), timeout=WORKER_TIMEOUT)
            finally:
                path = destination / f"round_{index}.json"
                if path.is_file():
                    report["rounds"].append(json.loads(path.read_text()))
                write_json(destination / "summary.json", report)
            require(completed.returncode == 0 and len(report["rounds"]) == index + 1 and
                    report["rounds"][-1].get("completed"), f"Worker {index} failed")
            require(report["rounds"][-1]["provenance_before"] == report["rounds"][0]["provenance_before"],
                    "Sources/assets/private OPP or environment changed between workers")
        require(all(len(r["results"]) == len(ENCODERS) * len(POINT_COUNTS) and
                    all(c["passed"] for c in r["results"]) for r in report["rounds"]), "Incomplete cases")
        for count in POINT_COUNTS:
            for encoder in (*ENCODERS, "G+D sum"):
                values = {n: [sum(c["variants"][n]["latency"]["median_ms"] for c in r["results"]
                    if c["point_count"] == count and (encoder == "G+D sum" or c["encoder"] == encoder))
                    for r in report["rounds"]] for n in VARIANTS}
                medians = {n: statistics.median(v) for n, v in values.items()}
                delta = medians["npu"] - medians["cpu"]
                report["aggregate"].append({"encoder": encoder, "point_count": count, "process_medians_ms": values,
                    "median_of_process_medians_ms": medians, "change_ms": delta, "change_percent": 100 * delta / medians["cpu"]})
                print(f"{encoder} N={count} CPU={medians['cpu']:.3f} NPU={medians['npu']:.3f} "
                      f"delta={delta:+.3f}ms ({100 * delta / medians['cpu']:+.2f}%)", flush=True)
        report["passed"] = True
    except (Exception, KeyboardInterrupt):
        report["failures"].append(traceback.format_exc())
        print(f"FAIL: {report['failures'][-1].splitlines()[-1]}", flush=True)
    finally:
        write_json(destination / "summary.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    require(not sys.argv[1:], "No CLI arguments; edit top configuration")
    if "CPE_POSTOPS_ROUND" not in os.environ:
        raise SystemExit(main())
    destination = Path(os.environ["CPE_POSTOPS_DEST"])
    report = {"round": int(os.environ["CPE_POSTOPS_ROUND"]), "pid": os.getpid(), "results": [], "failures": []}
    write_json(destination / f"round_{report['round']}.json", report)
    try:
        import numpy as np
        import torch
        import torch_npu
        sys.path.insert(0, str(REPO_ROOT))
        from ascend.benchmark import validate_ptv3 as validation, profile_ptv3_stages as profiler
        from graspgenx.models.ptv3.ptv3_ascend import AscendBlock, PointTransformerV3Ascend
        from ascend.custom_ops.grid_encode import grid_encode as grid_ops
        from ascend.custom_ops.submconv3d import submconv3d as cpe_ops
        report["provenance_before"] = provenance()
        write_json(destination / f"round_{report['round']}.json", report)
        try:
            with torch.inference_mode():
                run_round(report, destination)
        finally:
            report["provenance_after"] = provenance()
            require(report["provenance_after"] == report["provenance_before"], "Runtime artifacts changed during worker")
        report["completed"] = True
    except (Exception, KeyboardInterrupt):
        report["failures"].append(traceback.format_exc())
        print(f"p{report['round']} FAIL: {report['failures'][-1].splitlines()[-1]}", flush=True)
    finally:
        write_json(destination / f"round_{report['round']}.json", report)
    raise SystemExit(0 if report.get("completed") and not report["failures"] else 1)
