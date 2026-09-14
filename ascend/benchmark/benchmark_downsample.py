#!/usr/bin/env python3
"""Four-way paired pooling experiment; source ascend/env.sh before running.

Only down.forward changes on each resident, frozen-weight Ascend encoder.
FP16 means offload + padded-gather equivalent group max, NOT precision alone:
NPU Linear -> gather/amax -> eval BatchNorm -> GELU -> CPU FP32, without BN folding.
No builds, model switches, coordinate probes, golden writes or fallback claims.
"""

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "ascend/baselines/ptv3-cuda-fp32-eager"
RESULT_ROOT = REPO_ROOT / "ascend/results"
CPU_THREADS, PROCESS_RUNS, WARMUP_RUNS, MEASURED_RUNS, REPEAT_CHECKS = 16, 3, 3, 20, 3
POINT_COUNTS, ENCODERS = (64, 2048, 3500), ("generator", "discriminator")
VARIANTS = ("reference", "drop_coords", "fp16", "drop_coords_fp16")
PROFILE_RUNS, PROFILE_POINTS, AUDIT_EXECUTION, WORKER_TIMEOUT = 2, 2048, True, 1200
METADATA = ("grid_coord", "batch", "offset", "serialized_code", "serialized_order", "serialized_inverse")
PAIRS = (("drop_coords", "reference"), ("fp16", "reference"), ("drop_coords_fp16", "reference"),
         ("drop_coords_fp16", "drop_coords"), ("drop_coords_fp16", "fp16"))
BYTE_PAIRS = (("drop_coords", "reference"), ("drop_coords_fp16", "fp16"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def compare(expected, actual, embedding=True):
    if expected is None or actual is None:
        return {"passed": False, "reason": "missing output"}
    if (expected.shape != actual.shape or (embedding and actual.shape != (1, 512))
            or not np.isfinite(expected).all() or not np.isfinite(actual).all()):
        return {"passed": False, "reference_shape": list(expected.shape), "shape": list(actual.shape),
                "reference_finite": bool(np.isfinite(expected).all()), "finite": bool(np.isfinite(actual).all())}
    return validation.compare(expected, actual)  # A zero norm has cosine=0, never a pass.


def byte_equal(a, b):
    return bool(a is not None and b is not None and a.shape == b.shape
                and a.dtype == b.dtype and a.tobytes() == b.tobytes())


def provenance():
    paths = {Path(__file__), Path(validation.__file__), Path(profiler.__file__), Path(helper.__file__),
             REPO_ROOT / "ascend/env.sh"}
    paths.update(REPO_ROOT / "graspgenx/models/ptv3" / f"ptv3_{n}.py" for n in ("ascend", "vanilla"))
    configured = {Path(p).resolve() for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p}
    configs, providers = {}, {}
    for root in configured:
        config = root / "op_impl/ai_core/tbe/kernel/config/ascend310p/binary_info_config.json"
        if config.is_file():
            configs[root] = (config, json.loads(config.read_text()))
    for module, vendor, names in ((sys.modules[grid_encode.__module__], "graspgenx_grid", ("GridEncode",)),
                                  (cpe_ops, "graspgenx_subm", ("BuildSubmMap", "SubmConv3d"))):
        source = Path(module.__file__).resolve().parent
        expected = (source / "opp/vendors" / vendor).resolve()
        paths.update((Path(module.__file__), Path(module._LIBRARY), source / "torch_bridge.cpp", source / "env.sh"))
        paths.update(p for p in source.glob("op_*/*") if p.is_file())
        for name in names:
            matches = [root for root, (_, info) in configs.items() if info.get(name, {}).get("binaryList")]
            require(matches == [expected], f"Require exactly the private {vendor} provider for {name}: {matches}")
            config, info = configs[expected]
            providers[name] = str(expected)
            paths.add(config)
            for binary in info[name]["binaryList"]:
                for key in ("binPath", "jsonPath"):
                    path = (config.parents[2] / binary[key]).resolve()
                    require(expected in path.parents, f"Kernel outside private provider: {path}")
                    paths.add(path)
        paths.update(expected.rglob("*.so"))
        paths.update(expected.glob("op_impl/ai_core/tbe/config/ascend310p/*.json"))
    return {"sha256": {str(p): validation.sha256_file(p) for p in sorted(paths)}, "providers": providers,
            "baseline_sha256": {n: validation.sha256_file(BASELINE_DIR / n) for n in validation.EXPECTED_SHA256},
            "host": platform.node(), "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "environment": {k: os.environ.get(k) for k in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
                "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH", "ASCEND_RT_VISIBLE_DEVICES")}}


def tensor_state(model):
    # Hash one tensor at a time, only before preparation / after preparation / after all runs.
    return {"checkpoint_keys": list(model.state_dict()), "tensors": {
        name: {**profiler.tensor_spec(value), "requires_grad": value.requires_grad,
               "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
        for name, value in (*model.named_parameters(), *model.named_buffers())}}


def run_case(model, prepared, data, golden, row, arrays, process):
    captures, helper_diagnostics = defaultdict(list), defaultdict(list)

    def capture(module, inputs, output):
        point = inputs[0]
        require(all(point[k].device.type == "cpu" for k in METADATA), "Non-CPU input geometry")
        require(point.feat.device.type == "cpu" and point.feat.dtype == torch.float32, "Non-CPU FP32 down input")
        depth = (math.ceil(module.stride) - 1).bit_length()
        depth = 0 if depth > point.serialized_depth else depth
        _, counts = torch.unique(point.serialized_code[0] >> (3 * depth), sorted=True, return_counts=True)
        stage = len(captures[name]) + 1
        geometry = {"stage": stage, "input_points": len(point.batch), "output_points": len(counts),
                    "max_cluster": int(counts.max()), "padded_rows": len(counts) * int(counts.max()) if "fp16" in name else None}
        require(int(counts.sum()) == len(point.batch) and len(output.batch) == len(counts), "Invalid CPU cluster geometry")
        require(type(output.serialized_depth) is int, "Depth must remain a host integer")
        specs = {}
        for key in (*METADATA, "feat", "coord"):
            if key == "coord" and key not in output:
                continue
            value = output[key]
            require(value.device.type == "cpu", f"Non-CPU down boundary: {key}")
            specs[key] = profiler.tensor_spec(value)
            arrays[f"{name}_down{stage}_{key}"] = value.detach().clone().numpy()
        require(output.feat.dtype == torch.float32, "Down output must be CPU FP32")
        captures[name].append({"stage": stage, "depth": output.serialized_depth, "specs": specs, "cpu_geometry": geometry})

    for phase, runs in (("preflight", 1), ("warmup", WARMUP_RUNS), ("measured", MEASURED_RUNS), ("repeat", REPEAT_CHECKS)):
        handles = [down.register_forward_hook(capture) for down, _, _ in prepared] if phase == "preflight" else []
        try:
            for step in range(runs):
                rotation = (process + step) % len(VARIANTS)
                order = VARIANTS[rotation:] + VARIANTS[:rotation]
                row["orders"].append({"phase": phase, "step": step, "variants": order})
                outputs = {}
                for name in order:
                    variant = row["variants"][name]
                    try:
                        helper.select(prepared, name, helper_diagnostics[name] if phase == "preflight" else None)
                        validation.synchronize()
                        started = time.perf_counter_ns() if phase != "preflight" else None
                        output = model(data)
                        validation.synchronize()
                        if phase == "measured":
                            variant["samples_ms"].append((time.perf_counter_ns() - started) / 1e6)
                        outputs[name] = output.detach().cpu().numpy().copy()
                        arrays[f"{name}_{phase}_{step}"] = outputs[name]
                    except Exception:
                        variant["errors"].append({"phase": phase, "step": step, "traceback": traceback.format_exc()})
                for name, variant in row["variants"].items():
                    actual = outputs.get(name)
                    checks = {"cuda": compare(golden, actual), "reference": compare(outputs.get("reference"), actual)}
                    variant["checks"].append({"phase": phase, "step": step, **checks})
                    variant["passed"] &= all(c["passed"] for c in checks.values())
                    if phase == "repeat":
                        variant.setdefault("repeat_drift_report_only", []).append(compare(arrays.get(f"{name}_measured_0"), actual))
                row["byte_equality_report_only"].append({"phase": phase, "step": step, **{
                    f"{a}_vs_{b}": byte_equal(outputs.get(a), outputs.get(b)) for a, b in BYTE_PAIRS}})
        finally:
            for handle in handles:
                handle.remove()
        if phase == "preflight":
            for name, variant in row["variants"].items():
                stages = captures[name]
                valid = len(stages) == len(captures["reference"]) == 4
                valid &= len(helper_diagnostics[name]) == (0 if name == "reference" else 4)
                for stage, reference in zip(stages, captures["reference"]):
                    prefix, ref_prefix = f"{name}_down{stage['stage']}_", f"reference_down{stage['stage']}_"
                    stage["topology_exact"] = stage["depth"] == reference["depth"] and all(
                        byte_equal(arrays.get(prefix + k), arrays.get(ref_prefix + k)) for k in METADATA)
                    stage["coordinates_valid"] = ("coord" not in stage["specs"] if "drop_coords" in name else
                        byte_equal(arrays.get(prefix + "coord"), arrays.get(ref_prefix + "coord")))
                    stage["feature_drift_report_only"] = compare(arrays.get(ref_prefix + "feat"), arrays.get(prefix + "feat"), False)
                    diag = next((d for d in helper_diagnostics[name] if d["stage"] == stage["stage"]), None)
                    stage["helper_diagnostics"] = diag
                    stage["geometry_valid"] = (name == "reference" or (diag is not None and all(
                        diag.get(k) == v for k, v in stage["cpu_geometry"].items())))
                    valid &= stage["topology_exact"] and stage["coordinates_valid"] and stage["geometry_valid"]
                variant["preflight"] = {"passed": bool(valid), "stages": stages}
                variant["passed"] &= bool(valid)
            row["down_feature_bytes_report_only"] = {f"{a}_vs_{b}": [byte_equal(
                arrays.get(f"{a}_down{s}_feat"), arrays.get(f"{b}_down{s}_feat")) for s in range(1, 5)] for a, b in BYTE_PAIRS}
    for variant in row["variants"].values():
        variant["latency"] = validation.summarize_ms(variant["samples_ms"]) if variant["samples_ms"] else None
        variant["passed"] &= not variant["errors"] and len(variant["samples_ms"]) == MEASURED_RUNS


def diagnostics(model, prepared, data, golden, row, direct, arrays, process):
    for name, variant in row["variants"].items():
        helper.select(prepared, name)
        samples = defaultdict(list)
        profile = variant["profile"] = {"diagnostic_only": True, "samples_ms": samples, "checks": []}
        for step in range(PROFILE_RUNS):
            try:
                actual = profiler.run_partitioned(model, data, samples)
                arrays[f"{name}_profile_{step}"] = actual.copy()
                checks = {"cuda": compare(golden, actual), "direct": compare(direct.get(f"{name}_measured_0"), actual),
                          "reference": compare(direct.get("reference_measured_0"), actual)}
                profile["checks"].append(checks)
                variant["passed"] &= all(c["passed"] for c in checks.values())
            except Exception:
                variant["passed"] = False
                profile.setdefault("errors", []).append(traceback.format_exc())
        profile["stages"] = {k: {"category": profiler.stage_category(k), **validation.summarize_ms(v)}
                             for k, v in samples.items() if v}
        downs = [v for k, v in samples.items() if profiler.stage_category(k) == "downsample"]
        if PROFILE_RUNS and len(downs) == 4 and all(len(v) == PROFILE_RUNS for v in downs):
            totals = [sum(v[i] for v in downs) for i in range(PROFILE_RUNS)]
            profile["downsample_category"] = {"samples_ms": totals, **validation.summarize_ms(totals)}
    if not (AUDIT_EXECUTION and process == 0 and row["encoder"] == ENCODERS[0]):
        return
    helper.select(prepared, "fp16")
    audit = profiler.ExecutionAudit()
    variant = row["variants"]["fp16"]
    try:
        with audit:
            actual = profiler.run_partitioned(model, data, audit=audit)
        arrays["fp16_audit"] = actual.copy()
        checks = {"cuda": compare(golden, actual), "direct": compare(direct.get("fp16_measured_0"), actual),
                  "reference": compare(direct.get("reference_measured_0"), actual)}
        variant["passed"] &= all(c["passed"] for c in checks.values())
        variant["audit_checks"] = checks
    except Exception:
        variant["passed"] = False
        variant["audit_error"] = traceback.format_exc()
    finally:
        evidence = variant["audit"] = audit.report()
        downs = [r for r in evidence["operations"] if profiler.stage_category(r["stage"]) == "downsample"]
        fp32 = [r for r in downs if any(t["device"].startswith("npu") and t["dtype"] == "torch.float32"
                                      for t in r["inputs"] + r["outputs"])]
        features, cpu_features = [], []
        channels = {c for down, _, _ in prepared for c in (down.in_channels, down.out_channels)}
        for operation in downs:
            operator = operation["operator"]
            # BN's first input/output are features; remaining tensors can be auxiliary statistics.
            tensors = (operation["inputs"][:1] + operation["outputs"][:1] if "batch_norm" in operator
                       else operation["inputs"] + operation["outputs"])
            known = any(k in operator for k in ("linear", "mm.", "index_select", "amax", "gelu", "batch_norm", "_to_copy", "copy_"))
            feature_tensors = [t for t in tensors if t["dtype"] == "torch.float32" and len(t["shape"]) >= 2
                               and t["shape"][-1] in channels]
            if known and any(t["device"].startswith("npu") for t in feature_tensors):
                features.append(operation)
            if known and "copy" not in operator and any(t["device"] == "cpu" for t in feature_tensors):
                cpu_features.append(operation)
        observed = sorted({r["stage"] for r in downs if any(t["device"].startswith("npu")
                           and t["dtype"] == "torch.float16" for t in r["inputs"] + r["outputs"])})
        evidence["down_path"] = {"npu_fp16_stages": observed, "explicit_npu_fp32_features": features,
            "explicit_cpu_fp32_feature_compute": cpu_features,
            "stats_or_ambiguous_report_only": [r for r in fp32 if r not in features],
            "status": "violated" if features or cpu_features else "no explicit FP32 feature path observed" if len(observed) == 4 else "inconclusive",
            "npu_fp16_amax_stages": sorted({r["stage"] for r in downs if "amax" in r["operator"] and any(
                t["device"].startswith("npu") and t["dtype"] == "torch.float16" for t in r["outputs"])}),
            "note": "CPU FP32 input/output and BN auxiliary statistics allowed; tensor audit does not prove no fallback"}
        variant["passed"] &= not features and not cpu_features


def run_round(report, destination):
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)
    torch.manual_seed(0)
    report["provenance_before"] = provenance()
    require(report["provenance_before"]["baseline_sha256"] == validation.EXPECTED_SHA256, "Frozen golden hashes differ")
    require(helper.VARIANTS == VARIANTS and validation.COSINE_GATE == 0.9999, "Benchmark contract changed")
    for encoder in ENCODERS:
        model, prepared = validation.make_model(encoder), []
        state = report.setdefault("models", {}).setdefault(encoder, {})
        try:
            require(type(model) is PointTransformerV3Ascend, "Validator must import the current Ascend implementation")
            require(not any(m.training or getattr(m, "shuffle_orders", False) or getattr(m, "traceable", False)
                            for m in model.modules()), "Require eval, unshuffled orders and traceable=False")
            state["execution"] = dict(model.execution_config)
            require(state["execution"]["jit_compile"] is False and state["execution"]["dense_resident"], "Wrong execution config")
            state["runtime_jit_false"] = torch.npu.is_jit_compile_false()
            require(state["runtime_jit_false"], "Runtime JIT must be disabled by model construction")
            require("serialize_point" in PointTransformerV3Ascend.forward.__code__.co_names
                    and "serialize_point" in profiler.serialize_point.__code__.co_names, "Direct/profile serialization must be shared")
            state["before"] = tensor_state(model)
            prepared = helper.prepare(model)
            require(len(prepared) == 4, "Expected four downsample modules")
            state["prepared"] = [{k: profiler.tensor_spec(v) for k, v in packed.items() if v is not None}
                                 for _, _, packed in prepared]
            require(all(v.device.type == "npu" and v.dtype == torch.float16
                        for _, _, packed in prepared for v in packed.values() if v is not None), "Non-FP16 prepared state")
            state["after_prepare"] = tensor_state(model)
            require(state["before"] == state["after_prepare"], "Preparation changed source tensor/checkpoint state")
            for count in POINT_COUNTS:
                arrays = {}
                row = {"encoder": encoder, "point_count": count, "orders": [], "byte_equality_report_only": [],
                       "variants": {n: {"samples_ms": [], "checks": [], "errors": [], "passed": True} for n in VARIANTS}}
                report["results"].append(row)
                try:
                    with np.load(BASELINE_DIR / f"reference_n{count}.npz") as reference:
                        data, golden = validation.make_input(reference), reference[f"{encoder}_embedding"].copy()
                    run_case(model, prepared, data, golden, row, arrays, report["round"])
                except Exception:
                    row["error"] = traceback.format_exc()
                    for variant in row["variants"].values():
                        variant["passed"] = False
                finally:
                    np.savez(destination / f"round_{report['round']}_{encoder}_n{count}.npz", **arrays)
                    write_json(destination / f"round_{report['round']}.json", report)
            # All primary point-count windows for this resident encoder finish before instrumentation.
            if PROFILE_RUNS or (AUDIT_EXECUTION and report["round"] == 0 and encoder == ENCODERS[0]):
                row = next(r for r in report["results"] if r["encoder"] == encoder and r["point_count"] == PROFILE_POINTS)
                arrays = {}
                try:
                    with np.load(BASELINE_DIR / f"reference_n{PROFILE_POINTS}.npz") as reference:
                        data, golden = validation.make_input(reference), reference[f"{encoder}_embedding"].copy()
                    with np.load(destination / f"round_{report['round']}_{encoder}_n{PROFILE_POINTS}.npz") as direct:
                        diagnostics(model, prepared, data, golden, row, direct, arrays, report["round"])
                finally:
                    np.savez(destination / f"round_{report['round']}_{encoder}_diagnostics.npz", **arrays)
        except Exception:
            state["error"] = traceback.format_exc()
            report["failures"].append(state["error"])
        finally:
            helper.restore(prepared)
            try:
                state["after"] = tensor_state(model)
                state["unchanged"] = state.get("before") == state["after"]
                require(state["unchanged"], "Source parameters/buffers/checkpoint state changed")
                require(all("forward" not in down.__dict__ for down, _, _ in prepared), "Forward not restored")
            except Exception:
                state["error"] = traceback.format_exc()
                report["failures"].append(state["error"])
            for row in report["results"]:
                if row["encoder"] == encoder:
                    for name, variant in row["variants"].items():
                        variant["passed"] &= state.get("unchanged", False) and "error" not in state
                        variant["adoption"] = "accuracy/integrity eligible; performance report-only" if variant["passed"] else "do not adopt"
                        latency = variant.get("latency")
                        median = f"{latency['median_ms']:.3f}ms" if latency else "unavailable"
                        print(f"p{report['round']} {encoder} N={row['point_count']} {name}: "
                              f"median={median} passed={variant['passed']}", flush=True)
            write_json(destination / f"round_{report['round']}.json", report)
        del prepared, model
        torch.npu.empty_cache()
    report["completed"] = True


def aggregate(report):
    complete = len(report["rounds"]) == PROCESS_RUNS
    for count in POINT_COUNTS:
        for encoder in (*ENCODERS, "G+D"):
            values, eligible = {n: [] for n in VARIANTS}, {}
            selected = [[r for r in p["results"] if r["point_count"] == count
                         and (encoder == "G+D" or r["encoder"] == encoder)] for p in report["rounds"]]
            for name in VARIANTS:
                for rows in selected:
                    valid = len(rows) == (2 if encoder == "G+D" else 1) and all(
                        len(r["variants"][name]["samples_ms"]) == MEASURED_RUNS and r["variants"][name].get("latency") for r in rows)
                    values[name].append(sum(r["variants"][name]["latency"]["median_ms"] for r in rows) if valid else None)
                eligible[name] = (complete and not report["failures"] and all(v is not None for v in values[name])
                                  and all(r["variants"][name]["passed"] for rows in selected for r in rows))
            medians = {n: statistics.median([v for v in vs if v is not None]) if any(v is not None for v in vs) else None
                       for n, vs in values.items()}
            result = {"encoder": encoder, "point_count": count, "process_ids": [p["round"] for p in report["rounds"]],
                      "process_medians_ms": values, "median_of_process_sums_ms": medians, "eligible": eligible, "pairs": {}}
            for candidate, control in PAIRS:
                paired = [{"process": p["round"], "candidate_ms": a, "control_ms": b, "change_ms": a - b,
                           "change_percent": 100 * (a - b) / b}
                          for p, a, b in zip(report["rounds"], values[candidate], values[control]) if a is not None and b is not None and b > 0]
                if paired:
                    control_median = statistics.median(p["control_ms"] for p in paired)
                    delta = statistics.median(p["candidate_ms"] for p in paired) - control_median
                    result["pairs"][f"{candidate}_vs_{control}"] = {"raw_process_pairs": paired,
                        "change_ms": delta, "change_percent": 100 * delta / control_median,
                        "median_paired_change_ms": statistics.median(p["change_ms"] for p in paired),
                        "median_paired_change_percent": statistics.median(p["change_percent"] for p in paired),
                        "eligible": eligible[candidate] and eligible[control]}
            report["aggregate"].append(result)
            print(f"{encoder} N={count}: " + " ".join(f"{n}={medians[n]}ms passed={eligible[n]}" for n in VARIANTS), flush=True)
            for name, pair in result["pairs"].items():
                print(f"  {name}: {pair['change_ms']:+.3f}ms ({pair['change_percent']:+.2f}%), "
                      f"paired median={pair['median_paired_change_ms']:+.3f}ms", flush=True)
    report["passed"] = complete and not report["failures"] and all(all(r["eligible"].values()) for r in report["aggregate"])


def main():
    destination = RESULT_ROOT / f"downsample_compare_{datetime.now():%Y%m%d_%H%M%S_%f}"
    destination.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "rounds": [], "aggregate": [], "failures": [], "passed": False,
              "contract": {"variants": VARIANTS, "cpu_threads": CPU_THREADS, "interop_threads": CPU_THREADS,
                  "processes": PROCESS_RUNS, "warmup": WARMUP_RUNS, "measured": MEASURED_RUNS, "repeat": REPEAT_CHECKS,
                  "point_counts": POINT_COUNTS, "encoders": ENCODERS, "profile_runs": PROFILE_RUNS,
                  "audit": "one total: process 0 / generator / N2048 / fp16" if AUDIT_EXECUTION else "disabled",
                  "control": "current resident PointTransformerV3Ascend, CPE/GridEncode/dense unchanged",
                  "experiment": "drop coordinates and/or NPU FP16 Linear before padded gather/amax, eval BN, GELU; no BN fold",
                  "timing": "synchronized full-encoder host wall ms including transfers; excludes preparation, hooks, validation, hashing",
                  "schedule": "cyclic rotation by process+trial; measured 20 gives five balanced four-order cycles; warm/repeat only three trials",
                  "gate": "every embedding shape 1x512, finite, cosine >=0.9999 vs CUDA and paired current reference; exact topology/coordinate contract",
                  "report_only": "feature drift/byte equality/repeat drift/latency; FP16 metadata features are NOT exact-gated",
                  "scope": "hybrid encoder, not fully NPU; P1-first; no significance or global no-fallback claim",
                  "combined": "median of per-process G+D separate encoder-median sums, NOT request latency",
                  "diagnostics": "N2048 after each encoder's primary windows; run_partitioned only, no profiler.main/probes"}}
    print(f"Results: {destination}", flush=True)
    try:
        for index in range(PROCESS_RUNS):
            try:
                with (destination / f"round_{index}.log").open("w") as log:
                    with subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve())], cwd=REPO_ROOT,
                            env=dict(os.environ, DOWNSAMPLE_BENCH_ROUND=str(index), DOWNSAMPLE_BENCH_DEST=str(destination)),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace") as child:
                        def tee():
                            for line in child.stdout:
                                log.write(line)
                                log.flush()
                                print(line, end="", flush=True)
                        reader = threading.Thread(target=tee, daemon=True)
                        reader.start()
                        try:
                            code = child.wait(timeout=WORKER_TIMEOUT)
                        finally:
                            if child.poll() is None:
                                child.kill()
                            child.wait()
                            reader.join()
                        require(code == 0, f"Worker {index} failed; see round_{index}.log and JSON")
            except Exception:
                report["failures"].append(traceback.format_exc())
            finally:
                path = destination / f"round_{index}.json"
                try:
                    require(path.is_file(), f"Missing worker {index} JSON")
                    report["rounds"].append(json.loads(path.read_text()))
                except Exception:
                    report["failures"].append(traceback.format_exc())
                write_json(destination / "summary.json", report)
        require(len(report["rounds"]) == PROCESS_RUNS, "Missing worker JSON")
        require(all(p.get("completed") for p in report["rounds"]), "Incomplete worker")
        require(all(p.get("provenance_before") == p.get("provenance_after") == report["rounds"][0].get("provenance_before")
                    for p in report["rounds"]), "Provenance changed during/between processes")
    except Exception:
        report["failures"].append(traceback.format_exc())
    finally:
        try:
            aggregate(report)
        finally:
            write_json(destination / "summary.json", report)
    print("Three processes only; no statistical significance claim. G+D is NOT request latency.", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    require(not sys.argv[1:], "No CLI arguments; edit top constants")
    if "DOWNSAMPLE_BENCH_ROUND" not in os.environ:
        raise SystemExit(main())
    destination = Path(os.environ["DOWNSAMPLE_BENCH_DEST"])
    report = {"round": int(os.environ["DOWNSAMPLE_BENCH_ROUND"]), "pid": os.getpid(), "results": [], "failures": []}
    write_json(destination / f"round_{report['round']}.json", report)
    try:
        import numpy as np
        import torch
        import torch_npu
        sys.path.insert(0, str(REPO_ROOT))
        from ascend.benchmark import validate_ptv3 as validation, profile_ptv3_stages as profiler, downsample_experiment as helper
        from graspgenx.models.ptv3.ptv3_ascend import PointTransformerV3Ascend
        from ascend.custom_ops.grid_encode.grid_encode import grid_encode
        from ascend.custom_ops.submconv3d import submconv3d as cpe_ops
        with torch.inference_mode():
            run_round(report, destination)
    except Exception:
        report["failures"].append(traceback.format_exc())
    finally:
        try:
            report["provenance_after"] = provenance()
            require(report.get("provenance_before") == report["provenance_after"], "Runtime artifacts/golden changed")
        except Exception:
            report["failures"].append(traceback.format_exc())
        for row in report["results"]:
            for variant in row["variants"].values():
                if report["failures"]:
                    variant.update(passed=False, adoption="do not adopt")
        report["passed"] = bool(report.get("completed") and not report["failures"] and all(
            v["passed"] for row in report["results"] for v in row["variants"].values()))
        write_json(destination / f"round_{report['round']}.json", report)
    # Accuracy failures live on individual variants; do not abort later processes/cases.
    raise SystemExit(0 if report.get("completed") and not report["failures"] else 1)
