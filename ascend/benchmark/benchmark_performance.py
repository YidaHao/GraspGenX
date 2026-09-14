#!/usr/bin/env python3
"""Private P1 performance experiments; immutable working-tree control, real goldens."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "ascend/results/performance_baseline_20260913/ptv3_ascend.py"
CONTROL_SOURCE = BASELINE
MODE = "compare"
TAG = "round2_verified"
CONTROL_ENV = {"TASK_QUEUE_ENABLE": "1"}
CANDIDATE_ENV = {"TASK_QUEUE_ENABLE": "2"}
THREADS = (16, 15, 14, 13, 12, 16)
CONTROL_THREADS = 16
CANDIDATE_THREADS = 14
PROCESSES = 3
POINT_COUNTS = (64, 2048, 3500)
WARMUP = 3
MEASURE = 20


def worker(output, threads, variant):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from ascend.benchmark import validate_ptv3 as validation

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(16)
    if variant == "control":
        name = "graspgenx.models.ptv3._performance_control"
        spec = importlib.util.spec_from_file_location(name, CONTROL_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        constructor = module.PointTransformerV3Ascend
        source = CONTROL_SOURCE
    else:
        constructor = validation.PointTransformerV3
        source = ROOT / "graspgenx/models/ptv3/ptv3_ascend.py"
    report = {"variant": variant, "threads": threads, "environment": {key: os.environ.get(key) for key in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME", "TASK_QUEUE_ENABLE")}, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cases": {}}
    models = {}
    for encoder in validation.ENCODERS:
        payload = torch.load(validation.BASELINE_DIR / validation.WEIGHT_FILES[encoder], map_location="cpu", weights_only=True)
        model = constructor(in_channels=3, output_dim=512, grid_size=0.01, enable_flash=False, shuffle_orders=False)
        model.load_state_dict(payload["model"], strict=True)
        for child in model.modules():
            if hasattr(child, "shuffle_orders"):
                child.shuffle_orders = False
            if hasattr(child, "traceable"):
                child.traceable = False
        models[encoder] = model.eval()
    with torch.inference_mode():
        for count in POINT_COUNTS:
            with np.load(validation.BASELINE_DIR / f"reference_n{count}.npz") as reference:
                data = validation.make_input(reference)
                goldens = {name: reference[f"{name}_embedding"].copy() for name in models}
            samples = {name: [] for name in models}
            paired = []
            accuracy = {name: [] for name in models}
            outputs = {name: [] for name in models}
            for run in range(WARMUP + MEASURE + 3):
                validation.synchronize()
                pair_start = time.perf_counter()
                current = {}
                for name, model in models.items():
                    start = time.perf_counter()
                    current[name] = model(data)
                    validation.synchronize()
                    if WARMUP <= run < WARMUP + MEASURE:
                        samples[name].append((time.perf_counter() - start) * 1000)
                elapsed = (time.perf_counter() - pair_start) * 1000
                if WARMUP <= run < WARMUP + MEASURE:
                    paired.append(elapsed)
                for name, tensor in current.items():
                    array = tensor.cpu().numpy().copy()
                    metric = validation.compare(goldens[name], array)
                    accuracy[name].append(metric)
                    outputs[name].append(array)
                    if not metric["passed"]:
                        report["failed"] = {"encoder": name, "N": count, "run": run, "metrics": metric}
                        output.write_text(json.dumps(report, indent=2) + "\n")
                        raise RuntimeError(report["failed"])
            case = {"paired": validation.summarize_ms(paired), "paired_samples_ms": paired, "encoders": {}}
            for name in models:
                case["encoders"][name] = {"latency": validation.summarize_ms(samples[name]), "samples_ms": samples[name], "accuracy": accuracy[name], "repeat_max_abs": float(np.max(np.abs(np.asarray(outputs[name]) - outputs[name][0])))}
                np.savez(output.with_name(f"{output.stem}_{name}_n{count}.npz"), outputs=np.asarray(outputs[name]))
            case["sum_medians_ms"] = sum(row["latency"]["median_ms"] for row in case["encoders"].values())
            report["cases"][str(count)] = case
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(variant, threads, count, "G+D", round(case["sum_medians_ms"], 3), "paired", round(case["paired"]["median_ms"], 3), "cos", min(m["cosine"] for a in accuracy.values() for m in a), flush=True)


def main():
    if len(sys.argv) > 1:
        worker(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
        return
    if not CONTROL_SOURCE.is_file():
        raise FileNotFoundError(f"Required immutable control is missing: {CONTROL_SOURCE}")
    destination = ROOT / "ascend/results" / f"performance_{TAG}_{time.strftime('%Y%m%d_%H%M%S')}"
    destination.mkdir(exist_ok=False)
    shutil.copyfile(__file__, destination / "benchmark_source.py")
    shutil.copyfile(ROOT / "graspgenx/models/ptv3/ptv3_ascend.py", destination / "candidate_source.py")
    shutil.copyfile(CONTROL_SOURCE, destination / "control_source.py")
    jobs = [(i, thread, "candidate") for i, thread in enumerate(THREADS)] if MODE == "threads" else [
        (run, CONTROL_THREADS if variant == "control" else CANDIDATE_THREADS, variant)
        for run in range(PROCESSES)
        for variant in (("control", "candidate") if run % 2 == 0 else ("candidate", "control"))
    ]
    reports = []
    print("Results:", destination, flush=True)
    for index, thread, variant in jobs:
        path = destination / f"run{index}_{variant}_t{thread}.json"
        environment = dict(os.environ, **(CONTROL_ENV if variant == "control" else CANDIDATE_ENV))
        subprocess.run([sys.executable, "-B", __file__, str(path), str(thread), variant], env=environment, check=True)
        reports.append(json.loads(path.read_text()))
    summary = {"mode": MODE, "tag": TAG, "control_source": str(CONTROL_SOURCE), "baseline_sha256": hashlib.sha256(CONTROL_SOURCE.read_bytes()).hexdigest(), "workers": reports}
    for variant in (("control", "candidate") if MODE == "compare" else ()):
        rows = [r for r in reports if r["variant"] == variant]
        if rows:
            summary[variant] = {str(n): {"sum_medians_ms": statistics.median(r["cases"][str(n)]["sum_medians_ms"] for r in rows), "paired_median_ms": statistics.median(r["cases"][str(n)]["paired"]["median_ms"] for r in rows)} for n in POINT_COUNTS}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
