#!/usr/bin/env python3
"""Validate and benchmark the full GraspGenX pipeline against CUDA golden.

Run on 310P1 after `source ascend/env.sh`. Required cuda_baselines.npz and
cuda_baselines.json must exist in GOLDEN_DIR (alongside the preserved golden).
Cosine and all error metrics are report-only. Exit 1 means invalid output or a
runtime failure, not a numerical difference. There are no CLI args.
Switch only the direct PTV3 import to run a CPU vanilla control.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from importlib.metadata import version

os.environ.setdefault("TASK_QUEUE_ENABLE", "2")

import numpy as np
import torch

from graspgenx_baseline_common import (
    Pipeline, REPO_ROOT, build_model, compare_outputs, load_cuda_baselines,
    output_arrays, print_cuda_comparisons, run_timed, save_json,
)

from graspgenx.models.ptv3.ptv3_ascend import PointTransformerV3Ascend as PointTransformerV3
# from graspgenx.models.ptv3.ptv3_vanilla import PointTransformerV3Vanilla as PointTransformerV3


CHECKPOINT_ROOT = REPO_ROOT / "ascend/release"
GOLDEN_DIR = REPO_ROOT / "ascend/baselines/graspgenx-cuda-reference-2048-v1"
RESULT_DIR = REPO_ROOT / "ascend/results"
RESULT_PATH = RESULT_DIR / f"graspgenx_{PointTransformerV3.__module__.rsplit('.', 1)[-1]}_{time.strftime('%Y%m%d_%H%M%S')}_validation.json"
OUTPUT_PATH = RESULT_PATH.with_suffix(".npz")

WARMUP_RUNS = 3
MEASURED_RUNS = 20
CPU_THREADS = 16


def main():
    torch.set_num_threads(CPU_THREADS)
    request, baselines, metadata = load_cuda_baselines(GOLDEN_DIR)
    model, cfg, device, acceleration = build_model(
        CHECKPOINT_ROOT, "npu", encoder_class=PointTransformerV3
    )
    pipeline = Pipeline(model, request, device)
    samples, timings = run_timed(pipeline, WARMUP_RUNS, MEASURED_RUNS)
    outputs, traces = pipeline.run(record=True)
    candidate = output_arrays(outputs, traces, pipeline)
    comparisons = {name: compare_outputs(golden, candidate)
                   for name, golden in baselines.items()}
    valid = all(metrics["valid"] for metrics in comparisons.values())
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH, **candidate)
    report = {
        "implementation": f"{PointTransformerV3.__module__}.{PointTransformerV3.__name__}",
        "runtime": "npu",
        "torch": torch.__version__,
        "torch_npu": version("torch-npu"),
        "point_count": len(request["points"]),
        "num_grasps": request["num_grasps"],
        "diffusion_steps": request["diffusion_steps"],
        "numerical_policy": "report_only",
        "output_valid": valid,
        "comparisons": comparisons,
        "cuda_baselines": metadata,
        "timings": timings,
        "samples_ms": samples,
        "output": str(OUTPUT_PATH),
    }
    save_json(RESULT_PATH, report)
    print(f"OUTPUT VALIDITY: {'VALID' if valid else 'INVALID'}", flush=True)
    print(f"steady median: {timings['steady']['median_ms']:.3f} ms", flush=True)
    print_cuda_comparisons(metadata, comparisons)
    print(f"Report: {RESULT_PATH}", flush=True)
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
