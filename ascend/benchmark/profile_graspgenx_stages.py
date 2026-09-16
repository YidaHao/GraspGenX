#!/usr/bin/env python3
"""Synchronized full-GraspGenX stage profile using the CUDA baseline request.

Run on 310P1 after `source ascend/env.sh`. This diagnostic script uses the same
request and model path as validate_graspgenx.py; its synchronized timings are not
the formal end-to-end latency. Switch only the direct PTV3 import for controls.
The golden directory requires cuda_baselines.npz and cuda_baselines.json with
all three CUDA deployments. Cross-backend and profiled/direct errors are reports,
not accuracy gates; invalid output or runtime failure still exits nonzero.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TASK_QUEUE_ENABLE", "2")

import torch

from graspgenx_baseline_common import (
    Pipeline, REPO_ROOT, StageTimer, build_model, compare_outputs,
    install_function_timers, install_stage_timers, load_cuda_baselines, output_arrays,
    print_cuda_comparisons, save_json,
)

RUNTIME = "npu"  # npu or cuda
CUDA_MODE = "native"  # native or tensorrt_fp16; used only for RUNTIME=cuda

WARMUP_RUNS = 3
PROFILE_RUNS = 10
CPU_THREADS = 16
CHECKPOINT_ROOTS = {
    "npu": REPO_ROOT / "ascend/release",
    "cuda": REPO_ROOT / ".artifacts/checkpoints/release",
}
GOLDEN_DIRS = {
    "npu": REPO_ROOT / "ascend/baselines/graspgenx-cuda-reference-2048-v1",
    "cuda": REPO_ROOT / ".artifacts/graspgenx-baseline/cuda-reference-2048-v1",
}


def main():
    torch.set_num_threads(CPU_THREADS)
    if RUNTIME == "npu":
        from graspgenx.models.ptv3.ptv3_ascend import PointTransformerV3Ascend as PointTransformerV3
        # from graspgenx.models.ptv3.ptv3_vanilla import PointTransformerV3Vanilla as PointTransformerV3
        encoder_class = PointTransformerV3
        implementation = PointTransformerV3.__module__.rsplit('.', 1)[-1]
    elif RUNTIME == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        encoder_class = None
        implementation = f"cuda_{CUDA_MODE}"
    else:
        raise ValueError(f"unsupported runtime: {RUNTIME}")
    result_path = (
        REPO_ROOT
        / ("ascend/results" if RUNTIME == "npu" else ".artifacts/graspgenx-baseline")
        / f"graspgenx_{implementation}_{time.strftime('%Y%m%d_%H%M%S')}_profile.json"
    )
    request, baselines, metadata = load_cuda_baselines(GOLDEN_DIRS[RUNTIME])
    model, cfg, device, acceleration = build_model(
        CHECKPOINT_ROOTS[RUNTIME],
        RUNTIME,
        encoder_class=encoder_class,
        cuda_acceleration=CUDA_MODE,
        cuda_ptv3_flash=RUNTIME == "cuda",
    )
    pipeline = Pipeline(model, request, device)
    reference_outputs, reference_traces = pipeline.run(record=True)
    reference = output_arrays(reference_outputs, reference_traces, pipeline)

    timer = StageTimer(device)
    install_stage_timers(model, timer)
    pipeline.set_timer(timer)
    with install_function_timers(timer):
        for _ in range(WARMUP_RUNS):
            pipeline.run(timer=timer)
        timer.reset()
        profiled_outputs = profiled_traces = None
        for _ in range(PROFILE_RUNS):
            pipeline.run(timer=timer)
        stage_timings = timer.summary()
        request_stage_timings = timer.request_summary(
            PROFILE_RUNS, request["diffusion_steps"]
        )
        profiled_outputs, profiled_traces = pipeline.run(record=True)
    profiled = output_arrays(profiled_outputs, profiled_traces, pipeline)
    parity = compare_outputs(reference, profiled)
    comparisons = {name: compare_outputs(golden, reference)
                   for name, golden in baselines.items()}
    valid = parity["valid"] and all(metrics["valid"] for metrics in comparisons.values())
    report = {
        "implementation": implementation,
        "runtime": RUNTIME,
        "acceleration": acceleration,
        "point_count": len(request["points"]),
        "num_grasps": request["num_grasps"],
        "diffusion_steps": request["diffusion_steps"],
        "profile_runs": PROFILE_RUNS,
        "stage_timings": stage_timings,
        "request_stage_timings": request_stage_timings,
        "profiled_vs_direct": parity,
        "numerical_policy": "report_only",
        "output_valid": valid,
        "comparisons": comparisons,
        "cuda_baselines": metadata,
    }
    save_json(result_path, report)
    for name, values in sorted(report["stage_timings"].items()):
        print(f"{name:34s} median={values['median_ms']:.3f} ms", flush=True)
    print(f"OUTPUT VALIDITY: {'VALID' if valid else 'INVALID'}", flush=True)
    print("Profiled vs direct differences are report-only; see JSON.", flush=True)
    print_cuda_comparisons(metadata, comparisons)
    print(f"Report: {result_path}", flush=True)
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
