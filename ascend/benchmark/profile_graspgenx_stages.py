#!/usr/bin/env python3
"""Synchronized full-GraspGenX stage profile using the CUDA baseline request.

Run on 310P1 after `source ascend/env.sh`. This diagnostic script uses the same
request and model path as validate_graspgenx.py; its synchronized timings are not
the formal end-to-end latency. Switch the direct generator/PTV3 imports below
independently for controls; no inference optimization toggles are provided.
The golden directory requires cuda_baselines.npz and cuda_baselines.json with
all three CUDA deployments. Cross-backend and profiled/direct errors are reports,
not accuracy gates; invalid output or runtime failure still exits nonzero.
Console tables distinguish per-call latency from per-request accumulated
latency. Means reconcile with each parent total; medians need not add up.
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

from graspgenx.models.generator_ascend import GraspGenGeneratorAscend as Generator
# from graspgenx.models.generator import GraspGenGenerator as Generator

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


def print_stage_timings(call_timings: dict, request_timings: dict) -> None:
    count = request_timings["pipeline.total"]["count"]
    print(f"\nSynchronized stage profile: {count} requests, all times in ms", flush=True)
    print("Call p50 = one invocation; Req p50/mean = accumulated time per request.")
    print("Req p50 is NOT Call p50 x Calls/req; medians are not additive.")
    print("Mean % uses the group TOTAL as denominator; TOTAL already includes its rows.")
    if "generator.likelihood" in request_timings:
        print("likelihood includes scale lookup, Normal construction, log_prob and reduction;")
        print("likelihood_setup prepares constants per request; history_write includes writes to CPU history.")

    for group in ("generator", "discriminator"):
        total = request_timings[f"{group}.total"]
        residual = request_timings[f"{group}.unattributed"]
        children = [name for name in request_timings if name.startswith(f"{group}.")
                    and name not in (f"{group}.total", f"{group}.unattributed")]
        print(f"\n[{group}]")
        print(f"{'Stage':26s} {'Calls/req':>9s} {'Call p50':>10s} "
              f"{'Req p50':>10s} {'Req mean':>10s} {'Mean %':>8s}")
        for name in (*children, f"{group}.unattributed", f"{group}.total"):
            values = request_timings[name]
            call = call_timings.get(name)
            calls = str(call["count"] // count) if call is not None else "-"
            call_p50 = f"{call['median_ms']:.3f}" if call is not None else "-"
            share = f"{100 * values['mean_ms'] / total['mean_ms']:.1f}%" if total["mean_ms"] else "n/a"
            label = "TOTAL" if name.endswith(".total") else name.split(".", 1)[1]
            print(f"{label:26s} {calls:>9s} {call_p50:>10s} "
                  f"{values['median_ms']:10.3f} {values['mean_ms']:10.3f} {share:>8s}")
        measured_mean = sum(request_timings[name]["mean_ms"] for name in children)
        print(f"Mean balance: measured {measured_mean:.3f} + "
              f"unattributed {residual['mean_ms']:.3f} = TOTAL {total['mean_ms']:.3f} ms.")
        if residual["min_ms"] < 0:
            print("WARNING: negative unattributed samples; check overlapping timers. Values were not clamped.")

    pipeline = request_timings["pipeline.total"]
    print("\n[pipeline = generator.total + discriminator.total, paired per request]")
    print(f"Req mean={pipeline['mean_ms']:.3f} ms  p50={pipeline['median_ms']:.3f} ms  "
          f"p95={pipeline['p95_ms']:.3f} ms", flush=True)
    print("Unattributed = parent time minus instrumented children, computed per request.")
    print("It includes uninstrumented work and instrumentation overhead, not a named kernel.")
    print("Scope: generator/discriminator infer only; excludes batch construction and final D2H/reporting.")
    print("Use validate_graspgenx.py for uninstrumented model-pipeline latency.\n", flush=True)


def main():
    torch.set_num_threads(CPU_THREADS)
    if RUNTIME == "npu":
        from graspgenx.models.ptv3.ptv3_ascend import PointTransformerV3Ascend as PointTransformerV3
        # from graspgenx.models.ptv3.ptv3_vanilla import PointTransformerV3Vanilla as PointTransformerV3
        encoder_class = PointTransformerV3
        implementation = PointTransformerV3.__module__.rsplit('.', 1)[-1]
        generator_class = Generator
    elif RUNTIME == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        encoder_class = None
        implementation = f"cuda_{CUDA_MODE}"
        generator_class = None  # Preserve the original CUDA generator by default.
    else:
        raise ValueError(f"unsupported runtime: {RUNTIME}")
    result_path = (
        REPO_ROOT
        / ("ascend/results" if RUNTIME == "npu" else ".artifacts/graspgenx-baseline")
        / f"graspgenx_{Generator.__module__.rsplit('.', 1)[-1] if RUNTIME == 'npu' else 'generator'}_{implementation}_{time.strftime('%Y%m%d_%H%M%S')}_profile.json"
    )
    request, baselines, metadata = load_cuda_baselines(GOLDEN_DIRS[RUNTIME])
    model, cfg, device, acceleration = build_model(
        CHECKPOINT_ROOTS[RUNTIME],
        RUNTIME,
        encoder_class=encoder_class,
        cuda_acceleration=CUDA_MODE,
        cuda_ptv3_flash=RUNTIME == "cuda",
        generator_class=generator_class,
    )
    print(f"Generator: {type(model.grasp_generator).__module__}.{type(model.grasp_generator).__name__}", flush=True)
    pipeline = Pipeline(model, request, device)
    reference_outputs, reference_traces = pipeline.run(record=True)
    reference = output_arrays(reference_outputs, reference_traces, pipeline)

    timer = StageTimer(device)
    install_stage_timers(model, timer)
    pipeline.set_timer(timer)
    with install_function_timers(timer, model):
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
        "generator_implementation": f"{type(model.grasp_generator).__module__}.{type(model.grasp_generator).__name__}",
        "runtime": RUNTIME,
        "acceleration": acceleration,
        "point_count": len(request["points"]),
        "num_grasps": request["num_grasps"],
        "diffusion_steps": request["diffusion_steps"],
        "profile_runs": PROFILE_RUNS,
        "stage_timings": stage_timings,
        "request_stage_timings": request_stage_timings,
        "stage_calls_per_request": {
            name: timing["count"] // PROFILE_RUNS for name, timing in stage_timings.items()
        },
        "timing_scope": {
            "stage_timings": "per invocation",
            "request_stage_timings": "sum invocations within each request, then summarize requests",
            "unattributed": "parent minus instrumented children within the same request; includes instrumentation overhead",
            "pipeline_total": "generator.total + discriminator.total within the same request; excludes batch construction and final D2H/reporting",
            "additivity": "means add up before rounding; medians need not",
        },
        "profiled_vs_direct": parity,
        "numerical_policy": "report_only",
        "output_valid": valid,
        "comparisons": comparisons,
        "cuda_baselines": metadata,
    }
    save_json(result_path, report)
    print_stage_timings(stage_timings, request_stage_timings)
    print(f"OUTPUT VALIDITY: {'VALID' if valid else 'INVALID'}", flush=True)
    print("Profiled vs direct differences are report-only; see JSON.", flush=True)
    print_cuda_comparisons(metadata, comparisons)
    print(f"Report: {result_path}", flush=True)
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
