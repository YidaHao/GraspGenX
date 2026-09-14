#!/usr/bin/env python3
"""Replay real PTV3 attention inputs, separating host walls from device traces.

Load IB_Robot/.shrc_local from its workspace and prepend this repository to
PYTHONPATH. Run `python ascend/benchmark/profile_attention_breakdown.py`.
Requires the existing PTV3 CUDA golden files and encoder weights used by
profile_ptv3_stages.py. Does not change model code or golden data.
"""

from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import csv
import json
import platform
import time

import numpy as np
import torch
import torch_npu

from ascend.benchmark import profile_ptv3_stages as base
from graspgenx.models.ptv3.ptv3_vanilla import VanillaPoint

# Fixed workload: base's 2048-point real input, both official encoder weights.
CPU_THREADS = 16
WARMUP = 5
RUNS = 30
TRACE_RUNS = 2
TRACE_WARMUP = 1
OUTPUT_DIR = base.RESULT_DIR / (
    "attention_breakdown_" + platform.node() + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

STAGES = {
    "01.norm": "CPU FP32",
    "02.checks": "CPU Python",
    "03.indices": "CPU metadata: no-padding views or padding/gather fallback",
    "04.h2d_features": "CPU/NPU transfer and FP32->FP16 cast",
    "05.h2d_indices": "CPU/NPU transfer and int64->int32 cast",
    "06.qkv": "NPU Linear plus host dispatch",
    "07.order_gather": "NPU gather plus host dispatch",
    "08.qkv_views": "Host tensor metadata",
    "09.qkv_contiguous": "NPU layout copies plus host dispatch",
    "10.scale": "NPU vector scale plus host dispatch",
    "11.qk": "NPU batched matmul plus host dispatch",
    "12.softmax": "NPU softmax plus host dispatch",
    "13.av": "NPU batched matmul plus host dispatch",
    "14.output_layout": "NPU layout copy plus host metadata",
    "15.inverse_gather": "NPU gather plus host dispatch",
    "16.projection": "NPU Linear plus host dispatch",
    "17.d2h_features": "NPU/CPU transfer, FP16->FP32 cast and possible queue wait",
    "18.residual": "CPU FP32",
}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


@torch.inference_mode()
def capture_cases(model, data):
    cases = {}
    handles = []
    for name, block in model.named_modules():
        if not hasattr(block, "attn"):
            continue
        cases[name] = {"block": block}

        def norm_input(module, inputs, name=name):
            cases[name]["shortcut"] = inputs[0].clone()

        def attention_input(module, inputs, name=name):
            cases[name]["point"] = dict(inputs[0])

        def attention_output(module, inputs, output, name=name):
            cases[name]["expected"] = (
                cases[name]["shortcut"] + output.feat
            ).clone()

        handles.append(block.norm1.register_forward_pre_hook(norm_input))
        handles.append(block.attn.register_forward_pre_hook(attention_input))
        handles.append(block.attn.register_forward_hook(attention_output))
    try:
        output = model(data)
        torch.npu.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    return cases, output.cpu().numpy()


def original(case):
    point = VanillaPoint(dict(case["point"]))
    point.feat = case["shortcut"]
    return base.run_block_attention(case["block"], point).feat


def split(case, synchronize_stages=False, samples=None, trace_label=None):
    block = case["block"]
    attn = block.attn
    point = VanillaPoint(dict(case["point"]))
    shortcut = case["shortcut"]
    device = attn.qkv.weight.device

    def stage(name, operation):
        context = (
            torch.profiler.record_function(f"ATTN/{trace_label}/{name}")
            if trace_label else nullcontext()
        )
        with context:
            started = time.perf_counter()
            result = operation()
            if synchronize_stages:
                torch.npu.synchronize()
            if samples is not None:
                samples[name].append((time.perf_counter() - started) * 1000)
            return result

    def checks():
        if attn.training or point.feat.device.type != "cpu" or point.feat.dtype != torch.float32:
            raise RuntimeError("Expected eval CPU FP32 features")
        for parameter in attn.parameters():
            if parameter.device.type != "npu" or parameter.dtype != torch.float16:
                raise RuntimeError("Expected NPU FP16 attention parameters")

    point.feat = stage("01.norm", lambda: block.norm1(shortcut))
    stage("02.checks", checks)
    order, inverse = stage("03.indices", lambda: attn.prepare_indices(point))
    features = stage("04.h2d_features", lambda: point.feat.to(device=device, dtype=torch.float16))
    order, inverse = stage("05.h2d_indices", lambda: (
        order.to(device=device, dtype=torch.int32),
        inverse.to(device=device, dtype=torch.int32),
    ))
    qkv = stage("06.qkv", lambda: attn.qkv(features))
    qkv = stage("07.order_gather", lambda: qkv.index_select(0, order))
    k, h, c = attn.patch_size, attn.num_heads, attn.channels
    query, key, value = stage("08.qkv_views", lambda: (
        qkv.reshape(-1, k, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(dim=0)
    ))
    query, key, value = stage("09.qkv_contiguous", lambda: (
        query.contiguous(), key.contiguous(), value.contiguous()
    ))
    query = stage("10.scale", lambda: query * attn.scale)
    scores = stage("11.qk", lambda: query @ key.transpose(-2, -1))
    probabilities = stage("12.softmax", lambda: attn.softmax(scores))
    features = stage("13.av", lambda: probabilities @ value)
    features = stage("14.output_layout", lambda: features.transpose(1, 2).reshape(-1, c))
    features = stage("15.inverse_gather", lambda: features.index_select(0, inverse))
    features = stage("16.projection", lambda: attn.proj(features))
    features = stage("17.d2h_features", lambda: features.to(device="cpu", dtype=torch.float32))
    return stage("18.residual", lambda: shortcut + block.drop_path(features))


@torch.inference_mode()
def measure(cases, mode):
    # Stage sync is measured in a separate run. It perturbs launch overlap and
    # includes synchronization overhead, so it is not pure kernel time.
    all_samples = {name: defaultdict(list) for name in cases}
    for _ in range(WARMUP):
        for case in cases.values():
            original(case) if mode == "original" else split(case, mode == "synchronized")
    torch.npu.synchronize()
    max_abs = 0.0
    for _ in range(RUNS):
        for name, case in cases.items():
            torch.npu.synchronize()
            started = time.perf_counter()
            if mode == "original":
                output = original(case)
            else:
                output = split(case, mode == "synchronized", all_samples[name])
            torch.npu.synchronize()
            all_samples[name]["total"].append((time.perf_counter() - started) * 1000)
            max_abs = max(max_abs, float((output - case["expected"]).abs().max()))
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError(f"Non-finite output in {mode}/{name}")
    if max_abs != 0:
        raise RuntimeError(f"Split/original replay changed outputs: max_abs={max_abs}")
    aggregate = {}
    for stage_name in ("total", *STAGES):
        if stage_name not in next(iter(all_samples.values())):
            continue
        totals = np.sum([values[stage_name] for values in all_samples.values()], axis=0)
        aggregate[stage_name] = stats(totals)
    return {
        "max_abs_vs_captured": max_abs,
        "aggregate": aggregate,
        "block_samples_ms": {name: dict(values) for name, values in all_samples.items()},
    }


def print_summary(report):
    print("\nSummed attention replay wall per encoder (ms)")
    for encoder, result in report["encoders"].items():
        for mode, measured in result["measurements"].items():
            total = measured["aggregate"]["total"]
            print(encoder, mode, f"median={total['median_ms']:.3f}",
                  f"mean={total['mean_ms']:.3f}", f"p95={total['p95_ms']:.3f}")
    print("\nSubstages: sum of both encoder means (ms); sync includes fence overhead")
    for name, role in STAGES.items():
        values = {}
        for mode in ("enqueue", "synchronized"):
            values[mode] = sum(r["measurements"][mode]["aggregate"][name]["mean_ms"]
                               for r in report["encoders"].values())
        print(f"{name:23s} enqueue={values['enqueue']:8.3f} "
              f"sync_wall={values['synchronized']:8.3f}  {role}")


def summarize_trace(trace_dir, expected_blocks):
    operator_path, = trace_dir.glob("*/ASCEND_PROFILER_OUTPUT/operator_details.csv")
    kernel_path, = trace_dir.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
    by_stage = defaultdict(float)
    counts = defaultdict(int)
    with operator_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Name"].startswith("ATTN/"):
                stage_name = row["Name"].rsplit("/", 1)[-1]
                by_stage[stage_name] += float(row["Device Total Duration(us)"])
                counts[stage_name] += 1
    if any(counts[name] != expected_blocks * TRACE_RUNS for name in STAGES):
        raise RuntimeError(f"Incomplete trace labels: {dict(counts)}")

    with kernel_path.open(newline="") as stream:
        kernels = list(csv.DictReader(stream))
    # Subtract the large epoch timestamp before converting to float so
    # microsecond interval unions retain sub-microsecond resolution.
    origin = min(Decimal(row["Start Time(us)"].strip()) for row in kernels)
    intervals = sorted((float(Decimal(row["Start Time(us)"].strip()) - origin),
                        float(row["Duration(us)"])) for row in kernels)
    union_us = 0.0
    end = float("-inf")
    for start, duration in intervals:
        union_us += max(0.0, start + duration - max(start, end))
        end = max(end, start + duration)
    task_sum_us = sum(duration for _, duration in intervals)
    # Ranges are disjoint CPU annotations. Each device task must be attributed
    # exactly once, even when multiple tasks overlap on the device timeline.
    if abs(sum(by_stage.values()) - task_sum_us) > 2.0:
        raise RuntimeError("Annotated task durations do not match kernel CSV")
    return {
        "device_task_sum_ms_per_replay": task_sum_us / TRACE_RUNS / 1000,
        "device_task_union_ms_per_replay": union_us / TRACE_RUNS / 1000,
        "task_count_per_replay": len(kernels) / TRACE_RUNS,
        "stage_device_task_sum_ms": {name: by_stage[name] / TRACE_RUNS / 1000 for name in STAGES},
        "stage_label_counts": dict(counts),
        "note": "Task sums include overlaps; union removes overlap. Neither includes all DMA/host waits. Trace instrumentation changes launch gaps.",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(CPU_THREADS)
    data, golden = base.load_input()
    report = {
        "host": platform.node(), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "point_count": base.POINT_COUNT,
        "cpu_threads": CPU_THREADS, "warmup": WARMUP, "runs": RUNS,
        "trace_runs": TRACE_RUNS, "trace_warmup": TRACE_WARMUP,
        "stage_roles": STAGES, "encoders": {},
        "scope": "Replay 14 captured real attention inputs per encoder; excludes CPE/FFN",
    }
    print(f"Output: {OUTPUT_DIR}", flush=True)
    for encoder in base.ENCODERS:
        model = base.make_model(encoder)
        cases, embedding = capture_cases(model, data)
        accuracy = base.compare(golden[f"{encoder}_embedding"], embedding)
        if not accuracy["passed"]:
            raise RuntimeError(f"Full encoder CUDA-golden accuracy failed: {accuracy}")
        result = {
            "accuracy": accuracy,
            "shapes": {name: {
                "points": len(case["shortcut"]),
                "channels": case["block"].attn.channels,
                "heads": case["block"].attn.num_heads,
                "patch_size": case["block"].attn.patch_size,
                "padding_cached_at_entry": "pad" in case["point"],
            } for name, case in cases.items()},
            "measurements": {},
        }
        for mode in ("original", "enqueue", "synchronized"):
            result["measurements"][mode] = measure(cases, mode)
            print(encoder, mode, result["measurements"][mode]["aggregate"]["total"], flush=True)
        report["encoders"][encoder] = result
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

        trace_dir = OUTPUT_DIR / f"trace_{encoder}"
        print(f"Capturing {encoder} device trace", flush=True)
        with torch.inference_mode(), torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=TRACE_WARMUP, active=TRACE_RUNS, repeat=1),
            record_shapes=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(export_type="text"),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
        ) as prof:
            for run in range(TRACE_WARMUP + TRACE_RUNS):
                for name, case in cases.items():
                    split(case, trace_label=f"{encoder}/{run}/{name}")
                torch.npu.synchronize()
                prof.step()
        result["trace"] = summarize_trace(trace_dir, len(cases))
        print(encoder, "device trace:", result["trace"], flush=True)
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        del cases, model
        torch.npu.empty_cache()
    print_summary(report)
    print("\nDevice task sums from traces, both encoders combined (ms, overlaps included)")
    for name in STAGES:
        duration = sum(r["trace"]["stage_device_task_sum_ms"][name] for r in report["encoders"].values())
        print(name, f"{duration:.4f}")
    print(f"Report: {OUTPUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
