#!/usr/bin/env python3
"""Same-host, paired full-encoder comparison; no implementation/device CLI flags."""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MethodType

import numpy as np
import torch
import torch_npu

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "ascend/baselines/ptv3-cuda-fp32-eager"
RESULT_ROOT = REPO_ROOT / "ascend/results"
POINT_COUNTS = (64, 2048, 3500)
ENCODERS = ("generator", "discriminator")
PROCESS_RUNS = 3
WARMUP_RUNS = 3
MEASURED_RUNS = 20
REPEAT_CHECKS = 3
PROFILE_RUNS = 5
CPU_THREADS = 16
BENCHMARK_HOST = "192.168.7.101"
BENCHMARK_DEVICE = "Ascend310P1"

sys.path.insert(0, str(REPO_ROOT))
from ascend.tools import validate_ptv3 as validation
from ascend.tools import profile_ptv3_stages as profiler
from graspgenx.models.ptv3.ptv3_ascend import (
    PointTransformerV3Ascend,
    CachedCPEConv,
    NPU_DEVICE,
)
from graspgenx.models.ptv3.ptv3_vanilla import HashSparseConv3d

# Historical labels identify benchmark-only ablations, not model API choices.
VARIANTS = ("current", "npu_subm", "mixed_subm")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def cpu_map_control(conv, grid, batch, point):
    cache = conv._get_neighbor_map(grid, batch, point)
    if "npu" not in cache:
        n, volume = grid.shape[0], conv.kernel_size**3
        packed = torch.full((n, (volume + 7) // 8 * 8), -1, dtype=torch.int32)
        packed[:, :volume] = cache["indices"].view(n, volume)
        packed[:, :volume].masked_fill_(~cache["found"].view(n, volume), -1)
        cache["npu"] = packed.to(NPU_DEVICE)
    return cache["npu"]


def model_from_weights(state, variant):
    model = PointTransformerV3Ascend(in_channels=3, output_dim=512, grid_size=0.01,
                                   enable_flash=False, shuffle_orders=False)
    model.load_state_dict(state, strict=True)
    if variant != "mixed_subm":
        for stage in model.enc:
            for block in stage.children():
                if hasattr(block, "attn"):
                    if variant == "current":
                        block.cpe_conv = CachedCPEConv(block.cpe_conv)
                    else:
                        block.cpe_conv._get_npu_map = MethodType(cpu_map_control, block.cpe_conv)
        model.execution_config["cpe_map"] = "cpu_full_map"
        if variant == "current":
            model.execution_config["cpe_compute"] = "cpu_fp32"
    for module in model.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = False
        if hasattr(module, "traceable"):
            module.traceable = False
        if getattr(module, "npu_weight", None) is not None:
            assert torch.equal(module.npu_weight.cpu(), module.weight.half())
    return model.eval()


def inspect_geometry(model, data):
    rows, hooks = [], []
    for stage_index, stage in enumerate(model.enc):
        def capture(module, args, stage_index=stage_index):
            feat, grid, batch = args[:3]
            keys = HashSparseConv3d._hash(batch, grid)
            unique = keys.unique().numel()
            rows.append({"stage": stage_index, "N": len(feat), "C": feat.shape[1],
                         "unique_hashes": unique,
                         "duplicate_hash_rows": len(feat) - unique})
        hooks.append(stage.block0.cpe_conv.register_forward_pre_hook(capture))
    try:
        model(data)
        torch.npu.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
    return rows


@torch.inference_mode()
def run_round(index, destination):
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(0)
    report = {"round": index, "pid": os.getpid(), "results": [], "failures": []}
    report_path = destination / f"round_{index}.json"
    try:
        for encoder in ENCODERS:
            state = torch.load(BASELINE_DIR / validation.WEIGHT_FILES[encoder],
                               map_location="cpu", weights_only=False)["model"]
            models = {name: model_from_weights(state, name) for name in VARIANTS}
            report["execution_configs"] = {name: model.execution_config for name, model in models.items()}
            del state
            for count in POINT_COUNTS:
                with np.load(BASELINE_DIR / f"reference_n{count}.npz") as reference:
                    data = validation.make_input(reference)
                    golden = reference[f"{encoder}_embedding"].copy()
                geometry = inspect_geometry(models["current"], data)
                print(f"round={index} {encoder} N={count} geometry={geometry}", flush=True)
                for model in models.values():
                    for _ in range(WARMUP_RUNS):
                        model(data)
                    torch.npu.synchronize()
                samples = {name: [] for name in models}
                outputs = {}
                names = list(models)
                # Rotate all controls/candidates each measurement and process.
                for step in range(MEASURED_RUNS):
                    first = (index + step) % len(names)
                    for name in names[first:] + names[:first]:
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        outputs[name] = models[name](data)
                        torch.npu.synchronize()
                        samples[name].append((time.perf_counter() - start) * 1000)
                arrays = {name: output.float().cpu().numpy().copy() for name, output in outputs.items()}
                for name, model in models.items():
                    actual = arrays[name]
                    drift = []
                    repeats_valid = True
                    for _ in range(REPEAT_CHECKS):
                        repeat = model(data).float().cpu().numpy()
                        torch.npu.synchronize()
                        repeats_valid &= repeat.shape == golden.shape and bool(np.isfinite(repeat).all())
                        drift.append(float(np.max(np.abs(repeat - actual))))
                    row = {
                        "encoder": encoder, "point_count": count, "variant": name,
                        "accuracy": validation.compare(golden, actual),
                        "vs_current": validation.compare(arrays["current"], actual),
                        "vs_cpu_map_subm": validation.compare(arrays["npu_subm"], actual),
                        "repeat_max_abs": max(drift), "repeats_valid": repeats_valid,
                        "latency": validation.summarize_ms(samples[name]),
                        "latency_samples_ms": samples[name], "geometry": geometry,
                    }
                    np.savez(destination / f"round_{index}_{name}_{encoder}_n{count}.npz",
                             embedding=actual, points=data["coord"].numpy())
                    report["results"].append(row)
                    print(f"{name:15s} {encoder:13s} N={count} "
                          f"median={row['latency']['median_ms']:.3f} ms "
                          f"cos={row['accuracy']['cosine']:.8f} "
                          f"max_abs={row['accuracy']['max_abs']:.6f} "
                          f"PASS={row['accuracy']['passed'] and repeats_valid}", flush=True)
                    write_json(report_path, report)
                # Separate diagnostics from the uninstrumented latency above.
                if count == 2048:
                    for name, model in models.items():
                        stage_samples = defaultdict(list)
                        for _ in range(PROFILE_RUNS):
                            profiled = profiler.run_partitioned(model, data, stage_samples)
                            if not validation.compare(arrays[name], profiled)["passed"]:
                                raise RuntimeError(f"invalid partitioned CPE profile: {name} {encoder}")
                        categories = defaultdict(lambda: [0.0] * PROFILE_RUNS)
                        for stage_name, values in stage_samples.items():
                            if profiler.is_leaf_stage(stage_name):
                                category = profiler.stage_category(stage_name)
                                categories[category] = [a + b for a, b in zip(categories[category], values)]
                        row = next(r for r in reversed(report["results"]) if r["variant"] == name)
                        row["profile"] = {
                            "runs": PROFILE_RUNS,
                            "vs_direct": validation.compare(arrays[name], profiled),
                            "categories": {k: validation.summarize_ms(v) for k, v in categories.items()},
                            "stage_samples_ms": dict(stage_samples),
                        }
                        write_json(report_path, report)
            del models, outputs
            torch.npu.empty_cache()
        report["completed"] = True
    except Exception as exc:
        report["failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        write_json(report_path, report)


def main():
    destination = RESULT_ROOT / f"subm_compare_{datetime.now():%Y%m%d_%H%M%S}"
    destination.mkdir(parents=True, exist_ok=False)
    for filename, expected in validation.EXPECTED_SHA256.items():
        assert validation.sha256_file(BASELINE_DIR / filename) == expected, filename
    report = {
        "contract": {"point_counts": POINT_COUNTS, "encoders": ENCODERS,
                     "process_runs": PROCESS_RUNS, "warmup": WARMUP_RUNS,
                     "measured_runs": MEASURED_RUNS, "repeat_checks": REPEAT_CHECKS,
                     "cpu_threads": CPU_THREADS, "cosine_gate": validation.COSINE_GATE,
                      "scope": "encoder-only; maps rebuilt once/stage/forward; all transfers included",
                      "reference": "benchmark-only cached CPU CPE (not the historical uncached control)",
                      "order": "cyclic variant rotation in each process", "profile_runs": PROFILE_RUNS},
        "environment": {"host": BENCHMARK_HOST, "device": BENCHMARK_DEVICE,
                        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                        "cann": os.environ.get("ASCEND_HOME_PATH")},
        "sources": {str(p.relative_to(REPO_ROOT)): validation.sha256_file(p) for p in (
            Path(__file__), REPO_ROOT / "graspgenx/models/ptv3/ptv3_ascend.py",
            REPO_ROOT / "graspgenx/models/ptv3/ptv3_vanilla.py",
            REPO_ROOT / "ascend/custom_ops/submconv3d/op_kernel/subm_conv3d.cpp",
            REPO_ROOT / "ascend/custom_ops/submconv3d/op_host/subm_conv3d.cpp",
            REPO_ROOT / "ascend/custom_ops/submconv3d/op_kernel/build_subm_map.cpp",
            REPO_ROOT / "ascend/custom_ops/submconv3d/op_host/build_subm_map.cpp",
            REPO_ROOT / "ascend/custom_ops/submconv3d/build/torch_bridge.so")},
        "baseline_dir": str(BASELINE_DIR), "rounds": [],
    }
    print(f"Results: {destination}", flush=True)
    write_json(destination / "summary.json", report)
    for index in range(PROCESS_RUNS):
        env = dict(os.environ, SUBM_BENCH_ROUND=str(index), SUBM_BENCH_DEST=str(destination))
        completed = subprocess.run([sys.executable, "-u", __file__], env=env,
                                   cwd=REPO_ROOT, timeout=600)
        report["rounds"].append(json.loads((destination / f"round_{index}.json").read_text()))
        write_json(destination / "summary.json", report)
        if completed.returncode:
            return completed.returncode
    report["aggregate"] = []
    for encoder in ENCODERS:
        for count in POINT_COUNTS:
            for name in VARIANTS:
                rows = [r for run in report["rounds"] for r in run["results"]
                        if (r["encoder"], r["point_count"], r["variant"]) == (encoder, count, name)]
                report["aggregate"].append({
                    "encoder": encoder, "point_count": count, "variant": name,
                    "process_medians_ms": [r["latency"]["median_ms"] for r in rows],
                    "median_of_process_medians_ms": float(np.median([r["latency"]["median_ms"] for r in rows])),
                    "all_samples": validation.summarize_ms([x for r in rows for x in r["latency_samples_ms"]]),
                    "min_cosine": min(r["accuracy"]["cosine"] for r in rows),
                    "max_abs": max(r["accuracy"]["max_abs"] for r in rows),
                    "max_mean_abs": max(r["accuracy"]["mean_abs"] for r in rows),
                    "max_relative_l2": max(r["accuracy"]["relative_l2"] for r in rows),
                    "min_cosine_vs_current": min(r["vs_current"]["cosine"] for r in rows),
                    "max_abs_vs_current": max(r["vs_current"]["max_abs"] for r in rows),
                    "max_repeat_drift": max(r["repeat_max_abs"] for r in rows),
                    "passed": all(r["accuracy"]["passed"] and r["repeats_valid"] for r in rows),
                })
    report["profile_aggregate"] = []
    for encoder in ENCODERS:
        for name in VARIANTS:
            profiles = [r["profile"] for run in report["rounds"] for r in run["results"]
                        if (r["encoder"], r["point_count"], r["variant"]) == (encoder, 2048, name)]
            report["profile_aggregate"].append({
                "encoder": encoder, "variant": name,
                "max_abs_vs_direct": max(p["vs_direct"]["max_abs"] for p in profiles),
                "category_median_ms": {
                    k: float(np.median([p["categories"][k]["median_ms"] for p in profiles]))
                    for k in profiles[0]["categories"]
                },
            })
    by_case = {(r["encoder"], r["point_count"], r["variant"]): r for r in report["aggregate"]}
    report["combined"] = []
    for count in POINT_COUNTS:
        medians = {name: sum(by_case[e, count, name]["median_of_process_medians_ms"]
                             for e in ENCODERS) for name in VARIANTS}
        report["combined"].append({
            "point_count": count, "sum_encoder_medians_ms": medians,
            "change_ms_vs_current": {k: v - medians["current"] for k, v in medians.items()},
            "change_percent_vs_current": {k: (v / medians["current"] - 1) * 100
                                          for k, v in medians.items()},
        })
    write_json(destination / "summary.json", report)
    print(json.dumps(report["combined"], indent=2), flush=True)
    return 0 if all(r["passed"] for r in report["aggregate"]) else 1


if __name__ == "__main__":
    if "SUBM_BENCH_ROUND" in os.environ:
        run_round(int(os.environ["SUBM_BENCH_ROUND"]), Path(os.environ["SUBM_BENCH_DEST"]))
    else:
        raise SystemExit(main())
