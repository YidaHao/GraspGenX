#!/usr/bin/env python3
"""Profile PTV3 vanilla stages against the shared CUDA golden baseline.

Configuration is intentionally kept in the block immediately below. The script
takes no command-line arguments. Its synchronized stage timings are diagnostic;
use ``validate_ptv3.py`` for uninstrumented correctness and latency results.

Prerequisites
-------------
Run from the GraspGenX repository after loading CANN::

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python3 ascend/tools/profile_ptv3_stages.py

The following files must exist under ``BASELINE_DIR``::

    generator_ptv3_state.pth
    discriminator_ptv3_state.pth
    reference_n2048.npz

The reference points are already sampled, mean-centered, and multiplied by
kappa. ``USE_CUDA_GOLDEN=False`` retains a synthetic random-weight CPU mode for
the historical profiler use case.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:  # CPU fallback mode remains usable without torch-npu
    torch_npu = None


# ---------------------------------------------------------------------------
# Configuration and required paths. Edit values here, not at the call site.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "ascend/baselines/ptv3-cuda-fp32-eager"
POINT_COUNT = 2048
RESULT_PATH = REPO_ROOT / (
    f"ascend/results/ptv3_vanilla_310p_profile_n{POINT_COUNT}.json"
)

USE_CUDA_GOLDEN = True
# Start with the original vanilla model on the 310P host CPU. Change to
# "npu:0" after the NPU implementation can execute the complete encoder.
DEVICE = "cpu"
SERIALIZATION_DEVICE = "cpu"
DOWNSAMPLE_DEVICE = "cpu"
FINAL_POOL_DEVICE = "cpu"
ENCODERS = ("generator", "discriminator")
WARMUP_RUNS = 1
PROFILE_RUNS = 20
CPU_THREADS = 16

GRID_SIZE = 0.01
KAPPA = 3.27
OUTPUT_DIM = 512
ENABLE_FLASH = False
SHUFFLE_ORDERS = False

MAX_ABS_GATE = 1.5e-2
MEAN_ABS_GATE = 3e-3
RELATIVE_L2_GATE = 5e-3
COSINE_GATE = 0.99999

WEIGHT_FILES = {
    "generator": "generator_ptv3_state.pth",
    "discriminator": "discriminator_ptv3_state.pth",
}
RANDOM_WEIGHT_SEEDS = {"generator": 2026, "discriminator": 2027}
REFERENCE_FILE = f"reference_n{POINT_COUNT}.npz"


def install_minimal_import_shims() -> None:
    """Provide eval-equivalent fallbacks for absent addict and timm packages."""
    if importlib.util.find_spec("addict") is None:
        addict_module = types.ModuleType("addict")

        class AddictDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            __setattr__ = dict.__setitem__

        addict_module.Dict = AddictDict
        addict_module.__spec__ = importlib.machinery.ModuleSpec(
            "addict", loader=None
        )
        sys.modules["addict"] = addict_module

    if importlib.util.find_spec("timm") is None:
        layers_module = types.ModuleType("timm.models.layers")

        class DropPath(torch.nn.Module):
            def __init__(self, drop_prob=0.0, *args, **kwargs):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, value):
                if self.training and self.drop_prob:
                    raise RuntimeError("DropPath shim is valid for eval mode only")
                return value

        layers_module.DropPath = DropPath
        layers_module.__spec__ = importlib.machinery.ModuleSpec(
            "timm.models.layers", loader=None
        )
        sys.modules["timm.models.layers"] = layers_module


os.environ.setdefault("GRASPGENX_GRIPPER_CFG_DIR", str(REPO_ROOT / "assets"))
os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", str(BASELINE_DIR))
install_minimal_import_shims()

# This direct import is shared with validate_ptv3.py by design.
from graspgenx.models.ptv3.ptv3_vanilla import (
    PointTransformerV3Vanilla,
    VanillaPoint,
    segment_csr_vanilla,
)


class StageFailure(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"stage {stage!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values),
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "passed": False,
        }
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    difference = candidate64 - reference64
    denominator = np.linalg.norm(reference64) * np.linalg.norm(candidate64)
    metrics = {
        "shape_match": True,
        "finite": bool(np.isfinite(candidate).all()),
        "max_abs": float(np.abs(difference).max()),
        "mean_abs": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference64), 1e-12)
        ),
        "cosine": (
            float(np.vdot(reference64, candidate64) / denominator)
            if denominator
            else 1.0
        ),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["max_abs"] <= MAX_ABS_GATE
        and metrics["mean_abs"] <= MEAN_ABS_GATE
        and metrics["relative_l2"] <= RELATIVE_L2_GATE
        and metrics["cosine"] >= COSINE_GATE
    )
    return metrics


def make_model(encoder: str, device: torch.device) -> torch.nn.Module:
    if USE_CUDA_GOLDEN:
        payload = torch.load(
            BASELINE_DIR / WEIGHT_FILES[encoder],
            map_location="cpu",
            weights_only=False,
        )
    else:
        torch.manual_seed(RANDOM_WEIGHT_SEEDS[encoder])
        payload = None

    model = PointTransformerV3Vanilla(
        in_channels=3,
        output_dim=OUTPUT_DIM,
        grid_size=GRID_SIZE,
        enable_flash=ENABLE_FLASH,
        shuffle_orders=SHUFFLE_ORDERS,
    )
    if payload is not None:
        model.load_state_dict(payload["model"], strict=True)
        del payload
    for module in model.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = SHUFFLE_ORDERS
        if hasattr(module, "traceable"):
            module.traceable = False
    model = model.to(device).eval()
    if DOWNSAMPLE_DEVICE == "cpu":
        for encoder_stage in model.enc:
            if "down" in encoder_stage._modules:
                encoder_stage.down.cpu()
    if FINAL_POOL_DEVICE == "cpu":
        model.projection.cpu()
    return model


def load_input(device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    if USE_CUDA_GOLDEN:
        reference_path = BASELINE_DIR / REFERENCE_FILE
        if not reference_path.is_file():
            raise FileNotFoundError(f"required reference is missing: {reference_path}")
        with np.load(reference_path) as reference:
            points = np.asarray(reference["points"], dtype=np.float32)
            offset = np.asarray(reference["offset"], dtype=np.int64)
            golden = {
                f"{encoder}_embedding": np.asarray(
                    reference[f"{encoder}_embedding"], dtype=np.float32
                )
                for encoder in ENCODERS
            }
    else:
        rng = np.random.default_rng(1234)
        points = rng.normal(0.0, 0.05, size=(POINT_COUNT, 3)).astype(np.float32)
        points -= points.mean(axis=0, keepdims=True)
        points *= np.float32(KAPPA)
        offset = np.asarray([POINT_COUNT], dtype=np.int64)
        golden = {}

    serialization_device = (
        torch.device("cpu") if SERIALIZATION_DEVICE == "cpu" else device
    )
    points_tensor = torch.from_numpy(points).to(serialization_device)
    data = {
        "coord": points_tensor,
        "feat": points_tensor,
        "offset": torch.from_numpy(offset).to(serialization_device),
        "grid_size": GRID_SIZE,
    }
    return data, golden


def run_direct(model: torch.nn.Module, data: dict, device: torch.device) -> np.ndarray:
    synchronize(device)
    with torch.inference_mode():
        if device.type == "cpu" or SERIALIZATION_DEVICE == "npu":
            output = model(data)
        else:
            point = serialize_point(model, data)
            point = move_point(point, device)
            point = model.embedding(point)
            point = run_encoder(model, point, device)
            if FINAL_POOL_DEVICE == "cpu":
                point = move_point(point, torch.device("cpu"))
            pooled = segment_csr_vanilla(
                point.feat,
                F.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            output = model.projection(pooled)
    synchronize(device)
    return output.detach().float().cpu().numpy()


def run_partitioned(
    model: torch.nn.Module,
    data: dict,
    device: torch.device,
    samples: dict[str, list[float]] | None = None,
) -> np.ndarray:
    aggregate_ms = defaultdict(float)

    def stage(name, operation, aggregates=()):
        synchronize(device)
        started = time.perf_counter()
        try:
            value = operation()
            synchronize(device)
        except Exception as exc:
            raise StageFailure(name, exc) from exc
        if samples is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            samples[name].append(elapsed_ms)
            for aggregate in aggregates:
                aggregate_ms[aggregate] += elapsed_ms
        return value

    total_started = time.perf_counter()
    point = stage(
        "1.serialization",
        lambda: serialize_point(model, data),
    )
    if SERIALIZATION_DEVICE == "cpu" and device.type != "cpu":
        point = stage("2.serialized_to_device", lambda: move_point(point, device))
    point = stage("3.embedding", lambda: model.embedding(point))
    for stage_index, encoder_stage in enumerate(model.enc):
        if "down" in encoder_stage._modules:
            if DOWNSAMPLE_DEVICE == "cpu":
                point = stage(
                    f"4.encoder_stage_{stage_index}.to_cpu",
                    lambda point=point: move_point(point, torch.device("cpu")),
                )
            point = stage(
                f"4.encoder_stage_{stage_index}.down",
                lambda encoder_stage=encoder_stage, point=point: encoder_stage.down(
                    point
                ),
            )
            if DOWNSAMPLE_DEVICE == "cpu":
                point = stage(
                    f"4.encoder_stage_{stage_index}.to_device",
                    lambda point=point: move_point(point, device),
                )
        for block_name, block in encoder_stage._modules.items():
            if block_name == "down":
                continue
            prefix = f"4.encoder_stage_{stage_index}.{block_name}"
            summary_prefix = f"4.encoder_stage_{stage_index}.summary"
            point = stage(
                f"{prefix}.cpe",
                lambda block=block, point=point: run_block_cpe(block, point),
                (f"{summary_prefix}.cpe", f"{summary_prefix}.blocks_total"),
            )
            point = stage(
                f"{prefix}.attention",
                lambda block=block, point=point: run_block_attention(block, point),
                (
                    f"{summary_prefix}.attention",
                    f"{summary_prefix}.blocks_total",
                ),
            )
            point = stage(
                f"{prefix}.ffn",
                lambda block=block, point=point: run_block_ffn(block, point),
                (f"{summary_prefix}.ffn", f"{summary_prefix}.blocks_total"),
            )
    if FINAL_POOL_DEVICE == "cpu":
        point = stage(
            "5.final_to_cpu",
            lambda: move_point(point, torch.device("cpu")),
        )
    pooled = stage(
        "6.global_mean_pooling",
        lambda: segment_csr_vanilla(
            point.feat,
            F.pad(point.offset, (1, 0)),
            reduce="mean",
        ),
    )
    embedding = stage("7.projection", lambda: model.projection(pooled))
    synchronize(device)
    if samples is not None:
        samples["0.total"].append((time.perf_counter() - total_started) * 1000.0)
        for name, elapsed_ms in aggregate_ms.items():
            samples[name].append(elapsed_ms)
    return embedding.detach().float().cpu().numpy()


def serialize_point(model: torch.nn.Module, data: dict) -> VanillaPoint:
    point = VanillaPoint(data)
    point.serialization(order=model.order, shuffle_orders=model.shuffle_orders)
    return point


def move_point(point: VanillaPoint, device: torch.device) -> VanillaPoint:
    for key, value in list(point.items()):
        if torch.is_tensor(value):
            point[key] = value.to(device)
    return point


def run_block_cpe(block, point: VanillaPoint) -> VanillaPoint:
    shortcut = point.feat
    cpe_out = block.cpe_conv(point.feat, point.grid_coord, point.batch)
    cpe_out = block.cpe_linear(cpe_out)
    cpe_out = block.cpe_norm(cpe_out)
    point.feat = shortcut + cpe_out
    return point


def run_block_attention(block, point: VanillaPoint) -> VanillaPoint:
    shortcut = point.feat
    if block.pre_norm:
        point.feat = block.norm1(point.feat)
    point = block.attn(point)
    point.feat = shortcut + block.drop_path(point.feat)
    return point


def run_block_ffn(block, point: VanillaPoint) -> VanillaPoint:
    shortcut = point.feat
    if block.pre_norm:
        point.feat = block.norm2(point.feat)
    point.feat = shortcut + block.drop_path(block.mlp(point.feat))
    return point


def run_stage_blocks(encoder_stage, point: VanillaPoint) -> VanillaPoint:
    for name, block in encoder_stage._modules.items():
        if name == "down":
            continue
        point = run_block_cpe(block, point)
        point = run_block_attention(block, point)
        point = run_block_ffn(block, point)
    return point


def run_encoder(model: torch.nn.Module, point: VanillaPoint, device: torch.device):
    for encoder_stage in model.enc:
        if "down" in encoder_stage._modules:
            if DOWNSAMPLE_DEVICE == "cpu":
                point = move_point(point, torch.device("cpu"))
            point = encoder_stage.down(point)
            if DOWNSAMPLE_DEVICE == "cpu":
                point = move_point(point, device)
        point = run_stage_blocks(encoder_stage, point)
    return point


def write_report(report: dict) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    report = {
        "contract": {
            "implementation": (
                "graspgenx.models.ptv3.ptv3_vanilla.PointTransformerV3Vanilla"
            ),
            "device": DEVICE,
            "serialization_device": SERIALIZATION_DEVICE,
            "downsample_device": DOWNSAMPLE_DEVICE,
            "final_pool_device": FINAL_POOL_DEVICE,
            "use_cuda_golden": USE_CUDA_GOLDEN,
            "point_count": POINT_COUNT,
            "encoders": list(ENCODERS),
            "precision": "fp32",
            "grid_size": GRID_SIZE,
            "enable_flash": ENABLE_FLASH,
            "shuffle_orders": SHUFFLE_ORDERS,
            "warmup_runs": WARMUP_RUNS,
            "profile_runs": PROFILE_RUNS,
        },
        "paths": {
            "repository": str(REPO_ROOT),
            "baseline_dir": str(BASELINE_DIR),
            "result": str(RESULT_PATH),
        },
        "environment": {
            "torch": torch.__version__,
            "torch_npu": (
                getattr(torch_npu, "__version__", "unknown")
                if torch_npu is not None
                else None
            ),
        },
        "models": {},
        "failures": [],
    }

    print("PTV3 synchronized stage profile", flush=True)
    print(
        f"device={DEVICE}, serialization={SERIALIZATION_DEVICE}, "
        f"points={POINT_COUNT}, golden={USE_CUDA_GOLDEN}, "
        f"warmup={WARMUP_RUNS}, runs={PROFILE_RUNS}",
        flush=True,
    )

    try:
        torch.set_num_threads(CPU_THREADS)
        device = torch.device(DEVICE)
        if device.type == "npu":
            if torch_npu is None:
                raise RuntimeError("torch_npu is required for DEVICE=npu:0")
            torch.npu.set_device(device)
            synchronize(device)
            report["environment"]["device_name"] = torch.npu.get_device_name(0)
        else:
            report["environment"]["device_name"] = "cpu"
        data, golden = load_input(device)

        for encoder in ENCODERS:
            print(f"Profiling {encoder} PTV3", flush=True)
            model = make_model(encoder, device)
            model_report = {}
            runtime_failed = False
            try:
                # Run explicit stages first so unsupported operations are
                # reported with a useful stage name.
                for _ in range(WARMUP_RUNS):
                    run_partitioned(model, data, device)

                samples = defaultdict(list)
                partitioned_output = None
                for run_index in range(PROFILE_RUNS):
                    partitioned_output = run_partitioned(
                        model, data, device, samples
                    )
                    print(
                        f"  staged run {run_index + 1}/{PROFILE_RUNS}", flush=True
                    )
                assert partitioned_output is not None

                direct_output = run_direct(model, data, device)
                model_report["direct_output"] = {
                    "shape": list(direct_output.shape),
                    "finite": bool(np.isfinite(direct_output).all()),
                }

                model_report["stage_timings"] = {
                    name: summarize_ms(values)
                    for name, values in sorted(samples.items())
                }
                model_report["partitioned_vs_direct"] = compare(
                    direct_output, partitioned_output
                )
                if USE_CUDA_GOLDEN:
                    reference = golden[f"{encoder}_embedding"]
                    model_report["direct_vs_cuda"] = compare(
                        reference, direct_output
                    )
                    model_report["partitioned_vs_cuda"] = compare(
                        reference, partitioned_output
                    )
                model_report["passed"] = bool(
                    model_report["partitioned_vs_direct"]["passed"]
                    and (
                        not USE_CUDA_GOLDEN
                        or (
                            model_report["direct_vs_cuda"]["passed"]
                            and model_report["partitioned_vs_cuda"]["passed"]
                        )
                    )
                )
                status = "PASS" if model_report["passed"] else "FAIL"
                print(f"[{status}] {encoder}", flush=True)
                for stage_name, timing in model_report["stage_timings"].items():
                    print(
                        f"  {stage_name:42s} mean={timing['mean_ms']:.3f} ms "
                        f"median={timing['median_ms']:.3f} ms "
                        f"p95={timing['p95_ms']:.3f} ms "
                        f"p99={timing['p99_ms']:.3f} ms",
                        flush=True,
                    )
            except Exception as exc:
                failure = {
                    "encoder": encoder,
                    "stage": getattr(exc, "stage", "unknown"),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                report["failures"].append(failure)
                model_report["passed"] = False
                runtime_failed = True
                print(
                    f"[FAIL] {encoder} stage={failure['stage']} error={exc!r}",
                    flush=True,
                )
            report["models"][encoder] = model_report
            write_report(report)
            del model
            if device.type == "npu":
                try:
                    torch.npu.empty_cache()
                except Exception as exc:
                    report.setdefault("cleanup_warnings", []).append(repr(exc))
            if runtime_failed:
                break
    except Exception as exc:
        report["failures"].append(
            {
                "stage": "setup",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[FAIL] setup: {exc!r}", flush=True)

    passed = bool(
        len(report["models"]) == len(ENCODERS)
        and not report["failures"]
        and all(model["passed"] for model in report["models"].values())
    )
    report["summary"] = "PASS" if passed else "FAIL"
    write_report(report)
    print(f"PROFILE VALIDATION: {report['summary']}", flush=True)
    print(f"Report: {RESULT_PATH}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
