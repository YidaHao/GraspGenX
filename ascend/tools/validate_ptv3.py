#!/usr/bin/env python3
"""Compare the repository's PTV3 vanilla implementation with CUDA golden data.

Configuration is intentionally kept in the block immediately below. The script
takes no command-line arguments so every run uses the same validation contract.

Prerequisites
-------------
Run from the GraspGenX repository after loading CANN::

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python3 ascend/tools/validate_ptv3.py

The following files must exist under ``BASELINE_DIR``::

    generator_ptv3_state.pth
    discriminator_ptv3_state.pth
    reference_n64.npz
    reference_n2048.npz
    reference_n3500.npz

Each reference contains already sampled, mean-centered, kappa-scaled points.
Do not preprocess ``points`` again before passing them to PTV3.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:  # CPU baseline remains usable without torch-npu
    torch_npu = None


# ---------------------------------------------------------------------------
# Configuration and required paths. Edit values here, not at the call site.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "ascend/baselines/ptv3-cuda-fp32-eager"
RESULT_PATH = REPO_ROOT / "ascend/results/ptv3_vanilla_310p_validation.json"

# The initial baseline runs the unmodified vanilla model on the 310P host CPU.
# Change this to "npu:0" when the NPU implementation is ready.
DEVICE = "cpu"
SERIALIZATION_DEVICE = "cpu"
DOWNSAMPLE_DEVICE = "cpu"
FINAL_POOL_DEVICE = "cpu"
POINT_COUNTS = (64, 2048, 3500)
ENCODERS = ("generator", "discriminator")
WARMUP_RUNS = 3
MEASURED_RUNS = 20
REPEAT_CHECKS = 3
CPU_THREADS = 16

GRID_SIZE = 0.01
OUTPUT_DIM = 512
ENABLE_FLASH = False
SHUFFLE_ORDERS = False

# Initial cross-backend gates include margin over the measured unmodified
# vanilla CPU-vs-CUDA baseline. Keep the raw metrics in the report when
# tightening these gates for an NPU implementation.
MAX_ABS_GATE = 1.5e-2
MEAN_ABS_GATE = 3e-3
RELATIVE_L2_GATE = 5e-3
COSINE_GATE = 0.99999
REPEAT_MAX_ABS_GATE = 1e-5
PER_ENCODER_TARGET_MS = 150.0

WEIGHT_FILES = {
    "generator": "generator_ptv3_state.pth",
    "discriminator": "discriminator_ptv3_state.pth",
}
REFERENCE_FILES = {
    point_count: f"reference_n{point_count}.npz" for point_count in POINT_COUNTS
}
EXPECTED_SHA256 = {
    "generator_ptv3_state.pth": "f017bc75315737a5eb088badb18dfebdd719483f669b694e113e47f45783b4e6",
    "discriminator_ptv3_state.pth": "0ed58567048bfde97b716d04d85dc58c487b27ad103a5fc8714ca86a7d756dd1",
    "reference_n64.npz": "e033f95d973202ed4c8c54d4d6c30078389f33a2e1d581d648425eea3a139675",
    "reference_n2048.npz": "e0d901235e755404522ad86bb9a651b7d0c93bc9dc6851b0178686be194d14ff",
    "reference_n3500.npz": "9df08f2e7b06631dfdd721356691b178409c4d5e11bd553e65fe68eaac38adca",
}


def install_minimal_import_shims() -> None:
    """Provide eval-equivalent fallbacks for two absent optional packages."""
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

# This direct import is deliberate: editing ptv3_vanilla.py changes what this
# validator exercises without an implementation registry or adapter layer.
from graspgenx.models.ptv3.ptv3_vanilla import (
    PointTransformerV3Vanilla,
    VanillaPoint,
    segment_csr_vanilla,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize() -> None:
    if DEVICE.startswith("npu"):
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
    reference_flat = reference64.reshape(-1)
    candidate_flat = candidate64.reshape(-1)
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat)
    metrics = {
        "shape_match": True,
        "finite": bool(np.isfinite(candidate).all()),
        "max_abs": float(np.abs(difference).max()),
        "mean_abs": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference_flat), 1e-12)
        ),
        "cosine": (
            float(np.dot(reference_flat, candidate_flat) / denominator)
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
    payload = torch.load(
        BASELINE_DIR / WEIGHT_FILES[encoder], map_location="cpu", weights_only=False
    )
    model = PointTransformerV3Vanilla(
        in_channels=3,
        output_dim=OUTPUT_DIM,
        grid_size=GRID_SIZE,
        enable_flash=ENABLE_FLASH,
        shuffle_orders=SHUFFLE_ORDERS,
    )
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict weight load failed: {incompatible}")
    for module in model.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = SHUFFLE_ORDERS
        if hasattr(module, "traceable"):
            module.traceable = False
    del payload
    model = model.to(device).eval()
    if DOWNSAMPLE_DEVICE == "cpu":
        for encoder_stage in model.enc:
            if "down" in encoder_stage._modules:
                encoder_stage.down.cpu()
    if FINAL_POOL_DEVICE == "cpu":
        model.projection.cpu()
    return model


def make_input(reference: np.lib.npyio.NpzFile) -> dict:
    points = torch.from_numpy(np.asarray(reference["points"], dtype=np.float32))
    offset = torch.from_numpy(np.asarray(reference["offset"], dtype=np.int64))
    return {
        "coord": points,
        "feat": points,
        "offset": offset,
        "grid_size": float(reference["grid_size"]),
    }


def move_point(point: VanillaPoint, device: torch.device) -> VanillaPoint:
    for key, value in list(point.items()):
        if torch.is_tensor(value):
            point[key] = value.to(device)
    return point


def run_encoder(model: torch.nn.Module, point: VanillaPoint, device: torch.device):
    for encoder_stage in model.enc:
        if "down" in encoder_stage._modules:
            if DOWNSAMPLE_DEVICE == "cpu":
                point = move_point(point, torch.device("cpu"))
            point = encoder_stage.down(point)
            if DOWNSAMPLE_DEVICE == "cpu":
                point = move_point(point, device)
        for name, module in encoder_stage._modules.items():
            if name != "down":
                point = module(point)
    return point


def forward_ptv3(model: torch.nn.Module, data: dict, device: torch.device):
    if device.type == "cpu":
        return model(data)
    if SERIALIZATION_DEVICE == "npu":
        npu_data = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        return model(npu_data)
    if SERIALIZATION_DEVICE != "cpu":
        raise ValueError(
            f"SERIALIZATION_DEVICE must be 'cpu' or 'npu', got {SERIALIZATION_DEVICE}"
        )

    point = VanillaPoint(data)
    point.serialization(order=model.order, shuffle_orders=model.shuffle_orders)
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
    return model.projection(pooled)


def run_case(
    model: torch.nn.Module,
    encoder: str,
    point_count: int,
    device: torch.device,
) -> dict:
    with np.load(BASELINE_DIR / REFERENCE_FILES[point_count]) as reference:
        model_input = make_input(reference)
        golden = np.asarray(reference[f"{encoder}_embedding"], dtype=np.float32)

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            forward_ptv3(model, model_input, device)
        synchronize()

        timings = []
        output = None
        for _ in range(MEASURED_RUNS):
            synchronize()
            started = time.perf_counter()
            output = forward_ptv3(model, model_input, device)
            synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)

        assert output is not None
        candidate = output.detach().float().cpu().numpy()
        repeat_max_abs = []
        for _ in range(REPEAT_CHECKS):
            repeated = forward_ptv3(model, model_input, device)
            synchronize()
            repeated_np = repeated.detach().float().cpu().numpy()
            repeat_max_abs.append(float(np.abs(repeated_np - candidate).max()))

    accuracy = compare(golden, candidate)
    repeatability = {
        "checks": REPEAT_CHECKS,
        "max_abs": max(repeat_max_abs),
        "passed": max(repeat_max_abs) <= REPEAT_MAX_ABS_GATE,
    }
    latency = summarize_ms(timings)
    return {
        "encoder": encoder,
        "point_count": point_count,
        "accuracy": accuracy,
        "repeatability": repeatability,
        "latency": latency,
        "performance_target_ms": PER_ENCODER_TARGET_MS,
        "performance_target_passed": latency["median_ms"]
        <= PER_ENCODER_TARGET_MS,
        "passed": accuracy["passed"] and repeatability["passed"],
    }


def write_report(report: dict) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")


def print_case(result: dict) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    accuracy = result["accuracy"]
    latency = result["latency"]
    repeatability = result["repeatability"]
    print(
        f"[{status}] {result['encoder']:13s} n={result['point_count']:4d} "
        f"max_abs={accuracy['max_abs']:.3e} "
        f"mean_abs={accuracy['mean_abs']:.3e} "
        f"rel_l2={accuracy['relative_l2']:.3e} "
        f"cosine={accuracy['cosine']:.8f} "
        f"repeat={repeatability['max_abs']:.3e} "
        f"mean={latency['mean_ms']:.3f} ms "
        f"median={latency['median_ms']:.3f} ms "
        f"p95={latency['p95_ms']:.3f} ms "
        f"p99={latency['p99_ms']:.3f} ms",
        flush=True,
    )


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
            "precision": "fp32",
            "point_counts": list(POINT_COUNTS),
            "encoders": list(ENCODERS),
            "grid_size": GRID_SIZE,
            "enable_flash": ENABLE_FLASH,
            "shuffle_orders": SHUFFLE_ORDERS,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "repeat_checks": REPEAT_CHECKS,
            "gates": {
                "max_abs": MAX_ABS_GATE,
                "mean_abs": MEAN_ABS_GATE,
                "relative_l2": RELATIVE_L2_GATE,
                "cosine": COSINE_GATE,
                "repeat_max_abs": REPEAT_MAX_ABS_GATE,
            },
        },
        "paths": {
            "repository": str(REPO_ROOT),
            "baseline_dir": str(BASELINE_DIR),
            "result": str(RESULT_PATH),
        },
        "baseline_sha256": {},
        "environment": {
            "torch": torch.__version__,
            "torch_npu": (
                getattr(torch_npu, "__version__", "unknown")
                if torch_npu is not None
                else None
            ),
        },
        "results": [],
        "failures": [],
    }

    print("PTV3 CUDA-golden comparison", flush=True)
    print(f"baseline: {BASELINE_DIR}", flush=True)
    print(
        f"device: {DEVICE}, serialization={SERIALIZATION_DEVICE}, fp32, "
        f"warmup={WARMUP_RUNS}, runs={MEASURED_RUNS}",
        flush=True,
    )

    try:
        for filename, expected in EXPECTED_SHA256.items():
            path = BASELINE_DIR / filename
            if not path.is_file():
                raise FileNotFoundError(f"required baseline file is missing: {path}")
            actual = sha256_file(path)
            report["baseline_sha256"][filename] = actual
            if actual != expected:
                raise RuntimeError(
                    f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}"
                )

        torch.set_num_threads(CPU_THREADS)
        torch.manual_seed(0)
        device = torch.device(DEVICE)
        if device.type == "npu":
            if torch_npu is None:
                raise RuntimeError("torch_npu is required for DEVICE=npu:0")
            torch.npu.set_device(device)
        synchronize()
        report["environment"]["device_name"] = (
            torch.npu.get_device_name(0) if device.type == "npu" else "310P host CPU"
        )
        print(f"Runtime: {report['environment']['device_name']}", flush=True)

        for encoder in ENCODERS:
            model = make_model(encoder, device)
            runtime_failed = False
            for point_count in POINT_COUNTS:
                try:
                    result = run_case(model, encoder, point_count, device)
                    report["results"].append(result)
                    print_case(result)
                except Exception as exc:  # preserve partial evidence
                    failure = {
                        "encoder": encoder,
                        "point_count": point_count,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    report["failures"].append(failure)
                    runtime_failed = True
                    print(
                        f"[FAIL] {encoder:13s} n={point_count:4d} "
                        f"runtime_error={exc!r}",
                        flush=True,
                    )
                    break
                finally:
                    write_report(report)
            del model
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

    expected_case_count = len(ENCODERS) * len(POINT_COUNTS)
    correctness_passed = bool(
        len(report["results"]) == expected_case_count
        and not report["failures"]
        and all(result["passed"] for result in report["results"])
    )
    performance_passed = bool(
        len(report["results"]) == expected_case_count
        and all(
            result["performance_target_passed"] for result in report["results"]
        )
    )
    report["summary"] = {
        "correctness": "PASS" if correctness_passed else "FAIL",
        "performance_target": "PASS" if performance_passed else "FAIL",
    }
    write_report(report)
    print(f"ACCURACY/STABILITY: {report['summary']['correctness']}", flush=True)
    print(f"PERFORMANCE TARGET: {report['summary']['performance_target']}", flush=True)
    print(f"Report: {RESULT_PATH}", flush=True)
    return 0 if correctness_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
