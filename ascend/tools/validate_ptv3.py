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

Select the implementation by commenting/uncommenting the imports below.
Ascend keeps attention/residual/norm/FFN on NPU FP16; vanilla uses CPU FP32.
No device or precision switch is needed. Timings include every feature/index copy.
Result names follow the imported implementation, preserving the CUDA golden.
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

try:
    import torch_npu  # noqa: F401
except ImportError:  # CPU baseline remains usable without torch-npu
    torch_npu = None


# ---------------------------------------------------------------------------
# Configuration and required paths. Edit values here, not at the call site.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "ascend/baselines/ptv3-cuda-fp32-eager"
RESULT_DIR = REPO_ROOT / "ascend/results"
EXPERIMENT_TAG = "grid_encode_four_orders"
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

# Only cosine gates numerical accuracy. Other error/stability metrics are
# diagnostic; wrong shapes, non-finite outputs and runtime errors remain invalid.
COSINE_GATE = 0.9999
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

# Select exactly one import. Nothing else needs changing for a CPU control.
from graspgenx.models.ptv3.ptv3_ascend import PointTransformerV3Ascend as PointTransformerV3
# from graspgenx.models.ptv3.ptv3_vanilla import PointTransformerV3Vanilla as PointTransformerV3

IMPLEMENTATION = PointTransformerV3.__module__
RESULT_PATH = RESULT_DIR / f"{IMPLEMENTATION.rsplit('.', 1)[-1]}_{EXPERIMENT_TAG}_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize() -> None:
    if torch_npu is not None and torch.npu.is_initialized():
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
            if denominator > 0
            else 0.0
        ),
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["cosine"] >= COSINE_GATE
    )
    return metrics


def make_model(encoder: str) -> torch.nn.Module:
    payload = torch.load(
        BASELINE_DIR / WEIGHT_FILES[encoder], map_location="cpu", weights_only=False
    )
    model = PointTransformerV3(
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
    return model.eval()


def make_input(reference: np.lib.npyio.NpzFile) -> dict:
    points = torch.from_numpy(np.asarray(reference["points"], dtype=np.float32))
    offset = torch.from_numpy(np.asarray(reference["offset"], dtype=np.int64))
    return {
        "coord": points,
        "feat": points,
        "offset": offset,
        "grid_size": float(reference["grid_size"]),
    }


def run_case(
    model: torch.nn.Module,
    encoder: str,
    point_count: int,
) -> dict:
    with np.load(BASELINE_DIR / REFERENCE_FILES[point_count]) as reference:
        model_input = make_input(reference)
        golden = np.asarray(reference[f"{encoder}_embedding"], dtype=np.float32)

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(model_input)
        synchronize()

        timings = []
        output = None
        for _ in range(MEASURED_RUNS):
            synchronize()
            started = time.perf_counter()
            output = model(model_input)
            synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)

        assert output is not None
        candidate = output.detach().float().cpu().numpy()
        repeat_max_abs = []
        repeat_finite = True
        repeat_shape_match = True
        for _ in range(REPEAT_CHECKS):
            repeated = model(model_input)
            synchronize()
            repeated_np = repeated.detach().float().cpu().numpy()
            repeat_finite = repeat_finite and bool(np.isfinite(repeated_np).all())
            repeat_shape_match &= repeated_np.shape == golden.shape
            if repeated_np.shape == candidate.shape:
                repeat_max_abs.append(float(np.abs(repeated_np - candidate).max()))

    accuracy = compare(golden, candidate)
    repeatability = {
        "checks": REPEAT_CHECKS,
        "max_abs": max(repeat_max_abs, default=None),
        "finite": repeat_finite,
        "shape_match": repeat_shape_match,
        "report_only": True,
    }
    latency = summarize_ms(timings)
    output_path = RESULT_PATH.with_name(
        f"{RESULT_PATH.stem}_{encoder}_n{point_count}.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, embedding=candidate, points=model_input["coord"].numpy())
    return {
        "encoder": encoder,
        "point_count": point_count,
        "accuracy": accuracy,
        "repeatability": repeatability,
        "latency": latency,
        "output_path": str(output_path),
        "latency_samples_ms": timings,
        "performance_target_ms": PER_ENCODER_TARGET_MS,
        "performance_target_passed": latency["median_ms"]
        <= PER_ENCODER_TARGET_MS,
        "passed": accuracy["passed"] and repeat_finite and repeat_shape_match,
    }


def write_report(report: dict) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")


def print_case(result: dict) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    accuracy = result["accuracy"]
    latency = result["latency"]
    repeatability = result["repeatability"]
    repeat_text = "n/a" if repeatability["max_abs"] is None else f"{repeatability['max_abs']:.3e}"
    print(
        f"[{status}] {result['encoder']:13s} n={result['point_count']:4d} "
        f"max_abs={accuracy['max_abs']:.3e} "
        f"mean_abs={accuracy['mean_abs']:.3e} "
        f"rel_l2={accuracy['relative_l2']:.3e} "
        f"cosine={accuracy['cosine']:.8f} "
        f"repeat={repeat_text} "
        f"mean={latency['mean_ms']:.3f} ms "
        f"median={latency['median_ms']:.3f} ms "
        f"p95={latency['p95_ms']:.3f} ms "
        f"p99={latency['p99_ms']:.3f} ms",
        flush=True,
    )


def print_encoder_totals(results: list[dict]) -> list[dict]:
    by_case = {(r["encoder"], r["point_count"]): r["latency"] for r in results}
    totals = []
    print("\nGenerator + discriminator latency (ms; sums of separate measurements)", flush=True)
    print(
        f"{'points':>6} {'gen mean':>10} {'dis mean':>10} {'sum mean':>10} "
        f"{'gen median':>11} {'dis median':>11} {'sum median':>11}",
        flush=True,
    )
    for point_count in POINT_COUNTS:
        generator = by_case.get(("generator", point_count))
        discriminator = by_case.get(("discriminator", point_count))
        if generator is None or discriminator is None:
            print(f"{point_count:6d} N/A (missing encoder result)", flush=True)
            continue
        mean_sum = generator["mean_ms"] + discriminator["mean_ms"]
        median_sum = generator["median_ms"] + discriminator["median_ms"]
        totals.append({
            "point_count": point_count,
            "mean_ms_sum": mean_sum,
            "median_ms_sum": median_sum,
        })
        print(
            f"{point_count:6d} {generator['mean_ms']:10.3f} "
            f"{discriminator['mean_ms']:10.3f} {mean_sum:10.3f} "
            f"{generator['median_ms']:11.3f} "
            f"{discriminator['median_ms']:11.3f} {median_sum:11.3f}",
            flush=True,
        )
    return totals


def main() -> int:
    report = {
        "contract": {
            "implementation": f"{IMPLEMENTATION}.{PointTransformerV3.__name__}",
            "point_counts": list(POINT_COUNTS),
            "encoders": list(ENCODERS),
            "grid_size": GRID_SIZE,
            "enable_flash": ENABLE_FLASH,
            "shuffle_orders": SHUFFLE_ORDERS,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
            "repeat_checks": REPEAT_CHECKS,
            "gates": {
                "cosine": COSINE_GATE,
            },
            "required_output": "matching shape and finite values",
            "report_only_metrics": [
                "max_abs", "mean_abs", "rmse", "relative_l2", "repeat_max_abs",
            ],
        },
        "paths": {
            "repository": str(REPO_ROOT),
            "baseline_dir": str(BASELINE_DIR),
            "result": str(RESULT_PATH),
            "implementation_sha256": sha256_file(
                Path(sys.modules[IMPLEMENTATION].__file__)
            ),
            "vanilla_sha256": sha256_file(
                REPO_ROOT / "graspgenx/models/ptv3/ptv3_vanilla.py"
            ),
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
    print(f"implementation: {IMPLEMENTATION}", flush=True)
    print(f"Accuracy gate: cosine >= {COSINE_GATE}; other errors are report-only", flush=True)
    print(
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
        for encoder in ENCODERS:
            model = make_model(encoder)
            report["contract"]["parameter_layout"] = sorted({
                f"{p.device}/{p.dtype}" for p in model.parameters()
            })
            report["contract"]["execution"] = getattr(model, "execution_config", {})
            runtime_failed = False
            for point_count in POINT_COUNTS:
                try:
                    result = run_case(model, encoder, point_count)
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
            if torch_npu is not None and torch.npu.is_initialized():
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
    print(f"ACCURACY: {report['summary']['correctness']}", flush=True)
    print(f"PERFORMANCE TARGET (report only): {report['summary']['performance_target']}", flush=True)
    report["latency_totals"] = print_encoder_totals(report["results"])
    write_report(report)
    print(f"Report: {RESULT_PATH}", flush=True)
    return 0 if correctness_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
