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
kappa. ``USE_CUDA_GOLDEN=False`` selects synthetic inputs and random weights;
device placement still follows the selected implementation import.

Select the implementation by commenting/uncommenting the imports below.
Ascend defaults to NPU FP16 attention; vanilla uses CPU FP32. Result names
follow the import. Attention timing includes outer LayerNorm/residual and
CPU/NPU transfers; profile timings include per-stage synchronization overhead.
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
RESULT_DIR = REPO_ROOT / "ascend/results"

USE_CUDA_GOLDEN = True
ENCODERS = ("generator", "discriminator")
WARMUP_RUNS = 1
PROFILE_RUNS = 20
CPU_THREADS = 16
RANKING_TABLE_LIMIT = 100

GRID_SIZE = 0.01
KAPPA = 3.27
OUTPUT_DIM = 512
ENABLE_FLASH = False
SHUFFLE_ORDERS = False

# Only cosine gates numerical accuracy; other errors remain in the report.
# Wrong shapes, non-finite outputs and runtime errors are still invalid.
COSINE_GATE = 0.9999

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

# Select exactly one import block. Nothing else needs changing for a CPU control.
from graspgenx.models.ptv3.ptv3_ascend import (
    PointTransformerV3Ascend as PointTransformerV3,
    VanillaPoint,
    segment_csr_vanilla,
)
# from graspgenx.models.ptv3.ptv3_vanilla import (
#     PointTransformerV3Vanilla as PointTransformerV3,
#     VanillaPoint,
#     segment_csr_vanilla,
# )

IMPLEMENTATION = PointTransformerV3.__module__
RESULT_PATH = RESULT_DIR / (
    f"{IMPLEMENTATION.rsplit('.', 1)[-1]}_profile_n{POINT_COUNT}.json"
)


class StageFailure(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"stage {stage!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


def synchronize() -> None:
    if torch_npu is not None and torch.npu.is_initialized():
        torch.npu.synchronize()


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }
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


OPTIMIZATION_DIRECTIONS = {
    "attention": "优化 QKV/softmax/gather，避免不必要的 FP32 upcast",
    "cpe": "缓存邻接索引，融合 hash-conv/linear/norm",
    "ffn": "融合 Linear-GELU-Linear；当前优先级较低",
    "serialization": "缓存四路 code/order/inverse，避免重复排序",
    "downsample": "缓存 unique/cluster/argsort 结果，减少动态索引重建",
    "embedding": "优化 stem 的 hash lookup 与加权聚合",
    "transfer": "合并 CPU/NPU 往返，静态索引一次性传输",
    "pooling": "使用固定 batch reduction；当前优先级较低",
    "projection": "不是当前热点，暂不优先优化",
    "other": "保留到下一轮 stage 诊断",
}


def stage_category(stage_name: str) -> str:
    if stage_name == "1.serialization":
        return "serialization"
    if stage_name == "3.embedding":
        return "embedding"
    if stage_name.endswith(".to_cpu") or stage_name.endswith(".to_device"):
        return "transfer"
    if stage_name.endswith(".down"):
        return "downsample"
    if stage_name == "6.global_mean_pooling":
        return "pooling"
    if stage_name == "7.projection":
        return "projection"
    if stage_name.endswith(".cpe"):
        return "cpe"
    if stage_name.endswith(".attention"):
        return "attention"
    if stage_name.endswith(".ffn"):
        return "ffn"
    return "other"


def is_leaf_stage(stage_name: str) -> bool:
    return stage_name != "0.total" and ".summary." not in stage_name


def add_sample_lists(left: list[float], right: list[float]) -> list[float]:
    if not left:
        return list(right)
    if not right:
        return list(left)
    if len(left) != len(right):
        raise ValueError(f"cannot combine samples with lengths {len(left)} and {len(right)}")
    return [a + b for a, b in zip(left, right)]


def build_optimization_ranking(
    timing_samples: dict[str, dict[str, list[float]]],
) -> dict[str, list[dict]]:
    """Build exact rankings from the per-run samples, not summed percentiles."""
    encoders = list(timing_samples)
    stage_names = sorted(
        {
            name
            for encoder_samples in timing_samples.values()
            for name in encoder_samples
            if is_leaf_stage(name)
        }
    )
    stage_rows = []
    for stage_name in stage_names:
        per_encoder = {
            encoder: summarize_ms(timing_samples[encoder].get(stage_name, []))
            for encoder in encoders
        }
        combined_samples = add_sample_lists(
            timing_samples[encoders[0]].get(stage_name, []),
            timing_samples[encoders[1]].get(stage_name, []),
        )
        combined = summarize_ms(combined_samples)
        stage_rows.append(
            {
                "stage": stage_name,
                "category": stage_category(stage_name),
                "generator": per_encoder.get("generator"),
                "discriminator": per_encoder.get("discriminator"),
                "combined": combined,
            }
        )
    stage_rows.sort(key=lambda row: row["combined"]["median_ms"], reverse=True)
    total_median = sum(row["combined"]["median_ms"] for row in stage_rows)
    for rank, row in enumerate(stage_rows, start=1):
        row["rank"] = rank
        row["median_share_percent"] = (
            100.0 * row["combined"]["median_ms"] / total_median
            if total_median
            else 0.0
        )

    category_samples: dict[str, dict[str, list[list[float]]]] = {
        encoder: defaultdict(list) for encoder in encoders
    }
    for encoder in encoders:
        for stage_name, samples in timing_samples[encoder].items():
            if is_leaf_stage(stage_name):
                category_samples[encoder][stage_category(stage_name)].append(samples)

    category_rows = []
    for category in sorted(
        {
            category
            for encoder_categories in category_samples.values()
            for category in encoder_categories
        }
    ):
        per_encoder = {}
        for encoder in encoders:
            category_runs = category_samples[encoder].get(category, [])
            if category_runs:
                per_encoder[encoder] = summarize_ms(
                    [
                        sum(run[index] for run in category_runs)
                        for index in range(PROFILE_RUNS)
                    ]
                )
            else:
                per_encoder[encoder] = summarize_ms([])
        combined_runs = add_sample_lists(
            [
                sum(
                    run[index]
                    for run in category_samples[encoders[0]].get(category, [])
                )
                for index in range(PROFILE_RUNS)
            ],
            [
                sum(
                    run[index]
                    for run in category_samples[encoders[1]].get(category, [])
                )
                for index in range(PROFILE_RUNS)
            ],
        )
        category_rows.append(
            {
                "category": category,
                "generator": per_encoder["generator"],
                "discriminator": per_encoder["discriminator"],
                "combined": summarize_ms(combined_runs),
                "direction": OPTIMIZATION_DIRECTIONS[category],
            }
        )
    category_rows.sort(key=lambda row: row["combined"]["median_ms"], reverse=True)
    category_total = sum(row["combined"]["median_ms"] for row in category_rows)
    for rank, row in enumerate(category_rows, start=1):
        row["rank"] = rank
        row["median_share_percent"] = (
            100.0 * row["combined"]["median_ms"] / category_total
            if category_total
            else 0.0
        )
    return {
        "stage_rows": stage_rows,
        "category_rows": category_rows,
        "sort_key": "combined.median_ms descending",
        "stage_table_limit": RANKING_TABLE_LIMIT,
        "p99_note": "combined p99 is calculated from per-run generator+discriminator sums",
    }


def print_optimization_tables(ranking: dict[str, list[dict]]) -> None:
    print("\nOptimization ranking: leaf stages by combined median (descending)", flush=True)
    print(
        f"{'#':>3} {'stage':<42} {'category':<13} "
        f"{'gen med':>10} {'dis med':>10} {'total med':>11} "
        f"{'total mean':>11} {'total p99':>11} {'share':>7}",
        flush=True,
    )
    for row in ranking["stage_rows"][:RANKING_TABLE_LIMIT]:
        combined = row["combined"]
        generator = row["generator"]
        discriminator = row["discriminator"]
        print(
            f"{row['rank']:3d} {row['stage']:<42.42s} {row['category']:<13} "
            f"{generator['median_ms']:10.3f} {discriminator['median_ms']:10.3f} "
            f"{combined['median_ms']:11.3f} {combined['mean_ms']:11.3f} "
            f"{combined['p99_ms']:11.3f} {row['median_share_percent']:6.1f}%",
            flush=True,
        )
    if len(ranking["stage_rows"]) > RANKING_TABLE_LIMIT:
        print(
            f"... {len(ranking['stage_rows']) - RANKING_TABLE_LIMIT} more leaf stages "
            "are preserved in the JSON report.",
            flush=True,
        )

    print("\nOptimization direction ranking: component categories", flush=True)
    print(
        f"{'#':>3} {'category':<13} {'gen med':>10} {'dis med':>10} "
        f"{'total med':>11} {'total mean':>11} {'total p99':>11} {'share':>7}  direction",
        flush=True,
    )
    for row in ranking["category_rows"]:
        combined = row["combined"]
        print(
            f"{row['rank']:3d} {row['category']:<13} "
            f"{row['generator']['median_ms']:10.3f} "
            f"{row['discriminator']['median_ms']:10.3f} "
            f"{combined['median_ms']:11.3f} {combined['mean_ms']:11.3f} "
            f"{combined['p99_ms']:11.3f} {row['median_share_percent']:6.1f}%  "
            f"{row['direction']}",
            flush=True,
        )


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
    if USE_CUDA_GOLDEN:
        payload = torch.load(
            BASELINE_DIR / WEIGHT_FILES[encoder],
            map_location="cpu",
            weights_only=False,
        )
    else:
        torch.manual_seed(RANDOM_WEIGHT_SEEDS[encoder])
        payload = None

    model = PointTransformerV3(
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
    return model.eval()


def load_input() -> tuple[dict, dict[str, np.ndarray]]:
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

    points_tensor = torch.from_numpy(points)
    data = {
        "coord": points_tensor,
        "feat": points_tensor,
        "offset": torch.from_numpy(offset),
        "grid_size": GRID_SIZE,
    }
    return data, golden


def run_direct(model: torch.nn.Module, data: dict) -> np.ndarray:
    synchronize()
    with torch.inference_mode():
        output = model(data)
    synchronize()
    return output.detach().float().cpu().numpy()


@torch.inference_mode()
def run_partitioned(
    model: torch.nn.Module,
    data: dict,
    samples: dict[str, list[float]] | None = None,
) -> np.ndarray:
    aggregate_ms = defaultdict(float)

    def stage(name, operation, aggregates=()):
        synchronize()
        started = time.perf_counter()
        try:
            value = operation()
            synchronize()
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
    point = stage("3.embedding", lambda: model.embedding(point))
    for stage_index, encoder_stage in enumerate(model.enc):
        if "down" in encoder_stage._modules:
            point = stage(
                f"4.encoder_stage_{stage_index}.down",
                lambda encoder_stage=encoder_stage, point=point: encoder_stage.down(
                    point
                ),
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
    pooled = stage(
        "6.global_mean_pooling",
        lambda: segment_csr_vanilla(
            point.feat,
            F.pad(point.offset, (1, 0)),
            reduce="mean",
        ),
    )
    embedding = stage("7.projection", lambda: model.projection(pooled))
    synchronize()
    if samples is not None:
        samples["0.total"].append((time.perf_counter() - total_started) * 1000.0)
        for name, elapsed_ms in aggregate_ms.items():
            samples[name].append(elapsed_ms)
    return embedding.detach().float().cpu().numpy()


def serialize_point(model: torch.nn.Module, data: dict) -> VanillaPoint:
    point = VanillaPoint(data)
    point.serialization(order=model.order, shuffle_orders=model.shuffle_orders)
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


def write_report(report: dict) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    report = {
        "contract": {
            "implementation": f"{IMPLEMENTATION}.{PointTransformerV3.__name__}",
            "use_cuda_golden": USE_CUDA_GOLDEN,
            "point_count": POINT_COUNT,
            "encoders": list(ENCODERS),
            "grid_size": GRID_SIZE,
            "enable_flash": ENABLE_FLASH,
            "shuffle_orders": SHUFFLE_ORDERS,
            "warmup_runs": WARMUP_RUNS,
            "profile_runs": PROFILE_RUNS,
            "gates": {"cosine": COSINE_GATE},
            "required_output": "matching shape and finite values",
            "report_only_metrics": ["max_abs", "mean_abs", "rmse", "relative_l2"],
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
    timing_samples: dict[str, dict[str, list[float]]] = {}

    print("PTV3 synchronized stage profile", flush=True)
    print(f"Accuracy gate: cosine >= {COSINE_GATE}; other errors are report-only", flush=True)
    print(
        f"implementation={IMPLEMENTATION}, "
        f"points={POINT_COUNT}, golden={USE_CUDA_GOLDEN}, "
        f"warmup={WARMUP_RUNS}, runs={PROFILE_RUNS}",
        flush=True,
    )

    try:
        torch.set_num_threads(CPU_THREADS)
        data, golden = load_input()

        for encoder in ENCODERS:
            print(f"Profiling {encoder} PTV3", flush=True)
            model = make_model(encoder)
            report["contract"]["parameter_layout"] = sorted({
                f"{p.device}/{p.dtype}" for p in model.parameters()
            })
            print(f"{encoder}: {report['contract']['parameter_layout']}", flush=True)
            model_report = {}
            runtime_failed = False
            try:
                # Run explicit stages first so unsupported operations are
                # reported with a useful stage name.
                for _ in range(WARMUP_RUNS):
                    run_partitioned(model, data)

                samples = defaultdict(list)
                timing_samples[encoder] = samples
                partitioned_output = None
                for run_index in range(PROFILE_RUNS):
                    partitioned_output = run_partitioned(
                        model, data, samples
                    )
                    print(
                        f"  staged run {run_index + 1}/{PROFILE_RUNS}", flush=True
                    )
                assert partitioned_output is not None

                direct_output = run_direct(model, data)
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

    ranking = build_optimization_ranking(timing_samples) if len(timing_samples) == len(ENCODERS) else {
        "stage_rows": [],
        "category_rows": [],
        "sort_key": "combined.median_ms descending",
        "stage_table_limit": RANKING_TABLE_LIMIT,
        "p99_note": "ranking unavailable because one or more encoders failed",
    }
    report["optimization_ranking"] = ranking
    if ranking["stage_rows"]:
        print_optimization_tables(ranking)

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
