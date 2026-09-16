#!/usr/bin/env python3
"""Generate and optionally visualize the full CUDA GraspGenX baseline.

Run on 136.209 from `/data/hyd/GraspGenX` with its isolated `.venv-ptv3`.
Required files are the release checkpoints and the existing PTV3 n2048 golden.
All configuration is below; the script has no command-line arguments.
"""

from __future__ import annotations

import os
from pathlib import Path

# reference: deterministic accuracy golden, manual FP32 PTV3 attention.
# native: release Flash/SDPA + TF32 performance baseline.
# tensorrt_fp16: native PTV3 plus optional TensorRT FP16 heads.
CUDA_MODE = "native"
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
if CUDA_MODE == "reference":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from graspgenx_baseline_common import (
    Pipeline, REPO_ROOT, build_model, compare_outputs, create_request, load_request,
    load_cuda_baselines, output_arrays, run_timed, save_json, write_cuda_baselines,
)


CHECKPOINT_ROOT = REPO_ROOT / ".artifacts/checkpoints/release"
PTV3_REFERENCE = REPO_ROOT / ".artifacts/ptv3-baseline/cuda-fp32-eager/reference_n2048.npz"

OUTPUT_DIR = REPO_ROOT / f".artifacts/graspgenx-baseline/cuda-{CUDA_MODE}-2048-v1"
REQUEST_PATH = OUTPUT_DIR / "request.npz"
OUTPUT_PATH = OUTPUT_DIR / "cuda_outputs.npz"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

CUDA_ACCELERATION = "tensorrt_fp16" if CUDA_MODE == "tensorrt_fp16" else "native"
PTV3_FLASH = CUDA_MODE != "reference"
ALLOW_TF32 = CUDA_MODE != "reference"
DETERMINISTIC_ALGORITHMS = CUDA_MODE == "reference"
WARMUP_RUNS = 3
MEASURED_RUNS = 20
SHOW_VISER = True
VISER_PORT = 8081
SHOW_TOP_GRASPS = 50

# Pack canonical frozen runs, not the newest candidate, alongside the reference.
CUDA_BASELINE_SOURCES = {
    "reference": REPO_ROOT / ".artifacts/graspgenx-baseline/cuda-reference-2048-v1",
    "native": REPO_ROOT / ".artifacts/graspgenx-baseline/cuda-native-2048-v1",
    "trt": REPO_ROOT / ".artifacts/graspgenx-baseline/cuda-tensorrt_fp16-2048-v1",
}
COMPARISON_DIR = CUDA_BASELINE_SOURCES["reference"]


def ensure_comparison_bundle():
    if (COMPARISON_DIR / "cuda_baselines.json").exists():
        load_cuda_baselines(COMPARISON_DIR)
        return
    if not all((directory / filename).is_file()
               for directory in CUDA_BASELINE_SOURCES.values()
               for filename in ("request.npz", "cuda_outputs.npz", "summary.json")):
        print("CUDA comparison bundle pending: generate all three modes first.", flush=True)
        return
    write_cuda_baselines(CUDA_BASELINE_SOURCES, COMPARISON_DIR)
    print(f"CUDA reference/native/TRT comparison bundle: {COMPARISON_DIR}", flush=True)


def add_viewer(points, grasps, confidence, sweep_volume):
    import time
    from graspgenx.utils.viser_utils import (
        create_visualizer,
        get_color_from_score,
        visualize_pointcloud,
        visualize_x_grasp,
    )

    server = create_visualizer(port=VISER_PORT)
    visualize_pointcloud(
        server, "/object", points, color=np.array([120, 180, 255]), size=0.002
    )
    sweep = {"extents": sweep_volume[:3], "offset": sweep_volume[3:6]}
    order = np.argsort(-confidence.reshape(-1))[:SHOW_TOP_GRASPS]
    for rank, index in enumerate(order):
        grasp = grasps.reshape(-1, 4, 4)[index]
        score = float(confidence.reshape(-1)[index])
        visualize_x_grasp(
            server,
            f"/grasps/{rank:03d}_score_{score:.4f}",
            transform=grasp,
            color=get_color_from_score(score, use_255_scale=True),
            sweep_volume=sweep,
            linewidth=3.0,
        )
    print(f"Viser: http://192.168.136.209:{VISER_PORT}", flush=True)
    while True:
        time.sleep(1)


def main():
    if OUTPUT_PATH.exists() or SUMMARY_PATH.exists():
        if not (OUTPUT_PATH.exists() and SUMMARY_PATH.exists() and REQUEST_PATH.exists()):
            raise FileExistsError(f"Incomplete frozen output under {OUTPUT_DIR}")
        print(f"Reusing frozen CUDA output under {OUTPUT_DIR}", flush=True)
        ensure_comparison_bundle()
        if SHOW_VISER:
            request = load_request(REQUEST_PATH)
            with np.load(OUTPUT_PATH) as source:
                add_viewer(
                    request["points"],
                    np.asarray(source["grasps"]),
                    np.asarray(source["confidence"]),
                    request["sweep_volume"],
                )
        return
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    torch.set_float32_matmul_precision("high" if ALLOW_TF32 else "highest")
    torch.use_deterministic_algorithms(DETERMINISTIC_ALGORITHMS, warn_only=False)
    create_request(REQUEST_PATH, PTV3_REFERENCE)
    request = load_request(REQUEST_PATH)
    model, cfg, device, acceleration = build_model(
        CHECKPOINT_ROOT,
        "cuda",
        cuda_acceleration=CUDA_ACCELERATION,
        cuda_ptv3_flash=PTV3_FLASH,
    )
    pipeline = Pipeline(model, request, device)
    samples, timings = run_timed(pipeline, WARMUP_RUNS, MEASURED_RUNS)
    outputs, traces = pipeline.run(record=True)
    arrays = output_arrays(outputs, traces, pipeline)
    accuracy = None
    if CUDA_MODE == "reference":
        repeated_outputs, repeated_traces = pipeline.run(record=True)
        repeated = output_arrays(repeated_outputs, repeated_traces, pipeline)
        accuracy = compare_outputs(arrays, repeated)
        if not accuracy["valid"] or any(
            not np.array_equal(arrays[key], repeated[key]) for key in arrays
        ):
            raise RuntimeError("deterministic CUDA reference did not reproduce")
    else:
        reference_path = (
            REPO_ROOT
            / ".artifacts/graspgenx-baseline/cuda-reference-2048-v1/cuda_outputs.npz"
        )
        if reference_path.is_file():
            with np.load(reference_path) as source:
                reference = {key: np.asarray(source[key]) for key in source.files}
            accuracy = compare_outputs(reference, arrays)
    np.savez(OUTPUT_PATH, **arrays)
    summary = {
        "runtime": "cuda",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "acceleration": acceleration,
        "tf32": ALLOW_TF32,
        "ptv3_flash": PTV3_FLASH,
        "deterministic_algorithms": DETERMINISTIC_ALGORITHMS,
        "point_count": len(request["points"]),
        "num_grasps": request["num_grasps"],
        "diffusion_steps": request["diffusion_steps"],
        "timings": timings,
        "samples_ms": samples,
        "confidence_min": float(arrays["confidence"].min()),
        "confidence_max": float(arrays["confidence"].max()),
        "confidence_mean": float(arrays["confidence"].mean()),
        "outputs_finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
        "accuracy": accuracy,
        "numerical_policy": "report_only",
    }
    save_json(SUMMARY_PATH, summary)
    ensure_comparison_bundle()
    print(summary, flush=True)
    if SHOW_VISER:
        add_viewer(
            request["points"], arrays["grasps"], arrays["confidence"],
            request["sweep_volume"],
        )


if __name__ == "__main__":
    main()
