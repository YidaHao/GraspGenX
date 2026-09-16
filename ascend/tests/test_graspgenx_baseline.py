import sys
from pathlib import Path

import numpy as np
import pytest
import json
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from graspgenx_baseline_common import (
    ReplayScheduler, compare_outputs, load_cuda_baselines, write_cuda_baselines,
)


def test_replay_scheduler_matches_diffusers_contiguous_step():
    scheduler = DDPMScheduler(
        num_train_timesteps=20,
        beta_schedule="scaled_linear",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(20)
    sample = torch.linspace(-1, 1, 30, dtype=torch.float32).reshape(10, 3)
    model_output = sample.flip(-1) * 0.2
    generator = torch.Generator().manual_seed(123)
    expected = scheduler.step(
        model_output, scheduler.timesteps[0], sample,
        generator=generator,
    )
    replay_noise = torch.randn(
        sample.shape, generator=torch.Generator().manual_seed(123)
    ).numpy()
    replay = ReplayScheduler(scheduler, replay_noise[None])
    actual = replay.step(model_output, scheduler.timesteps[0], sample)
    torch.testing.assert_close(actual.prev_sample, expected.prev_sample)
    torch.testing.assert_close(
        actual.pred_original_sample, expected.pred_original_sample
    )


def make_outputs():
    grasps = np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
    grasps[..., :3, 3] = [0.1, 0.2, 0.3]
    return {
        "grasps": grasps,
        "confidence": np.ones((1, 2, 1), dtype=np.float32),
        "generator_embedding": np.ones((1, 1, 512), dtype=np.float32),
        "discriminator_embedding": np.ones((1, 1, 512), dtype=np.float32),
        "noise_prediction": np.ones((2, 2, 6), dtype=np.float32),
        "diffusion_latent": np.ones((2, 2, 6), dtype=np.float32),
        "logits": np.ones((1, 2, 1), dtype=np.float32),
        "likelihood": np.ones((1, 2, 1), dtype=np.float32),
        "grasps_per_iteration": np.repeat(grasps[:, None], 2, axis=1),
    }


def test_numerical_differences_are_report_only():
    base = make_outputs()
    assert compare_outputs(base, {key: value.copy() for key, value in base.items()})[
        "valid"
    ]
    candidate = {key: value.copy() for key, value in base.items()}
    candidate["confidence"][0, 0, 0] = 0
    report = compare_outputs(base, candidate)
    assert report["confidence"]["cosine"] < 0.9999
    assert report["valid"]
    assert "passed" not in json.dumps(report)
    assert report["numerical_policy"] == "report_only"


@pytest.mark.parametrize("kind", ["nan", "inf", "shape", "missing", "steps", "homogeneous"])
def test_invalid_outputs_fail_without_computing_pose_errors(kind):
    base = make_outputs()
    candidate = {key: value.copy() for key, value in base.items()}
    if kind in ("nan", "inf"):
        candidate["confidence"][0, 0, 0] = float(kind)
    elif kind == "shape":
        candidate["grasps"] = candidate["grasps"][..., :3, :]
    elif kind == "missing":
        del candidate["likelihood"]
    elif kind == "steps":
        candidate["diffusion_latent"] = candidate["diffusion_latent"][:1]
    else:
        candidate["grasps"][..., 3, 3] = 0
    report = compare_outputs(base, candidate)
    assert not report["valid"]
    assert report["errors"]
    json.dumps(report, allow_nan=False)


def test_zero_vector_cosine_is_undefined_not_accuracy_failure():
    candidate = make_outputs()
    candidate["generator_embedding"][:] = 0
    report = compare_outputs(make_outputs(), candidate)
    assert report["valid"] and report["generator_embedding"]["cosine"] is None


def create_sources(tmp_path):
    sources = {name: tmp_path / name for name in ("reference", "native", "trt")}
    for directory in sources.values():
        directory.mkdir()
        np.savez(directory / "request.npz", points=np.ones((8, 3), dtype=np.float32),
                 num_grasps=np.int64(2), diffusion_steps=np.int64(2))
        np.savez(directory / "cuda_outputs.npz", **make_outputs())
        (directory / "summary.json").write_text(json.dumps({
            "timings": {"steady": {"median_ms": 42.0}}, "samples_ms": [42.0],
        }))
    return sources


def test_bundle_preserves_frozen_outputs_and_refuses_overwrite(tmp_path):
    sources = create_sources(tmp_path)
    original = {p: p.read_bytes() for d in sources.values() for p in d.iterdir()}
    destination = tmp_path / "bundle"
    write_cuda_baselines(sources, destination)
    request, baselines, metadata = load_cuda_baselines(destination)
    assert int(request["num_grasps"]) == 2
    assert set(baselines) == {"reference", "native", "trt"}
    for name, outputs in baselines.items():
        for key, value in make_outputs().items():
            np.testing.assert_array_equal(outputs[key], value)
        assert metadata["baselines"][name]["vs_reference"]["valid"]
    assert all(p.read_bytes() == content for p, content in original.items())
    with pytest.raises(FileExistsError):
        write_cuda_baselines(sources, destination)


def test_bundle_rejects_different_requests(tmp_path):
    sources = create_sources(tmp_path)
    np.savez(sources["native"] / "request.npz", points=np.zeros((8, 3)),
             num_grasps=np.int64(2), diffusion_steps=np.int64(2))
    with pytest.raises(ValueError, match="identical frozen request"):
        write_cuda_baselines(sources, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_bundle_rejects_corruption(tmp_path):
    sources = create_sources(tmp_path)
    destination = tmp_path / "bundle"
    write_cuda_baselines(sources, destination)
    path = destination / "cuda_baselines.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_cuda_baselines(destination)
