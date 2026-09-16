import sys
from pathlib import Path

import numpy as np
import pytest
import json
from types import SimpleNamespace
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from graspgenx_baseline_common import (
    AttrDict, ReplayScheduler, StageTimer, compare_outputs, install_function_timers,
    load_cuda_baselines, write_cuda_baselines,
)
from profile_graspgenx_stages import print_stage_timings
from graspgenx.models.generator_ascend import GraspGenGeneratorAscend
from graspgenx.models.generator import GraspGenGenerator
from graspgenx.models.grasp_gen import GraspGen


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


def make_stage_timer(steps=2):
    timer = StageTimer(torch.device("cpu"))
    for name in ("generator.ptv3", "generator.gripper_encoder", "discriminator.ptv3",
                 "discriminator.pose_conversion", "discriminator.sample_encoder",
                 "discriminator.gripper_encoder", "discriminator.prediction_head"):
        timer.samples[name] = [0.0] * 3
    for name in ("generator.diffusion_head", "generator.scheduler_position",
                 "generator.scheduler_rotation", "generator.pose_conversion"):
        timer.samples[name] = [1.0] * (3 * steps)
    timer.samples["generator.likelihood_log_prob"] = [1.0] * (3 * 2 * (steps - 1))
    timer.samples["generator.total"] = [200.0, 300.0, 400.0]
    timer.samples["discriminator.total"] = [2.0, 4.0, 6.0]
    return timer


def test_profile_aggregates_before_median_and_reconciles_means(capsys):
    timer = make_stage_timer()
    timer.samples["generator.diffusion_head"] = [1.0, 20.0, 2.0, 30.0, 3.0, 40.0]
    timer.samples["generator.ptv3"] = [100.0, 1.0, 100.0]
    raw = timer.summary()
    grouped = timer.request_summary(runs=3, diffusion_steps=2)
    assert raw["generator.diffusion_head"]["median_ms"] == 11.5
    assert grouped["generator.diffusion_head"]["median_ms"] == 32.0  # median(21,32,43), not 11.5*2
    for group in ("generator", "discriminator"):
        children = [row for name, row in grouped.items()
                    if name.startswith(group + ".") and name != group + ".total"]
        assert sum(row["mean_ms"] for row in children) == pytest.approx(grouped[group + ".total"]["mean_ms"])
    assert grouped["pipeline.total"]["mean_ms"] == 304.0
    assert grouped["pipeline.total"]["median_ms"] == 304.0

    print_stage_timings(raw, grouped)
    text = capsys.readouterr().out
    row = next(line for line in text.splitlines() if line.startswith("diffusion_head"))
    assert row.split()[1:5] == ["2", "11.500", "32.000", "32.000"]
    assert "[generator]" in text and "[discriminator]" in text
    assert "unattributed" in text and "Mean balance:" in text
    assert "medians are not additive" in text


def test_profile_prints_diffusion_call_counts_from_samples(capsys):
    timer = make_stage_timer(steps=20)
    print_stage_timings(timer.summary(), timer.request_summary(3, 20))
    rows = {line.split()[0]: line.split()[1:]
            for line in capsys.readouterr().out.splitlines()
            if line.startswith(("diffusion_head ", "likelihood_log_prob "))}
    assert rows["diffusion_head"][:4] == ["20", "1.000", "20.000", "20.000"]
    assert rows["likelihood_log_prob"][:4] == ["38", "1.000", "38.000", "38.000"]


def test_profile_preserves_negative_unattributed_and_warns(capsys):
    timer = make_stage_timer()
    timer.samples["generator.total"] = [0.0, 0.0, 0.0]
    grouped = timer.request_summary(3, 2)
    assert grouped["generator.unattributed"]["mean_ms"] < 0
    print_stage_timings(timer.summary(), grouped)
    text = capsys.readouterr().out
    assert "WARNING: negative unattributed samples" in text
    assert "n/a" in text  # zero parent time has no defined percentage


def test_profile_rejects_missing_diffusion_calls():
    timer = make_stage_timer(20)
    timer.samples["generator.diffusion_head"].pop()
    with pytest.raises(RuntimeError, match="expected 60"):
        timer.request_summary(3, 20)


def generator_helpers():
    model = GraspGenGeneratorAscend.__new__(GraspGenGeneratorAscend)
    torch.nn.Module.__init__(model)
    return model


def test_likelihood_cache_matches_original_expression():
    model = generator_helpers()
    betas = torch.linspace(0.0001, 0.02, 20)
    saved_betas = betas.clone()
    means = torch.linspace(-2, 2, 600).reshape(100, 6)
    samples = means.flip(-1)
    scales = model._prepare_likelihood_scales(betas, "cpu")
    assert len(scales) == len(betas)
    for k in range(19, 0, -1):
        for section in (slice(None, 3), slice(3, None)):
            sample, mean = samples[:, section], means[:, section]
            expected = torch.distributions.Normal(mean, betas[k].sqrt()).log_prob(sample).sum(-1, keepdim=True)
            actual = model._inference_likelihood(sample, mean, k, scales)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(betas, saved_betas, rtol=0, atol=0)
    assert torch.distributions.Distribution._validate_args  # no global disable
    expected = torch.distributions.Normal(means, betas[5].sqrt()).log_prob(samples).sum(-1, keepdim=True)
    torch.testing.assert_close(expected, model._inference_likelihood(samples, means, 5, scales), rtol=0, atol=0)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_likelihood_rejects_invalid_scheduler_constants(value):
    model = generator_helpers()
    with pytest.raises(ValueError, match="finite and positive"):
        model._prepare_likelihood_scales(torch.tensor([0.01, value]), "cpu")


def test_history_keeps_cpu_contract_and_timestep_order():
    model = generator_helpers()
    history = model._allocate_inference_history((1, 4, 2, 4, 4))
    for k in (3, 2, 1, 0):
        model._write_inference_history(history, k, torch.full((1, 2, 4, 4), k + 1.0))
    output = history
    assert output.device.type == "cpu" and output.dtype == torch.float32
    assert output.shape == (1, 4, 2, 4, 4)
    for k in range(4):
        torch.testing.assert_close(output[:, k], torch.full((1, 2, 4, 4), k + 1.0))
    torch.testing.assert_close(model._allocate_inference_history((1, 2)), torch.zeros(1, 2))


def test_profile_likelihood_and_history_are_disjoint(capsys):
    timer = make_stage_timer(20)
    del timer.samples["generator.likelihood_log_prob"]
    for name, calls in (("likelihood", 38), ("likelihood_setup", 2), ("likelihood_accumulate", 19),
                         ("history_allocate", 1), ("history_write", 20)):
        timer.samples[f"generator.{name}"] = [1.0] * (3 * calls)
    grouped = timer.request_summary(3, 20)
    assert grouped["generator.likelihood"]["mean_ms"] == 38
    assert "generator.likelihood_log_prob" not in grouped
    print_stage_timings(timer.summary(), grouped)
    assert "history_write" in capsys.readouterr().out
    parts = sum(v["mean_ms"] for k, v in grouped.items() if k.startswith("generator.") and k != "generator.total")
    assert parts == pytest.approx(grouped["generator.total"]["mean_ms"])


def test_function_timers_restore_methods_on_error():
    generator = generator_helpers()
    model = SimpleNamespace(grasp_generator=generator)
    timer = StageTimer(torch.device("cpu"))
    original = generator._inference_likelihood
    normal_log_prob = torch.distributions.Normal.log_prob
    with pytest.raises(RuntimeError, match="intentional"):
        with install_function_timers(timer, model):
            assert torch.distributions.Normal.log_prob is normal_log_prob
            assert generator._inference_likelihood is not original
            raise RuntimeError("intentional")
    assert generator._inference_likelihood == original
    assert "_inference_likelihood" not in generator.__dict__
    assert torch.distributions.Normal.log_prob is normal_log_prob


def test_likelihood_accumulation_preserves_inplace_and_rounding_order():
    model = generator_helpers()
    position, rotation = torch.tensor([[0.1], [1e5]]), torch.tensor([[0.3], [-1e5]])
    original = torch.ones(2, 1)
    current = original.clone()
    pointer = current.data_ptr()
    model._accumulate_inference_likelihood(current, position, rotation)
    torch.testing.assert_close(current, original + (position + rotation), rtol=0, atol=0)
    assert current.data_ptr() == pointer
    model._accumulate_inference_likelihood(current, position)
    torch.testing.assert_close(current, original + (position + rotation) + position, rtol=0, atol=0)


def small_pipeline_configs(compositional=True):
    ptv3 = AttrDict(
        enc_depths=[1, 1, 1, 1, 1], enc_channels=[4, 8, 16, 16, 16],
        enc_num_head=[1, 1, 2, 2, 2], enc_patch_size=[8] * 5,
        enable_flash=False, drop_path=0.0,
    )
    common = dict(
        num_object_dim=16, num_embed_dim=16, object_backbone="ptv3vanilla",
        gripper_backbone="sweep_volume_v2", grasp_repr="r3_so3", kappa=3.27,
        pose_repr="mlp", checkpoint_object_encoder_pretrained=None,
        ptv3=AttrDict(grid_size=0.01), ptv3vanilla=ptv3,
    )
    generator = AttrDict(
        **common, diffusion_embed_dim=16, image_size=32, num_diffusion_iters=4,
        num_diffusion_iters_eval=4, compositional_schedular=compositional,
        loss_pointmatching=False, loss_l1_pos=False, loss_l1_rot=False,
        clip_sample=True, beta_schedule="scaled_linear", attention="cat",
        num_grasps_per_object=3,
    )
    return generator, AttrDict(**common, topk_ratio=0.4)


def test_factory_uses_requested_class_and_preserves_reference_default():
    gen_cfg, dis_cfg = small_pipeline_configs()
    reference = GraspGen.from_config(gen_cfg, dis_cfg)
    selected = GraspGen.from_config(gen_cfg, dis_cfg, generator_class=GraspGenGeneratorAscend)
    assert type(reference.grasp_generator) is GraspGenGenerator
    assert type(selected.grasp_generator) is GraspGenGeneratorAscend
    assert not hasattr(reference.grasp_generator, "_prepare_likelihood_scales")
    before = reference.grasp_generator.state_dict()
    after = selected.grasp_generator.state_dict()
    assert before.keys() == after.keys()
    assert all(before[key].shape == after[key].shape for key in before)
    selected.grasp_generator.load_state_dict(before, strict=True)
    assert GraspGenGeneratorAscend.__init__ is GraspGenGenerator.__init__
    assert GraspGenGeneratorAscend.forward_train is GraspGenGenerator.forward_train


@pytest.mark.parametrize("compositional", [True, False])
@pytest.mark.parametrize("initial_noise", [True, False])
def test_generator_inference_preserves_outputs(compositional, initial_noise):
    cfg, _ = small_pipeline_configs(compositional)
    reference = GraspGenGenerator.from_config(cfg).eval()
    selected = GraspGenGeneratorAscend.from_config(cfg).eval()
    selected.load_state_dict(reference.state_dict(), strict=True)
    for model in (reference, selected):
        for module in model.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = False
    inputs = {
        "points": torch.linspace(-0.02, 0.02, 48).reshape(2, 8, 3),
        "sweep_volume_open_and_mid": torch.ones(2, 12) * 0.05,
    }
    if initial_noise:
        inputs["initial_noise"] = torch.linspace(-0.7, 0.7, 36).reshape(6, 6)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(405)
        expected, _, _ = reference.infer({key: value.clone() for key, value in inputs.items()})
        torch.manual_seed(405)
        actual, _, _ = selected.infer({key: value.clone() for key, value in inputs.items()})
    finally:
        torch.set_num_threads(threads)
    assert expected.keys() == actual.keys()
    for key in expected:
        assert actual[key].device == expected[key].device
        assert actual[key].dtype == expected[key].dtype
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["grasps_per_iteration"].shape == (2, 4, 3, 4, 4)
    torch.testing.assert_close(actual["grasps_pred"], actual["grasps_per_iteration"][:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("generator_class", [GraspGenGenerator, GraspGenGeneratorAscend])
def test_pose_profile_tracks_selected_generator_module(generator_class, monkeypatch):
    generator = generator_class.__new__(generator_class)
    torch.nn.Module.__init__(generator)
    module = sys.modules[generator.forward_inference.__module__]
    original = lambda *args, **kwargs: "pose"
    monkeypatch.setattr(module, "rt_to_matrix", original)
    timer = StageTimer(torch.device("cpu"))
    with install_function_timers(timer, SimpleNamespace(grasp_generator=generator)):
        assert module.rt_to_matrix() == "pose"
        assert len(timer.samples["generator.pose_conversion"]) == 1
    assert module.rt_to_matrix is original
