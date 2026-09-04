#!/usr/bin/env python3
"""Benchmark the current CPU-PTV3/NPU-head GraspGenX baseline.

This entry point uses full-size randomly initialized release modules. It keeps
both PTV3 object encoders on CPU and runs the dense generator/discriminator
modules on an Ascend NPU. Model loading is intentionally excluded so the
benchmark can run without release checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover - hardware dependent
    raise RuntimeError("torch-npu is required for this benchmark") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()

    def forward(self, x):
        return x


def install_import_shims() -> None:
    """Stub optional training/data dependencies absent from the NPU image."""
    package = types.ModuleType("graspgenx")
    package.__path__ = [str(REPO_ROOT / "graspgenx")]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "graspgenx", loader=None, is_package=True
    )

    addict = types.ModuleType("addict")
    addict.Dict = AttrDict

    timm_layers = types.ModuleType("timm.models.layers")
    timm_layers.DropPath = DropPath

    robot = types.ModuleType("graspgenx.robot")
    robot.get_gripper_info = lambda *args, **kwargs: None
    robot.GripperInfo = object
    robot.get_canonical_gripper_control_points = (
        lambda *args, **kwargs: np.zeros((4, 3), np.float32)
    )

    dataset = types.ModuleType("graspgenx.dataset.dataset")
    dataset.MAPPING_ID2NAME = {}

    metrics = types.ModuleType("graspgenx.metrics")
    metrics.compute_metrics_given_two_sets_of_xgripper_poses = (
        lambda *args, **kwargs: {}
    )
    metrics.compute_recall = lambda *args, **kwargs: 0.0

    sklearn = types.ModuleType("sklearn")
    sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    sklearn_metrics = types.ModuleType("sklearn.metrics")
    sklearn_metrics.average_precision_score = lambda *args, **kwargs: 0.0
    sklearn.metrics = sklearn_metrics

    pointnet2_ops = types.ModuleType("pointnet2_ops")
    pointnet2_ops._ext = None

    sys.modules.update(
        {
            "graspgenx": package,
            "addict": addict,
            "timm.models.layers": timm_layers,
            "graspgenx.robot": robot,
            "graspgenx.dataset.dataset": dataset,
            "graspgenx.metrics": metrics,
            "sklearn": sklearn,
            "sklearn.metrics": sklearn_metrics,
            "pointnet2_ops": pointnet2_ops,
        }
    )


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": float(np.percentile(values, 95)) * 1000,
        "min_ms": min(values) * 1000,
        "max_ms": max(values) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-count", type=int, default=2048)
    parser.add_argument("--num-grasps", type=int, default=100)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(
        args.point_count,
        args.num_grasps,
        args.diffusion_steps,
        args.warmup,
        args.profile_runs,
        args.runs,
        args.cpu_threads,
    ) < 1:
        parser.error("all numeric arguments must be positive")

    install_import_shims()

    from graspgenx.models.discriminator import GraspGenDiscriminator
    from graspgenx.models.generator import GraspGenGenerator
    from graspgenx.models.model_utils import (
        convert_to_ptv3_pc_format,
        offset2batch,
    )
    from graspgenx.utils.transformations import matrix_to_rt, rt_to_matrix

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    ptv3_config = AttrDict(enable_flash=False)

    setup_started = time.perf_counter()
    torch.manual_seed(2026)
    generator = GraspGenGenerator(
        num_embed_dim=256,
        num_object_dim=512,
        num_gripper_dim=512,
        diffusion_embed_dim=512,
        image_size=256,
        num_diffusion_iters=20,
        num_diffusion_iters_eval=args.diffusion_steps,
        object_backbone="ptv3vanilla",
        gripper_backbone="sweep_volume_v2",
        compositional_schedular=True,
        loss_pointmatching=False,
        loss_l1_pos=True,
        loss_l1_rot=True,
        grasp_repr="r3_so3",
        kappa=3.27,
        clip_sample=True,
        beta_schedule="squaredcos_cap_v2",
        attention="cat_attn",
        grid_size=0.01,
        pose_repr="mlp",
        num_grasps_per_object=args.num_grasps,
        pointnet_version="v2",
        ptv3vanilla_config=ptv3_config,
    )
    torch.manual_seed(2027)
    discriminator = GraspGenDiscriminator(
        num_object_dim=512,
        num_gripper_dim=512,
        object_backbone="ptv3vanilla",
        gripper_backbone="sweep_volume_v2",
        grasp_repr="r3_so3",
        grid_size=0.01,
        sample_embed_dim=256,
        pose_repr="mlp",
        topk_ratio=1.0,
        kappa=3.27,
        pointnet_version="v2",
        ptv3vanilla_config=ptv3_config,
    )

    generator_ptv3 = generator.object_encoder.cpu().eval()
    discriminator_ptv3 = discriminator.object_encoder.cpu().eval()
    generator.object_encoder = nn.Identity()
    discriminator.object_encoder = nn.Identity()
    generator = generator.to(device).eval()
    discriminator = discriminator.to(device).eval()
    generator.object_encoder = generator_ptv3
    discriminator.object_encoder = discriminator_ptv3
    for encoder in (generator_ptv3, discriminator_ptv3):
        encoder.shuffle_orders = False

    point_rng = np.random.default_rng(1234)
    points_np = point_rng.normal(
        0.0, 0.05, size=(args.point_count, 3)
    ).astype(np.float32)
    points_np -= points_np.mean(axis=0, keepdims=True)
    points = torch.from_numpy(points_np).unsqueeze(0).to(device)

    noise_np = np.random.default_rng(5678).standard_normal(
        (args.num_grasps, 6), dtype=np.float32
    )
    initial_noise = torch.from_numpy(noise_np).to(device)
    sweep_np = np.concatenate(
        [
            np.array(
                [
                    0.08647013588950439,
                    0.015360593925194512,
                    0.047533627175553844,
                    0,
                    0,
                    0.16079110012540068,
                ],
                np.float32,
            ),
            np.array(
                [
                    0.043235067944752195,
                    0.015360593925194512,
                    0.047533627175553844,
                    0,
                    0,
                    0.16079110012540068,
                ],
                np.float32,
            ),
        ]
    )
    sweep = torch.from_numpy(sweep_np).unsqueeze(0).to(device)
    torch.npu.synchronize()
    setup_s = time.perf_counter() - setup_started

    def sync() -> None:
        torch.npu.synchronize()

    def run_pipeline(stage_totals: dict[str, float] | None = None):
        def stage(name, function):
            if stage_totals is None:
                return function()
            sync()
            started = time.perf_counter()
            result = function()
            sync()
            stage_totals[name] += time.perf_counter() - started
            return result

        if stage_totals is not None:
            sync()
            end_to_end_started = time.perf_counter()
            generator_started = end_to_end_started

        depth = stage("generator.scale", lambda: points * generator.kappa)
        depth_cpu = stage("generator.points_to_cpu", depth.cpu)
        generator_pc = stage(
            "generator.ptv3_format",
            lambda: convert_to_ptv3_pc_format(
                depth_cpu, grid_size=generator.grid_size
            ),
        )
        object_cpu = stage(
            "generator.ptv3_encoder", lambda: generator_ptv3(generator_pc)
        )
        object_embedding = stage(
            "generator.embedding_to_device", lambda: object_cpu.to(device)
        )

        def generator_batch_mapping():
            offset = (
                torch.tensor([args.num_grasps])
                .repeat(1)
                .cumsum(dim=0)
                .to(device)
            )
            return offset2batch(offset)

        grasp_batch = stage(
            "generator.batch_mapping", generator_batch_mapping
        )
        object_embedding = stage(
            "generator.object_repeat", lambda: object_embedding[grasp_batch]
        )
        gripper_embedding = stage(
            "generator.gripper_encoder",
            lambda: generator.gripper_encoder(sweep),
        )
        gripper_embedding = stage(
            "generator.gripper_repeat", lambda: gripper_embedding[grasp_batch]
        )
        observation_embedding = stage(
            "generator.conditioning",
            lambda: torch.cat(
                [object_embedding, gripper_embedding], dim=-1
            ),
        )

        def scheduler_setup():
            generator.noise_scheduler_pos.set_timesteps(args.diffusion_steps)
            generator.noise_scheduler_rot.set_timesteps(args.diffusion_steps)
            return generator.noise_scheduler_pos.timesteps

        timesteps = stage("generator.scheduler_setup", scheduler_setup)
        noisy_grasps = stage("generator.noise_init", initial_noise.clone)
        likelihood = stage(
            "generator.likelihood_init",
            lambda: torch.zeros((args.num_grasps, 1), device=device),
        )
        grasp_history = stage(
            "generator.history_init",
            lambda: torch.zeros(
                (1, args.diffusion_steps, args.num_grasps, 4, 4),
                device=device,
            ),
        )

        pred_grasps = None
        for step_index, timestep in enumerate(timesteps):
            noise_pred = stage(
                "generator.diffusion_head",
                lambda: generator.diffusion_head(
                    observation_embedding, timestep, noisy_grasps
                ),
            )

            def scheduler_step():
                position = generator.noise_scheduler_pos.step(
                    model_output=noise_pred[..., :3],
                    timestep=timestep,
                    sample=noisy_grasps[..., :3],
                )
                rotation = generator.noise_scheduler_rot.step(
                    model_output=noise_pred[..., 3:6],
                    timestep=timestep,
                    sample=noisy_grasps[..., 3:6],
                )
                return position, rotation

            position_result, rotation_result = stage(
                "generator.scheduler_step", scheduler_step
            )

            if int(timestep) > 0:
                def update_likelihood():
                    position_variance = generator.noise_scheduler_pos.betas[
                        timestep
                    ]
                    rotation_variance = generator.noise_scheduler_rot.betas[
                        timestep
                    ]
                    position_score = torch.distributions.Normal(
                        position_result.pred_original_sample,
                        torch.sqrt(
                            torch.tensor(position_variance, device=device)
                        ),
                    ).log_prob(noisy_grasps[..., :3]).sum(-1, keepdim=True)
                    rotation_score = torch.distributions.Normal(
                        rotation_result.pred_original_sample,
                        torch.sqrt(
                            torch.tensor(rotation_variance, device=device)
                        ),
                    ).log_prob(noisy_grasps[..., 3:6]).sum(-1, keepdim=True)
                    return likelihood + position_score + rotation_score

                likelihood = stage(
                    "generator.likelihood", update_likelihood
                )

            noisy_grasps = stage(
                "generator.sample_update",
                lambda: torch.hstack(
                    [position_result.prev_sample, rotation_result.prev_sample]
                ),
            )
            pred_grasps = stage(
                "generator.pose_conversion",
                lambda: rt_to_matrix(
                    noisy_grasps, "r3_so3", generator.kappa
                ),
            )
            stage(
                "generator.history_store",
                lambda: grasp_history.__setitem__(
                    (slice(None), step_index),
                    pred_grasps.reshape(1, args.num_grasps, 4, 4),
                ),
            )

        grasps = pred_grasps.reshape(1, args.num_grasps, 4, 4)
        if stage_totals is not None:
            sync()
            stage_totals["generator.total"] = (
                time.perf_counter() - generator_started
            )
            discriminator_started = time.perf_counter()

        discriminator_depth = stage(
            "discriminator.scale", lambda: points * discriminator.kappa
        )
        discriminator_depth_cpu = stage(
            "discriminator.points_to_cpu", discriminator_depth.cpu
        )
        discriminator_pc = stage(
            "discriminator.ptv3_format",
            lambda: convert_to_ptv3_pc_format(
                discriminator_depth_cpu, grid_size=discriminator.grid_size
            ),
        )
        discriminator_object_cpu = stage(
            "discriminator.ptv3_encoder",
            lambda: discriminator_ptv3(discriminator_pc),
        )
        discriminator_object = stage(
            "discriminator.embedding_to_device",
            lambda: discriminator_object_cpu.to(device),
        )
        grasp_features = stage(
            "discriminator.pose_conversion",
            lambda: matrix_to_rt(
                grasps.reshape(-1, 4, 4),
                "r3_so3",
                discriminator.kappa,
            ),
        )

        def discriminator_batch_mapping():
            offset = (
                torch.tensor([args.num_grasps])
                .repeat(1)
                .cumsum(dim=0)
                .to(device)
            )
            return offset2batch(offset)

        discriminator_batch = stage(
            "discriminator.batch_mapping", discriminator_batch_mapping
        )
        sample_embedding = stage(
            "discriminator.sample_encoder",
            lambda: discriminator.sample_encoder(grasp_features),
        )
        discriminator_object = stage(
            "discriminator.object_repeat",
            lambda: discriminator_object[discriminator_batch],
        )
        discriminator_gripper = stage(
            "discriminator.gripper_encoder",
            lambda: discriminator.gripper_encoder(sweep),
        )
        discriminator_gripper = stage(
            "discriminator.gripper_repeat",
            lambda: discriminator_gripper[discriminator_batch],
        )
        discriminator_embedding = stage(
            "discriminator.conditioning",
            lambda: torch.cat(
                [
                    sample_embedding,
                    discriminator_object,
                    discriminator_gripper,
                ],
                dim=-1,
            ),
        )
        logits = stage(
            "discriminator.head",
            lambda: discriminator.prediction_head(discriminator_embedding),
        )
        confidence = stage(
            "discriminator.sigmoid",
            lambda: logits.sigmoid().reshape(1, args.num_grasps, 1),
        )

        if stage_totals is not None:
            sync()
            stage_totals["discriminator.total"] = (
                time.perf_counter() - discriminator_started
            )
            stage_totals["end_to_end_profiled"] = (
                time.perf_counter() - end_to_end_started
            )
        return grasps, confidence

    warmup_values = []
    for index in range(args.warmup):
        print(f"warmup {index + 1}/{args.warmup}", flush=True)
        sync()
        started = time.perf_counter()
        with torch.inference_mode():
            grasps, confidence = run_pipeline()
        sync()
        warmup_values.append(time.perf_counter() - started)

    profiled_samples = defaultdict(list)
    for index in range(args.profile_runs):
        print(f"profiled run {index + 1}/{args.profile_runs}", flush=True)
        totals = defaultdict(float)
        with torch.inference_mode():
            grasps, confidence = run_pipeline(totals)
        for name, value in totals.items():
            profiled_samples[name].append(value)

    unprofiled_values = []
    for index in range(args.runs):
        print(f"unprofiled run {index + 1}/{args.runs}", flush=True)
        sync()
        started = time.perf_counter()
        with torch.inference_mode():
            grasps, confidence = run_pipeline()
        sync()
        unprofiled_values.append(time.perf_counter() - started)

    grasps_np = grasps.detach().float().cpu().numpy()
    confidence_np = confidence.detach().float().cpu().numpy()
    report = {
        "environment": {
            "device": torch.npu.get_device_name(0),
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "precision": "fp32",
            "weights": "random",
            "ptv3_device": "cpu",
            "dense_heads_device": str(device),
        },
        "workload": {
            "point_count": args.point_count,
            "num_grasps": args.num_grasps,
            "diffusion_steps": args.diffusion_steps,
            "grid_size": 0.01,
            "kappa": 3.27,
            "ptv3_enable_flash": False,
            "ptv3_shuffle_orders": False,
        },
        "call_counts_per_request": {
            "generator.ptv3_encoder": 1,
            "generator.diffusion_head": args.diffusion_steps,
            "discriminator.ptv3_encoder": 1,
            "discriminator.head": 1,
        },
        "setup_s": setup_s,
        "warmup": summarize(warmup_values),
        "profiled_stages": {
            name: summarize(values)
            for name, values in sorted(profiled_samples.items())
        },
        "end_to_end_unprofiled": summarize(unprofiled_values),
        "output": {
            "grasps_shape": list(grasps_np.shape),
            "confidence_shape": list(confidence_np.shape),
            "finite": bool(
                np.isfinite(grasps_np).all()
                and np.isfinite(confidence_np).all()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
