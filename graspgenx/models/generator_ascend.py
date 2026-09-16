#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ascend inference variant of the GraspGen generator.

The network definition, checkpoint layout, training path, and public output
contract remain in ``generator.py``. This subclass owns inference-only changes
that are useful on Ascend: scheduler standard deviations are prepared once per
request and Normal argument validation is not repeated for every diffusion step.
History remains a CPU tensor to preserve the original contract.
PTV3 placement is independent and is selected by the caller's model assembly.
"""

import numpy as np
import torch
from scipy.spatial import KDTree

from graspgenx.models.generator import GraspGenGenerator
from graspgenx.metrics import (
    compute_metrics_given_two_sets_of_xgripper_poses,
    compute_recall,
)
from graspgenx.models.model_utils import (
    convert_to_ptv3_pc_format,
    offset2batch,
)
from graspgenx.utils.transformations import rt_to_matrix


class GraspGenGeneratorAscend(GraspGenGenerator):
    """Reference-compatible generator with the default Ascend inference path."""

    def _prepare_likelihood_scales(self, betas, device):
        # Validate fixed scheduler constants once, avoiding device-to-host
        # boolean checks inside every Normal construction/log_prob call.
        host_betas = betas.detach().cpu()
        if not bool(torch.isfinite(host_betas).all() and (host_betas > 0).all()):
            raise ValueError("Likelihood scheduler betas must be finite and positive")
        return betas.to(device=device).sqrt().unbind(0)

    def _inference_likelihood(self, sample, mean, timestep, scales):
        distribution = torch.distributions.Normal(
            mean, scales[timestep], validate_args=False
        )
        return distribution.log_prob(sample).sum(-1, keepdim=True)

    def _accumulate_inference_likelihood(self, likelihood, position, rotation=None):
        likelihood += position if rotation is None else position + rotation

    def _allocate_inference_history(self, shape):
        return torch.zeros(shape, device="cpu")

    def _write_inference_history(self, history, timestep, grasps):
        history[:, timestep, :, ::] = grasps

    def forward_inference(self, data, return_metrics=False):
        """Reference inference flow with precomputed likelihood scales."""
        device = data["points"].device

        num_objects_in_batch = len(data["points"])
        if "grasps" in data:
            if type(data["grasps"][0]) == list:
                data["grasps"][0] = np.array(data["grasps"][0])
            num_grasps_per_batch = data["grasps"][0].shape[0]
        else:
            num_grasps_per_batch = self.num_grasps_per_object
            return_metrics = False

        batch_size = data["points"].shape[0] * num_grasps_per_batch
        depth = data["points"]
        num_points = depth.shape[-2]
        depth = depth.reshape([-1, num_points, 3]).to(device)
        grasps_init_size = [num_objects_in_batch, num_grasps_per_batch, 4, 4]

        if self.kappa is not None:
            depth = self.kappa * depth
        if self.object_backbone in ("ptv3", "ptv3vanilla"):
            depth = convert_to_ptv3_pc_format(depth, grid_size=self.grid_size)

        grasps_per_iteration = self._allocate_inference_history(
            [
                num_objects_in_batch,
                self.num_diffusion_iters_eval,
                num_grasps_per_batch,
                4,
                4,
            ],
        )

        with torch.no_grad():
            if "initial_noise" in data:
                noisy_init = data["initial_noise"].to(
                    device=device, dtype=data["points"].dtype
                )
                noisy_init = noisy_init.reshape(batch_size, self.output_dim).clone()
            else:
                noisy_init = torch.randn([batch_size, self.output_dim], device=device)
            noisy_grasps = noisy_init
            likelihood = torch.zeros((batch_size, 1), device=device)

            offset = (
                torch.tensor([num_grasps_per_batch])
                .repeat(num_objects_in_batch)
                .cumsum(dim=0)
                .to(device)
            )
            mask_batch = offset2batch(offset)

            if self.pose_repr == "mlp":
                object_embedding = self.object_encoder(depth)
                object_embedding = object_embedding[mask_batch]

            if self.gripper_backbone == "onehot":
                gripper_embedding = self.gripper_encoder(data["onehot"])
            elif self.gripper_backbone == "gripper_type":
                gripper_type = torch.eye(
                    3, dtype=torch.float32, device=object_embedding.device
                )[
                    torch.tensor(
                        data["gripper_type"],
                        dtype=torch.long,
                        device=object_embedding.device,
                    )
                ]
                gripper_embedding = self.gripper_encoder(gripper_type)
            elif self.gripper_backbone == "z_offset":
                gripper_embedding = self.gripper_encoder(data["z_offset"])
            elif self.gripper_backbone == "none":
                gripper_embedding = torch.zeros(
                    (num_objects_in_batch, self.num_gripper_dim),
                    dtype=object_embedding.dtype,
                    device=object_embedding.device,
                )
            elif self.gripper_backbone == "sweep_volume":
                gripper_embedding = self.gripper_encoder(data["sweep_volume"])
            elif self.gripper_backbone == "sweep_volume_v2":
                gripper_embedding = self.gripper_encoder(
                    data["sweep_volume_open_and_mid"]
                )
            elif self.gripper_backbone == "gripper_type+sweep_volume_v2":
                gripper_type = torch.eye(
                    3, dtype=torch.float32, device=object_embedding.device
                )[
                    torch.tensor(
                        data["gripper_type"],
                        dtype=torch.long,
                        device=object_embedding.device,
                    )
                ]
                gripper_embedding = self.gripper_encoder(
                    torch.concat([data["sweep_volume_open_and_mid"], gripper_type], dim=-1)
                )
            elif self.gripper_backbone == "pointcloud":
                gripper_embedding = torch.concat(
                    [
                        self.gripper_encoder(data["gripper_open_ptc"]),
                        self.gripper_encoder(data["gripper_close_ptc"]),
                    ],
                    dim=-1,
                )
            elif self.gripper_backbone == "selected_pointcloud":
                gripper_embedding = self.gripper_encoder(data["gripper_selected_ptc"])
            elif self.gripper_backbone == "selected_pointcloud_v2":
                gripper_embedding = self.gripper_mlp_encoder(
                    torch.concat(
                        [
                            self.gripper_ptc_encoder(data["gripper_selected_open_ptc"]),
                            self.gripper_ptc_encoder(data["gripper_selected_mid_ptc"]),
                            self.gripper_ptc_encoder(data["gripper_selected_close_ptc"]),
                        ],
                        dim=-1,
                    )
                )
            elif self.gripper_backbone == "selected_pointcloud_v3":
                gripper_embedding = torch.concat(
                    [
                        self.gripper_encoder(data["gripper_selected_open_ptc"]),
                        self.gripper_encoder(data["gripper_selected_close_ptc"]),
                    ],
                    dim=-1,
                )
            elif self.gripper_backbone == "volume_tsdf":
                gripper_3d_embedding = self.gripper_backbone_3dconv(
                    data["gripper_vol_tsdf"]
                ).squeeze(-2)
                gripper_2d_embedding = self.gripper_backbone_2dconv(
                    gripper_3d_embedding
                ).reshape(num_objects_in_batch, -1)
                gripper_embedding = self.gripper_backbone_mlp(gripper_2d_embedding)
            elif self.gripper_backbone == "pointnet_repr":
                gripper_embedding = self.gripper_encoder(data["gripper_pointnet_repr"])
            else:
                raise NotImplementedError(
                    f"Gripper Backbone {self.gripper_backbone} not implemented"
                )

            gripper_embedding = gripper_embedding[mask_batch]
            obs_embedding = torch.concat([object_embedding, gripper_embedding], dim=-1)

            if self.compositional_schedular:
                self.noise_scheduler_pos.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler_pos.timesteps
                self.noise_scheduler_rot.set_timesteps(self.num_diffusion_iters_eval)
                scales_pos = self._prepare_likelihood_scales(
                    self.noise_scheduler_pos.betas, device
                )
                scales_rot = self._prepare_likelihood_scales(
                    self.noise_scheduler_rot.betas, device
                )
            else:
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters_eval)
                timesteps = self.noise_scheduler.timesteps
                scales = self._prepare_likelihood_scales(
                    self.noise_scheduler.betas, device
                )

            for k in timesteps:
                samples = noisy_grasps if self.pose_repr == "mlp" else None
                if self.pose_repr in ["grasp_cloud", "grasp_cloud_pe", "pc_feature"]:
                    ctrl_pts = self.ctr_pts.to(device=device)
                    noisy_grasps_mat = rt_to_matrix(
                        noisy_grasps, self.grasp_repr, self.kappa
                    )
                    grasp_pc = (noisy_grasps_mat @ ctrl_pts).transpose(-2, -1)[..., :3]

                if self.pose_repr == "pc_feature":
                    depth_full = depth[mask_batch]
                    depth_full = torch.cat([depth_full, grasp_pc], dim=1)
                    pc_feature = torch.cat(
                        [
                            torch.zeros(
                                [batch_size, num_points, 1]
                            ),
                            torch.ones(
                                [batch_size, grasp_pc.shape[1], 1]
                            ),
                        ],
                        dim=1,
                    ).to(device=device)
                    object_embedding = torch.cat([depth_full, pc_feature], dim=-1)
                    object_embedding = self.object_encoder(object_embedding)

                noise_pred = self.diffusion_head(obs_embedding, k, samples)
                if self.compositional_schedular:
                    res_pos = self.noise_scheduler_pos.step(
                        model_output=noise_pred[..., :3],
                        timestep=k,
                        sample=noisy_grasps[..., :3],
                    )
                    res_rot = self.noise_scheduler_rot.step(
                        model_output=noise_pred[..., 3 : self.output_dim],
                        timestep=k,
                        sample=noisy_grasps[..., 3 : self.output_dim],
                    )
                    if k > 0:
                        self._accumulate_inference_likelihood(
                            likelihood,
                            self._inference_likelihood(
                                noisy_grasps[..., :3],
                                res_pos.pred_original_sample,
                                k,
                                scales_pos,
                            ),
                            self._inference_likelihood(
                                noisy_grasps[..., 3 : self.output_dim],
                                res_rot.pred_original_sample,
                                k,
                                scales_rot,
                            ),
                        )
                    noisy_grasps = torch.hstack([res_pos.prev_sample, res_rot.prev_sample])
                else:
                    res = self.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=noisy_grasps
                    )
                    if k > 0:
                        self._accumulate_inference_likelihood(
                            likelihood,
                            self._inference_likelihood(
                                noisy_grasps, res.pred_original_sample, k, scales
                            ),
                        )
                    noisy_grasps = res.prev_sample

                pred_grasps = rt_to_matrix(noisy_grasps, self.grasp_repr, self.kappa)
                grasps_pred = pred_grasps.reshape(grasps_init_size)
                self._write_inference_history(grasps_per_iteration, k, grasps_pred)

        grasps_pred = pred_grasps.reshape(grasps_init_size)
        stats_batch = []
        if return_metrics:
            all_stats = []
            for i in range(num_objects_in_batch):
                grasps_pred_i = grasps_pred[i].cpu().numpy()
                grasps_gt_i = data["grasps_highres"][i].cpu().numpy()
                tree = KDTree(grasps_gt_i[:, :3, 3])
                dist, idx = tree.query(grasps_pred_i[:, :3, 3])
                matched = dist < 4.0
                grasps_pred_matched = grasps_pred_i[matched]
                grasps_gt_for_pred = grasps_gt_i[idx[matched]]
                gripper_depth = torch.tensor(
                    [data["gripper_depth"][i]], device=grasps_pred.device
                ).unsqueeze(0).repeat_interleave(len(grasps_gt_for_pred), dim=0)
                gripper_sym = torch.tensor(
                    [data["gripper_symmetry"][i]], device=grasps_pred.device
                ).unsqueeze(0).repeat_interleave(len(grasps_gt_for_pred), dim=0)
                stats = compute_metrics_given_two_sets_of_xgripper_poses(
                    torch.from_numpy(grasps_gt_for_pred).to(grasps_pred.device),
                    torch.from_numpy(grasps_pred_matched).to(grasps_pred.device),
                    gripper_depth,
                    gripper_sym,
                    consider_symmetry=True,
                )
                stats["recall"] = torch.tensor(
                    compute_recall(grasps_gt_i, grasps_pred_i)
                ).to(device)
                stats["precision"] = torch.tensor(
                    compute_recall(grasps_pred_i, grasps_gt_i)
                ).to(device)
                all_stats.append(stats)
            stats_batch = {
                key: torch.mean(torch.tensor([stats[key] for stats in all_stats]).to(device))
                for key in all_stats[0]
            }

        outputs = {
            "grasps_pred": grasps_pred,
            "grasps_per_iteration": grasps_per_iteration,
            "grasp_confidence": torch.zeros(grasps_pred.shape[:2]),
            "grasping_masks": torch.zeros(grasps_pred.shape[:2]),
            "grasp_contacts": torch.zeros(grasps_pred.shape[:2]),
            "instance_masks": torch.zeros(grasps_pred.shape[:2]),
            "likelihood": likelihood.reshape(num_objects_in_batch, num_grasps_per_batch, 1),
        }
        return outputs, {}, stats_batch
