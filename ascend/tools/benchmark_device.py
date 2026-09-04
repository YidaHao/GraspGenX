#!/usr/bin/env python3
"""Small, device-agnostic GraspGenX stability benchmark.

The script intentionally times only the complete ``GraspGen.infer`` call.  Model
construction/checkpoint loading and warmup are reported separately. It supports
released checkpoints or release-shaped random weights and uses a deterministic
synthetic point cloud across CUDA and torch-npu.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import types
import time
import importlib.machinery
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

# torch-npu must be loaded before diffusers/torch._dynamo on some Ascend
# images; otherwise the backend extension can attempt a second triton
# registration.  CUDA hosts simply skip this optional import.
try:  # pragma: no cover - hardware dependent
    import torch_npu  # noqa: F401
except Exception:
    pass


try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    class DictConfig(dict):
        """Minimal OmegaConf-compatible mapping for the benchmark config."""

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        __setattr__ = dict.__setitem__

    def _to_config(value):
        if isinstance(value, dict):
            return DictConfig({key: _to_config(item) for key, item in value.items()})
        if isinstance(value, list):
            return [_to_config(item) for item in value]
        return value

    class OmegaConf:
        @staticmethod
        def load(path):
            with Path(path).open() as stream:
                return _to_config(yaml.safe_load(stream))

    omegaconf_mod = types.ModuleType("omegaconf")
    omegaconf_mod.DictConfig = DictConfig
    omegaconf_mod.OmegaConf = OmegaConf
    sys.modules.setdefault("omegaconf", omegaconf_mod)

# The benchmark only needs the model, not dataset loading, mesh collision, or
# URDF support.  Keeping these tiny compatibility shims here lets the same
# script run in the deliberately minimal torch-npu image on 310P.
dataset_mod = types.ModuleType("graspgenx.dataset.dataset")
dataset_mod.MAPPING_ID2NAME = {}
sys.modules.setdefault("graspgenx.dataset.dataset", dataset_mod)
robot_mod = types.ModuleType("graspgenx.robot")
robot_mod.GripperInfo = object
robot_mod.get_gripper_info = lambda *a, **k: None
robot_mod.get_canonical_gripper_control_points = lambda w, d: np.zeros((4, 3), np.float32)
sys.modules.setdefault("graspgenx.robot", robot_mod)
eval_mod = types.ModuleType("graspgenx.dataset.eval_utils")
eval_mod.load_urdf_scene = lambda *a, **k: None
sys.modules.setdefault("graspgenx.dataset.eval_utils", eval_mod)
if importlib.util.find_spec("sklearn") is None:
    sklearn_mod = types.ModuleType("sklearn")
    sklearn_metrics_mod = types.ModuleType("sklearn.metrics")
    sklearn_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    sklearn_metrics_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
    sklearn_metrics_mod.average_precision_score = lambda *a, **k: 0.0
    sklearn_metrics_mod.roc_curve = lambda *a, **k: (np.array([]), np.array([]), np.array([]))
    sklearn_mod.metrics = sklearn_metrics_mod
    sys.modules.setdefault("sklearn", sklearn_mod)
    sys.modules.setdefault("sklearn.metrics", sklearn_metrics_mod)
# Metrics are imported by the model but are not evaluated by this benchmark.
# Keep the benchmark runnable in the minimal NPU image where trimesh is absent.
if importlib.util.find_spec("trimesh") is None:
    trimesh_mod = types.ModuleType("trimesh")
    transformations_mod = types.ModuleType("trimesh.transformations")
    transformations_mod.euler_matrix = lambda *a, **k: np.eye(4, dtype=np.float64)
    transformations_mod.translation_matrix = lambda xyz: np.array(
        [[1, 0, 0, xyz[0]], [0, 1, 0, xyz[1]], [0, 0, 1, xyz[2]], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    trimesh_mod.transformations = transformations_mod
    trimesh_mod.__spec__ = importlib.machinery.ModuleSpec("trimesh", loader=None)
    transformations_mod.__spec__ = importlib.machinery.ModuleSpec(
        "trimesh.transformations", loader=None
    )
    sys.modules.setdefault("trimesh", trimesh_mod)
    sys.modules.setdefault("trimesh.transformations", transformations_mod)
if importlib.util.find_spec("addict") is None:
    addict_mod = types.ModuleType("addict")

    class _AddictDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        __setattr__ = dict.__setitem__

    addict_mod.Dict = _AddictDict
    addict_mod.__spec__ = importlib.machinery.ModuleSpec("addict", loader=None)
    sys.modules.setdefault("addict", addict_mod)
layers_mod = types.ModuleType("timm.models.layers")
class _DropPath(torch.nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
    def forward(self, x):
        return x
layers_mod.DropPath = _DropPath
sys.modules.setdefault("timm.models.layers", layers_mod)

# Importing graspgenx normally bootstraps external assets. The benchmark uses
# its explicit CLI paths and checked-in gripper metadata, so disable downloads.
repo_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("GRASPGENX_GRIPPER_CFG_DIR", str(repo_root / "assets"))
os.environ.setdefault(
    "GRASPGENX_CHECKPOINT_DIR",
    str(repo_root / "ascend" / "sample-data" / "release-config"),
)

from graspgenx.models.grasp_gen import GraspGen


def resolve_device(name: str) -> torch.device:
    if name == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is not available")
        return torch.device("npu:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device("cuda:0")


def synchronize(device: torch.device) -> None:
    getattr(torch, device.type).synchronize(device)


def collate(items):
    out = {key: [item[key] for item in items] for key in items[0]}
    out["task_is_pick"] = torch.ones(len(items), dtype=torch.bool, device=items[0]["points"].device)
    out["task_is_place"] = torch.zeros(len(items), dtype=torch.bool, device=items[0]["points"].device)
    for key in ("inputs", "points", "sweep_volume_open_and_mid", "z_offset"):
        out[key] = torch.stack(out[key])
    out.pop("task", None)
    return out


def sync(device: torch.device) -> None:
    synchronize(device)


class StageTimer:
    """Synchronous wall-clock timer for nested model stages.

    Synchronizing at both boundaries is intentional: without it, accelerator
    kernels would be timed asynchronously and the CPU/NPU transfer cost would
    be attributed to a later stage (or to the final benchmark sync).
    """

    def __init__(self):
        self.samples = {}

    @contextmanager
    def section(self, name: str, device: torch.device):
        sync(device)
        started = time.perf_counter()
        try:
            yield
        finally:
            sync(device)
            self.samples.setdefault(name, []).append(time.perf_counter() - started)

    def reset(self):
        self.samples.clear()

    def summary(self):
        return {name: stats(values) for name, values in sorted(self.samples.items())}


class CPUEncoderBridge(torch.nn.Module):
    """Run a detached PTV3 encoder on CPU and return its embedding to NPU."""

    def __init__(self, encoder, output_device: torch.device):
        super().__init__()
        self.encoder = encoder.cpu().eval()
        self.output_device = output_device
        self.samples = []

    def forward(self, data_dict):
        sync(self.output_device)
        started = time.perf_counter()
        cpu_data = {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in data_dict.items()
        }
        embedding = self.encoder(cpu_data).to(self.output_device)
        sync(self.output_device)
        self.samples.append(time.perf_counter() - started)
        return embedding

    def reset(self):
        self.samples.clear()


class ConstantEncoder(torch.nn.Module):
    """Return a correctly shaped embedding to expose the non-PTV3 latency floor."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, data_dict):
        feat = data_dict["feat"]
        batch_size = data_dict["offset"].numel()
        return torch.zeros(
            (batch_size, self.output_dim), device=feat.device, dtype=feat.dtype
        )


class TimedModule(torch.nn.Module):
    """Profile an accelerator module with synchronized stage boundaries."""

    def __init__(self, module, timer: StageTimer, name: str, device: torch.device):
        super().__init__()
        self.module = module
        self.timer = timer
        self.name = name
        self.device = device

    def forward(self, *args, **kwargs):
        with self.timer.section(self.name, self.device):
            return self.module(*args, **kwargs)


def load_cfg(root: Path):
    gen_cfg = OmegaConf.load(root / "gen" / "config.yaml")
    dis_cfg = OmegaConf.load(root / "dis" / "config.yaml")
    cfg = dis_cfg
    cfg.diffusion = gen_cfg.diffusion
    cfg.eval.gen_checkpoint = str((root / "gen" / "epoch_736.pth").resolve())
    cfg.eval.dis_checkpoint = str((root / "dis" / "epoch_1056.pth").resolve())
    # Flash SDPA is CUDA-specific; the eager attention path is the fair common
    # denominator for CUDA and Ascend NPU.
    cfg.diffusion.ptv3vanilla.enable_flash = False
    cfg.discriminator.ptv3vanilla.enable_flash = False
    return cfg


def make_batch(
    cfg,
    gripper_name: str,
    assets_dir: Path,
    device: torch.device,
    point_count: int = 2048,
    input_npz: Path | None = None,
):
    rng = np.random.default_rng(1234)
    if input_npz is not None:
        golden = np.load(input_npz)
        points = np.asarray(golden["points"], dtype=np.float32)
        if "initial_noise" not in golden:
            raise ValueError(f"{input_npz} must contain initial_noise")
        initial_noise = np.asarray(golden["initial_noise"], dtype=np.float32)
    else:
        points = rng.normal(0.0, 0.05, size=(point_count, 3)).astype(np.float32)
        points -= points.mean(axis=0, keepdims=True)
        initial_noise = None
    pc = torch.from_numpy(points).to(device)
    # parallel_2f_v1_1014 values from its checked-in config.json.  The model
    # only consumes the 12-D open+mid sweep vector and z offset in this test.
    if gripper_name != "parallel_2f_v1_1014":
        raise ValueError("benchmark currently supports parallel_2f_v1_1014")
    gi_depth = 0.12679110012540068
    gi_sweep = np.array([0.08647013588950439, 0.015360593925194512, 0.047533627175553844, 0, 0, 0.16079110012540068], np.float32)
    gi_sweep_mid = np.array([0.043235067944752195, 0.015360593925194512, 0.047533627175553844, 0, 0, 0.16079110012540068], np.float32)
    sweep = torch.from_numpy(
        np.concatenate([gi_sweep, gi_sweep_mid]).astype(np.float32)
    ).to(device)
    item = {
        "task": "pick",
        "inputs": torch.cat([pc, torch.zeros_like(pc)], dim=-1),
        "points": pc,
        "sweep_volume_open_and_mid": sweep,
        "z_offset": torch.tensor([gi_depth], dtype=torch.float32, device=device),
    }
    batch = collate([item])
    if initial_noise is not None:
        batch["initial_noise"] = torch.from_numpy(initial_noise).to(device)
    return batch, points, initial_noise


def stats(values):
    return {
        "count": len(values),
        "mean_s": statistics.mean(values) if values else None,
        "median_s": statistics.median(values) if values else None,
        "p95_s": float(np.percentile(values, 95)) if values else None,
        "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
        "std_s": statistics.pstdev(values) if values else None,
        "hz": (1.0 / statistics.mean(values)) if values else None,
    }


def run_infer(model, batch, stage_timer=None):
    """Run a fresh end-to-end request without reusing discriminator embeddings.

    ``GraspGen.forward`` updates its input dictionary with discriminator outputs,
    including ``object_embedding``.  Reusing the same dictionary would turn all
    calls after the first into a cached-discriminator benchmark and would omit
    the second PTv3 encoder from steady-state latency.
    """
    batch.pop("object_embedding", None)
    if stage_timer is not None:
        device = batch["points"].device
        with stage_timer.section("generator.total", device):
            outputs, _, model_stats = model.grasp_generator.infer(
                batch, return_metrics=True
            )
        batch.update(outputs)
        batch["grasp_key"] = "grasps_pred"
        with stage_timer.section("discriminator.total", device):
            outputs, _, _ = model.grasp_discriminator.infer(batch)
        return outputs, {}, model_stats
    return model.infer(batch)


def validate_output(out, num_grasps: int):
    grasps = out[0]["grasps_pred"]
    confidence = out[0]["grasp_confidence"]
    if grasps.shape[1] != num_grasps:
        raise RuntimeError(f"unexpected grasp count: {grasps.shape}")
    grasps_np = grasps.detach().float().cpu().numpy()
    confidence_np = confidence.detach().float().cpu().numpy()
    if not np.isfinite(grasps_np).all():
        raise RuntimeError("non-finite grasp output")
    if not np.isfinite(confidence_np).all():
        raise RuntimeError("non-finite confidence output")
    return list(grasps.shape), list(confidence.shape)


def precision_context(device: torch.device, precision: str):
    """Return the device-native inference autocast context."""
    if precision == "fp32":
        return nullcontext()
    if device.type == "npu":
        return torch.npu.amp.autocast(dtype=torch.float16)
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "npu"], required=True)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--assets-dir", type=Path, required=True)
    ap.add_argument("--gripper", default="parallel_2f_v1_1014")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--num-grasps", type=int, default=100)
    ap.add_argument("--diffusion-steps", type=int, default=None)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="Inference autocast precision (default: fp32)")
    ap.add_argument("--point-count", type=int, default=2048)
    ap.add_argument("--input-npz", type=Path, default=None,
                    help="Reuse points and initial_noise saved by a CUDA golden run")
    ap.add_argument("--save-npz", type=Path, default=None,
                    help="Save points, initial_noise, grasps and confidence")
    ap.add_argument("--single-order", action="store_true",
                    help="Use one Morton-z serialization order for both CUDA and NPU smoke tests")
    encoder_group = ap.add_mutually_exclusive_group()
    encoder_group.add_argument("--ptv3-cpu", action="store_true",
                               help="Run complete PTv3 object encoders on CPU and transfer only embeddings")
    encoder_group.add_argument("--stub-ptv3", action="store_true",
                               help="Replace both PTv3 encoders with zero embeddings to measure the dense floor")
    ap.add_argument("--random-weights", action="store_true",
                    help="Build release-compatible random weights without loading checkpoint files")
    ap.add_argument("--cpu-threads", type=int, default=16,
                    help="CPU intra-op thread count (default: 16)")
    ap.add_argument("--profile-stages", action="store_true",
                    help="Record synchronized per-stage timings for generator/discriminator")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.warmup < 1:
        args.warmup = 1
    if args.cpu_threads < 1:
        ap.error("--cpu-threads must be at least 1")

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)

    device = resolve_device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    if args.ptv3_cpu:
        os.environ["GRASPGENX_PTV3_CPU_ENCODER"] = "1"

    t0 = time.perf_counter()
    cfg = load_cfg(args.checkpoint_root)
    if args.diffusion_steps is not None:
        cfg.diffusion.num_diffusion_iters_eval = int(args.diffusion_steps)
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    if not args.random_weights:
        model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)

    # Detach PTV3 before moving the remaining model to NPU. This avoids briefly
    # allocating both large encoders on the accelerator only to move them back.
    cpu_encoders = None
    if args.ptv3_cpu or args.stub_ptv3:
        cpu_encoders = (
            model.grasp_generator.object_encoder,
            model.grasp_discriminator.object_encoder,
        )
        model.grasp_generator.object_encoder = torch.nn.Identity()
        model.grasp_discriminator.object_encoder = torch.nn.Identity()
    model = model.to(device).eval()
    if args.ptv3_cpu:
        model.grasp_generator.object_encoder = CPUEncoderBridge(
            cpu_encoders[0], device
        ).eval()
        model.grasp_discriminator.object_encoder = CPUEncoderBridge(
            cpu_encoders[1], device
        ).eval()
    elif args.stub_ptv3:
        model.grasp_generator.object_encoder = ConstantEncoder(
            model.grasp_generator.num_object_dim
        ).eval()
        model.grasp_discriminator.object_encoder = ConstantEncoder(
            model.grasp_discriminator.num_object_dim
        ).eval()
    # Evaluation must map the same serialization order to the same attention
    # blocks on both devices.  Keep all four trained orders, but disable their
    # random permutation.  NPU executes their integer construction on CPU when
    # GRASPGENX_PTV3_CPU_SERIALIZATION=1 is set.
    for module in model.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = False
        if args.single_order and hasattr(module, "order"):
            module.order = ["z"]
    model.grasp_generator.num_grasps_per_object = args.num_grasps
    batch, points_np, initial_noise_np = make_batch(
        cfg, args.gripper, args.assets_dir, device,
        point_count=args.point_count, input_npz=args.input_npz,
    )
    if initial_noise_np is None:
        output_dim = 9 if str(cfg.diffusion.grasp_repr) == "r3_6d" else 6
        initial_noise_np = np.random.default_rng(5678).standard_normal(
            (args.num_grasps, output_dim), dtype=np.float32
        )
        batch["initial_noise"] = torch.from_numpy(initial_noise_np).to(device)
    stage_timer = StageTimer() if args.profile_stages else None
    if stage_timer is not None:
        model.grasp_generator.diffusion_head = TimedModule(
            model.grasp_generator.diffusion_head,
            stage_timer,
            "generator.diffusion_head",
            device,
        )
        model.grasp_discriminator.prediction_head = TimedModule(
            model.grasp_discriminator.prediction_head,
            stage_timer,
            "discriminator.prediction_head",
            device,
        )
    sync(device)
    load_s = time.perf_counter() - t0

    warmup_values = []
    for _ in range(args.warmup):
        sync(device)
        started = time.perf_counter()
        with torch.inference_mode(), precision_context(device, args.precision):
            out = run_infer(model, batch, stage_timer)
        sync(device)
        warmup_values.append(time.perf_counter() - started)
    grasp_shape, confidence_shape = validate_output(out, args.num_grasps)

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            points=points_np,
            initial_noise=initial_noise_np,
            grasps=out[0]["grasps_pred"].detach().float().cpu().numpy(),
            confidence=out[0]["grasp_confidence"].detach().float().cpu().numpy(),
        )

    # Warmup compilation is reported separately and excluded from stage
    # summaries, which represent steady-state calls only.
    if stage_timer is not None:
        stage_timer.reset()
    if args.ptv3_cpu:
        model.grasp_generator.object_encoder.reset()
        model.grasp_discriminator.object_encoder.reset()
    values = []
    failures = []
    started_window = time.perf_counter()
    while time.perf_counter() - started_window < args.duration:
        try:
            sync(device)
            started = time.perf_counter()
            with torch.inference_mode(), precision_context(device, args.precision):
                out = run_infer(model, batch, stage_timer)
            sync(device)
            elapsed = time.perf_counter() - started
            grasp_shape, confidence_shape = validate_output(out, args.num_grasps)
            values.append(elapsed)
        except Exception as exc:  # keep the failure evidence in the report
            failures.append({"time_s": time.perf_counter() - started_window, "error": repr(exc)})
            break

    report = {
        "device": str(device),
        "torch": torch.__version__,
        "precision": args.precision,
        "weights": "random" if args.random_weights else "checkpoint",
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "gripper": args.gripper,
        "point_count": int(len(points_np)),
        "num_grasps": args.num_grasps,
        "ptv3_encoder_device": (
            "cpu" if args.ptv3_cpu else "stub" if args.stub_ptv3 else str(device)
        ),
        "cpu_threads": torch.get_num_threads(),
        "diffusion_device": str(device),
        "diffusion_eval_steps": int(cfg.diffusion.num_diffusion_iters_eval),
        "grasp_output_shape": grasp_shape,
        "confidence_output_shape": confidence_shape,
        "outputs_finite": not failures,
        "load_and_setup_s": load_s,
        "warmup": stats(warmup_values),
        "warmup_samples_s": warmup_values,
        "stable_window_requested_s": args.duration,
        "stable_window_measured_s": time.perf_counter() - started_window,
        "stable_runs": stats(values),
        "stable_samples_s": values,
        "failures": failures,
    }
    if stage_timer is not None:
        report["stage_timings"] = stage_timer.summary()
    if args.ptv3_cpu:
        report["encoder_bridge_timings"] = {
            "generator": stats(model.grasp_generator.object_encoder.samples),
            "discriminator": stats(model.grasp_discriminator.object_encoder.samples),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
