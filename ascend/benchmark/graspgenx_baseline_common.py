"""Shared deterministic full-GraspGenX benchmark support."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import hashlib
import json
import os
import statistics
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


REPO_ROOT = Path(__file__).resolve().parents[2]
KAPPA = 3.27
GRID_SIZE = 0.01
PTV3_OUTPUT_DIM = 512
SWEEP_VOLUME = np.asarray(
    [
        0.08647013588950439, 0.015360593925194512, 0.047533627175553844,
        0.0, 0.0, 0.16079110012540068,
        0.043235067944752195, 0.015360593925194512, 0.047533627175553844,
        0.0, 0.0, 0.16079110012540068,
    ],
    dtype=np.float32,
)
GRIPPER_DEPTH = np.float32(0.12679110012540068)


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__


def _to_config(value):
    if isinstance(value, dict):
        return AttrDict({key: _to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_config(item) for item in value]
    return value


def install_import_shims() -> None:
    """Stub dependencies that are imported but unused by this benchmark."""
    if importlib.util.find_spec("omegaconf") is None:
        module = types.ModuleType("omegaconf")
        module.DictConfig = AttrDict

        class OmegaConf:
            @staticmethod
            def load(path):
                with Path(path).open() as stream:
                    return _to_config(yaml.safe_load(stream))

        module.OmegaConf = OmegaConf
        sys.modules["omegaconf"] = module

    if importlib.util.find_spec("addict") is None:
        module = types.ModuleType("addict")
        module.Dict = AttrDict
        module.__spec__ = importlib.machinery.ModuleSpec("addict", loader=None)
        sys.modules["addict"] = module

    if importlib.util.find_spec("timm") is None:
        module = types.ModuleType("timm.models.layers")

        class DropPath(torch.nn.Module):
            def __init__(self, drop_prob=0.0, *args, **kwargs):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, value):
                if self.training and self.drop_prob:
                    raise RuntimeError("DropPath shim is valid in eval mode only")
                return value

        module.DropPath = DropPath
        module.__spec__ = importlib.machinery.ModuleSpec(
            "timm.models.layers", loader=None
        )
        sys.modules["timm.models.layers"] = module

    dataset = types.ModuleType("graspgenx.dataset.dataset")
    dataset.MAPPING_ID2NAME = {}
    sys.modules.setdefault("graspgenx.dataset.dataset", dataset)

    robot = types.ModuleType("graspgenx.robot")
    robot.GripperInfo = object
    robot.get_gripper_info = lambda *args, **kwargs: None
    robot.get_canonical_gripper_control_points = (
        lambda *args, **kwargs: np.zeros((4, 3), dtype=np.float32)
    )
    sys.modules.setdefault("graspgenx.robot", robot)

    evaluation = types.ModuleType("graspgenx.dataset.eval_utils")
    evaluation.load_urdf_scene = lambda *args, **kwargs: None
    sys.modules.setdefault("graspgenx.dataset.eval_utils", evaluation)

    if importlib.util.find_spec("sklearn") is None:
        sklearn = types.ModuleType("sklearn")
        metrics = types.ModuleType("sklearn.metrics")
        sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        metrics.__spec__ = importlib.machinery.ModuleSpec(
            "sklearn.metrics", loader=None
        )
        metrics.average_precision_score = lambda *args, **kwargs: 0.0
        metrics.roc_curve = lambda *args, **kwargs: (
            np.asarray([]), np.asarray([]), np.asarray([])
        )
        sklearn.metrics = metrics
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.metrics"] = metrics

    if importlib.util.find_spec("trimesh") is None:
        trimesh = types.ModuleType("trimesh")
        transformations = types.ModuleType("trimesh.transformations")
        transformations.euler_matrix = lambda *args, **kwargs: np.eye(4)
        transformations.translation_matrix = lambda xyz: np.asarray(
            [[1, 0, 0, xyz[0]], [0, 1, 0, xyz[1]],
             [0, 0, 1, xyz[2]], [0, 0, 0, 1]], dtype=np.float64
        )
        trimesh.transformations = transformations
        trimesh.__spec__ = importlib.machinery.ModuleSpec("trimesh", loader=None)
        transformations.__spec__ = importlib.machinery.ModuleSpec(
            "trimesh.transformations", loader=None
        )
        sys.modules["trimesh"] = trimesh
        sys.modules["trimesh.transformations"] = transformations


os.environ.setdefault("GRASPGENX_GRIPPER_CFG_DIR", str(REPO_ROOT / "assets"))
_checkpoint_parent = REPO_ROOT / ".artifacts/checkpoints"
if not _checkpoint_parent.is_dir():
    _checkpoint_parent = REPO_ROOT / "ascend"
os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", str(_checkpoint_parent))
install_import_shims()

from omegaconf import OmegaConf
from graspgenx.models.grasp_gen import GraspGen


def synchronize(device: torch.device) -> None:
    getattr(torch, device.type).synchronize(device)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in (
            "count", "mean_ms", "median_ms", "p95_ms", "p99_ms",
            "min_ms", "max_ms", "std_ms",
        )}
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


def load_config(checkpoint_root: Path, runtime: str, cuda_ptv3_flash: bool = True):
    generator = OmegaConf.load(checkpoint_root / "gen/config.yaml")
    discriminator = OmegaConf.load(checkpoint_root / "dis/config.yaml")
    cfg = discriminator
    cfg.diffusion = generator.diffusion
    cfg.eval.gen_checkpoint = str(checkpoint_root / "gen/epoch_736.pth")
    cfg.eval.dis_checkpoint = str(checkpoint_root / "dis/epoch_1056.pth")
    flash = runtime == "cuda" and cuda_ptv3_flash
    cfg.diffusion.ptv3vanilla.enable_flash = flash
    cfg.discriminator.ptv3vanilla.enable_flash = flash
    return cfg


def _replacement_encoder(source, encoder_class):
    encoder = encoder_class(
        in_channels=3,
        output_dim=PTV3_OUTPUT_DIM,
        grid_size=GRID_SIZE,
        enable_flash=False,
        shuffle_orders=False,
    )
    encoder.load_state_dict(source.state_dict(), strict=True)
    for module in encoder.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = False
    return encoder.eval()


class EncoderDeviceBridge(torch.nn.Module):
    """Adapt the current CPU-bound PTV3 interface to an NPU pipeline."""

    def __init__(self, encoder, output_device):
        super().__init__()
        self.encoder = encoder
        self.output_device = output_device

    def forward(self, data):
        cpu_data = {
            key: value.cpu() if torch.is_tensor(value) else value
            for key, value in data.items()
        }
        return self.encoder(cpu_data).to(self.output_device)


def build_model(
    checkpoint_root: Path,
    runtime: str,
    encoder_class=None,
    cuda_acceleration: str = "native",
    cuda_ptv3_flash: bool = True,
):
    device = torch.device("cuda:0" if runtime == "cuda" else "npu:0")
    if runtime == "npu":
        if torch_npu is None or not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable")
        torch.npu.set_device(device)
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    cfg = load_config(checkpoint_root, runtime, cuda_ptv3_flash)
    model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
    model.grasp_generator.num_grasps_per_object = 100

    if runtime == "npu":
        if encoder_class is None:
            raise ValueError("NPU runtime requires an explicit PTV3 encoder class")
        generator_encoder = _replacement_encoder(
            model.grasp_generator.object_encoder, encoder_class
        )
        discriminator_encoder = _replacement_encoder(
            model.grasp_discriminator.object_encoder, encoder_class
        )
        model.grasp_generator.object_encoder = torch.nn.Identity()
        model.grasp_discriminator.object_encoder = torch.nn.Identity()
        model = model.to(device).eval()
        model.grasp_generator.object_encoder = EncoderDeviceBridge(
            generator_encoder, device
        ).eval()
        model.grasp_discriminator.object_encoder = EncoderDeviceBridge(
            discriminator_encoder, device
        ).eval()
    else:
        model = model.to(device).eval()

    for module in model.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = False

    acceleration = {"requested": cuda_acceleration, "active": "native"}
    if runtime == "cuda" and cuda_acceleration == "tensorrt_fp16":
        from graspgenx.models.tensorrt_utils import accelerate_sampler

        holder = SimpleNamespace(model=model)
        if accelerate_sampler(holder, precision="fp16"):
            acceleration["active"] = "tensorrt_fp16"
        else:
            acceleration["fallback"] = "native"
    elif cuda_acceleration != "native":
        raise ValueError(f"unsupported acceleration: {cuda_acceleration}")

    synchronize(device)
    return model, cfg, device, acceleration


def create_request(
    path: Path,
    ptv3_reference: Path,
    num_grasps: int = 100,
    diffusion_steps: int = 20,
) -> None:
    with np.load(ptv3_reference) as reference:
        points = np.asarray(reference["points"], dtype=np.float32) / np.float32(
            reference["kappa"]
        )
    initial = np.random.default_rng(5678).standard_normal(
        (num_grasps, 6), dtype=np.float32
    )
    nonzero_steps = diffusion_steps - 1
    position_noise = np.random.default_rng(6789).standard_normal(
        (nonzero_steps, num_grasps, 3), dtype=np.float32
    )
    rotation_noise = np.random.default_rng(7890).standard_normal(
        (nonzero_steps, num_grasps, 3), dtype=np.float32
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        points=points,
        initial_noise=initial,
        position_noise=position_noise,
        rotation_noise=rotation_noise,
        sweep_volume=SWEEP_VOLUME,
        gripper_depth=GRIPPER_DEPTH,
        num_grasps=np.int64(num_grasps),
        diffusion_steps=np.int64(diffusion_steps),
        kappa=np.float32(KAPPA),
        grid_size=np.float32(GRID_SIZE),
    )


def load_request(path: Path) -> dict[str, np.ndarray | int | float]:
    with np.load(path) as request:
        result = {key: np.asarray(request[key]).copy() for key in request.files}
    result["num_grasps"] = int(result["num_grasps"])
    result["diffusion_steps"] = int(result["diffusion_steps"])
    return result


def make_batch(request: dict, device: torch.device) -> dict:
    points = torch.from_numpy(request["points"]).to(device=device, dtype=torch.float32)
    return {
        "task": ["pick"],
        "inputs": torch.cat([points, torch.zeros_like(points)], dim=-1).unsqueeze(0),
        "points": points.unsqueeze(0),
        "sweep_volume_open_and_mid": torch.from_numpy(
            request["sweep_volume"]
        ).to(device=device, dtype=torch.float32).unsqueeze(0),
        "z_offset": torch.tensor(
            [[float(request["gripper_depth"])]], device=device, dtype=torch.float32
        ),
        "task_is_pick": torch.ones(1, dtype=torch.bool, device=device),
        "task_is_place": torch.zeros(1, dtype=torch.bool, device=device),
        "initial_noise": torch.from_numpy(request["initial_noise"]).to(
            device=device, dtype=torch.float32
        ),
    }


class StageTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.samples: dict[str, list[float]] = {}

    @contextmanager
    def section(self, name: str):
        synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.samples.setdefault(name, []).append(
                (time.perf_counter() - started) * 1000.0
            )

    def reset(self):
        self.samples.clear()

    def summary(self):
        return {name: distribution(values) for name, values in self.samples.items()}

    def request_summary(self, runs: int, diffusion_steps: int):
        grouped = {}
        specifications = {
            "generator.ptv3": 1,
            "generator.gripper_encoder": 1,
            "generator.diffusion_head": diffusion_steps,
            "generator.scheduler_position": diffusion_steps,
            "generator.scheduler_rotation": diffusion_steps,
            "generator.pose_conversion": diffusion_steps,
            "generator.likelihood_log_prob": 2 * (diffusion_steps - 1),
            "discriminator.ptv3": 1,
            "discriminator.pose_conversion": 1,
            "discriminator.sample_encoder": 1,
            "discriminator.gripper_encoder": 1,
            "discriminator.prediction_head": 1,
        }
        for name, calls_per_request in specifications.items():
            samples = self.samples.get(name, [])
            expected = runs * calls_per_request
            if len(samples) != expected:
                raise RuntimeError(
                    f"{name} recorded {len(samples)} calls; expected {expected}"
                )
            grouped[name] = [
                sum(samples[index:index + calls_per_request])
                for index in range(0, len(samples), calls_per_request)
            ]

        for parent, children in (
            ("generator.total", tuple(name for name in grouped if name.startswith("generator."))),
            ("discriminator.total", tuple(name for name in grouped if name.startswith("discriminator."))),
        ):
            totals = self.samples[parent]
            if len(totals) != runs:
                raise RuntimeError(f"{parent} recorded {len(totals)} calls; expected {runs}")
            grouped[parent] = totals
            grouped[parent.replace(".total", ".unattributed")] = [
                totals[index] - sum(grouped[child][index] for child in children)
                for index in range(runs)
            ]
        grouped["pipeline.total"] = [
            grouped["generator.total"][index] + grouped["discriminator.total"][index]
            for index in range(runs)
        ]
        return {name: distribution(values) for name, values in grouped.items()}


class TimedModule(torch.nn.Module):
    def __init__(self, module, timer: StageTimer, name: str):
        super().__init__()
        self.module = module
        self.timer = timer
        self.name = name

    def forward(self, *args, **kwargs):
        with self.timer.section(self.name):
            return self.module(*args, **kwargs)


@contextmanager
def install_function_timers(timer: StageTimer):
    import graspgenx.models.discriminator as discriminator_module
    import graspgenx.models.generator as generator_module

    original_rt_to_matrix = generator_module.rt_to_matrix
    original_matrix_to_rt = discriminator_module.matrix_to_rt
    original_log_prob = torch.distributions.Normal.log_prob

    def rt_to_matrix(*args, **kwargs):
        with timer.section("generator.pose_conversion"):
            return original_rt_to_matrix(*args, **kwargs)

    def matrix_to_rt(*args, **kwargs):
        with timer.section("discriminator.pose_conversion"):
            return original_matrix_to_rt(*args, **kwargs)

    def log_prob(distribution, *args, **kwargs):
        with timer.section("generator.likelihood_log_prob"):
            return original_log_prob(distribution, *args, **kwargs)

    generator_module.rt_to_matrix = rt_to_matrix
    discriminator_module.matrix_to_rt = matrix_to_rt
    torch.distributions.Normal.log_prob = log_prob
    try:
        yield
    finally:
        generator_module.rt_to_matrix = original_rt_to_matrix
        discriminator_module.matrix_to_rt = original_matrix_to_rt
        torch.distributions.Normal.log_prob = original_log_prob


class ReplayScheduler:
    """Version-independent DDPM step using explicit frozen variance noise."""

    def __init__(self, scheduler, noise: np.ndarray, timer=None, name=None):
        self.scheduler = scheduler
        self.noise = noise
        self.timer = timer
        self.name = name
        self.cursor = 0
        self.record = False
        self.samples = []

    def reset(self, record=False):
        self.cursor = 0
        self.record = record
        self.samples = []

    def step(self, model_output, timestep, sample, *args, **kwargs):
        context = self.timer.section(self.name) if self.timer else _null_context()
        with context:
            result = self._step(model_output, timestep, sample)
        if self.record:
            self.samples.append(result.prev_sample.detach().float().cpu().numpy())
        return result

    def _step(self, model_output, timestep, sample):
        t = int(timestep)
        scheduler = self.scheduler
        alpha_t = scheduler.alphas_cumprod[t].to(sample.device, sample.dtype)
        alpha_prev = (
            scheduler.alphas_cumprod[t - 1]
            if t > 0 else scheduler.one
        ).to(sample.device, sample.dtype)
        beta_t = 1 - alpha_t
        beta_prev = 1 - alpha_prev
        current_beta = scheduler.betas[t].to(sample.device, sample.dtype)
        original = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
        if scheduler.config.clip_sample:
            original = original.clamp(-1, 1)
        original_coeff = alpha_prev.sqrt() * current_beta / beta_t
        sample_coeff = scheduler.alphas[t].to(
            sample.device, sample.dtype
        ).sqrt() * beta_prev / beta_t
        previous = original_coeff * original + sample_coeff * sample
        if t > 0:
            if self.cursor >= len(self.noise):
                raise RuntimeError("scheduler noise package exhausted")
            noise = torch.from_numpy(self.noise[self.cursor]).to(
                sample.device, sample.dtype
            )
            self.cursor += 1
            variance = scheduler._get_variance(t).to(sample.device, sample.dtype)
            previous = previous + variance.sqrt() * noise
        return SimpleNamespace(prev_sample=previous, pred_original_sample=original)


@contextmanager
def _null_context():
    yield


class Pipeline:
    def __init__(self, model, request: dict, device: torch.device):
        self.model = model
        self.request = request
        self.device = device
        generator = model.grasp_generator
        self.position = ReplayScheduler(
            generator.noise_scheduler_pos, request["position_noise"]
        )
        self.rotation = ReplayScheduler(
            generator.noise_scheduler_rot, request["rotation_noise"]
        )
        generator.noise_scheduler_pos.step = self.position.step
        generator.noise_scheduler_rot.step = self.rotation.step

    def set_timer(self, timer: StageTimer | None):
        self.position.timer = timer
        self.position.name = "generator.scheduler_position"
        self.rotation.timer = timer
        self.rotation.name = "generator.scheduler_rotation"

    def run(self, record=False, timer: StageTimer | None = None):
        self.position.reset(record=record)
        self.rotation.reset(record=record)
        batch = make_batch(self.request, self.device)
        traces = {"generator_embedding": [], "discriminator_embedding": [],
                  "noise_prediction": [], "logits": []}
        hooks = []
        if record:
            def capture(name):
                def hook(_module, _inputs, output):
                    traces[name].append(output.detach().float().cpu().numpy())
                return hook

            hooks = [
                self.model.grasp_generator.object_encoder.register_forward_hook(
                    capture("generator_embedding")
                ),
                self.model.grasp_generator.diffusion_head.register_forward_hook(
                    capture("noise_prediction")
                ),
                self.model.grasp_discriminator.object_encoder.register_forward_hook(
                    capture("discriminator_embedding")
                ),
                self.model.grasp_discriminator.prediction_head.register_forward_hook(
                    capture("logits")
                ),
            ]
        try:
            with torch.inference_mode():
                with fixed_initial_noise(self.request, self.device):
                    if timer:
                        with timer.section("generator.total"):
                            outputs, _, _ = self.model.grasp_generator.infer(
                                batch, return_metrics=True
                            )
                    else:
                        outputs, _, _ = self.model.grasp_generator.infer(
                            batch, return_metrics=True
                        )
                batch.update(outputs)
                batch["grasp_key"] = "grasps_pred"
                if timer:
                    with timer.section("discriminator.total"):
                        outputs, _, _ = self.model.grasp_discriminator.infer(batch)
                else:
                    outputs, _, _ = self.model.grasp_discriminator.infer(batch)
            synchronize(self.device)
        finally:
            for hook in hooks:
                hook.remove()

        expected = self.request["diffusion_steps"] - 1
        if self.position.cursor != expected or self.rotation.cursor != expected:
            raise RuntimeError(
                f"unexpected scheduler noise use: position={self.position.cursor}, "
                f"rotation={self.rotation.cursor}, expected={expected}"
            )
        return outputs, traces


@contextmanager
def fixed_initial_noise(request: dict, device: torch.device):
    """Support upstream checkouts predating the explicit initial_noise input."""
    original = torch.randn
    target_shape = tuple(request["initial_noise"].shape)
    used = False

    def replay_randn(*args, **kwargs):
        nonlocal used
        shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else tuple(args)
        if not used and shape == target_shape:
            used = True
            dtype = kwargs.get("dtype", torch.float32)
            target = kwargs.get("device", device)
            return torch.from_numpy(request["initial_noise"]).to(target, dtype=dtype)
        return original(*args, **kwargs)

    torch.randn = replay_randn
    try:
        yield
    finally:
        torch.randn = original


def install_stage_timers(model, timer: StageTimer) -> None:
    generator = model.grasp_generator
    discriminator = model.grasp_discriminator
    for owner, attribute, name in (
        (generator, "object_encoder", "generator.ptv3"),
        (generator, "gripper_encoder", "generator.gripper_encoder"),
        (generator, "diffusion_head", "generator.diffusion_head"),
        (discriminator, "object_encoder", "discriminator.ptv3"),
        (discriminator, "sample_encoder", "discriminator.sample_encoder"),
        (discriminator, "gripper_encoder", "discriminator.gripper_encoder"),
        (discriminator, "prediction_head", "discriminator.prediction_head"),
    ):
        setattr(owner, attribute, TimedModule(getattr(owner, attribute), timer, name))


def output_arrays(outputs, traces, pipeline: Pipeline) -> dict[str, np.ndarray]:
    result = {
        "grasps": outputs["grasps_pred"].detach().float().cpu().numpy(),
        "confidence": outputs["grasp_confidence"].detach().float().cpu().numpy(),
        "likelihood": outputs["likelihood"].detach().float().cpu().numpy(),
        "grasps_per_iteration": outputs["grasps_per_iteration"].detach().float().cpu().numpy(),
    }
    for name, values in traces.items():
        if values:
            result[name] = np.stack(values)
    if pipeline.position.samples and pipeline.rotation.samples:
        result["diffusion_latent"] = np.concatenate(
            [np.stack(pipeline.position.samples), np.stack(pipeline.rotation.samples)],
            axis=-1,
        )
    return result


def run_timed(pipeline: Pipeline, warmup: int, runs: int) -> tuple[list[float], dict]:
    warmups = []
    for _ in range(warmup):
        synchronize(pipeline.device)
        started = time.perf_counter()
        pipeline.run()
        synchronize(pipeline.device)
        warmups.append((time.perf_counter() - started) * 1000.0)
    values = []
    for _ in range(runs):
        synchronize(pipeline.device)
        started = time.perf_counter()
        pipeline.run()
        synchronize(pipeline.device)
        values.append((time.perf_counter() - started) * 1000.0)
    return values, {"warmup": distribution(warmups), "steady": distribution(values)}


def cosine(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    a = reference.astype(np.float64).reshape(-1)
    b = candidate.astype(np.float64).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator > 0 else None


def compare_array(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {"shape_match": False, "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape), "cosine": None,
                "valid": False}
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        return {"shape_match": True, "finite": False, "cosine": None, "valid": False}
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    value = cosine(reference, candidate)
    return {
        "shape_match": True,
        "finite": bool(np.isfinite(candidate).all()),
        "cosine": value,
        "max_abs": float(np.abs(difference).max()),
        "mean_abs": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_l2": float(
            np.linalg.norm(difference) /
            max(np.linalg.norm(reference.astype(np.float64)), 1e-12)
        ),
        "valid": True,
    }


def output_errors(outputs: dict) -> list[str]:
    """Check the batch-one pipeline contract, not closeness to another backend."""
    required = (
        "grasps", "confidence", "likelihood", "grasps_per_iteration",
        "generator_embedding", "discriminator_embedding", "noise_prediction",
        "logits", "diffusion_latent",
    )
    errors = [f"missing {name}" for name in required if name not in outputs]
    if errors:
        return errors
    grasps, predictions = outputs["grasps"], outputs["noise_prediction"]
    if grasps.ndim != 4 or grasps.shape[0] != 1 or grasps.shape[2:] != (4, 4) or grasps.shape[1] < 1:
        return [f"invalid grasp shape: {grasps.shape}"]
    if predictions.ndim != 3 or predictions.shape[0] < 1:
        return [f"invalid noise prediction shape: {predictions.shape}"]
    count, steps = grasps.shape[1], predictions.shape[0]
    shapes = {
        "confidence": (1, count, 1), "likelihood": (1, count, 1),
        "grasps_per_iteration": (1, steps, count, 4, 4),
        "generator_embedding": (1, 1, 512), "discriminator_embedding": (1, 1, 512),
        "noise_prediction": (steps, count, 6), "diffusion_latent": (steps, count, 6),
        "logits": (1, count, 1),
    }
    for name, shape in shapes.items():
        if outputs[name].shape != shape:
            errors.append(f"{name}: expected {shape}, got {outputs[name].shape}")
    for name in required:
        if not np.isfinite(outputs[name]).all():
            errors.append(f"{name}: non-finite values")
    if not np.allclose(grasps[..., 3, :], [0, 0, 0, 1]):
        errors.append("invalid homogeneous row")
    return errors


def compare_outputs(reference: dict, candidate: dict) -> dict:
    errors = [f"reference: {error}" for error in output_errors(reference)]
    errors += [f"candidate: {error}" for error in output_errors(candidate)]
    if not errors:
        errors += [f"shape mismatch: {key}" for key in reference
                   if key in candidate and reference[key].shape != candidate[key].shape]
    if errors:
        return {"valid": False, "numerical_policy": "report_only", "errors": errors}
    components = {
        "generator_embedding": (reference["generator_embedding"], candidate["generator_embedding"]),
        "noise_prediction": (reference["noise_prediction"], candidate["noise_prediction"]),
        "diffusion_latent": (reference["diffusion_latent"], candidate["diffusion_latent"]),
        "translation": (reference["grasps"][..., :3, 3], candidate["grasps"][..., :3, 3]),
        "rotation": (reference["grasps"][..., :3, :3], candidate["grasps"][..., :3, :3]),
        "discriminator_embedding": (reference["discriminator_embedding"], candidate["discriminator_embedding"]),
        "logits": (reference["logits"], candidate["logits"]),
        "confidence": (reference["confidence"], candidate["confidence"]),
        "likelihood": (reference["likelihood"], candidate["likelihood"]),
    }
    report = {name: compare_array(*arrays) for name, arrays in components.items()}
    report["noise_prediction_steps"] = [
        compare_array(expected, actual)
        for expected, actual in zip(
            reference["noise_prediction"], candidate["noise_prediction"]
        )
    ]
    report["diffusion_latent_steps"] = [
        compare_array(expected, actual)
        for expected, actual in zip(
            reference["diffusion_latent"], candidate["diffusion_latent"]
        )
    ]
    grasps = candidate["grasps"].astype(np.float64)
    rotations = grasps[..., :3, :3]
    reference_grasps = reference["grasps"].astype(np.float64)
    translation_error = np.linalg.norm(
        grasps[..., :3, 3] - reference_grasps[..., :3, 3], axis=-1
    )
    relative_rotation = np.swapaxes(
        reference_grasps[..., :3, :3], -1, -2
    ) @ rotations
    rotation_cosine = np.clip(
        (np.trace(relative_rotation, axis1=-2, axis2=-1) - 1) / 2,
        -1,
        1,
    )
    rotation_degrees = np.degrees(np.arccos(rotation_cosine))
    reference_scores = reference["confidence"].reshape(-1)
    candidate_scores = candidate["confidence"].reshape(-1)
    topk_overlap = {}
    for count in (1, 5, 10, 20):
        count = min(count, len(reference_scores))
        expected = set(np.argsort(-reference_scores)[:count].tolist())
        actual = set(np.argsort(-candidate_scores)[:count].tolist())
        topk_overlap[str(count)] = len(expected & actual) / count
    report["pose_error_report_only"] = {
        "translation_mean_mm": float(translation_error.mean() * 1000),
        "translation_p95_mm": float(np.percentile(translation_error, 95) * 1000),
        "translation_max_mm": float(translation_error.max() * 1000),
        "rotation_mean_deg": float(rotation_degrees.mean()),
        "rotation_p95_deg": float(np.percentile(rotation_degrees, 95)),
        "rotation_max_deg": float(rotation_degrees.max()),
    }
    report["ranking_report_only"] = {"topk_overlap": topk_overlap}
    report["validity"] = {
        "finite": bool(all(np.isfinite(value).all() for value in candidate.values())),
        "homogeneous_row": bool(np.allclose(grasps[..., 3, :], [0, 0, 0, 1])),
        "max_rotation_orthogonality_error": float(
            np.abs(rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3)).max()
        ),
        "rotation_determinant_min": float(np.linalg.det(rotations).min()),
        "rotation_determinant_max": float(np.linalg.det(rotations).max()),
    }
    report["valid"] = True
    report["numerical_policy"] = "report_only"
    return report


CUDA_BASELINE_NAMES = ("reference", "native", "trt")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def write_cuda_baselines(sources: dict[str, Path], destination: Path) -> None:
    """Freeze three existing runs together; never replace an existing bundle."""
    array_path = destination / "cuda_baselines.npz"
    json_path = destination / "cuda_baselines.json"
    if array_path.exists() or json_path.exists():
        raise FileExistsError(f"CUDA comparison bundle already exists: {destination}")
    outputs, arrays, entries = {}, {}, {}
    request = None
    for name in CUDA_BASELINE_NAMES:
        source = sources[name]
        current = load_arrays(source / "request.npz")
        if request is None:
            request = current
            arrays.update({f"request__{key}": value for key, value in request.items()})
        elif current.keys() != request.keys() or any(
            current[key].dtype != request[key].dtype
            or not np.array_equal(current[key], request[key]) for key in request
        ):
            raise ValueError(f"{name} does not use the identical frozen request")
        outputs[name] = load_arrays(source / "cuda_outputs.npz")
        errors = output_errors(outputs[name])
        if errors:
            raise ValueError(f"invalid CUDA {name} outputs: {errors}")
        arrays.update({f"{name}__{key}": value for key, value in outputs[name].items()})
        summary = json.loads((source / "summary.json").read_text())
        # Keep original measurements/config, but do not inherit obsolete gates.
        entries[name] = {
            "source_directory": str(source),
            "source_sha256": {
                filename: file_sha256(source / filename)
                for filename in ("request.npz", "cuda_outputs.npz", "summary.json")
            },
            "configuration": {key: summary[key] for key in (
                "runtime", "device", "torch", "cuda", "acceleration", "tf32",
                "ptv3_flash", "deterministic_algorithms", "point_count",
                "num_grasps", "diffusion_steps",
            ) if key in summary},
            "timings": summary["timings"],
            "samples_ms": summary["samples_ms"],
        }
    for name in CUDA_BASELINE_NAMES:
        comparison = compare_outputs(outputs["reference"], outputs[name])
        if not comparison["valid"]:
            raise ValueError(f"CUDA {name} contract mismatch: {comparison}")
        entries[name]["vs_reference"] = comparison
    metadata = {
        "schema_version": 1,
        "numerical_policy": "report_only",
        "note": "Valid data is not an accuracy acceptance. Timings describe the selected frozen run, not a multi-process aggregate.",
        "baselines": entries,
    }
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(array_path, **arrays)
    metadata["arrays_sha256"] = file_sha256(array_path)
    save_json(json_path, metadata)


def load_cuda_baselines(directory: Path) -> tuple[dict, dict, dict]:
    metadata = json.loads((directory / "cuda_baselines.json").read_text())
    array_path = directory / "cuda_baselines.npz"
    if metadata["schema_version"] != 1 or file_sha256(array_path) != metadata["arrays_sha256"]:
        raise ValueError("Invalid CUDA comparison bundle version/checksum")
    arrays = load_arrays(array_path)
    groups = {
        name: {key.split("__", 1)[1]: value for key, value in arrays.items()
               if key.startswith(f"{name}__")}
        for name in ("request", *CUDA_BASELINE_NAMES)
    }
    request = groups.pop("request")
    request["num_grasps"] = int(request["num_grasps"])
    request["diffusion_steps"] = int(request["diffusion_steps"])
    for name, output in groups.items():
        errors = output_errors(output)
        if errors:
            raise ValueError(f"invalid CUDA {name}: {errors}")
        if output["grasps"].shape[1] != int(request["num_grasps"]):
            errors.append("request/output grasp count mismatch")
        if output["noise_prediction"].shape[0] != int(request["diffusion_steps"]):
            errors.append("request/output step count mismatch")
        if errors:
            raise ValueError(f"invalid CUDA {name}: {errors}")
        if not compare_outputs(groups["reference"], output)["valid"]:
            raise ValueError(f"CUDA {name} does not match the reference output contract")
    return request, groups, metadata


def print_cuda_comparisons(metadata: dict, comparisons: dict) -> None:
    def value(report, component):
        metric = report.get(component, {}).get("cosine")
        return f"{metric:.8f}" if metric is not None else "n/a"

    print("NUMERICAL COMPARISON: REPORT ONLY (no cosine threshold)", flush=True)
    print("CUDA baseline      frozen-run ms   rotation/ref   confidence/ref", flush=True)
    for name in CUDA_BASELINE_NAMES:
        baseline = metadata["baselines"][name]
        print(f"{name:18s} {baseline['timings']['steady']['median_ms']:12.3f}   "
              f"{value(baseline['vs_reference'], 'rotation'):14s} "
              f"{value(baseline['vs_reference'], 'confidence')}", flush=True)
    print("Candidate versus   translation    rotation       confidence", flush=True)
    for name, metrics in comparisons.items():
        print(f"{name:18s} {value(metrics, 'translation'):14s} "
              f"{value(metrics, 'rotation'):14s} {value(metrics, 'confidence')}", flush=True)
        if not metrics["valid"]:
            print(f"  INVALID: {metrics.get('errors')}", flush=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
