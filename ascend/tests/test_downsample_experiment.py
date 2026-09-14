"""Validation for benchmark-only downsampling, not a new production model."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ascend.benchmark import downsample_experiment as experiment
from ascend.benchmark import validate_ptv3 as validation
from graspgenx.models.ptv3.ptv3_vanilla import VanillaPoint, VanillaSerializedPooling

METADATA = ("grid_coord", "batch", "offset", "serialized_code", "serialized_order", "serialized_inverse")
TEST_NPU = os.environ.get("ASCEND_TEST_NPU") == "1"


def fixture(n=65):
    torch.manual_seed(7)
    grid = torch.randint(0, 8, (n, 3), dtype=torch.int32)
    if n > 4:
        grid[1:4] = grid[0]
    point = VanillaPoint(coord=grid.float() * .01, grid_coord=grid,
                         feat=torch.randn(n, 32), batch=torch.arange(n).long() % 2)
    point.serialization(shuffle_orders=False)
    pool = VanillaSerializedPooling(32, 64, stride=2, norm_layer=torch.nn.BatchNorm1d,
                                   act_layer=torch.nn.GELU, traceable=False).eval()
    stage = torch.nn.Module()
    stage.add_module("down", pool)
    return point, pool, SimpleNamespace(enc=[stage])


class DownsampleCPUTests(unittest.TestCase):
    def test_cpu_replay_and_coordinate_removal(self):
        with torch.inference_mode():
            for n in (1, 17, 65):
                point, pool, _ = fixture(n)
                for shuffled in (False, True):
                    pool.shuffle_orders = shuffled
                    torch.manual_seed(33)
                    reference = pool(point)
                    expected_rng = torch.random.get_rng_state()
                    for remove in (False, True):
                        torch.manual_seed(33)
                        actual = experiment.forward(pool, point, remove_coords=remove)
                        for key in METADATA:
                            self.assertTrue(torch.equal(reference[key], actual[key]), key)
                        self.assertTrue(torch.equal(reference.feat, actual.feat))
                        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
                        self.assertEqual("coord" in actual, not remove)
                        if not remove:
                            self.assertTrue(torch.equal(reference.coord, actual.coord))

    def test_no_coordinate_path_rejects_traceable_contract(self):
        point, pool, _ = fixture()
        pool.traceable = True
        with self.assertRaisesRegex(RuntimeError, "non-traceable"):
            experiment.forward(pool, point, remove_coords=True)


@unittest.skipUnless(TEST_NPU, "set ASCEND_TEST_NPU=1 in the CANN environment")
class DownsampleNPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(16)
        torch.npu.set_device(0)
        torch.npu.set_compile_mode(jit_compile=False)

    def test_four_paths_metadata_dtype_and_state(self):
        point, pool, model = fixture()
        pool.shuffle_orders = False
        before = {key: value.clone() for key, value in pool.state_dict().items()}
        prepared = experiment.prepare(model)
        outputs = {}
        try:
            with torch.inference_mode():
                for variant in experiment.VARIANTS:
                    diagnostics = []
                    experiment.select(prepared, variant, diagnostics)
                    outputs[variant] = pool(point)
                    self.assertEqual(outputs[variant].feat.dtype, torch.float32)
                    self.assertEqual(outputs[variant].feat.device.type, "cpu")
                    self.assertTrue(torch.isfinite(outputs[variant].feat).all())
                    self.assertEqual("coord" in outputs[variant], "drop_coords" not in variant)
                    for key in METADATA:
                        self.assertTrue(torch.equal(outputs["reference"][key], outputs[variant][key]), key)
                    if "fp16" in variant:
                        self.assertGreaterEqual(diagnostics[0]["padding_ratio"], 1)
                self.assertTrue(torch.equal(outputs["reference"].feat, outputs["drop_coords"].feat))
                self.assertTrue(torch.equal(outputs["fp16"].feat, outputs["drop_coords_fp16"].feat))
                metrics = validation.compare(outputs["reference"].feat.numpy(), outputs["fp16"].feat.numpy())
                self.assertTrue(metrics["passed"], metrics)
        finally:
            experiment.restore(prepared)
        self.assertNotIn("forward", pool.__dict__)
        for key, value in pool.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)

    def test_real_generator_n64_smoke(self):
        model = validation.make_model("generator")
        prepared = experiment.prepare(model)
        with np.load(validation.BASELINE_DIR / "reference_n64.npz") as frozen:
            data, golden = validation.make_input(frozen), frozen["generator_embedding"].copy()
        try:
            with torch.inference_mode():
                for variant in experiment.VARIANTS:
                    experiment.select(prepared, variant)
                    actual = model(data).cpu().numpy()
                    metrics = validation.compare(golden, actual)
                    print(variant, metrics, flush=True)
                    self.assertTrue(metrics["passed"], metrics)
        finally:
            experiment.restore(prepared)


if __name__ == "__main__":
    unittest.main()
