"""CPE regressions: python -B ascend/tests/test_ptv3_cpe.py -v.

Run from the repository with the existing validation environment/PYTHONPATH.
Default tests execute only CPU CPE. PTV3_CPE_NPU=1 also tests SubMCPEConv with
the already-built custom op; nothing is installed or built here. These are not
full-encoder accuracy/performance acceptance tests for the Ascend implementation.
Map/cache/checkpoint/RNG checks are structural; only cosine >= 0.9999 gates
numerical agreement. Shapes and finite outputs are required.
"""

import json
import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from ascend.tools.validate_ptv3 import install_minimal_import_shims

install_minimal_import_shims()

from graspgenx.models.ptv3.ptv3_ascend import (
    AscendBlock, CachedCPEConv, NPU_DEVICE, NPU_JIT_COMPILE, SubMCPEConv,
)
from graspgenx.models.ptv3.ptv3_vanilla import (
    HashSparseConv3d, PointTransformerV3Vanilla, VanillaBlock,
)


COSINE_GATE = 0.9999
TEST_NPU = os.environ.get("PTV3_CPE_NPU") == "1"


def vanilla_map(source, grid_coord, batch):
    """Preserve vanilla's exact representative, including its non-stable sort."""
    n, volume = grid_coord.shape[0], source.offsets.shape[0]
    sorted_keys, order = HashSparseConv3d._hash(batch, grid_coord).sort()
    queries = grid_coord[:, None, :] + source.offsets[None, :, :]
    keys = HashSparseConv3d._hash(
        batch[:, None].expand(-1, volume).reshape(-1), queries.reshape(-1, 3),
    )
    positions = torch.searchsorted(sorted_keys, keys).clamp(max=n - 1)
    return {"indices": order[positions], "found": sorted_keys[positions] == keys}


class CachedCPETests(unittest.TestCase):
    wrapper = CachedCPEConv

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.random.default_generator.manual_seed(703)

    def compare(self, actual, expected, label):
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertEqual(actual.device.type, "cpu", label)
        self.assertEqual(actual.dtype, torch.float32, label)
        self.assertTrue(torch.isfinite(actual).all().item(), label)
        self.assertTrue(torch.isfinite(expected).all().item(), label)
        a, b = actual.flatten().double(), expected.flatten().double()
        self.assertGreater(a.norm().item(), 0, "zero output has no cosine")
        self.assertGreater(b.norm().item(), 0, "zero reference has no cosine")
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        print(json.dumps({"cpe": self.wrapper.__name__, "case": label,
                          "cosine": cosine, "max_abs": (actual - expected).abs().max().item()}),
              flush=True)
        self.assertGreaterEqual(cosine, COSINE_GATE, label)

    def check_map(self, actual, expected):
        for key in ("indices", "found"):
            self.assertEqual(actual[key].device.type, "cpu")
            self.assertEqual(actual[key].dtype, expected[key].dtype)
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def check_point_map(self, candidate, grid, batch, point):
        expected = vanilla_map(candidate, grid, batch)
        if self.wrapper is SubMCPEConv:
            # Observe the forward's actual cache, not a CPU map created by the test.
            cache = point[f"_cpe_npu_map_{candidate.kernel_size}"]
            self.assertIs(candidate._get_npu_map(grid, batch, point), cache)
            volume = candidate.kernel_size**3
            packed = torch.full((len(grid), (volume + 7) // 8 * 8), -1, dtype=torch.int32)
            packed[:, :volume] = expected["indices"].view(-1, volume)
            packed[:, :volume].masked_fill_(~expected["found"].view(-1, volume), -1)
            self.assertEqual(cache.dtype, torch.int32)
            self.assertEqual(cache.device.type, "npu")
            self.assertTrue(cache.is_contiguous())
            self.assertTrue(torch.equal(cache.cpu(), packed), "map rows/missing/padding")
            coordinates = torch.cat((batch[:, None], grid), dim=1)
            if coordinates.min() >= -(2**31) and coordinates.max() < 2**31:
                self.assertNotIn(f"_cpe_hash_map_{candidate.kernel_size}", point)
            else:
                self.check_map(point[f"_cpe_hash_map_{candidate.kernel_size}"], expected)
        else:
            cache = point[f"_cpe_hash_map_{candidate.kernel_size}"]
            self.assertIs(candidate._get_neighbor_map(grid, batch, point), cache)
            self.check_map(cache, expected)
        return cache

    @torch.no_grad()
    def test_hash_representatives_duplicates_collisions_and_permutations(self):
        fixtures = {
            "duplicates": [[0, 0, 0, 0]] * 17 + [[0, 1, 0, 0], [1, 0, 0, 0]],
            "collision": [[0, 19349669, -73856093, 0], [0, 0, 0, 0],
                          [0, 1, 0, 0], [1, 0, 0, 0]],
            "query_collision": [[0, 19349668, -73856093, 0], [0, 0, 0, 0],
                                [1, 0, 0, 0], [0, 4, 0, 0]],
            # batch * P_batch cancels x * P_x. Large batch IDs are map-only:
            # do not create VanillaPoint's dense batch-count/offset array here.
            "crossbatch_collision": [[73856093, -334214467, 0, 0], [0, 0, 0, 0],
                                     [1, 0, 0, 0]],
            "crossbatch_query_collision": [[73856093, -334214468, 0, 0],
                                           [0, 0, 0, 0], [1, 0, 0, 0]],
            # Hash arithmetic stays inside int64. Narrowing x to int32 would
            # alias a distinct row, changing both representatives and outputs.
            "outside_int32_upper": [[0, 2**31 + 17, 0, 0]] * 17
                                   + [[0, 2**31 + 18, 0, 0], [0, -(2**31) + 17, 0, 0],
                                      [1, 2**31 + 17, 0, 0]],
            "outside_int32_lower": [[0, -(2**31) - 17, 0, 0], [0, -(2**31) - 18, 0, 0],
                                   [0, 2**31 - 17, 0, 0]],
        }
        for kernel_size in (1, 3, 5):
            source = HashSparseConv3d(16, 48, kernel_size, bias=True).eval()
            source.bias.copy_(torch.randn(48) * 0.01)
            candidate = self.wrapper(source).eval()
            for name, rows in fixtures.items():
                rows = torch.tensor(rows, dtype=torch.int64)
                features = torch.randn(len(rows), 16)  # Duplicate voxels have different features.
                for permutation in (torch.arange(len(rows)), torch.arange(len(rows)).flip(0),
                                    torch.randperm(len(rows))):
                    with self.subTest(K=kernel_size, fixture=name,
                                      order=permutation.tolist()), ExitStack() as stack:
                        batch, grid = rows[permutation, 0], rows[permutation, 1:]
                        feat, point = features[permutation], {}
                        builder = None
                        if self.wrapper is SubMCPEConv and name.startswith("outside_int32"):
                            builder = stack.enter_context(patch.object(
                                torch.ops.graspgenx_subm, "build_subm_map",
                                side_effect=AssertionError("out-of-int32 coordinates reached NPU builder"),
                            ))
                        # Recompute after permutation; never assume min-row/stable tie-breaking.
                        actual = candidate(feat, grid, batch, point)
                        self.check_point_map(candidate, grid, batch, point)
                        self.compare(actual, source(feat, grid, batch), f"{name}_k{kernel_size}")
                        if builder is not None:
                            builder.assert_not_called()

    @torch.no_grad()
    def test_point_local_cache_feature_changes_and_new_geometry(self):
        grid = torch.zeros(17, 3, dtype=torch.int64)
        grid[:, 0] = torch.arange(17)
        batch = torch.zeros(17, dtype=torch.int64)
        features = torch.randn(17, 16)
        sources = [HashSparseConv3d(16, 48, 3).eval() for _ in range(2)]
        candidates = [self.wrapper(source).eval() for source in sources]
        point = {}
        candidates[0](features, grid, batch, point)
        cache = self.check_point_map(candidates[0], grid, batch, point)
        for source, candidate in zip(sources, candidates):
            features.copy_(torch.randn_like(features))
            # A second block and changed feature storage reuse geometry, never outputs.
            with patch.object(candidate, "_hash", side_effect=AssertionError("map rebuilt")):
                actual = candidate(features, grid, batch, point)
            self.assertIs(self.check_point_map(candidate, grid, batch, point), cache)
            self.compare(actual, source(features, grid, batch), "changed_features_shared_point")

        new_grid, new_point = grid * 4, {}
        actual = candidates[0](features, new_grid, batch, new_point)
        new_cache = self.check_point_map(candidates[0], new_grid, batch, new_point)
        self.assertIsNot(new_cache, cache)
        if self.wrapper is SubMCPEConv:
            self.assertFalse(torch.equal(new_cache.cpu(), cache.cpu()))
        else:
            self.assertFalse(torch.equal(new_cache["found"], cache["found"]))
        self.assertIs(self.check_point_map(candidates[0], grid, batch, point), cache)
        self.compare(actual, sources[0](features, new_grid, batch), "new_same_shape_geometry")

    @torch.no_grad()
    def test_subm_unsupported_channels_fall_back_before_npu(self):
        source = HashSparseConv3d(3, 16, 3, bias=True).eval()
        source.bias.copy_(torch.randn(16) * 0.01)
        grid = torch.zeros(17, 3, dtype=torch.int64)
        grid[:, 0] = torch.arange(17)
        batch, feat, point = torch.zeros(17, dtype=torch.int64), torch.randn(17, 3), {}
        # Cin=3 exercises the shape guard without NPU construction or a large N.
        with ExitStack() as stack:
            calls = [stack.enter_context(patch.object(
                owner, name, create=True,
                side_effect=AssertionError(f"unsupported shape reached {name}"),
            )) for owner, name in (
                (torch.Tensor, "to"), (SubMCPEConv, "_get_npu_map"),
                (torch.ops.graspgenx_subm, "build_subm_map"),
                (torch.ops.graspgenx_subm, "subm_conv3d"),
            )]
            candidate = SubMCPEConv(source).eval()
            self.assertFalse(candidate._supported_shape)
            self.assertIsNone(candidate.npu_weight)
            actual = candidate(feat, grid, batch, point)
            for call in calls:
                call.assert_not_called()
        self.assertNotIn("_cpe_npu_map_3", point)
        self.check_map(point["_cpe_hash_map_3"], vanilla_map(source, grid, batch))
        self.compare(actual, source(feat, grid, batch), "subm_unsupported_cin3_cpu_fallback")

    @torch.no_grad()
    def test_wrapping_rng_and_strict_checkpoint_reload(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                source = HashSparseConv3d(16, 48, 3, bias=bias).eval()
                rng_before = torch.random.get_rng_state().clone()
                candidate = self.wrapper(source).eval()
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
                self.assertIs(candidate.weight, source.weight)
                self.assertIs(candidate.bias, source.bias)
                self.assertIs(candidate.offsets, source.offsets)
                keys = {"weight", "offsets"} | ({"bias"} if bias else set())
                self.assertEqual(set(candidate.state_dict()), keys)
                replacement = HashSparseConv3d(16, 48, 3, bias=bias).eval()
                if bias:
                    replacement.bias.copy_(torch.randn(48) * 0.01)
                incompatible = candidate.load_state_dict(replacement.state_dict(), strict=True)
                self.assertFalse(incompatible.missing_keys or incompatible.unexpected_keys)
                grid = torch.zeros(17, 3, dtype=torch.int64)
                grid[:, 0] = torch.arange(17)
                batch, feat = torch.zeros(17, dtype=torch.int64), torch.randn(17, 16)
                self.compare(candidate(feat, grid, batch, {}), replacement(feat, grid, batch),
                             "strict_reload_repacked_weights")
                self.assertEqual(set(candidate.state_dict()), keys, "runtime cache must not persist")
                with self.assertRaises(RuntimeError):
                    candidate.load_state_dict({k: v for k, v in replacement.state_dict().items()
                                               if k != "weight"}, strict=True)
                with self.assertRaises(RuntimeError):
                    candidate.load_state_dict({**replacement.state_dict(), "unexpected": torch.zeros(1)},
                                              strict=True)

    @torch.no_grad()
    def test_model_forward_owns_fresh_stage_caches(self):
        model = PointTransformerV3Vanilla(
            in_channels=3, output_dim=32, order=("z",), stride=(2,),
            enc_depths=(2, 2), enc_channels=(16, 32), enc_num_head=(2, 4),
            enc_patch_size=(16, 16), drop_path=0, shuffle_orders=False,
            upcast_attention=False, upcast_softmax=False,
        ).eval()
        checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
        caches = []

        def observe(module, args, output):
            feat, grid, batch, point = args
            cache = self.check_point_map(module, grid, batch, point)
            self.compare(output, HashSparseConv3d.forward(module, feat, grid, batch), "model_cpe")
            caches.append(cache)

        for stage in model.enc:
            for block in stage.children():
                if isinstance(block, VanillaBlock):
                    block.cpe_conv = self.wrapper(block.cpe_conv).eval()
                    handle = block.cpe_conv.register_forward_hook(observe)
                    self.addCleanup(handle.remove)
        self.assertEqual(set(model.state_dict()), set(checkpoint))
        model.load_state_dict(checkpoint, strict=True)
        grid = torch.zeros(17, 3, dtype=torch.int32)
        grid[:, 0] = torch.arange(17)
        data = dict(grid_coord=grid, coord=grid.float(), feat=torch.randn(17, 3),
                    offset=torch.tensor([17]))
        # Exercise real point creation and pooling, using the production CPE body
        # only. Attention/FFN placement and full-encoder acceptance are out of scope.
        with patch.object(VanillaBlock, "forward", AscendBlock.forward_cpe):
            for version in range(3):
                data["feat"].copy_(torch.randn_like(data["feat"]))
                if version == 2:
                    data["grid_coord"].mul_(4)
                    data["coord"].copy_(data["grid_coord"].float())
                output = model(data)
                self.assertEqual(tuple(output.shape), (1, 32))
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertFalse(any(key.startswith("_cpe_") for key in data))
        self.assertEqual(len(caches), 12)
        for start in (0, 4, 8):
            self.assertIs(caches[start], caches[start + 1])
            self.assertIs(caches[start + 2], caches[start + 3])
        self.assertEqual(len({id(caches[i]) for i in (0, 2, 4, 6, 8, 10)}), 6)


@unittest.skipUnless(TEST_NPU, "optional NPU CPE; set PTV3_CPE_NPU=1 after device acceptance")
class SubMCPETests(CachedCPETests):
    wrapper = SubMCPEConv

    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(NPU_DEVICE)
        torch.npu.set_compile_mode(jit_compile=NPU_JIT_COMPILE)

    @torch.no_grad()
    def test_npu_runtime_errors_propagate_without_cpu_fallback(self):
        candidate = SubMCPEConv(HashSparseConv3d(16, 16, 3).eval()).eval()
        grid = torch.zeros(17, 3, dtype=torch.int64)
        grid[:, 0] = torch.arange(17)
        batch, feat = torch.zeros(17, dtype=torch.int64), torch.randn(17, 16)
        for op_name in ("build_subm_map", "subm_conv3d"):
            with self.subTest(op=op_name):
                sentinel = RuntimeError(f"sentinel {op_name} runtime failure")
                with patch.object(CachedCPEConv, "forward", side_effect=AssertionError(
                    "NPU runtime failure must not fall back to CPU",
                )) as cpu, patch.object(torch.ops.graspgenx_subm, op_name,
                                        side_effect=sentinel) as op:
                    with self.assertRaises(RuntimeError) as raised:
                        candidate(feat, grid, batch, {})
                    self.assertIs(raised.exception, sentinel)
                    op.assert_called_once()
                    cpu.assert_not_called()


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
