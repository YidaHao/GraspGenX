#!/usr/bin/env python3
"""Replay real CPE blocks with the existing CANN/OPP environment; no CLI/build.

For paired old/new runs, keep FROZEN_INPUT_DIR fixed and change RUN_TAG. Optional
traces replay the same fixtures, separately from warm component wall timings.
"""

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "ascend/baselines/ptv3-cuda-fp32-eager"
RESULT_ROOT = REPO_ROOT / "ascend/results"
OP_ROOT = REPO_ROOT / "ascend/custom_ops/submconv3d"
RUN_TAG = "gather_g4_mixed"
FROZEN_INPUT_DIR = None  # Set an earlier cpe_ops_* directory for fixed-input comparisons.
PROFILE = False  # Same FROZEN_INPUT_DIR + different RUN_TAG pairs old/new traces.
PROFILE_RUNS = 2
MEASURE_MIXED_MAP = True
POINT_COUNTS = (64, 2048, 3500)
ENCODERS = ("generator", "discriminator")
CPU_THREADS, WARMUP, SAMPLES = 16, 2, 10
COSINE_GATE = 0.9999
MAP_COMPONENTS = ("cpu_full_map", "cpu_packed_map_upload")
BLOCK_COMPONENTS = ("features_h2d", "resident_subm_conv3d", "d2h", "cpu_postops")

sys.path.insert(0, str(REPO_ROOT))
from ascend.tools import validate_ptv3 as validation  # Install shims before model imports.
from graspgenx.models.ptv3.ptv3_ascend import CachedCPEConv, PointTransformerV3Ascend
import torch_npu


def fingerprint(path):
    stat = path.stat()
    return {"sha256": validation.sha256_file(path), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def measure(operation, npu=False):
    samples = []
    for step in range(WARMUP + SAMPLES):
        if npu:
            torch.npu.synchronize()
        started = time.perf_counter()
        output = operation()
        if npu:
            torch.npu.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if step >= WARMUP:
            samples.append(elapsed)
    return output, {**validation.summarize_ms(samples), "samples_ms": samples}


def representatives(oracle, grid, batch):
    keys, rows = oracle._hash(batch, grid).sort()  # Original sort, NOT stable/min-row.
    first = torch.ones_like(keys, dtype=torch.bool)
    first[1:] = keys[1:] != keys[:-1]
    return keys[first].contiguous(), rows[first].to(torch.int32).contiguous()


def benchmark_block(block, snapshot, trace_dir):
    feat, grid, batch = (snapshot[key] for key in ("feat", "grid", "batch"))
    source = block.cpe_conv
    oracle = CachedCPEConv(source).eval()
    oracle.bias = None  # Raw kernel has no bias; source's CPU bias remains unchanged.
    n, cin = feat.shape
    volume, weight_cin, cout = source.weight.shape
    assert feat.dtype == torch.float32 and cin == weight_cin
    assert grid.shape == (n, 3) and batch.shape == (n,)
    assert grid.dtype in (torch.int32, torch.int64) and batch.dtype == torch.int64
    assert 1 <= n <= 4096 and source.kernel_size in (1, 3, 5) and volume == source.kernel_size**3
    assert all(16 <= c <= 512 and c % 16 == 0 for c in (cin, cout))
    for tensor in (*source.parameters(), *block.cpe_linear.parameters(), *block.cpe_norm.parameters()):
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    metrics = {}
    cache, metrics["cpu_full_map"] = measure(lambda: oracle._get_neighbor_map(grid, batch, {}))
    (keys, rows), metrics["cpu_sorted_representatives"] = measure(
        lambda: representatives(oracle, grid, batch))
    found, indices = cache["found"].view(n, volume), cache["indices"].view(n, volume)

    def packed_upload():
        packed = torch.full((n, (volume + 7) // 8 * 8), -1, dtype=torch.int32)
        packed[:, :volume] = indices
        packed[:, :volume].masked_fill_(~found, -1)
        return packed.to("npu:0")

    packed, metrics["cpu_packed_map_upload"] = measure(packed_upload, npu=True)
    if MEASURE_MIXED_MAP:
        coordinates = torch.cat((batch[:, None], grid.long()), dim=1)
        assert coordinates.min() >= -(2**31) and coordinates.max() < 2**31
        coordinates_npu = coordinates.to("npu:0", torch.int32)
        keys_npu, rows_npu = keys.to("npu:0"), rows.to("npu:0")
        builder = torch.ops.graspgenx_subm.build_subm_map
        mixed, metrics["resident_hash_query"] = measure(
            lambda: builder(coordinates_npu, source.kernel_size, keys_npu, rows_npu), npu=True)
        assert torch.equal(mixed.cpu(), packed.cpu()), "mixed builder changed reference map"

        def mixed_map():
            new_keys, new_rows = representatives(oracle, grid, batch)
            coords = torch.cat((batch[:, None], grid.long()), dim=1).to("npu:0", torch.int32)
            return builder(coords, source.kernel_size, new_keys.to("npu:0"), new_rows.to("npu:0"))

        mixed, metrics["mixed_map_total"] = measure(mixed_map, npu=True)
        assert torch.equal(mixed.cpu(), packed.cpu()), "mixed pipeline changed reference map"
    features, metrics["features_h2d"] = measure(lambda: feat.to("npu:0", torch.float16), npu=True)
    weight = source.npu_weight
    assert weight is not None and torch.equal(weight.cpu(), source.weight.half())
    assert weight.dtype == torch.float16 and weight.device.type == "npu"
    raw_op = lambda: torch.ops.graspgenx_subm.subm_conv3d(features, packed, weight)
    raw, metrics["resident_subm_conv3d"] = measure(raw_op, npu=True)
    assert raw.dtype == torch.float16 and raw.device.type == "npu"
    actual, metrics["d2h"] = measure(lambda: raw.to("cpu", torch.float32), npu=True)
    expected = oracle(feat, grid, batch, {})
    assert expected.shape == (n, cout) and torch.isfinite(expected).all()
    accuracy = {"shape_match": actual.shape == expected.shape, "finite": bool(torch.isfinite(actual).all())}
    if all(accuracy.values()):
        accuracy.update(validation.compare(expected.numpy(), actual.numpy()))
    accuracy["passed"] = bool(accuracy["shape_match"] and accuracy["finite"] and accuracy.get("cosine", 0) >= COSINE_GATE)

    def postops():
        biased = actual if source.bias is None else actual + source.bias
        return feat + block.cpe_norm(block.cpe_linear(biased))

    post, metrics["cpu_postops"] = measure(postops)
    accuracy["postops_valid"] = post.shape == feat.shape and bool(torch.isfinite(post).all())
    accuracy["passed"] &= accuracy["postops_valid"]
    coords = torch.cat((batch[:, None], grid.long()), dim=1)
    unique_coords = torch.unique(coords, dim=0).shape[0]
    center = int((source.offsets == 0).all(dim=1).nonzero().item())
    assert found[:, center].all() and torch.equal(
        indices[:, center], rows.long()[torch.searchsorted(keys, oracle._hash(batch, grid))])
    geometry = {"unique_coords": unique_coords, "duplicate_coord_rows": n - unique_coords,
                "unique_hashes": len(keys), "duplicate_hash_rows": n - len(keys),
                "distinct_coord_hash_collisions": unique_coords - len(keys),
                "representative_count": len(rows),
                "nonidentity_center_rows": int((indices[:, center] != torch.arange(n)).sum()),
                "neighbor_hits": int(found.sum()), "neighbor_slots": n * volume,
                "neighbor_occupancy": float(found.float().mean()),
                "hits_per_offset": found.sum(dim=0).tolist()}
    if PROFILE:
        # Warm isolated components, not a chained pipeline; profiler overhead is not a latency sample.
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
        ) as profiler:
            for _ in range(PROFILE_RUNS):
                for label, operation in (("cpu_full_map", lambda: oracle._get_neighbor_map(grid, batch, {})),
                                         ("cpu_sorted_representatives", lambda: representatives(oracle, grid, batch)),
                                         ("cpu_packed_map_upload", packed_upload),
                                         ("features_h2d", lambda: feat.to("npu:0", torch.float16)),
                                         ("resident_subm_conv3d", raw_op),
                                         ("d2h", lambda: raw.to("cpu", torch.float32)), ("cpu_postops", postops)):
                    with torch.profiler.record_function(label):
                        operation()
                torch.npu.synchronize()
                profiler.step()
    return {"shape": [n, cin, cout], "kernel_size": source.kernel_size,
            "geometry": geometry, "accuracy": accuracy, "components": metrics}


@torch.inference_mode()
def main():
    assert RUN_TAG and all(c.isalnum() or c in "_-" for c in RUN_TAG)
    destination = RESULT_ROOT / f"cpe_ops_{datetime.now():%Y%m%d_%H%M%S_%f}_{RUN_TAG}"
    destination.mkdir(parents=True, exist_ok=False)
    fixture_root = Path(FROZEN_INPUT_DIR).resolve() if FROZEN_INPUT_DIR is not None else destination
    sources = [Path(__file__), Path(validation.__file__), OP_ROOT / "submconv3d.py", OP_ROOT / "torch_bridge.cpp"]
    sources += list((REPO_ROOT / "graspgenx/models/ptv3").glob("ptv3_*.py"))
    sources += list(OP_ROOT.glob("op_*/*.cpp")) + list(OP_ROOT.glob("op_*/*.h"))
    opp_roots = [Path(p) for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p]
    artifacts = {OP_ROOT / "build/torch_bridge.so"}
    for root in [*opp_roots, OP_ROOT / "opp/vendors/graspgenx_subm"]:
        artifacts.update(p for p in root.rglob("*") if p.suffix in (".o", ".bin", ".so"))
    report = {"tag": RUN_TAG, "completed": False, "passed": False, "blocks": [], "weighted": [],
              "fixture_root": str(fixture_root), "sources": {str(p): fingerprint(p) for p in sources},
              "artifacts": {str(p): fingerprint(p) for p in sorted(artifacts) if p.is_file()},
              "environment": {"host": platform.node(), "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                              "cann": os.environ.get("ASCEND_HOME_PATH"), "opp_paths": list(map(str, opp_roots))},
              "contract": {"encoders": ENCODERS, "point_counts": POINT_COUNTS, "threads": CPU_THREADS,
                           "warmup": WARMUP, "samples": SAMPLES, "cosine_gate": COSINE_GATE,
                            "profile_runs": PROFILE_RUNS if PROFILE else 0, "builder_called": MEASURE_MIXED_MAP,
                           "control": "CPU vanilla hash map + packed upload; resident raw SubM FP16; CPU FP32 postops",
                           "timing": "diagnostic warm wall ms, not end-to-end/device-only; CPU no sync; NPU pre-sync excluded/post-sync included",
                           "overhead": "no-op timer/sync probes reported, not subtracted; host dispatch/allocation included",
                           "transfers": "H2D includes FP32->FP16; D2H includes FP16->FP32; packed upload includes CPU packing",
                           "cpu_postops": "original bias + cpe_linear + cpe_norm (LayerNorm) + feat residual, CPU FP32",
                           "excluded": "model load, snapshot inference, weight upload, oracle, profiler, cold graph time",
                           "weighting": "first block map+pack once/stage; transfers/kernel/postops each block; sorted prep separate"}}
    print(f"Results: {destination}", flush=True)
    try:
        report["baseline_sha256"] = {f: validation.sha256_file(BASELINE_DIR / f) for f in validation.EXPECTED_SHA256}
        assert report["baseline_sha256"] == validation.EXPECTED_SHA256
        torch.set_num_threads(CPU_THREADS)
        torch.manual_seed(0)
        for encoder in ENCODERS:
            model = PointTransformerV3Ascend(in_channels=3, output_dim=512, grid_size=0.01,
                                          enable_flash=False, shuffle_orders=False)
            model.load_state_dict(torch.load(BASELINE_DIR / validation.WEIGHT_FILES[encoder],
                                             map_location="cpu", weights_only=False)["model"], strict=True)
            for module in model.modules():
                if hasattr(module, "shuffle_orders"):
                    module.shuffle_orders = False
                if hasattr(module, "traceable"):
                    module.traceable = False
            model.eval()
            report["execution_config"] = model.execution_config
            if "overhead_ms" not in report:
                report["overhead_ms"] = {"cpu_noop": measure(lambda: None)[1],
                                         "npu_noop_sync": measure(lambda: None, npu=True)[1]}
            blocks = {name: block for name, block in model.named_modules() if hasattr(block, "cpe_conv")}
            for count in POINT_COUNTS:
                snapshots, hooks = {}, []
                for name, block in blocks.items():
                    def capture(module, args, name=name):
                        feat, grid, batch, point = args
                        # Freeze point geometry too, never retain its mutable object or map cache.
                        values = dict(feat=feat, grid=grid, batch=batch, point_coord=point.coord, point_offset=point.offset)
                        snapshots[name] = {key: value.detach().cpu().clone() for key, value in values.items()}
                    hooks.append(block.cpe_conv.register_forward_pre_hook(capture))
                try:
                    with np.load(BASELINE_DIR / f"reference_n{count}.npz", allow_pickle=False) as reference:
                        output = model(validation.make_input(reference))  # One untimed snapshot inference per case.
                    torch.npu.synchronize()
                    assert output.shape == (1, 512) and torch.isfinite(output).all()
                finally:
                    for hook in hooks:
                        hook.remove()
                assert snapshots.keys() == blocks.keys() and snapshots
                case_rows, stages = [], {}
                for name, live in snapshots.items():
                    assert all(torch.isfinite(value).all() for value in live.values()), name
                    fixture = fixture_root / f"{encoder}_n{count}_{name}.npz"
                    if FROZEN_INPUT_DIR is not None:
                        with np.load(fixture, allow_pickle=False) as frozen:
                            snapshot = {key: torch.from_numpy(frozen[key].copy()) for key in live}
                        for key in live:
                            assert snapshot[key].shape == live[key].shape and snapshot[key].dtype == live[key].dtype, (name, key)
                            assert key == "feat" or torch.equal(snapshot[key], live[key]), (name, key, "geometry changed")
                    else:
                        snapshot = live
                        with fixture.open("xb") as stream:
                            np.savez(stream, **{key: value.numpy() for key, value in snapshot.items()})
                    assert all(torch.isfinite(value).all() for value in snapshot.values()), name
                    stage = name.rsplit(".", 1)[0]
                    if stage in stages:
                        assert all(torch.equal(snapshot[k], stages[stage][0][k]) for k in ("grid", "batch")), stage
                        assert torch.equal(blocks[name].cpe_conv.offsets, blocks[stages[stage][1]["block"]].cpe_conv.offsets)
                    row = {"encoder": encoder, "point_count": count, "block": name, "stage": stage,
                           "fixture": str(fixture), "fixture_sha256": validation.sha256_file(fixture)}
                    row.update(benchmark_block(blocks[name], snapshot, destination / f"trace_{RUN_TAG}_{fixture.stem}"))
                    stages.setdefault(stage, (snapshot, row))
                    case_rows.append(row)
                    report["blocks"].append(row)
                weighted = {key: sum(row["components"][key]["median_ms"] for row in
                                    ([entry[1] for entry in stages.values()] if key in MAP_COMPONENTS else case_rows))
                            for key in (*MAP_COMPONENTS, *BLOCK_COMPONENTS)}
                report["weighted"].append({"encoder": encoder, "point_count": count, "component_medians_ms": weighted,
                                            "diagnostic_sum_not_end_to_end_ms": sum(weighted.values()),
                                            "mixed_map_once_per_stage_ms": sum(
                                                entry[1]["components"]["mixed_map_total"]["median_ms"] for entry in stages.values()) if MEASURE_MIXED_MAP else None,
                                           "map_stage_blocks": [entry[1]["block"] for entry in stages.values()],
                                           "sorted_prep_once_per_stage_excluded_ms": sum(
                                                entry[1]["components"]["cpu_sorted_representatives"]["median_ms"] for entry in stages.values())})
                print(json.dumps(report["weighted"][-1]), flush=True)
            del blocks, model
        report["completed"] = True
        report["passed"] = all(row["accuracy"]["passed"] for row in report["blocks"])
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        with (destination / f"summary_{RUN_TAG}.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
