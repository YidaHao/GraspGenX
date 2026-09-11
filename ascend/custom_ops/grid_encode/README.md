# GridEncode (Ascend 310P)

Four-route integer spatial encoding for the current `PointTransformerV3Ascend`.
The `GridEncode` branch starts from `eff0a76`; `ptv3_vanilla.py` remains the
unchanged reference. Only initial spatial-code generation is replaced, not CPE
or the dense feature path. This is not a fully NPU encoder.

## Raw Contract

`grid_encode(grid_coord, depth) -> spatial_codes`

| Item | Contract |
| --- | --- |
| Input | Contiguous, strided ND NPU `int32` tensor `[N, 3]`, columns `(x, y, z)` |
| Limits | `1 <= N <= 4096`; integer `1 <= depth <= 16` |
| Contents | Caller guarantees `0 <= coordinate < 2**depth` |
| Output | Contiguous ND NPU `int64` `[N, 4]`, on the input device |
| Columns | `z`, `z-trans`, `hilbert`, `hilbert-trans`, in that order |
| Transpose orders | Swap x/y only; z is unchanged |

Shape, dtype, device, layout/contiguity and depth are validated. Coordinate
contents are **not** scanned, copied to the host for validation, or clamped.
Out-of-range contents violate the caller precondition. Depth zero is rejected,
not automatically changed to one. A singleton still returns `[1, 4]`.

Codes exactly match the reference `encode` with zero batch IDs and the same
depth/order. There are no batch bits, floating-point quantization, origin
offsets, sorting, permutations, feature inputs or backward operation. Non-finite
feature handling is therefore outside this raw integer API, not an accuracy waiver.

## Integer Algorithm

Morton codes use integer bit interleaving. Hilbert codes reproduce the reference
bit-exchange/inversion loop, MSB-first interleaving and whole-code Gray decoding.
This is not a copy of an SDK axis-rotation or coordinate-clamping algorithm.

One kernel shares coordinate loads across all four routes, uses up to eight
AI cores, and requests zero GM workspace. Eight-point groups align DMA starts;
scalar loads cover tails and unaligned contiguous storage-offset views without
reading beyond `[N, 3]`. Each output row occupies 32 bytes.

## Model Integration

The single current implementation remains `PointTransformerV3Ascend` in
`graspgenx/models/ptv3/ptv3_ascend.py`. Its default `serialize_point` is shared by
direct `forward` and the partitioned profiler; there is no separate GridEncode
model class or user-facing implementation switch.

- CPU grid quantization, depth inference and the reference batch-bit budget stay unchanged.
- Canonical CPU int32/int64 grids are uploaded as contiguous NPU int32; only spatial codes run in GridEncode.
- Codes return to CPU for `[N, 4] -> [4, N]`, batch packing, default `argsort`, inverse and optional order shuffling.
- Grid quantization and downsampling remain CPU FP32. The continuous NPU attention/residual/LayerNorm/FFN region remains FP16, with no feature FP32/upcast path.
- CPE still uses CPU reference-hash representative selection and CPU post-ops, with NPU neighbor queries and FP16 sparse convolution.

Noncanonical orders, unsupported grid shape/dtype/domain, negative coordinates
and depth zero use reference CPU serialization handling. Existing reference
errors, including empty/depth-zero failures, are preserved rather than repaired.
Missing libraries, platform failures and runtime errors on the canonical NPU
path propagate; they are not caught and retried on CPU.

## Build And Use

Run commands from the repository root in the intended native Torch/Torch-NPU
Python environment. The package has its own `env.sh` and `build.sh`; its private
OPP vendor is `graspgenx_grid`. It preserves existing CPE environment paths and
does not install dependencies or modify the system OPP.

On P1, when native CPE and GridEncode artifacts are already built:

```bash
source ascend/env.sh
```

Sourcing an environment does not build the operator. For a new P1 GridEncode
build, source the package alone first, then load the whole-encoder environment:

```bash
source ascend/custom_ops/grid_encode/env.sh
bash ascend/custom_ops/grid_encode/build.sh
source ascend/env.sh
```

On P3, explicitly select its CANN and the Python containing Torch/Torch-NPU.
Do not source the P1-oriented root `ascend/env.sh`, which hardcodes a system
CANN path. In the P3 private checkout, use:

```bash
export ASCEND_HOME_PATH=/home/huawei/Ascend/ascend-toolkit/8.3.RC1
export PYTHON_BIN=/home/huawei/hyd/Workspace/IB_Robot/venv/bin/python
source ascend/custom_ops/grid_encode/env.sh
bash ascend/custom_ops/grid_encode/build.sh
# Whole encoder: CPE must also have been built natively in this checkout.
source ascend/custom_ops/submconv3d/env.sh
source ascend/custom_ops/grid_encode/env.sh
```

Builds use the selected CANN generator/compiler, CMake and native Torch ABI.
The bridge allocates via `at::empty`, without relying on an unresolved
`OpPreparation` function. Rebuild on each host: **do not copy P1 aarch64 `.so`
files to P3 x86_64**. The recorded P3 run used both a natively rebuilt private
CPE package and native GridEncode, without dependency or system-OPP changes.

Raw usage needs only the GridEncode package environment and native artifacts,
not CPE or model weights. Use the canonical package import for registration:

```python
import torch
from ascend.custom_ops.grid_encode.grid_encode import grid_encode

torch.npu.set_device("npu:0")
grid = torch.tensor([[0, 0, 0], [1, 2, 3], [15, 15, 15]],
                    dtype=torch.int32, device="npu:0")
codes = grid_encode(grid, depth=4)
print(codes.cpu())  # int64 [3, 4], in the four documented column orders
```

## Recorded Validation

These commands document the existing suites; no new run is implied by this
README. After sourcing the package environment and building native artifacts:

```bash
ASCEND_TEST_NPU=1 "$PYTHON_BIN" -B ascend/tests/test_grid_encode.py -v
ASCEND_TEST_NPU=1 "$PYTHON_BIN" -B ascend/tests/test_ptv3_grid_encode.py -v
```

Without `ASCEND_TEST_NPU=1`, only CPU checks run; opted-in missing artifacts are
errors, not skips. Both P1 and P3 recorded **11 raw-suite + 16 integration tests
PASS**, plus source/artifact integrity and frozen CUDA golden checks.

Coverage includes depths 1..16, N=1..4096 boundary/workload cases, alignment,
tails, contiguous storage offsets, duplicates, permutations, changed/reused
storage, exact batch packing/sort/inverse/shuffle and reference fallback/errors.
Eager `jit_compile=False` and both JIT modes, fake/meta, and real TorchAir
`fullgraph=True` raw-op execution are covered. Fullgraph coverage is for the
raw operator, **not compilation of the whole model**; import does not change JIT mode.
OPP registration targets `ascend310p` only. No unsupported-SoC hardware tests
were performed or are claimed.

| Host | Native environment | Minimum CUDA cosine | Old/new output bytes | Repeat max abs |
| --- | --- | ---: | --- | ---: |
| P1, `192.168.7.101` | CANN 8.1.RC1, Python 3.11, Torch/Torch-NPU 2.5.1 | 0.9999591909124736 | Equal | 0 |
| P3, `192.168.136.109` | CANN 8.3.RC1, Python 3.10.12, Torch 2.5.1+cpu / Torch-NPU 2.5.1 | 0.9999688914724082 | Equal | 0 |

Raw spatial codes and CPU serialization metadata must match exactly. The
floating-output accuracy gate is cosine >= 0.9999, with correct shapes and
finite outputs required; runtime failures cannot pass. Other errors, byte
equality, repeat drift and latency are report-only. Frozen goldens are not
replaced with candidate outputs.

The separate P1 `validate_ptv3.py` run passed all six G/D cases, with repeat drift
zero; the default N=2048 stage profiler also passed. Its serialization medians
were 4.107/4.033 ms for G/D. The validator's G/D medians were 137.808/138.908 ms
at N=2048 and 156.358/153.889 ms at N=3500, so the report-only 150 ms target is
not met at every size. These single-process regressions have no paired control;
use the three-process comparison below for the optimization verdict.
Validator/profiler selection remains via direct imports and top-of-file
configuration, not impl/device/precision CLI switches.

The unified `SUBM_SMOKE=1 bash ascend/tests/run_tests.sh -v` runner loads both
private packages and runs the three CPE suites plus both GridEncode suites in
separate processes. Both packages must already be built. The final P1 unified
run passed all 60 tests (12 + 10 + 11 + 11 + 16), with no skips; see
`build/unified_validation.log` for the two-package coexistence regression.

## Performance

P1 N=2048 G+D: 377.704 -> 276.818 ms, -100.886 ms (-26.71%). P3: 96.907 ->
77.501 ms, -19.406 ms (-20.03%). The control is original CPU initial serialization
with current CPE/dense code fixed; the candidate uses four-route GridEncode.
Verdict: **improved on these measured workloads**, encoder-only, with only three
processes per host and no statistical-significance claim.

| Host | Points | Before | After | Change (ms) | Change (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1 | 64 | 259.089 | 214.504 | -44.585 | -17.21% |
| P1 | 2048 | 377.704 | 276.818 | -100.886 | -26.71% |
| P1 | 3500 | 446.941 | 308.530 | -138.411 | -30.97% |
| P3 | 64 | 57.750 | 46.616 | -11.135 | -19.28% |
| P3 | 2048 | 96.907 | 77.501 | -19.406 | -20.03% |
| P3 | 3500 | 123.542 | 99.301 | -24.241 | -19.62% |

The control is `eff0a76`-based current optimized CPE/dense with benchmark-only
original CPU serialization, not CPU-CPE. The same resident model switches only
its bound serialization method in paired AB/BA order. Protocol: 3 independent
processes, 3 warmups, 20 measured runs, 3 repeat checks, 16 CPU threads; 20
serialization-entry/raw runs and 3 diagnostic profile runs at N=2048.

Timing is synchronized host wall time, including encoder transfers, without
rescaling. G+D is the median of per-process sums of G and D encoder medians,
**not request latency**. Individual G/D and entry values below are medians of
process medians; independently aggregated G and D need not sum to the G+D table.
Calculations use unrounded JSON; tables round encoder/entry to 3 decimals.

| Host | Points | G encoder old -> new | D encoder old -> new | G entry old -> new | D entry old -> new |
| --- | ---: | ---: | ---: | ---: | ---: |
| P1 | 64 | 129.732 -> 107.345 | 129.420 -> 107.254 | 23.650 -> 2.021 | 23.691 -> 2.014 |
| P1 | 2048 | 188.666 -> 138.391 | 189.123 -> 138.427 | 54.374 -> 4.181 | 54.239 -> 4.009 |
| P1 | 3500 | 222.733 -> 153.568 | 224.207 -> 155.600 | 73.953 -> 5.428 | 76.004 -> 5.540 |
| P3 | 64 | 28.898 -> 23.270 | 28.841 -> 23.346 | 5.708 -> 0.446 | 5.678 -> 0.427 |
| P3 | 2048 | 48.391 -> 38.678 | 48.516 -> 38.708 | 10.405 -> 0.976 | 10.459 -> 0.980 |
| P3 | 3500 | 61.778 -> 49.679 | 61.765 -> 49.538 | 13.456 -> 1.430 | 13.387 -> 1.421 |

Raw diagnostics, depth=7, once per N/process with shared G/D geometry; median
of process medians in ms (4 decimals). Resident kernel numbers include host
dispatch/synchronization: **not device-event kernel time**. These independently
timed parts omit some entry work, including quantization/depth and sort/inverse;
their sum is not serialization-entry latency or a separate end-to-end speedup.

| Host | Points | CPU four codes | H2D | Resident kernel host-sync | D2H | CPU layout | CPU batch pack |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | 64 | 22.1730 | 0.1692 | 0.3379 | 0.1264 | 0.1419 | 0.1541 |
| P1 | 2048 | 51.6912 | 0.2158 | 0.6927 | 0.1695 | 0.2788 | 0.2349 |
| P1 | 3500 | 70.1606 | 0.2334 | 0.9330 | 0.1929 | 0.3474 | 0.2883 |
| P3 | 64 | 2.1975 | 0.0393 | 0.1010 | 0.0258 | 0.0177 | 0.0203 |
| P3 | 2048 | 10.0344 | 0.0651 | 0.2845 | 0.0562 | 0.0601 | 0.0455 |
| P3 | 3500 | 12.4793 | 0.0708 | 0.4282 | 0.0642 | 0.0767 | 0.0534 |

All samples, including long tails, are retained. For example, P1 D at N=3500
has old/new maxima 258.972 / 219.679 ms despite medians 224.207 / 155.600 ms.
JSON retains per-process samples, p95/p99, min/max and standard deviation.
No P3 regression from GridEncode was observed in this comparison. This does
**not** establish that the earlier CPE regression versus CPU-CPE is resolved;
that baseline was not measured here. Default adoption still targets P1 first.

## Artifacts

- [P1 final three-process summary](../../results/grid_encode_compare_20260911_135156_786032/summary.json)
- [P1 default validator](../../results/ptv3_ascend_grid_encode_four_orders_validation.json)
- [P1 default stage profiler](../../results/ptv3_ascend_grid_encode_four_orders_profile_n2048.json)
- [P3 private three-process summary (SSH)](ssh://huawei@192.168.136.109/home/huawei/hyd/Workspace/GraspgenX/grid_encode_validation_20260911_aKpQaedG/ascend/results/grid_encode_compare_20260911_135746_664985/summary.json)

Per-process JSON and raw NPZ outputs reside beside each summary. P3 links refer
to the remote private checkout, not a local copy. Both summaries report PASS,
exact metadata/raw codes, unchanged source/bridge/private OPP across processes,
and equal old/new embedding bytes. No new measurements accompany this README.
