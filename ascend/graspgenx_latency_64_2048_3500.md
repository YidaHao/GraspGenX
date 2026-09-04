# GraspGenX 64 / 2048 / 3500 Point Latency Report

Date: 2026-08-29

## Conclusion

On the Ascend310P1 host at `192.168.7.101`, with both complete PTV3 encoders
running on CPU and the dense generator/discriminator modules running on NPU,
the typical end-to-end latency is:

| Point count | End-to-end median, no fine-grained sync | End-to-end profiled median |
|---:|---:|---:|
| 64 | 721.72 ms | 734.47 ms |
| 2048 | 5956.25 ms | 5937.50 ms |
| 3500 | 12393.98 ms | 12524.32 ms |

The 64-point result is sensitive to host load. An independent longer run with
the same configuration measured 651.71 ms unprofiled and 662.44 ms profiled,
so the practical 64-point range observed in this session is approximately
0.65-0.72 seconds.

At 2048 points, the two CPU PTV3 calls account for 99.22% of profiled latency.
At 3500 points, they account for 99.61%.

## Benchmark Contract

Only point count changes between the three cases.

| Setting | Value |
|---|---|
| Device | Ascend310P1, `npu:0` |
| PyTorch / torch-npu | 2.5.1 / 2.5.1 |
| PTV3 device | CPU |
| Dense heads device | NPU |
| CPU intra-op threads | 16 |
| Precision | FP32 |
| Weights | Random, full release-compatible shapes |
| PTV3 parameters per encoder | 38,660,640 |
| Grasps | 1 |
| Diffusion steps | 1 |
| Grid size | 0.01 |
| Kappa | 3.27 |
| PTV3 flash attention | Disabled |
| Serialization order shuffling | Disabled |
| Input distribution | `default_rng(1234)`, normal(0, 0.05), mean-centered |
| Initial noise | `default_rng(5678)`, shape `[1, 6]` |

The generated output contract was `[1, 1, 4, 4]` grasps and `[1, 1, 1]`
confidence for every point count. All outputs were finite.

This is the same one-grasp, one-diffusion-step microbenchmark contract as the
provided `npu_ptv3_cpu_profile_fresh.json`. It is not a 100-grasp, 20-step
production benchmark.

## End-to-End Results

The profiled run synchronizes NPU work at each stage boundary. The unprofiled
run synchronizes only around the complete pipeline and is the better estimate
of normal request latency. Medians are preferred because the 64-point run had
several host-side outliers.

| Points | Runs | Profiled mean | Profiled median | Profiled min-max | Unprofiled mean | Unprofiled median | Unprofiled min-max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 10 | 799.37 ms | 734.47 ms | 609.95-1312.34 ms | 773.72 ms | 721.72 ms | 614.08-1132.86 ms |
| 2048 | 5 | 5959.15 ms | 5937.50 ms | 5854.37-6142.26 ms | 6116.02 ms | 5956.25 ms | 5856.35-6704.79 ms |
| 3500 | 5 | 12637.94 ms | 12524.32 ms | 12389.57-13291.91 ms | 12390.25 ms | 12393.98 ms | 12344.67-12422.44 ms |

## Generator Stage Means

All values are synchronized stage means in milliseconds.

| Generator stage | 64 points | 2048 points | 3500 points |
|---|---:|---:|---:|
| Scale point cloud by kappa | 0.357 | 0.330 | 0.324 |
| Points NPU to CPU | 0.178 | 0.188 | 0.210 |
| PTV3 input formatting | 0.381 | 0.365 | 0.372 |
| **Generator PTV3 encoder, CPU** | **333.226** | **2974.279** | **6237.672** |
| Object embedding CPU to NPU | 0.206 | 0.217 | 0.227 |
| Gripper encoder | 2.352 | 2.419 | 3.355 |
| Conditioning concat | 0.295 | 0.298 | 0.347 |
| Scheduler setup | 0.512 | 0.526 | 0.553 |
| Diffusion head, NPU | 15.659 | 15.802 | 17.291 |
| Scheduler step and SO(3) pose conversion | 9.445 | 9.394 | 9.645 |
| **Complete generator** | **363.614** | **3004.809** | **6271.060** |

The generator stages after the object embedding are almost independent of
point count. The small increase at 3500 points is run-to-run NPU/host jitter;
their tensor shapes do not depend on point count in this contract.

## Discriminator Stage Means

All values are synchronized stage means in milliseconds.

| Discriminator stage | 64 points | 2048 points | 3500 points |
|---|---:|---:|---:|
| Scale point cloud by kappa | 0.333 | 0.339 | 0.358 |
| Points NPU to CPU | 0.123 | 0.198 | 0.197 |
| PTV3 input formatting | 0.353 | 0.354 | 0.351 |
| **Discriminator PTV3 encoder, CPU** | **421.011** | **2938.389** | **6351.272** |
| Object embedding CPU to NPU | 0.205 | 0.220 | 0.213 |
| Grasp matrix to `r3_so3` | 7.369 | 7.790 | 7.674 |
| Sample encoder | 1.156 | 1.340 | 1.277 |
| Gripper encoder | 1.954 | 2.167 | 1.955 |
| Conditioning concat | 0.284 | 0.291 | 0.283 |
| Discriminator head, NPU | 1.601 | 1.871 | 1.843 |
| Sigmoid | 0.224 | 0.227 | 0.302 |
| **Complete discriminator** | **435.578** | **2954.163** | **6366.707** |

The 64-point discriminator PTV3 mean includes a 940.11 ms outlier. Its median
was 313.56 ms, which is why the end-to-end median is more representative than
the mean for that case.

## Bottleneck Breakdown

| Points | Two PTV3 encoders | Other profiled work | PTV3 share | Four CPU/NPU transfers |
|---:|---:|---:|---:|---:|
| 64 | 754.24 ms | 45.14 ms | 94.35% | 0.71 ms |
| 2048 | 5912.67 ms | 46.48 ms | 99.22% | 0.82 ms |
| 3500 | 12588.94 ms | 49.00 ms | 99.61% | 0.85 ms |

The transfers are not a meaningful bottleneck. Even at 3500 points, both point
cloud transfers plus both embedding transfers total less than 1 ms.

The non-PTV3 latency floor for this one-step, one-grasp contract is about
46-49 ms. To reach a 100 ms end-to-end target, both accelerated PTV3 calls
would therefore need to total roughly 50 ms or less.

## Scaling

Using unprofiled medians:

| Comparison | Point-count ratio | Latency ratio |
|---|---:|---:|
| 2048 vs 64 | 32.00x | 8.25x |
| 3500 vs 2048 | 1.71x | 2.08x |
| 3500 vs 64 | 54.69x | 17.17x |

The 2048-to-3500 increase is super-linear. PTV3 runtime is not expected to
scale linearly with point count because serialized attention, neighborhood
construction, pooling occupancy, sorting, and hash lookup all change with the
number and spatial distribution of points.

## Comparison With The Supplied 64-Point Profile

The supplied profile came from a different host/software snapshot using
PyTorch `2.10.0+cpu` and reported:

| Metric | Supplied profile | This 7.101 run |
|---|---:|---:|
| Generator PTV3 | 4057.91 ms | 333.23 ms mean, 303.70 ms median |
| Discriminator PTV3 | 4504.65 ms | 421.01 ms mean, 313.56 ms median |
| End-to-end | 8636.70 ms mean | 721.72 ms unprofiled median |

The stage counts in the supplied JSON prove that each PTV3 encoder ran once per
request. The difference is therefore environmental rather than repeated model
execution. Host CPU, PyTorch CPU build, thread configuration, and code snapshot
must be recorded before treating the supplied 8.64-second result as comparable
to this board.

## Operational Notes

- The first 64-point request in the new process took 68.47 seconds because NPU
  kernels compiled. This cold compile time is excluded from steady-state data.
- First-shape warmup took 11.18 seconds for 2048 points and 17.74 seconds for
  3500 points. Those values are also excluded.
- The NPU runtime warned that `sinc`, used by the SO(3) conversion path, falls
  back to CPU. The two pose conversion stages total about 17 ms and become a
  secondary target after PTV3 acceleration.
- Random weights are valid for this performance comparison because all tensor
  shapes, topology construction, pooling, attention, and dense kernels are
  unchanged. They are not valid for accuracy evaluation.
- These historical measurements manually orchestrated the CPU-PTV3/NPU-head
  split. The benchmark harness now contains self-contained encoder bridges and
  stage wrappers; release-scale results are in
  `ascend/graspgenx_latency_2048_release_scale.md`.
