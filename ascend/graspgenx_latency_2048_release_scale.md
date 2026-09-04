# GraspGenX 2048-Point Release-Scale Latency Report

Date: 2026-08-29

## Conclusion

On Ascend310P1 at `192.168.7.101`, the release-scale FP32 pipeline with 2048
points, 100 grasps, and 20 diffusion steps has a steady-state median latency of
**6.734 seconds** and P95 latency of **6.983 seconds**. Throughput is about
**0.149 Hz**.

This is too slow for online grasp planning. A 1-second service objective needs
roughly a 6.7x end-to-end improvement. A 500 ms objective cannot be reached by
accelerating PTV3 alone because the measured NPU dense-path floor, with both
PTV3 encoders replaced by zero embeddings, is already **663 ms median**.

The official GraspGenX paper and project do not publish an inference-latency or
FPS benchmark, so these results should be compared with application-level
latency budgets rather than an official GraspGenX timing target.

## Benchmark Contract

| Setting | Value |
|---|---|
| Device | Ascend310P1, `npu:0` |
| PyTorch / torch-npu | 2.5.1 / 2.5.1 |
| Precision | FP32 |
| Point count | 2048 |
| Grasps per object | 100 |
| Diffusion steps | 20 |
| Generator representation | `r3_so3` |
| Scheduler | Compositional |
| Diffusion attention | `cat_attn` |
| Grid size | 0.01 |
| Kappa | 3.27 |
| Gripper conditioning | `sweep_volume_v2` |
| PTV3 encoders | CPU, one generator call and one discriminator call per request |
| Dense generator/discriminator modules | NPU |
| CPU intra-op threads | 16 |
| Weights | Deterministic random, release-compatible shapes |
| Input | `default_rng(1234)`, normal(0, 0.05), mean-centered |

The deployment fields were copied from the official release configs:

- `https://huggingface.co/adithyamurali/GraspGenXModel/resolve/main/release/gen/config.yaml`
- `https://huggingface.co/adithyamurali/GraspGenXModel/resolve/main/release/dis/config.yaml`

Random weights preserve topology, tensor shapes, operator selection, and PTV3
workload for this performance measurement. They are not valid for accuracy or
grasp-success evaluation.

Each formal run used a fresh process, one cold/warmup request, then a 30-second
steady-state window. Output shapes were `[1, 100, 4, 4]` for grasps and
`[1, 100, 1]` for confidence. All checked outputs were finite.

## Formal Results

| Process | Runs | Mean | Median | P95 | Min-max | Std. dev. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 6539.38 ms | 6522.24 ms | 6621.87 ms | 6484.12-6639.64 ms | 54.90 ms |
| 2 | 5 | 6769.74 ms | 6734.08 ms | 6845.89 ms | 6704.86-6846.59 ms | 62.01 ms |
| 3 | 5 | 6887.33 ms | 6840.39 ms | 7038.81 ms | 6786.77-7061.23 ms | 104.07 ms |
| **Pooled** | **15** | **6732.15 ms** | **6734.08 ms** | **6982.74 ms** | **6484.12-7061.23 ms** | **163.65 ms** |

The process medians were 6.522, 6.734, and 6.840 seconds. Their spread is much
larger than 1%, so CPU load and frequency state remain relevant sources of
variance. The pooled median and P95 are the most useful operational values.

## Dense-Path Floor

Both PTV3 encoders were replaced with correctly shaped zero embeddings while
the complete 100-grasp, 20-step generator, pose conversion, scheduler, and
discriminator paths still ran on NPU.

| Runs | Mean | Median | P95 | Min-max | Throughput |
|---:|---:|---:|---:|---:|---:|
| 46 | 662.32 ms | **663.08 ms** | 676.25 ms | 638.87-682.33 ms | 1.510 Hz |

This is a diagnostic lower bound, not a valid model output. It shows:

- Perfectly eliminating both PTV3 calls would reduce 6.732 seconds to about
  0.662 seconds, a theoretical maximum speedup of about 10.2x.
- A 1-second target leaves only about 337 ms for both PTV3 encoders combined.
- A 500 ms target is impossible without also accelerating the dense diffusion,
  scheduler, and pose-conversion path.

## Synchronized Stage Profile

The profile uses NPU synchronization at stage boundaries and is diagnostic; its
end-to-end time should not replace the unprofiled formal latency above.

| Stage | Mean per request | Share of profiled E2E |
|---|---:|---:|
| Generator total | 3724.67 ms | 54.95% |
| Generator CPU PTV3 | 3038.23 ms | 44.82% |
| Diffusion head, 20 calls total | 298.92 ms | 4.41% |
| Other generator work | 387.52 ms | 5.72% |
| Discriminator total | 3053.06 ms | 45.04% |
| Discriminator CPU PTV3 | 3034.46 ms | 44.77% |
| Discriminator prediction head | 1.99 ms | 0.03% |
| Other discriminator work | 16.61 ms | 0.25% |
| **Profiled end-to-end** | **6778.32 ms** | **100%** |

Each diffusion-head call averaged 14.946 ms, so 20 calls consume about 299 ms.
The two PTV3 calls consume about 6.073 seconds, or 89.6% of profiled latency.

The remaining generator work includes gripper conditioning, scheduler steps,
likelihood calculation, SO(3) conversion, and per-iteration output handling.
The NPU runtime reports that `sinc` falls back to CPU, making this path the next
optimization priority after PTV3.

## Cold Start

| Measurement | Observed value |
|---|---:|
| Setup median across three formal processes | 40.93 s |
| First request median across three formal processes | 93.31 s |
| Setup plus first result | about 134.24 s |
| First-ever fresh-shape compile observed in dense-floor run | 434.59 s |

Cold compilation is not part of steady latency, but it makes an on-demand
process unsuitable for production. Deployment must use a persistent prewarmed
service. A health check should not report ready until the production shape has
completed at least one inference.

## Comparison With The One-Step Microbenchmark

| Contract | Median |
|---|---:|
| 2048 points, 1 grasp, 1 step | 5956.25 ms |
| 2048 points, 100 grasps, 20 steps | 6734.08 ms |
| Increase | 777.83 ms, or 13.1% |

PTV3 is nearly shape-independent with respect to grasp count and diffusion
steps, so most of the increase comes from the 20-step generator loop. PTV3's
share falls from 99.22% in the microbenchmark to about 89.6% at release scale.

## Engineering Gates

### Minimum Online Target: 1 Second

Keeping the current 663 ms dense floor leaves about 337 ms for both PTV3 calls.
Their current pooled mean is 6.034 seconds, so they need approximately an 18x
speedup, with a practical combined target of at most 300 ms.

### Recommended Industrial Target: 500 Milliseconds

PTV3 acceleration alone cannot reach this target. A plausible budget is:

| Component | Target budget |
|---|---:|
| Both PTV3 encoders | <=250 ms total |
| Complete dense path | <=250 ms |
| End-to-end | <=500 ms |

That requires about 24x acceleration for the two PTV3 encoders and about 2.7x
for the measured dense path.

## Recommended Optimization Order

1. Keep inference in a persistent, prewarmed process; cold-start latency is a
   separate release blocker.
2. Move, replace, or distill both PTV3 encoders. For a 1-second first milestone,
   target no more than 150 ms per encoder.
3. Optimize the 20-step generator loop. Target 5-6 ms per diffusion-head call
   and remove avoidable per-step host synchronization or CPU output copies.
4. Eliminate the NPU `sinc` fallback in SO(3) conversion and profile scheduler
   operations separately.
5. Repeat performance and accuracy validation with the real release checkpoint;
   random weights establish performance only.

## Artifacts

- `ascend/tools/benchmark_device.py`: reproducible benchmark harness.
- `ascend/sample-data/release-config/`: deployment-relevant official config fields.
- `ascend/results/npu_2048_g100_s20_ptv3_cpu_run1.json`: formal process 1.
- `ascend/results/npu_2048_g100_s20_ptv3_cpu_run2.json`: formal process 2.
- `ascend/results/npu_2048_g100_s20_ptv3_cpu_run3.json`: formal process 3.
- `ascend/results/npu_2048_g100_s20_ptv3_cpu_profiled.json`: synchronized stage profile.
- `ascend/results/npu_2048_g100_s20_dense_floor.json`: dense-path diagnostic floor.

## Reproduction

Run from the repository root on `192.168.7.101` after loading CANN:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHONPATH=$PWD:$PYTHONPATH python3 ascend/tools/benchmark_device.py \
  --device npu \
  --checkpoint-root ascend/sample-data/release-config \
  --assets-dir assets \
  --warmup 1 \
  --duration 30 \
  --point-count 2048 \
  --num-grasps 100 \
  --diffusion-steps 20 \
  --precision fp32 \
  --random-weights \
  --ptv3-cpu \
  --output ascend/results/npu_2048_g100_s20_ptv3_cpu_run.json
```
