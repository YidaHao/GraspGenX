# Full GraspGenX CUDA Reference

This directory is the frozen deterministic reference for the full pipeline
comparison on Ascend310P1.

- Input: 2048 points from the current PTV3 reference, 100 grasps, 20 diffusion steps.
- CUDA reference: FP32 eager, PTV3 Flash/SDPA disabled, TF32 disabled, deterministic algorithms enabled.
- `request.npz`: points, gripper condition, initial noise, and both per-step scheduler noise streams.
- `cuda_outputs.npz`: generator/discriminator embeddings, every diffusion prediction and latent, final grasps, logits, confidence.
- `summary.json`: generation configuration and timing summary.

## Three-Way Comparison

- `cuda_baselines.npz`: the identical frozen request plus complete outputs for
  CUDA `reference`, `native` and `trt` (keys use `request__` or `<name>__` prefixes).
- `cuda_baselines.json`: source configurations, selected-run timings, checksums
  and all three CUDA results compared to reference. These timings are the
  selected frozen runs, not the median of multiple processes.
- Existing request/output/summary files above remain untouched.

The CUDA generator packages existing canonical runs when all three exist. It
checks identical requests, preserves original outputs and refuses to overwrite
the comparison bundle. Transfer both sidecar files to this directory on 310P.

Full-model validator/profile read this bundle. Cosine and other numerical
errors are report-only; there is no accuracy threshold. Valid shapes and finite
outputs are still required. Exit 0 means a valid completed comparison, not
accuracy or grasp-success acceptance. PTV3-only accuracy gates are unchanged.
