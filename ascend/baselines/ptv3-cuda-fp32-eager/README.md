# PTV3 CUDA FP32 Eager Baseline

The tracked `reference_n*.npz` files contain deterministic point inputs and
CUDA outputs for the release generator and discriminator PTV3 encoders. They
were generated on an RTX 3090 with PyTorch 2.5.1+cu124, FP32, Flash attention
disabled, and serialization-order shuffling disabled on every module.

`validate_ptv3.py` and `profile_ptv3_stages.py` also require two local weight
files in this directory. They are intentionally ignored by Git because each is
about 148 MiB:

| File | SHA-256 |
|---|---|
| `generator_ptv3_state.pth` | `f017bc75315737a5eb088badb18dfebdd719483f669b694e113e47f45783b4e6` |
| `discriminator_ptv3_state.pth` | `0ed58567048bfde97b716d04d85dc58c487b27ad103a5fc8714ca86a7d756dd1` |

They contain only the `object_encoder.*` state dictionaries extracted from the
official GraspGenX release checkpoints. The validation script verifies these
hashes before running.

Reference hashes:

| File | SHA-256 |
|---|---|
| `reference_n64.npz` | `e033f95d973202ed4c8c54d4d6c30078389f33a2e1d581d648425eea3a139675` |
| `reference_n2048.npz` | `e0d901235e755404522ad86bb9a651b7d0c93bc9dc6851b0178686be194d14ff` |
| `reference_n3500.npz` | `9df08f2e7b06631dfdd721356691b178409c4d5e11bd553e65fe68eaac38adca` |
