#!/usr/bin/env bash
root="$(dirname "$(realpath "$0")")"
source "$root/../custom_ops/submconv3d/env.sh"
set -eo pipefail
# Separate processes keep JIT settings and import shims local to each suite.
for test in test_submconv3d.py test_build_subm_map.py test_ptv3_cpe.py; do
    PTV3_CPE_NPU=1 "$PYTHON_BIN" -B -u "$root/$test" "$@"
done
