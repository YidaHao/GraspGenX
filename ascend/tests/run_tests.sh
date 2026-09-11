#!/usr/bin/env bash
root="$(dirname "$(realpath "$0")")"
source "$root/../custom_ops/submconv3d/env.sh"
source "$root/../custom_ops/grid_encode/env.sh" || exit $?
set -eo pipefail
# Separate processes keep JIT settings and import shims local to each suite.
for test in test_submconv3d.py test_build_subm_map.py test_ptv3_cpe.py test_grid_encode.py test_ptv3_grid_encode.py; do
    ASCEND_TEST_NPU=1 PTV3_CPE_NPU=1 "$PYTHON_BIN" -B -u "$root/$test" "$@"
done
