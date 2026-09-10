#!/usr/bin/env bash
root="$(dirname "$(realpath "$0")")"
source "$root/env.sh"
set -eo pipefail
cd "$root"
exec "$PYTHON_BIN" -u test_submconv3d.py "$@"
