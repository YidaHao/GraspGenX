#!/usr/bin/env bash
# Usage from the repository root: source ascend/env.sh
# Keep the caller's current Python/venv; do not activate another environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Use "source ascend/env.sh" so the environment stays in your shell.\n' >&2
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh || return
source ascend/custom_ops/submconv3d/env.sh
export PYTHONPATH="$(dirname -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")")${PYTHONPATH:+:$PYTHONPATH}"