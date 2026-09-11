#!/usr/bin/env bash
# Source this file. All CANN/cache changes are process-local.
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/8.1.RC1}"
source "$ASCEND_HOME_PATH/$(uname -m)-linux/bin/setenv.bash"
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export TOOLCHAIN_HOME="$ASCEND_HOME_PATH/toolkit"
# CANN 8.1 warns on an unset switch, although it already behaves like 0.
export TILINGKEY_PAR_COMPILE="${TILINGKEY_PAR_COMPILE:-0}"
export PYTHONPATH="$ASCEND_HOME_PATH/toolkit/python/site-packages:$ASCEND_HOME_PATH/compiler/python/site-packages:${PYTHONPATH:-}"
# Keep openEuler's installed /usr/local Python packages visible to CANN.
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON_BIN"):$ASCEND_HOME_PATH/compiler/ccec_compiler/bin:$ASCEND_HOME_PATH/compiler/bin:$PATH"
# ccec does not discover the host libstdc++ headers automatically.
_subm_gcc="$(g++ -dumpversion)"
_subm_target="$(g++ -dumpmachine)"
_subm_config="/usr/include/c++/$_subm_gcc/$_subm_target"
if [[ ! -d "$_subm_config" ]]; then
    _subm_config="/usr/include/$_subm_target/c++/$_subm_gcc"
fi
export CPLUS_INCLUDE_PATH="/usr/include/c++/$_subm_gcc:$_subm_config:/usr/include/c++/$_subm_gcc/backward:$(g++ -print-file-name=include)${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
unset _subm_gcc _subm_target _subm_config
_subm_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
if [[ -f "$_subm_dir/opp/vendors/graspgenx_subm/bin/set_env.bash" ]]; then
    source "$_subm_dir/opp/vendors/graspgenx_subm/bin/set_env.bash"
fi
export ASCEND_WORK_PATH="$_subm_dir/build/runtime"
export PYTHONPATH="$(realpath "$_subm_dir/../../.."):${PYTHONPATH:-}"
unset _subm_dir
