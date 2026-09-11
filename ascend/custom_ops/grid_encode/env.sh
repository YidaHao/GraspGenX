#!/usr/bin/env bash
# Source this file. All changes are process-local; private paths are relocatable.
_grid_pythonpath="${PYTHONPATH:-}"
_grid_opp_path="${ASCEND_CUSTOM_OPP_PATH:-}"
_grid_library_path="${LD_LIBRARY_PATH:-}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/8.1.RC1}}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
source "$ASCEND_HOME_PATH/$(uname -m)-linux/bin/setenv.bash" || return $?
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export TOOLCHAIN_HOME="$ASCEND_HOME_PATH/toolkit"
export TILINGKEY_PAR_COMPILE="${TILINGKEY_PAR_COMPILE:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$ASCEND_HOME_PATH/compiler/ccec_compiler/bin:$ASCEND_HOME_PATH/compiler/bin:$PATH"
_grid_gcc="$(g++ -dumpversion)"
_grid_target="$(g++ -dumpmachine)"
_grid_config="/usr/include/c++/$_grid_gcc/$_grid_target"
if [[ ! -d "$_grid_config" ]]; then
    _grid_config="/usr/include/$_grid_target/c++/$_grid_gcc"
fi
export CPLUS_INCLUDE_PATH="/usr/include/c++/$_grid_gcc:$_grid_config:/usr/include/c++/$_grid_gcc/backward:$(g++ -print-file-name=include)${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
_grid_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
_grid_vendor="$_grid_dir/opp/vendors/graspgenx_grid"
# CANN may reset these lists. Retain caller entries, including other vendors.
if [[ -n "$_grid_pythonpath" && "$_grid_pythonpath" != "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$_grid_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ASCEND_HOME_PATH/toolkit/python/site-packages:$ASCEND_HOME_PATH/compiler/python/site-packages:$(realpath "$_grid_dir/../../..")"
if [[ -n "$_grid_opp_path" && "$_grid_opp_path" != "${ASCEND_CUSTOM_OPP_PATH:-}" ]]; then
    export ASCEND_CUSTOM_OPP_PATH="$_grid_opp_path${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
fi
if [[ -n "$_grid_library_path" && "$_grid_library_path" != "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$_grid_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Do not source generated set_env.bash: it embeds the install-time absolute path.
case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
    *":$_grid_vendor:"*) ;;
    *) export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:+$ASCEND_CUSTOM_OPP_PATH:}$_grid_vendor" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
    *":$_grid_vendor/op_api/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_grid_vendor/op_api/lib" ;;
esac
export ASCEND_WORK_PATH="$_grid_dir/build/runtime"
unset _grid_pythonpath _grid_opp_path _grid_library_path _grid_gcc _grid_target _grid_config _grid_dir _grid_vendor
