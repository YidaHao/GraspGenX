#!/usr/bin/env bash
root="$(dirname "$(realpath "$0")")"
source "$root/env.sh"
set -eo pipefail
mkdir -p "$root/build/runtime" "$root/opp"
project="$root/build/opp_project"
generator="$ASCEND_HOME_PATH/toolkit/python/site-packages/op_gen/msopgen.py"
if [[ ! -f "$project/CMakeLists.txt" ]]; then
    "$PYTHON_BIN" "$generator" gen -i "$root/ops.json" -op BuildSubmMap \
        -f pytorch -c ai_core-ascend310p -out "$project" -lan cpp
    "$PYTHON_BIN" "$generator" gen -i "$root/ops.json" -op SubmConv3d \
        -f pytorch -c ai_core-ascend310p -out "$project" -lan cpp -m 1
fi
cp "$root"/op_host/* "$project/op_host/"
cp "$root"/op_kernel/* "$project/op_kernel/"
cmake -G "Unix Makefiles" -S "$project" -B "$root/build/cmake_opp" \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DASCEND_COMPUTE_UNIT=ascend310p -Dvendor_name=graspgenx_subm \
    -DASCEND_PYTHON_EXECUTABLE="$PYTHON_BIN" \
    -DENABLE_SOURCE_PACKAGE=ON -DENABLE_BINARY_PACKAGE=ON -DENABLE_TEST=OFF
# CANN 8.1 leaves these temporary directories behind after failed compilation.
rm -rf "$root/build/cmake_opp/op_kernel/BuildSubmMap_ascend310p" \
       "$root/build/cmake_opp/op_kernel/SubmConv3d_ascend310p"
cmake --build "$root/build/cmake_opp" --target binary -j2
cmake --build "$root/build/cmake_opp" --target package -j2
packages=("$project"/build_out/*.run)
[[ ${#packages[@]} == 1 && -f "${packages[0]}" ]]
bash "${packages[0]}" --install-path="$root/opp" --quiet

torch_dir="$($PYTHON_BIN -c 'import torch; print(torch.__path__[0])')"
npu_dir="$($PYTHON_BIN -c 'import torch_npu; print(torch_npu.__path__[0])')"
python_include="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_path("include"))')"
abi="$($PYTHON_BIN -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
${CXX:-c++} -O2 -std=c++17 -fPIC -shared "$root/torch_bridge.cpp" \
    -o "$root/build/torch_bridge.so" -D_GLIBCXX_USE_CXX11_ABI="$abi" \
    -DTORCH_EXTENSION_NAME=submconv3d_bridge \
    -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
    -I"$npu_dir/include" -I"$python_include" \
    -L"$torch_dir/lib" -L"$npu_dir/lib" \
    -Wl,-rpath,"$torch_dir/lib" -Wl,-rpath,"$npu_dir/lib" \
    -ltorch -ltorch_cpu -ltorch_python -lc10 -ltorch_npu
