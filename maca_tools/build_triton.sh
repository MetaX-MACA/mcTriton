#!/bin/bash
cur_dir="$(cd "$(dirname "$0")" ; pwd -P)"
extmathlib_dir=$cur_dir/../third_party/mcExtMathLib/
extmathlib_install_dir=$cur_dir/../third_party/metax/backend/
llvm_release_dir=${2:-$cur_dir/../third_party/llvm_release/}
pybind11_release_dir=$cur_dir/../third_party/pybind11/pybind11-2.11.1/
json_release_dir=$cur_dir/../third_party/json/
python_triton_dir=$cur_dir/../python/

export LLVM_SYSPATH=$llvm_release_dir
export PYBIND11_SYSPATH=$pybind11_release_dir
export JSON_SYSPATH=$json_release_dir
export TRITON_BUILD_PROTON=OFF
export THIRDPARTY_MANUAL=1
if [[ $1 == "debug" ]]; then
  export DEBUG=1
else
  export DEBUG=0
fi

# mathlib func

# mlir-opt
mlir_opt_path=$llvm_release_dir/bin/mlir-opt
backend_path=$cur_dir/../third_party/metax/backend/bin/
if [[ ! -e "${backend_path}/mlir-opt" ]]; then
  mkdir -p $backend_path
  if [[ ! -e $mlir_opt_path ]]; then
      echo "mlir-opt does not exist"
      exit 1
  fi
  cp $mlir_opt_path $backend_path
fi

# mctriton
cd $python_triton_dir
python -m pip install cmake
python -m pip install .
