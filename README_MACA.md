# 源码编译

### 1. 安装沐曦软件栈

### 2. 编译 mcTriton

``` shell

export MACA_PATH=/opt/maca/
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib/:$LD_LIBRARY_PATH
./maca_tools/build_triton.sh release ${metax_llvm_path}

````
