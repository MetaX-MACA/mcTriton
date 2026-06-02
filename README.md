## 源码编译

### 1. 依赖安装

##### 1.1 沐曦软件栈

参考[环境准备](https://github.com/MetaX-MACA/mcPytorch/blob/2.4/README.md#1-%E5%AE%89%E8%A3%85)准备沐曦软件栈环境。

### 2. 编译 mcTriton

##### 2.1 拉取源代码

``` shell
git clone https://github.com/MetaX-MACA/mcTriton.git
```

##### 2.2 获取编译器

编译 mcTriton 需要的编译器可以从沐曦开发者社区中的 mcPytorch 安装包中获取，具体步骤如下：

1. 访问 https://developer.metax-tech.com/softnova/search?package_name=torch 搜索 mcPytorch 安装包。

2. 选取 mcPytorch 2.4/2.6/2.8 版本的任一安装包下载，例如 https://developer.metax-tech.com/softnova/search?package_name=maca-pytorch2.8-py312-3.7.2.0-x86_64.tar.xz。

3. 解压后找到 metax_llvm*.tar.xz 文件并再次解压该文件获得 metax_llvm 目录即可。


##### 2.3 编译

``` shell

export MACA_PATH=/opt/maca/
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib/:$LD_LIBRARY_PATH
./maca_tools/build_triton.sh release /path_to_metax_llvm_in_step2.2

```
