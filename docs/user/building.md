---
meta:
  - name: keywords
    content: NumPy 从源码构建
  - name: description
    content: 此处给出了从源代码构建NumPy的一般概述，以及单独给出的特定平台的详细说明。
---

# 从源码构建

此处给出了从源代码构建NumPy的一般概述，以及单独给出的特定平台的详细说明。

## 先决条件

构建NumPy需要安装以下软件：

## 基本安装

要安装NumPy运行：

``` bash
pip install .
```

要执行可以从源文件夹运行的就地构建：

``` bash
python setup.py build_ext --inplace
```

NumPy构建系统使用``setuptools``（从numpy 1.11.0开始，之前很简单``distutils``）和``numpy.distutils``。使用``virtualenv``应该按预期工作。

*注意：有关在NumPy上进行开发工作的构建说明，请参阅* 
[设置和使用开发环境](/dev/development_environment.html)。

## 测试

确保测试您的构建。为了确保一切都保持稳定，请查看是否所有测试都通过：

``` python
$ python runtests.py -v -m full
```

有关测试的详细信息，请参阅[测试版本](/dev/development_environment.html#测试构建)。

### 并行构建

从NumPy 1.10.0起，它也可以用以下方式进行并行构建：

``` python
python setup.py build -j 4 install --prefix $HOME/.local
```

这将在4个CPU上编译numpy并将其安装到指定的前缀中。要执行并行的就地构建，请运行：

``` python
python setup.py build_ext --inplace -j 4
```

也可以通过环境变量指定构建作业的数量
 ``NPY_NUM_BUILD_JOBS``。

## FORTRAN ABI不匹配

两个最受欢迎的开源fortran编译器是g77和gfortran。不幸的是，它们不兼容ABI，
这意味着你应该避免混合使用彼此构建的库。特别是，如果您的blas / lapack / atlas是使用g77构建的，
那么在构建numpy和scipy时 *必须* 使用g77; 
相反，如果你的地图集是用gfortran 构建的，你 *必须* 用gfortran
建立numpy / scipy。这适用于可能使用了不同FORTRAN编译器的大多数其他情况。

### 选择fortran编译器

用gfortran构建：

``` python
python setup.py build --fcompiler=gnu95
```

有关更多信息，请运行帮助命令：

``` python
python setup.py build --help-fcompiler
```

### 如何检查BLAS / LAPACK /地图集ABI 

检查用于构建库的编译器的一种相对简单且可靠的方法是在库上使用ldd。
如果libg2c.so是依赖项，则表示已使用g77。
如果libgfortran.so是依赖项，则使用gfortran。
如果两者都是依赖关系，这意味着两者都已被使用，这几乎总是一个非常糟糕的主意。

## 加速BLAS / LAPACK库

NumPy搜索优化的线性代数库，如BLAS和LAPACK。搜索这些库有特定的顺序，如下所述。

### BLAS 

库的默认顺序是：

1. MKL
1. BLIS
1. OpenBLAS
1. ATLAS
1. 加速（MacOS）
1. BLAS（NetLIB）

如果您希望针对OpenBLAS进行构建，但您也可以使用BLIS，则可以通过环境变量预定义搜索顺序，该变量
 ``NPY_BLAS_ORDER``是用于确定要搜索内容的上述名称的逗号分隔列表，例如：

``` python
NPY_BLAS_ORDER=ATLAS,blis,openblas,MKL python setup.py build
```

我更喜欢使用ATLAS，然后是BLIS，然后是OpenBLAS，最后是MKL。如果这些都不存在，则构建将失败（名称将比较小写）。

### LAPACK 

库的默认顺序是：

1. MKL
1. OpenBLAS
1. libFLAME
1. ATLAS
1. 加速（MacOS）
1. LAPACK（NetLIB）

如果您希望针对OpenBLAS进行构建，但您也可以使用MKL，则可以通过环境变量预定义搜索顺序，该变量
 ``NPY_LAPACK_ORDER``是以逗号分隔的上述名称列表，例如：

``` python
NPY_LAPACK_ORDER=ATLAS,openblas,MKL python setup.py build
```

我希望使用ATLAS，然后使用OpenBLAS，作为最后的手段使用MKL。如果这些都不存在，则构建将失败（名称将比较小写）。

### 禁用ATLAS和其他加速库

可以通过以下方式禁用在NumPy中使用ATLAS和其他加速库：

``` python
NPY_BLAS_ORDER= NPY_LAPACK_ORDER= python setup.py build
```

要么：

``` python
BLAS=None LAPACK=None ATLAS=None python setup.py build
```

## 提供额外的编译器标志

额外的编译器标记可以通过设置来提供``OPT``，
 ``FOPT``（Fortran的），和``CC``环境变量。提供应该提高代码性能的选项时，请确保还要设置``-DNDEBUG``为不执行调试代码。

## 使用ATLAS支持构建

### Ubuntu的

您可以使用以下命令为优化的ATLAS安装必要的包：

``` python
sudo apt-get install libatlas-base-dev
```
