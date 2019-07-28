<title>numpy从源代码构建 - <%-__DOC_NAME__ %></title>
<meta name="keywords" content="numpy从源代码构建" />

# 从源代码构建

此处给出了从源代码构建NumPy的一般概述，以及单独给出的特定平台的详细说明。

## 先决条件
构建NumPy需要安装以下软件：

1. Python 2.7.x、3.4.x的版本或是最新版本。

    在Debian和其衍生版本（Ubuntu）中需要：python，python-dev（或python3-dev）。

    在Windows上：[www.python.org](http://www.python.org/)上的官方python安装程序就足够了。
 
    在继续之前，请确保已安装Python包distutils。 例如，在Debian GNU / Linux中，安装python-dev也会安装distutils。

    还必须在启用zlib模块的情况下编译Python。对于预先包装好的Pythons来说，必要模块几乎已经全部搞定。

1. 编译程序
    要为Python构建任何扩展模块，你需要一个C编译器。NumPy的各种模块使用都使用了 FORTRAN 77 库，因此你还需要安装FORTRAN 77的编译器。

    请注意，NumPy主要是使用GNU编译器开发的。 其他供应商（如Intel，Absoft，Sun，NAG，Compaq，Vast，Portland，Lahey，HP，IBM，Microsoft）的编译器仅以社区反馈的形式提供支持，并且可能无法开箱即用。 建议使用GCC 4.x（及更高版本）编译器。

1. 线性代数库
    NumPy不需要安装任何额外线性代数库。但是，如果你有一些可用的库，NumPy的安装脚本也可以检测到它们并将它们构建到numpy里面。你可以使用许多不同的LAPACK库设置，包括优化的LAPACK库，如ATLAS，MKL或OS X上的Accelerate/vecLib框架。

1. Cython
    要构建NumPy的开发版本，你需要一个Cython的最新版本。PyPI上发布的NumPy源代码包括了Cython代码生成的C文件，因此不需要安装具有Cython的发行版本。

## 基本安装

安装NumPy运行下面的命令:

```
python setup.py install
```

如果要就地构建，可以从源代码文件夹里运行：

```
python setup.py build_ext --inplace
```

NumPy的构建系统使用的是``setuptools`` (从numpy 1.11.0开始, 之前都是用很普通的 distutils) 和 numpy.distutils. 使用 ``virtualenv`` 会按照预期的那样进行。

*请注意: 有关在NumPy本身进行开发工作的构建的说明，请参阅* [设置和使用开发环境](https://docs.scipy.org/doc/numpy/dev/development_environment.html#development-environment).

### 并行构建

在NumPy 1.10.0上，还可以使用以下命令进行并行构建：

```
python setup.py build -j 4 install --prefix $HOME/.local
```

这将在4个CPU上编译numpy并将其安装到指定的前缀中。 要执行并行的就地构建，请运行：

```
python setup.py build_ext --inplace -j 4
```

也可以通过环境变量``NPY_NUM_BUILD_JOBS``来指定构建作业的数量。


## FORTRAN ABI 不匹配

两个最受欢迎的开源fortran编译器是g77和gfortran。 不幸的是，它们不兼容ABI，这意味着你应该避免混合使用彼此构建的库。特别是，如果你的blas/lapack/atlas是使用g77构建的，那么在构建numpy和scipy时必须使用g77; 相反，如果你的atlas是用gfortran构建的，你必须用gfortran建立numpy/scipy。这适用于可能使用了不同FORTRAN编译器的大多数其他情况。

### 选择fortran编译器

使用gfortran来构建:

```
python setup.py build --fcompiler=gnu95
```

查看更多信息请运行:

```
python setup.py build --help-fcompiler
```

### 如何检查 blas/lapack/atlas 的 ABI

检查用于构建库的编译器的一种相对简单且可靠的方法是在库上使用ldd。 如果libg2c.so是依赖项，则表示已使用g77。 如果libgfortran.so是依赖项，则使用gfortran。如果两者都是依赖关系，这意味着两者都被使用，不过这绝对是一个非常傻比的想法。

## 禁用 ATLAS 和其他加速库

可以通过以下方式禁用NumPy中ATLAS和其他加速库的使用：

```
BLAS=None LAPACK=None ATLAS=None python setup.py build
```

## 提供额外的编译器标志

可以通过设置``OPT``，``FOPT``（对于Fortran）和``CC``环境变量来提供额外的编译器标志。

## 通过ATLAS的支持来构建

### Ubuntu

你可以使用以下命令为优化后的ATLAS安装必要的包：

```
sudo apt-get install libatlas-base-dev
```