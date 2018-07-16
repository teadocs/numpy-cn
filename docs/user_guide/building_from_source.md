# 从源代码构建

此处给出了从源代码构建NumPy的一般概述，以及单独给出的特定平台的详细说明。

## 先决条件
构建NumPy需要安装以下软件：

1. Python 2.7.x、3.4.x的版本或是最新版本。

    在Debian和其衍生版本（Ubuntu）中需要：python，python-dev（或python3-dev）。

    在Windows上：www.python.org上的官方python安装程序就足够了。
 
    在继续之前，请确保已安装Python包distutils。 例如，在Debian GNU / Linux中，安装python-dev也会安装distutils。

    还必须在启用zlib模块的情况下编译Python。对于预先包装好的Pythons来说，必要模块几乎已经全部搞定。

1. 编译程序
    To build any extension modules for Python, you’ll need a C compiler. Various NumPy modules use FORTRAN 77 libraries, so you’ll also need a FORTRAN 77 compiler installed.

    Note that NumPy is developed mainly using GNU compilers. Compilers from other vendors such as Intel, Absoft, Sun, NAG, Compaq, Vast, Portland, Lahey, HP, IBM, Microsoft are only supported in the form of community feedback, and may not work out of the box. GCC 4.x (and later) compilers are recommended.

1. Linear Algebra libraries
    NumPy does not require any external linear algebra libraries to be installed. However, if these are available, NumPy’s setup script can detect them and use them for building. A number of different LAPACK library setups can be used, including optimized LAPACK libraries such as ATLAS, MKL or the Accelerate/vecLib framework on OS X.

1. Cython
    To build development versions of NumPy, you’ll need a recent version of Cython. Released NumPy sources on PyPi include the C files generated from Cython code, so for released versions having Cython installed isn’t needed.

## Basic Installation

To install NumPy run:

```sh
python setup.py install
```

To perform an in-place build that can be run from the source folder run:

```sh
python setup.py build_ext --inplace
```

The NumPy build system uses ``setuptools`` (from numpy 1.11.0, before that it was plain distutils) and numpy.distutils. Using ``virtualenv`` should work as expected.

*Note: for build instructions to do development work on NumPy itself, see* [Setting up and using your development environment](https://docs.scipy.org/doc/numpy/dev/development_environment.html#development-environment).

### Parallel builds

From NumPy 1.10.0 on it’s also possible to do a parallel build with:

```sh
python setup.py build -j 4 install --prefix $HOME/.local
```

This will compile numpy on 4 CPUs and install it into the specified prefix. to perform a parallel in-place build, run:

```sh
python setup.py build_ext --inplace -j 4
```

The number of build jobs can also be specified via the environment variable ``NPY_NUM_BUILD_JOBS``.


## FORTRAN ABI mismatch

The two most popular open source fortran compilers are g77 and gfortran. Unfortunately, they are not ABI compatible, which means that concretely you should avoid mixing libraries built with one with another. In particular, if your blas/lapack/atlas is built with g77, you must use g77 when building numpy and scipy; on the contrary, if your atlas is built with gfortran, you must build numpy/scipy with gfortran. This applies for most other cases where different FORTRAN compilers might have been used.

### Choosing the fortran compiler

To build with gfortran:

```sh
python setup.py build --fcompiler=gnu95
```

For more information see:

```sh
python setup.py build --help-fcompiler
```

### How to check the ABI of blas/lapack/atlas

One relatively simple and reliable way to check for the compiler used to build a library is to use ldd on the library. If libg2c.so is a dependency, this means that g77 has been used. If libgfortran.so is a dependency, gfortran has been used. If both are dependencies, this means both have been used, which is almost always a very bad idea.

## Disabling ATLAS and other accelerated libraries

Usage of ATLAS and other accelerated libraries in NumPy can be disabled via:

```sh
BLAS=None LAPACK=None ATLAS=None python setup.py build
```

## Supplying additional compiler flags

Additional compiler flags can be supplied by setting the ``OPT``, ``FOPT`` (for Fortran), and ``CC`` environment variables.

## Building with ATLAS support

### Ubuntu

You can install the necessary package for optimized ATLAS with this command:

```sh
sudo apt-get install libatlas-base-dev
```