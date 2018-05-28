==================================
从源代码构建
==================================

这里给出了从源代码构建NumPy的总体概述，并分别给出了特定平台的详细说明。

----------------------------------
先决条件
----------------------------------

构建NumPy需要安装以下软件：

	1. Python 2.7.x，3.4.x或更新版本

		在Debian和衍生产品（Ubuntu）上：python，python-dev（或python3-dev）在Windows上：http://www.python.org/官方的python安装程序就足够了。确保在继续之前安装了Python程序包distutils。例如，在Debian GNU / Linux中，安装python-dev也会安装distutils。还必须在启用zlib模块的情况下编译Python。事实上，预打包的Pythons就是这种情况。

	2. 编译器

		要为Python构建任何扩展模块，您需要一个C编译器。各种NumPy模块使用FORTRAN 77库，因此您还需要安装FORTRAN 77编译器。

		请注意，NumPy主要是使用GNU编译器开发的。来自其他供应商（如英特尔，Absoft，Sun，NAG，Compaq，Vast，Porland，Lahey，HP，IBM，Microsoft）的编译器仅以社区反馈的形式提供支持，并且可能无法使用。建议使用GCC 4.x（及更高版本）编译器。

	3. 线性代数库

		NumPy不需要安装任何外部线性代数库。但是，如果这些可用，NumPy的安装脚本可以检测到它们并将其用于构建。可以使用许多不同的LAPACK库设置，包括优化的LAPACK库，如ATLAS，MKL或OS X上的Accelerate / vecLib框架。

	4.	用Cython

			要构建NumPy的开发版本，您需要最新版本的Cython。在PyPi上发布的NumPy源代码包括从Cython代码生成的C文件，因此不需要安装Cython的发布版本。

----------------------------------
基本安装
----------------------------------

要安装NumPy运行：

.. code-block:: python

    > python setup.py install

To perform an in-place build that can be run from the source folder run:

.. code-block:: python

    > python setup.py build_ext --inplace

The NumPy build system uses ``setuptools`` (from numpy 1.11.0, before that it was plain ``distutils``) and ``numpy.distutils``. Using ``virtualenv`` should work as expected.

*Note: for build instructions to do development work on NumPy itself, see Setting up and using your development environment.*

----------------------------------
Parallel builds
----------------------------------

From NumPy 1.10.0 on its also possible to do a parallel build with:

.. code-block:: python

 	> python setup.py build -j 4 install --prefix $HOME/.local

This will compile numpy on 4 CPUs and install it into the specified prefix. to perform a parallel in-place build, run:

.. code-block:: python

 	> python setup.py build_ext --inplace -j 4

The number of build jobs can also be specified via the environment variable ``NPY_NUM_BUILD_JOBS`` .

----------------------------------
FORTRAN ABI mismatch
----------------------------------

The two most popular open source fortran compilers are g77 and gfortran. Unfortunately, they are not ABI compatible, which means that concretely you should avoid mixing libraries built with one with another. In particular, if your blas/lapack/atlas is built with g77, you must use g77 when building numpy and scipy; on the contrary, if your atlas is built with gfortran, you must build numpy/scipy with gfortran. This applies for most other cases where different FORTRAN compilers might have been used.

----------------------------------
Choosing the fortran compiler
----------------------------------

To build with gfortran:

.. code-block:: python

	> python setup.py build --fcompiler=gnu95

For more information see:

.. code-block:: python

	> python setup.py build --help-fcompiler

------------------------------------------
How to check the ABI of blas/lapack/atlas
------------------------------------------
One relatively simple and reliable way to check for the compiler used to build a library is to use ldd on the library. If libg2c.so is a dependency, this means that g77 has been used. If libgfortran.so is a dependency, gfortran has been used. If both are dependencies, this means both have been used, which is almost always a very bad idea.

-------------------------------------------------------
Disabling ATLAS and other accelerated libraries
-------------------------------------------------------
Usage of ATLAS and other accelerated libraries in NumPy can be disabled via:

.. code-block:: python

	> BLAS=None LAPACK=None ATLAS=None python setup.py build

--------------------------------------------
Supplying additional compiler flags
--------------------------------------------

Additional compiler flags can be supplied by setting the ``OPT``, ``FOPT`` (for Fortran), and ``CC`` environment variables.

--------------------------------------------
Building with ATLAS support
--------------------------------------------

^^^^^^^^^^^^^^^^
Ubuntu
^^^^^^^^^^^^^^^^

You can install the necessary package for optimized ATLAS with this command:

.. code-block:: python

	> sudo apt-get install libatlas-base-dev
