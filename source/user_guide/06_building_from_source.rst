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