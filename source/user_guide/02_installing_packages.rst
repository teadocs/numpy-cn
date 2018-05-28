==================================
安装
==================================
以下是在SciPy生态系统中的工具软件包的安装说明。

----------------------------------
各种发行版
----------------------------------
对于许多用户来说，特别是在Windows上，最简单的方法就是下载下面的Python发行版进行安装，重要发行版的有以下几个：

* Anaconda: 这是一个免费Python发行包且自带丰富的科学计算库。它支持Linux，Windows和Mac。
* Enthought Canopy：它拥有免费版本和商业版本且携带了核心科学计算库。 它支持Linux，Windows和Mac。
* Python(x, y)：基于Spyder IDE的免费发行版，也自带科学计算库。 仅限Windows。
* WinPython：免费的发行版，也自带科学计算库。 仅限Windows。
* Pyzo：基于Anaconda和IEP交互式开发环境的免费发行版。 支持Linux，Windows和Mac。

----------------------------------
通过 pip 命令来安装
----------------------------------

大多数的流行的Python开源项目，作者都会将自己的包上传到 Python Package 包管理库当中去。

这样就可以使用Python的标准pip包管理器安装在大多数操作系统上。

请注意，你需要在系统上安装 Python 和 pip。

你可以通过以下命令来安装软件包：

.. code-block:: python

    > python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

我们建议使用普通用户来安装你所需的包。

你可以在 pip 命令后使用使用 ``--user`` 参数来指定用户进行安装（注意：不要使用sudo pip，这会导致某些严重的问题），运行命令后 pip 会为你的本地用户安装软件包，不会写入系统目录。

----------------------------------
通过Linux包管理器安装
----------------------------------

Linux上的用户可以从Linux发行版提供的包管理器安装我们的Python软件包。 

值得注意的是：用Linux自带的包管理器安装会导致使用最高权限安装，并且可能导致安装的包的版本会比pip命令安装的包版本更旧。

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Ubuntu & Debian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    > sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

用户也可能想添加 `Neuro Debian <http://neuro.debian.net/>`_ 来获取更多的SciPy包。

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Fedora
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fedora 22及更高版本：

.. code-block:: python

    > sudo dnf install numpy scipy python-matplotlib ipython python-pandas sympy python-nose atlas-devel

----------------------------------
Mac系统的安装方式
----------------------------------

Mac 系统没有预装软件的包管理器，但可以安装一些常用的软件包管理器。

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Macports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

假如你使用的是 Python3.5，可以在终端中执行 macports 命令来安装python的包：

.. code-block:: python

    > sudo port install py35-numpy py35-scipy py35-matplotlib py35-ipython +notebook py35-pandas py35-sympy py35-nose

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Homebrew
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

通过下面的这行命令，你可以直接安装 NumPy、SciPy 和 Matplotlib

.. code-block:: python

    > brew tap homebrew/science && brew install python numpy scipy matplotlib

----------------------------------
其他的选择
----------------------------------

正如之前所说的，大部分的官方二进制包和源代码包都可以通过pip来获取。
二进制包也可以从第三方获得，比如上面所说的各种Python发行版。 
对于Windows来说，Christoph Gohlke 为许多Python的包提供了 `预构建的Windows安装程序。 <http://www.lfd.uci.edu/~gohlke/pythonlibs>`_

----------------------------------
源码包
----------------------------------

您可以从源代码构建任何包，例如，如果您想参与开发Python包，那么用Python来编写的包是最直接也是最简单的办法，但是如果是类似于NumPy这样的包则需要编译C代码。

请参阅个别项目了解更多详情。