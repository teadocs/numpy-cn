---
meta:
  - name: keywords
    content: 设置和使用您的 Numpy 开发环境
  - name: description
    content: 由于NumPy包含用C和Cython编写的部分，需要在使用前进行编译，因此请确保安装了必要的编译器和Python开发头...
---

# 设置和使用您的开发环境

## 推荐的开发设置

由于NumPy包含用C和Cython编写的部分，需要在使用前进行编译，因此请确保安装了必要的编译器和Python开发头 - 请参阅[从源代码构建](https://numpy.org/devdocs/user/building.html#building-from-source)。从版本开始构建NumPy ``1.17``需要符合C99的编译器。对于一些较旧的编译器，这可能需要。``export CFLAGS='-std=c99'``

编译代码也意味着从开发源导入NumPy需要一些额外的步骤，这将在下面解释。对于本章的其余部分，我们假设您已经按照[Git for development中的](gitwash/index.html#using-git)描述设置了git repo
 。

## 测试构建

要构建NumPy的开发版本并运行测试，使用正确设置的Python导入路径生成交互式shell，请执行以下操作之一：

``` python
$ python runtests.py -v
$ python runtests.py -v -s random
$ python runtests.py -v -t numpy/core/tests/test_nditer.py::test_iter_c_order
$ python runtests.py --ipython
$ python runtests.py --python somescript.py
$ python runtests.py --bench
$ python runtests.py -g -m full
```

这会首先构建NumPy，因此第一次可能需要几分钟。如果指定``-n``，则测试将针对当前PYTHONPATH上找到的NumPy版本（如果有）运行。

当使用``-s``,, ``-t``或者指定目标时，``--python``可以``runtests.py``通过在裸机之后传递额外的参数将其他参数转发到嵌入的目标``--``。例如，要运行将``--pdb``标志转发到目标的测试方法，请运行以下命令：

``` python
$ python runtests.py -t numpy/tests/test_scripts.py:test_f2py -- --pdb
```

当使用pytest作为目标（默认）时，您可以
 通过将参数传递给pytest [来使用python运算符匹配测试名称](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests)``-k``：

``` python
$ python runtests.py -v -t numpy/core/tests/test_multiarray.py -- -k "MatMul and not vector"
```

::: tip 注意

请记住，在提交更改之前，所有NumPy测试都应该通过。

:::

使用``runtests.py``是推荐的运行测试的方法。还有许多替代方案，例如就地构建或安装到virtualenv。有关详细信息，请参阅下面的FAQ

## 就地构建

对于开发，您可以设置就地构建，以便对``.py``文件所做的更改
 无需重建即可生效。第一次运行：

``` python
$ python setup.py build_ext -i
```

这允许您 *仅从repo基目录* 导入就地构建的NumPy 。如果希望就地构建在基础目录之外可见，则需要将``PYTHONPATH``环境变量指向此目录。某些IDE（例如[Spyder](https://www.spyder-ide.org/)）具有要管理的实用程序
 ``PYTHONPATH``。在Linux和OSX上，您可以运行以下命令：

``` python
$ export PYTHONPATH=$PWD
```

在Windows上：

``` python
$ set PYTHONPATH=/path/to/numpy
```

现在，在NumPy中编辑Python源文件``.py``，只需重新启动解释器，即可立即测试和使用您的更改（在文件中）。

请注意，在repo base dir外部进行可见内部构建的另一种方法是使用。这不是调整，而是将文件安装到您的站点包中以及调整
 那里，因此它是一个更永久（和神奇）的操作。``python setup.py develop``PYTHONPATH``.egg-link``easy-install.pth``

## 其他构建选项

``numpy.distutils``使用该``-j``选项可以进行并行构建; 有关详细信息，请参阅[并行构建](https://numpy.org/devdocs/user/building.html#parallel-builds)

``PYTHONPATH``在源树之外的就地构建和使用的类似方法是使用：

``` python
$ pip install . --prefix /some/owned/folder
$ export PYTHONPATH=/some/owned/folder/lib/python3.4/site-packages
```

## 使用virtualenvs 

一个常见问题是“如何将NumPy的开发版本与我用来完成工作/研究的发布版本并行设置？”。

实现此目的的一种简单方法是在site-packages中安装已发布的版本，例如使用二进制安装程序或pip，并在virtualenv中设置开发版本。首先安装
 [virtualenv](http://www.virtualenv.org/)（可选择使用[virtualenvwrapper](http://www.doughellmann.com/projects/virtualenvwrapper/)），然后创建你的virtualenv（这里名为numpy-dev）：

``` python
$ virtualenv numpy-dev
```

现在，只要您想切换到虚拟环境，就可以使用该命令，退出虚拟环境并返回到以前的shell。``source numpy-dev/bin/activate``deactivate``

## 运行测试

除了使用之外``runtests.py``，还有各种方法来运行测试。在解释器内部，测试可以像这样运行：

``` python
>>> np.test()
>>> np.test('full')   # Also run tests marked as slow
>>> np.test('full', verbose=2)   # Additionally print test name/file

An example of a successful test :
``4686 passed, 362 skipped, 9 xfailed, 5 warnings in 213.99 seconds``
```

或者从命令行以类似的方式：

``` python
$ python -c "import numpy as np; np.test()"
```

也可以运行测试，但是找不到导致奇怪副作用的NumPy特定插件``pytest numpy``

运行单个测试文件可能很有用; 它比运行整个测试套件或整个模块的速度快得多（例如:) ``np.random.test()``。这可以通过以下方式完成：

``` python
$ python path_to_testfile/test_file.py
```

这也需要额外的参数，比如``--pdb``在测试失败或引发异常时将您放入Python调试器。

还支持使用[tox](https://tox.readthedocs.io/)运行测试。例如，要构建NumPy并使用Python 3.7运行测试套件，请使用：

``` python
$ tox -e py37
```

有关更多信息，请参阅[测试指南](https://numpy.org/devdocs/reference/testing.html#testing-guidelines)

 *注意：不要在没有``runtests.py``的情况下从你的numpy git repo的根目录运行测试，这将导致奇怪的测试错误。* 

## 重建和清理工作区

在对已编译代码进行更改后，可以使用与先前相同的构建命令重建NumPy  - 仅重新构建已更改的文件。执行完整构建（有时是必需的）需要首先清理工作区。执行此操作的标准方法是（ *注意：删除所有未提交的文件！* ）：

``` python
$ git clean -xdf
```

如果要放弃所有更改并返回到repo中的最后一次提交，请使用以下方法之一：

``` python
$ git checkout .
$ git reset --hard
```

## 调试

另一个常见问题是“如何在NumPy中调试C代码？”。最简单的方法是首先编写一个Python脚本，调用要调试其执行的C代码。例如``mytest.py``：

``` python
from numpy import linspace
x = np.arange(5)
np.empty_like(x)
```

现在，您可以运行：

``` python
$ gdb --args python runtests.py -g --python mytest.py
```

然后在调试器中：

``` python
(gdb) break array_empty_like
(gdb) run
```

执行现在将停止在相应的C函数上，您可以照常执行它。安装了gdb的Python扩展（通常是Linux上的默认扩展），可以使用许多有用的特定于Python的命令。例如，要查看Python代码中的位置，请使用``py-list``。有关更多详细信息，请参阅[DebuggingWithGdb](https://wiki.python.org/moin/DebuggingWithGdb)。

``gdb``您可以使用自己喜欢的替代调试器，而不是简单的; 用参数在python二进制文件上运行它
 。``runtests.py -g --python mytest.py``

使用通过调试支持构建的Python构建NumPy（在Linux发行版中通常打包为``python-dbg``）强烈建议使用。

## 理解代码和入门

更好地理解代码库的最佳策略是选择您想要更改的内容并开始阅读代码以了解其工作原理。如有疑问，您可以在邮件列表中提问。如果您的拉动请求不完美，社区总是很乐意提供帮助，这是完全可以的。作为一个志愿者项目，事情确实有时会被取消，如果事情在没有响应的情况下持续大约两到四周，那么对我们来说是完全没问题的。

因此，请继续选择一些令您烦恼或困惑的东西，尝试代码，闲逛讨论或浏览参考文档以尝试修复它。事情将会落实到位，很快您就会对整个项目有一个很好的理解。祝好运！
