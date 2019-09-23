---
meta:
  - name: keywords
    content: 使用 ``numpy.distutils`` 模块
  - name: description
    content: numpy.distutils 是NumPy扩展标准Python distutils的一部分，用于处理Fortran源代码和F2PY签名文件...
---

# 使用 ``numpy.distutils`` 模块

[``numpy.distutils``](https://numpy.org/devdocs/reference/distutils.html#module-numpy.distutils)
是NumPy扩展标准Python distutils的一部分，
用于处理Fortran源代码和F2PY签名文件，
例如编译Fortran源代码，调用F2PY构造扩展模块等。

**示例**

请思考下面的``setup 文件``：

``` python
from __future__ import division, absolute_import, print_function

from numpy.distutils.core import Extension

ext1 = Extension(name = 'scalar',
                 sources = ['scalar.f'])
ext2 = Extension(name = 'fib2',
                 sources = ['fib2.pyf', 'fib1.f'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name = 'f2py_example',
          description       = "F2PY Users Guide examples",
          author            = "Pearu Peterson",
          author_email      = "pearu@cens.ioc.ee",
          ext_modules = [ext1, ext2]
          )
# End of setup_example.py
```

运行

``` python
python setup_example.py build
```

将构建两个扩展模块 ``标量`` 和 ``fib2`` 到构建目录。

[``numpy.distutils``](https://numpy.org/devdocs/reference/distutils.html#module-numpy.distutils)
使用以下功能扩展 ``distutils``：

- ``扩展`` 类参数 ``源`` 可能包含Fortran源文件。此外，列表 ``源`` 最多可包含一个F2PY签名文件，然后扩展模块的名称必须与签名文件中使用的``<modulename>`` 匹配。 假设F2PY签名文件恰好包含一个 ``python模块`` 块。

    如果 ``源`` 文件不包含签名文件，则使用 F2PY 扫描Fortran  源文件中的例程签名，以构造 Fortran 代码的包装器。

    可以使用 ``扩展`` 类参数 ``f2py_options`` 给出 F2py 进程的其他选项。
    
- 定义了以下新的 ``distutils`` 命令：

  ``build_src``

    构建Fortran包装器扩展模块，以及其他许多事情。

  ``config_fc``

    更改Fortran编译器选项

  以及 ``build_ext`` 和 ``build_clib`` 命令都得到了增强，以支持Fortran源代码。

  运行

  ``` python
  python <setup.py file> config_fc build_src build_ext --help
  ```

  要查看这些命令的可用选项，请执行以下操作。

- 在构建包含 Fortran 源代码的 Python 包时，可以使用 ``build_ext`` 命令选项 ``--fcompiler=<Vendor>.`` 来选择不同的Fortran编译器。此处``<Vendor>`` 可以是以下名称之一：

  ``` python
  absoft sun mips intel intelv intele intelev nag compaq compaqv gnu vast pg hpux
  ```

  有关支持的编译器或运行的最新列表，请参见 ``numpy_distutils/fCompiler.py``。

  ``` python
  f2py -c --help-fcompiler
  ```
  