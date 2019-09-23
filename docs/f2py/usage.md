---
meta:
  - name: keywords
    content: 使用 NumPy F2PY
  - name: description
    content: F2PY既可以用作命令行工具 f2py，也可以用作Python模块 numpy.f2py。 虽然我们尝试将命令行工具安装为numpy设置的一部分...
---

# 使用 F2PY

F2PY既可以用作命令行工具 ``f2py``，也可以用作Python模块 ``numpy.f2py``。
虽然我们尝试将命令行工具安装为numpy设置的一部分，但是像Windows这样的某些平台却难以将可执行文件可靠地放在 ``PATH`` 上。
我们将在本文档中引用 ``f2py``，但您可能必须将其作为模块运行：

```
python -m numpy.f2py
```

如果你运行没有参数的 ``f2py``，并且最后的行 ``numpy Version`` 与从 ``python -m numpy.f2py`` 打印的NumPy版本匹配，
那么你可以使用更短的版本。如果没有，或者如果你不能运行 ``f2py``，你应该用更长的版本替换所有对 ``f2py`` 的调用。

## ``f2py`` 命令

当用作命令行工具时，``f2py`` 有三种主要模式，区别在于使用 ``-c`` 和 ``-h`` 开关：要扫描Fortran源并生成签名文件，请使用：

1. 要扫描Fortran源并生成签名文件，请使用

    ``` python
    f2py -h <filename.pyf> <options> <fortran files>   \
      [[ only: <fortran functions>  : ]                \
      [ skip: <fortran functions>  : ]]...            \
      [<fortran files> ...]
    ```

    请注意，Fortran源文件可以包含许多例程，并且不一定需要从Python中使用所有例程。
    因此，您可以指定应该包装哪些例程（ 在 ``only: .. :`` 部分）或者应忽略哪些例程F2PY（在 ``skip: .. :`` 部分）。

    如果将 ``<filename.pyf>`` 指定为 ``stdout``，则签名将发送到标准输出而不是文件。

    在其他选项（见下文）中，可以在此模式中使用以下选项：

    ``--overwrite-signature``

    覆盖现有签名文件。
1. 要构建扩展模块，请使用：

    ``` python
    f2py <options> <fortran files>          \
      [[ only: <fortran functions>  : ]     \
      [ skip: <fortran functions>  : ]]... \
      [<fortran files> ...]
    ```

    构造的扩展模块作为 ``<modulename>module.c`` 保存到当前目录。

    这里 ``<fortran files>`` 也可能包含签名文件。在其他选项（见下文）中，可以在此模式中使用以下选项：

    - ``--debug-capi``

        将调试挂钩添加到扩展模块。使用此扩展模块时，有关包装器的各种信息将打印到标准输出，例如，变量值，所采取的步骤等。

    - ``-include'<includefile>'``

        将CPP ``#include`` 语句添加到扩展模块源。 ``<includefile>`` 应以下列形式之一给出：

        ```
        "filename.ext"
        <filename.ext>
        ```

        include语句就在包装函数之前插入。此功能允许在F2PY生成的包装器中使用任意C函数（在 ``<includefile>`` 中定义）。

        不推荐使用此选项。使用``usercode``语句直接在签名文件中指定C代码片段。

    - ``--[no-]wrap-functions``

        为 Fortran 函数创建 Fortran子例程包装器。``--wrap-functions`` 是默认的，因为它确保了最大的可移植性和编译器独立性。

    - ``--include-paths <path1>:<path2>:..``

        搜索包含给定目录中的文件。

    - ``--help-link [<list of resources names>]``
        列出 numpy_distutils/system_info.py 找到的系统资源。 例如，尝试 ``f2py --help-link lapack_opt``。
1. 要构建扩展模块，请使用

    ``` python
    f2py -c <options> <fortran files>       \
      [[ only: <fortran functions>  : ]     \
      [ skip: <fortran functions>  : ]]... \
      [ <fortran/c source files> ] [ <.o, .a, .so files> ]
    ```

    如果 ``<fortran files>`` 包含签名文件，则构建扩展模块的源，编译所有Fortran和C源，最后将所有对象和库文件链接到扩展模块 ``<modulename>``.so，保存到 当前目录。

    如果 ``<fortran files>`` 不包含签名文件，则通过扫描所有Fortran源代码以构建常规签名来构建扩展模块。

    在以前模式中描述的其他选项（参见下文）和选项中，可以在此模式中使用以下选项：

    - ``--help-fcompiler``

        列出可用的Fortran编译器。

    - ``--help-compiler [depreciated]``

        列出可用的Fortran编译器。

    - ``--fcompiler=<Vendor>``

        按供应商指定Fortran编译器类型。

    - ``--f77exec=<path>``
        
        指定F77编译器的路径。

    - ``--fcompiler-exec=<path> [depreciated]``

        指定F77编译器的路径。

    - ``--f90exec=<path>``

        指定F90编译器的路径。

    - ``--f90compiler-exec=<path> [depreciated]``

        指定F90编译器的路径。

    - ``--f77flags=<string>``

        指定F77编译器标志。

    -  ``--f90flags=<string>``

        指定F90编译器标志。

    - ``--opt=<string>``

        指定优化标志。

    - ``--arch=<string>``

        指定架构特定的优化标志。

    - ``--noopt``

        无需优化即可编译。

    - ``--noarch``

        编译时不依赖于arch的优化。

    - ``--debug``

        编译调试信息。

    - ``-l<libname>``

        链接时使用库``<libname>``。

    - ``-D<macro>[=<defn=1>]``

        将宏 ``<macro>`` 定义为 ``<defn>`` 。

    - ``-U<macro>``

        定义宏``<macro>``。

    - ``-I<dir>``

        将目录 ``<dir>`` 添加到搜索包含文件的目录列表中。

    - ``-L<dir>``
        
        将目录 ``<dir>`` 添加到要搜索 -l 的目录列表中。

    - ``link-<resource>``

        使用 numpy_distutils/system_info.py 定义的 ``<resource>`` 链接扩展模块。例如。要链接优化的LAPACK库（MacOSX上的vecLib，其他地方的ATLAS），请使用 --link-lapack_opt。 另请参阅 --help-link 开关。

    构建扩展模块时，非gcc Fortran编译器可能需要以下宏的组合：

    ```
    -DPREPEND_FORTRAN
    -DNO_APPEND_FORTRAN
    -DUPPERCASE_FORTRAN
    ```

    要测试 F2PY 生成的接口的性能，请使用 -DF2PY_REPORT_ATEXIT。
    然后在Python的出口处打印出各种计时的报告。
    此功能可能无法在所有平台上运行，目前仅支持Linux平台。

    要查看F2PY生成的接口是否执行数组参数的副本，请使用 ``-DF2PY_REPORT_ON_ARRAY_COPY=<int>``。
    当数组参数的大小大于 ``<int>`` 时，会将有关应对的消息发送到stderr。

其他选择：

- ``-m <modulename>``

    扩展模块的名称。默认为无标题。如果使用签名文件（*.pyf），请不要使用此选项。

- ``--[no-]lower``

    不要降低 ``<fortran files>`` 中的大小写。 默认情况下，``--lower`` 假定为 ``-h`` 开关， ``--no-lower`` 假定为 -h 开关。

- ``--build-dir <dirname>``

    所有F2PY生成的文件都在 ``<dirname>`` 中创建。默认值为 ``tempfile.mkdtemp()`` 。

- ``--quiet``

   安静地跑（不打印日志）。

- ``--verbose``

    额外冗长的跑（打印大量日志）。

- ``-v``

    打印f2py版本ID并退出。

在没有任何选项的情况下执行 ``f2py`` 以获取可用选项的最新列表。

## Python 模块 ``numpy.f2py``

::: danger 警告

``f2py`` 模块的当前Python接口尚未成熟，将来可能会发生变化。

:::

Fortran到Python接口生成器。

- ``numpy.f2py``.run_main( *comline_list* )[[点击查看源代码]](https://github.com/numpy/numpy/blob/master/numpy/f2py/f2py2e.py#L398-L461)

    相当于运行：

    ``` python
    f2py <args>
    ```

    其中 ``=string.join(,' ')``，但在Python中。除非使用 ``-h``，否则此函数将返回一个字典，其中包含有关生成的模块及其对源文件的依赖关系的信息。例如，可以从Python执行命令 ``f2py -m scalar scalar.f`` ，如下所示：

    您无法使用此功能构建扩展模块，即不允许使用 ``-c``。请改用 ``compile`` 命令。

    **示例：**

    ``` python
    >>> import numpy.f2py
    >>> r = numpy.f2py.run_main(['-m','scalar','doc/source/f2py/scalar.f'])
    Reading fortran codes...
            Reading file 'doc/source/f2py/scalar.f' (format:fix,strict)
    Post-processing...
            Block: scalar
                            Block: FOO
    Building modules...
            Building module "scalar"...
            Wrote C/API module "scalar" to file "./scalarmodule.c"
    >>> print(r)
    {'scalar': {'h': ['/home/users/pearu/src_cvs/f2py/src/fortranobject.h'],
            'csrc': ['./scalarmodule.c', 
                      '/home/users/pearu/src_cvs/f2py/src/fortranobject.c']}}
    ```

- ``numpy.f2py``.compile( *source* ,  *modulename='untitled'* ,  *extra_args=''* ,  *verbose=True* ,  *source_fn=None* ,  *extension='.f'* )[[点击查看源代码]](https://github.com/numpy/numpy/blob/master/numpy/f2py/__init__.py#L23-L117)

    使用f2py从Fortran 77源字符串构建扩展模块。

    **参数：**

    类型 | 描述
    ---|---
    source : str or bytes | 要编译的Fortran源模块/子程序。*在版本1.16.0中更改：* 接受str以及字节。
    modulename : str, optional | 已编译的python模块的名称
    extra_args : str or list, optional | 传递给f2py的其他参数。*版本1.16.0中已更改：* 也可能提供args列表。
    verbose : bool, optional | 将f2py输出打印到屏幕
    source_fn : str, optional | 写入fortran源的文件的名称。 默认设置是使用扩展参数提供的扩展名的临时文件。
    extension : {‘.f’, ‘.f90’}, optional | 如果未提供source_fn，则为文件扩展名。扩展名告诉我使用了哪个fortran标准。默认值为f，表示F77标准。 *版本1.11.0中的新功能。*

    **返回值：** 

    类型 | 描述
    ---|---
    result : int | 0 表示成功

    **示例：**

    ``` python
    >>> import numpy.f2py
    >>> fsource = '''
    ...       subroutine foo
    ...       print*, "Hello world!"
    ...       end 
    ... '''
    >>> numpy.f2py.compile(fsource, modulename='hello', verbose=0)
    0
    >>> import hello
    >>> hello.foo()
    Hello world!
    ```
