---
meta:
  - name: keywords
    content: NumPy Distutils-用户指南
  - name: description
    content: 本文档的目的是描述如何向SciPy添加新工具。
---

# NumPy Distutils-用户指南

## SciPy的结构

当前，SciPy项目包含两个软件包：

- NumPy-它提供以下软件包：
    - numpy.distutils-Python distutils的扩展
    - numpy.f2py-将Fortran / C代码绑定到Python的工具
    - numpy.core-将来替换数值和numarray软件包
    - numpy.lib-额外的实用程序功能
    - numpy.testing-用于单元测试的numpy风格的工具
    - 等等
- SciPy-Python的科学工具的集合。

本文档的目的是描述如何向SciPy添加新工具。

## SciPy软件包的要求

SciPy由称为SciPy软件包的Python软件包组成，可通过``scipy``名称空间供Python用户使用。
每个SciPy程序包可能包含其他SciPy程序包。等等。
因此，SciPy目录树是具有任意深度和宽度的包的树。
任何SciPy软件包都可能依赖于NumPy软件包，
但对其他SciPy软件包的依赖关系应保持最小或为零。

一个SciPy软件包，除了其来源外，还包含以下文件和目录：

- ``setup.py`` —— 构建脚本
- ``__init__.py`` —— 包初始化
- ``tests/`` —— 单元测试目录

其内容如下。

## ``setup.py`` 文件

为了将Python包添加到SciPy，其构建脚本（``setup.py``）必须满足某些要求。最重要的要求是，程序包定义一个``configuration(parent_package='',top_path=None)``函数，该函数返回适合传递给的字典
 ``numpy.distutils.core.setup(..)``。为了简化此字典的构造，请``numpy.distutils.misc_util``提供``Configuration``下面描述的
 类。

### SciPy纯Python包示例

以下是``setup.py``纯SciPy软件包的最小文件示例：

``` python
#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mypackage',parent_package,top_path)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    #setup(**configuration(top_path='').todict())
    setup(configuration=configuration)
```

该``configuration``函数的参数指定父SciPy包的名称（``parent_package``）和主``setup.py``脚本的目录位置（``top_path``）。这些参数以及当前包的名称应传递给
 ``Configuration``构造函数。

该``Configuration``构造函数有第四个可选参数
 ``package_path``，可以在包文件位于不同的位置比的目录中使用的``setup.py``文件。

其余``Configuration``参数是将用于初始化``Configuration``
实例属性的所有关键字参数。通常情况下，这些关键字是一样的那些
 ``setup(..)``功能所期望的，例如``packages``，
 ``ext_modules``，``data_files``，``include_dirs``，``libraries``，
 ``headers``，``scripts``，``package_dir``，等。然而，不建议使用这些关键字的直接指定为这些关键字参数的内容将不会被处理或检查SciPy构建系统的一致性。

最后，``Configuration``具有``.todict()``方法，该方法将所有配置数据作为适合于传递给``setup(..)``函数的字典返回
 。

### ``Configuration``实例属性

除了可以通过``Configuration``构造函数的关键字参数指定的属性外，``Configuration``实例（让我们表示为``config``）具有以下属性，这些属性在编写安装脚本时可能会有用：

- ``config.name``-当前软件包的全名。父包的名称可以提取为``config.name.split('.')``。
- ``config.local_path``-当前``setup.py``文件位置的路径。
- ``config.top_path``-主``setup.py``文件位置的路径。

### ``Configuration``实例方法

- ``config.todict()``—返回适合传递给``numpy.distutils.core.setup(..)``函数的配置字典。
- ``config.paths(*paths) --- applies ``glob.glob(..)``到``paths``必要的项目
 。修复``paths``相对于的项目
 ``config.local_path``。
- ``config.get_subpackage(subpackage_name,subpackage_path=None)``—返回子包配置的列表。子包在名称下的当前目录中查找，``subpackage_name``但是也可以通过可选``subpackage_path``参数指定路径。如果``subpackage_name``指定为，``None``则子包名称将作为的基本名称``subpackage_path``。``*``用于子包名称的任何名称都将扩展为通配符。
- ``config.add_subpackage(subpackage_name,subpackage_path=None)``—将SciPy子程序包配置添加到当前配置中。参数的含义和用法已在上面说明，请参见
 ``config.get_subpackage()``方法。
- ``config.add_data_files(*files)``-在前面加上``files``要``data_files``
列出。如果``files``item是一个元组，则其第一个元素定义相对于软件包安装目录复制数据文件的后缀，第二个元素指定数据文件的路径。默认情况下，数据文件复制在软件包安装目录下。例如，

``` python
config.add_data_files('foo.dat',
                      ('fun',['gun.dat','nun/pun.dat','/tmp/sun.dat']),
                      'bar/car.dat'.
                      '/full/path/to/can.dat',
                      )
```

将数据文件安装到以下位置

``` python
<installation path of config.name package>/
  foo.dat
  fun/
    gun.dat
    pun.dat
    sun.dat
  bar/
    car.dat
  can.dat
```

数据文件的路径可以是不带参数且返回数据文件路径的函数-当在构建程序包时生成数据文件时，这很有用。（XXX：说明确切调用此函数的步骤）
- ``config.add_data_dir(data_path)``—将目录``data_path``
递归添加到中``data_files``。始于的整个目录树
 ``data_path``将复制到软件包安装目录下。如果``data_path``是一个元组，则其第一个元素定义相对于软件包安装目录复制数据文件的后缀，第二个元素指定数据目录的路径。默认情况下，数据目录被复制到基本名称为的软件包安装目录下``data_path``。例如，

``` python
config.add_data_dir('fun')  # fun/ contains foo.dat bar/car.dat
config.add_data_dir(('sun','fun'))
config.add_data_dir(('gun','/full/path/to/fun'))
```

将数据文件安装到以下位置

``` python
<installation path of config.name package>/
  fun/
     foo.dat
     bar/
        car.dat
  sun/
     foo.dat
     bar/
        car.dat
  gun/
     foo.dat
     bar/
        car.dat
```
- ``config.add_include_dirs(*paths)``-在前面加上``paths``要
 ``include_dirs``列出。该列表将对当前软件包的所有扩展模块可见。
- ``config.add_headers(*files)``-在前面加上``files``要``headers``
列出。默认情况下，头文件将安装在``/include/pythonX.X//``
目录下
 。如果``files``item是一个元组，则它的第一个参数指定相对于``/include/pythonX.X/``path 的安装后缀
 。这是Python的distutils方法；不建议NumPy和SciPy使用它，而推荐使用
 ``config.add_data_files(*files)``。
- ``config.add_scripts(*files)``-在前面加上``files``要``scripts``
列出。脚本将安装在``/bin/``目录下。
- ``config.add_extension(name,sources,**kw)``—创建``Extension``实例并将其添加
 到``ext_modules``列表。第一个参数
 ``name``定义将在``config.name``软件包下安装的扩展模块的名称。第二个参数是来源列表。``add_extension``方法还接受传递给``Extension``构造函数的关键字参数。允许关键字的列表如下所示：``include_dirs``，
 ``define_macros``，``undef_macros``，``library_dirs``，``libraries``，
 ``runtime_library_dirs``，``extra_objects``，``extra_compile_args``，
 ``extra_link_args``，``export_symbols``，``swig_opts``，``depends``，
 ``language``，``f2py_options``，``module_dirs``，``extra_info``，
 ``extra_f77_compile_args``，``extra_f90_compile_args``。

请注意，该``config.paths``方法将应用于可能包含路径的所有列表。``extra_info``是将内容添加到关键字参数的字典或词典列表。该列表``depends``包含扩展模块源所依赖的文件或目录的路径。如果``depends``列表中的任何路径比扩展模块新，则将重建该模块。

源列表可能包含带有模式的函数（“源生成器”）。如果返回，则不生成任何源。并且，如果实例在处理完所有源生成器之后没有源，则不会构建扩展模块。这是建议的有条件定义扩展模块的方法。源生成器函数由的子命令调用
 。``def (ext, build_dir): return
``funcname``None``Extension``build_src``numpy.distutils``

例如，这是典型的源生成器函数：

``` python
def generate_source(ext,build_dir):
    import os
    from distutils.dep_util import newer
    target = os.path.join(build_dir,'somesource.c')
    if newer(target,__file__):
        # create target file
    return target
```

第一个参数包含扩展实例可能是有用的访问它的属性一样``depends``，``sources``在建设过程中，等名单，并对其进行修改。第二个参数提供了到磁盘创建文件时必须使用的构建目录的路径。
- ``config.add_library(name, sources, **build_info)``—将库添加到``libraries``列表。允许关键字参数是
 ``depends``，``macros``，``include_dirs``，``extra_compiler_args``，
 ``f2py_options``，``extra_f77_compile_args``，
 ``extra_f90_compile_args``。有关``.add_extension()``参数的更多信息，请参见method。
- ``config.have_f77c()`` —如果有Fortran 77编译器可用，则返回True（阅读：成功编译的简单Fortran 77代码）。
- ``config.have_f90c()`` —如果有Fortran 90编译器可用，则返回True（阅读：成功编译的简单Fortran 90代码）。
- ``config.get_version()``— ``None``如果无法检测到版本信息，则返回当前软件包的版本字符串
 。这种方法扫描文件``__version__.py``，``_version.py``，
 ``version.py``，``__svn_version__.py``对于字符串变量
 ``version``，``__version__``，``_version``。
- ``config.make_svn_version_py()``—将数据功能追加到
 ``data_files``列表，该列表将生成``__svn_version__.py``文件到当前程序包目录。Python退出时，该文件将从源目录中删除。
- ``config.get_build_temp_dir()``—返回临时目录的路径。在这里应该建立临时文件。
- ``config.get_distribution()``—返回distutils ``Distribution``
实例。
- ``config.get_config_cmd()``—返回``numpy.distutils``配置命令实例。
- ``config.get_info(*names)`` -

### 转换``.src``使用模板文件

NumPy distutils支持自动转换名为 ``<somefile>.src`` 的源文件。此功能可用于维护非常相似的代码块，仅需在块之间进行简单更改即可。在安装程序的构建阶段，如果遇到名为 ``<somefile>.src`` 的模板文件，则将从模板中构建一个名为 ``<somefile>`` 的新文件，并将其放置在构建目录中以供使用。支持两种形式的模板转换。第一种形式出现在名为 ``<file>.ext.src`` 的文件中，其中ext是公认的Fortran扩展名（f，f90，f95，f77，用于ftn，pyf）。第二种形式用于所有其他情况。

### Fortran文件

该模板转换器将复制所有**功能**和
 **子例程**根据“ <…>”中的规则，使用名称包含“ <…>”的文件中的块。“ <…>”中逗号分隔的单词数决定了该块重复的次数。这些词表示在每个块中应将重复规则“ <…>”替换为什么。块中的所有重复规则必须包含相同数量的逗号分隔的单词，以指示该块应重复的次数。如果重复规则中的单词需要逗号，左箭头或右箭头，则在其前面加上反斜杠“”。如果重复规则中的单词与 “ \  ``<index>`` ” 匹配，则它将被同一重复规范中的第 ``<index>`` 个单词替换。重复规则有两种形式：named和short。

#### 命名重复规则

当同一重复块必须在一个块中多次使用时，命名重复规则很有用。它使用<rule1 = item1，item2，item3，…，itemN>指定，其中N是应重复执行块的次数。在该块的每个重复中，整个表达式“ <…>”将首先被item1替换，然后被item2替换，依此类推，直到完成N次重复。一旦引入了命名的重复规范，就可以通过仅引用名称（即 ``<rule1>`` ）**在当前块中**使用相同的重复规则。

#### 短重复规则

一个简短的重复规则看起来像<item1，item2，item3，…，itemN>。该规则指定整个表达式“ <…>”应首先用item1替换，然后再用item2替换，依此类推，直到完成N次重复为止。

#### 预先定义的名称

可以使用以下预定义的命名重复规则：

- \<prefix=s,d,c,z>
- \<_c=s,d,c,z>
- \<_t=real, double precision, complex, double complex>
- \<ftype=real, double precision, complex, double complex>
- \<ctype=float, double, complex_float, complex_double>
- \<ftypereal=float, double precision, \0, \1>
- \<ctypereal=float, double, \0, \1>

### 其他文件

非Fortran文件使用单独的语法来定义模板块，应使用类似于Fortran特定重复的命名重复规则的变量扩展来重复这些模板块。

NumPy Distutils预处理``.c.src``以自定义模板语言编写的C源文件（扩展名：）以生成C代码。该``@``符号用于包装宏样式变量，以启用可能描述（例如）一组数据类型的字符串替换机制。

模板语言块由 **/\*\*begin repeat** 和 **/\*\*end repeat\*\*/** 行分隔，也可以使用  **/\*\*begin repeat1** 和  **/\*\*end repeat1\*\*/** 等连续编号的定界行进行嵌套：

1. 在一行上的 “/\*\*begin repeat “ 本身标志着应重复的片段的开始。
2. 命名变量扩展使用 ``#name=item1, item2, item3, ..., itemN#`` 定义，并放置在连续的行上。在每个重复块中用相应的字替换这些变量。同一重复块中的所有命名变量必须定义相同的字数。
3. 在为命名变量指定重复规则时，``item*N`` 是重复N次的简称。此外，圆括号与 \*N 结合可用于对应重复的几个项目进行分组。因此，``＃name =（item1，item2）* 4＃`` 等效于 ``#name=item1, item2, item1, item2, item1, item2, item1, item2#``
4. 一行上的 “\*/” 本身标志着变量扩展命名的结束。下一行是将使用命名规则重复的第一行。
5. 在要重复的块内，应扩展的变量指定为 ``@name@``。
6. 一行上的 “/\*\*end repeat\*\*/” 本身将前一行标记为要重复的块的最后一行。
7. NumPy C源代码中的循环可能具有 ``@TYPE@`` 目标为字符串替换的变量，该变量已预处理为多个带有几个字符串（如INT，LONG，UINT，ULONG）的其他相同循环。因此， ``@TYPE@``样式语法通过模仿具有通用类型支持的语言来减少代码重复和维护负担。

在以下模板源示例中，上述规则可能更清晰：

``` C
 /* TIMEDELTA to non-float types */

 /**begin repeat
  *
  * #TOTYPE = BYTE, UBYTE, SHORT, USHORT, INT, UINT, LONG, ULONG,
  *           LONGLONG, ULONGLONG, DATETIME,
  *           TIMEDELTA#
  * #totype = npy_byte, npy_ubyte, npy_short, npy_ushort, npy_int, npy_uint,
  *           npy_long, npy_ulong, npy_longlong, npy_ulonglong,
  *           npy_datetime, npy_timedelta#
  */

 /**begin repeat1
  *
  * #FROMTYPE = TIMEDELTA#
  * #fromtype = npy_timedelta#
  */
 static void
 @FROMTYPE@_to_@TOTYPE@(void *input, void *output, npy_intp n,
         void *NPY_UNUSED(aip), void *NPY_UNUSED(aop))
 {
     const @fromtype@ *ip = input;
     @totype@ *op = output;

     while (n--) {
         *op++ = (@totype@)*ip++;
     }
 }
 /**end repeat1**/

 /**end repeat**/
```

通用类型的C源文件的预处理（无论是使用NumPy还是使用NumPy Distutils的任何第三方软件包）由[conv_template.py进行](https://github.com/numpy/numpy/blob/master/numpy/distutils/conv_template.py)。这些模块在构建过程中生成的特定于类型的C文件（扩展名：.c）已准备好进行编译。C头文件（经过预处理可生成.h文件）也支持这种形式的通用类型。

### 在有用的功能``numpy.distutils.misc_util``

- ``get_numpy_include_dirs()``—返回NumPy基本包含目录的列表。NumPy基本包含目录包含头文件，例如``numpy/arrayobject.h``，``numpy/funcobject.h``
等。对于已安装的NumPy，返回的列表的长度为1，但是在构建NumPy时，列表可能包含更多目录，例如，``config.h``该``numpy/base/setup.py``文件生成的文件
 路径，并且``numpy``
由头文件使用。
- ``append_path(prefix,path)``—聪明地附加``path``到``prefix``。
- ``gpaths(paths, local_path='')``-将glob应用于路径，并``local_path``在需要时添加前缀
 。
- ``njoin(*path)``-加入路径组件+转换``/``-分隔路径``os.sep``-分隔路径和解决``..``，``.``从路径。防爆。。``njoin('a',['b','./c'],'..','g') -> os.path.join('a','b','g')``
- ``minrelpath(path)``—解析中的点``path``。
- ``rel_path(path, parent_path)``— ``path``相对于返回``parent_path``。
- ``def get_cmd(cmdname,_cache={})``—返回``numpy.distutils``
命令实例。
- ``all_strings(lst)``
- ``has_f_sources(sources)``
- ``has_cxx_sources(sources)``
- ``filter_sources(sources)`` —返回 ``c_sources, cxx_sources,
f_sources, fmodule_sources``
- ``get_dependencies(sources)``
- ``is_local_src_dir(directory)``
- ``get_ext_source_files(ext)``
- ``get_script_files(scripts)``
- ``get_lib_source_files(lib)``
- ``get_data_files(data)``
- ``dot_join(*args)`` —将非零参数加一个点。
- ``get_frame(level=0)`` —从调用堆栈返回给定级别的框架对象。
- ``cyg2win32(path)``
- ``mingw32()``— ``True``使用mingw32环境时返回。
- ``terminal_has_colors()``，``red_text(s)``，``green_text(s)``，
 ``yellow_text(s)``，``blue_text(s)``，``cyan_text(s)``
- ``get_path(mod_name,parent_path=None)``—给定的相对于parent_path的模块的返回路径。还处理``__main__``和
 ``__builtin__``模块。
- ``allpath(name)``—替换``/``为``os.sep``in ``name``。
- ``cxx_ext_match``，``fortran_ext_match``，``f90_ext_match``，
``f90_module_name_match``

### ``numpy.distutils.system_info`` 模块

- ``get_info(name,notfound_action=0)``
- ``combine_paths(*args,**kws)``
- ``show_all()``

### ``numpy.distutils.cpuinfo`` 模块

- ``cpuinfo``

### ``numpy.distutils.log`` 模块

- ``set_verbosity(v)``

### ``numpy.distutils.exec_command`` 模块

- ``get_pythonexe()``
- ``find_executable(exe, path=None)``
- ``exec_command( command, execute_in='', use_shell=None, use_tee=None, **env )``

## ``__init__.py`` 文件

典型的SciPy的标头``__init__.py``是：

``` python
"""
Package docstring, typically with a brief description and function listing.
"""

# py3k related imports
from __future__ import division, print_function, absolute_import

# import functions into module namespace
from .subpackage import *
...

__all__ = [s for s in dir() if not s.startswith('_')]

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
```

请注意，NumPy子模块仍使用名为的文件``info.py``，该文件``__all__``中定义了模块docstring和dict。这些文件有时会被删除。

## NumPy Distutils中的附加功能

### 在setup.py脚本中为库指定config_fc选项

可以在setup.py脚本中指定config_fc选项。例如，使用：

``` python
config.add_library(‘library’,
  sources=[…], config_fc={‘noopt’:(__file__,1)})
```

将编译``library``没有优化标志的源代码。

建议仅以与编译器无关的方式指定那些config_fc选项。

### 从源代码获取额外的Fortran 77编译器选项

一些旧的Fortran代码需要特殊的编译器选项才能正常工作。为了指定每个源文件的编译器选项，``numpy.distutils``Fortran编译器将寻找以下模式：

``` python
CF77FLAGS(<fcompiler type>) = <fcompiler f77flags>
```

在源代码的前20行中，并使用``f77flags``fcompiler的指定类型（第一个字符``C``是可选的）。

待办事项：此功能也可以轻松扩展为Fortran 90代码。让我们知道您是否需要这样的功能。
