# numpy.distutils中的模块

NumPy提供增强的distutils功能，以便更容易地构建和安装子包，自动生成代码以及使用Fortran编译库的扩展模块。 要使用NumPy distutils的功能，请使用numpy.distutils.core中的setup命令。 [numpy.distutils.misc_util](#misc_util) 中还提供了一个有用的 [Configuration](#类numpy.distutils.misc_util.Configuration) 类，它可以更容易地构造关键字参数以传递给setup函数（通过传递从类的todict()方法获得的字典）。 有关详细信息，请参阅 ``<site-packages>/numpy/doc/DISTUTILS.txt`` 中的 NumPy Distutils用户指南。

## distutils中的模块

### misc_util

- get_numpy_include_dirs()	
- dict_append(d, **kws)	
- appendpath(prefix, path)	
- allpath(name)	 使用OS的路径分隔符将/-分隔的路径名转换为。
- dot_join(*args)	
- generate_config_py(target)	生成config.py文件，其中包含构建包期间使用的system_info信息。
- get_cmd(cmdname[, _cache])	
- terminal_has_colors()	
- red_text(s)	
- green_text(s)	
- yellow_text(s)	
- blue_text(s)	
- cyan_text(s)	
- cyg2win32(path)	
- all_strings(lst)	如果lst中的所有项都是字符串对象，则返回True。
- has_f_sources(sources)	如果sources包含Fortran文件，则返回True
- has_cxx_sources(sources)	如果sources包含C ++文件，则返回True
- filter_sources(sources)	返回分别包含C，C ++，Fortran和Fortran 90模块源的四个文件名列表。
- get_dependencies(sources)	
- is_local_src_dir(directory)	如果directory是本地目录，则返回true。
- get_ext_source_files(ext)	
- get_script_files(scripts)	

## 类numpy.distutils.misc_util.Configuration

### 参数概览

```python
numpy.distutils.misc_util.Configuration(package_name=None, parent_name=None, top_path=None, package_path=None, **attrs)
```

### 类的源码

[http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L730-L2088](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L730-L2088)

### 类的简介

构造给定包名称的配置实例。 如果parent_name不是None，则将包构造为parent_name包的子包。 如果top_path和package_path为None，则假定它们等于此实例创建的文件的路径.numpy发行版中的setup.py文件是如何使用[Configuration](https://docs.scipy.org/doc/numpy/reference/distutils.html#numpy.distutils.misc_util.Configuration)实例的很好示例。 

### 方法todict()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L837-L854)

#### 简介

返回与distutils setup函数的关键字参数兼容的字典。

#### 例子

```python
>>> setup(**config.todict())   
```

### 方法get_distribution() 

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L837-L854) 

#### 简介

返回distutils分发对象以供自己使用。

### 方法get_subpackage()

#### 参数概览

```python
get_subpackage(subpackage_name, subpackage_path=None, parent_name=None, caller_level=1)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L948-L1007)

#### 简介

返回子包配置列表。

#### 参数说明

- **subpackage_name** : 字符串 或者 None
    获取配置的子包的名称。 subpackage_name中的'*'作为通配符处理。
- **subpackage_path** : 字符串
    如果为None，则假定路径为本地路径加上subpackage_name。 如果在subpackage_path中找不到setup.py文件，则使用默认配置。
- **parent_name** : 字符串
    顾名思义获取父名。

### 方法add_subpackage()

#### 参数概览

```python
add_subpackage(subpackage_name, subpackage_path=None, standalone=False)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1009-L1050)

#### 简介

将子包添加到当前的Configuration实例。

这在用于将子包添加到包的setup.py脚本中很有用。

#### 参数说明
- **subpackage_name** : 字符串
    子包的名称
- **subpackage_path** : 字符串
    如果给定，则子包路径（例如子包）位于subpackage_path / subpackage_name中。 如果为None，则假定子包位于本地路径/ subpackage_name中。
- **standalone** : bool

### 方法add_data_files()

#### 参数概览

```python
add_data_files(*files)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1182-L1331)

#### 简介

将数据文件添加到配置data_files。

#### 参数说明

- **files** : 序列
    参数可以是
    - 2元序列（< datadir prefix >，<数据文件的路径>）
    - 数据文件的路径，其中python datadir前缀默认为package dir。

#### 提示

文件序列的每个元素的形式非常灵活，允许从包中获取文件的位置以及最终应该在系统上安装的位置的许多组合。最基本的用法是将files参数序列的元素设置为简单文件名。这将导致将本地路径中的文件安装到self.name包的安装路径（包路径）。 file参数也可以是相对路径，在这种情况下，整个相对路径将安装到包目录中。最后，该文件可以是绝对路径名，在这种情况下，文件将在绝对路径名中找到，但安装到包路径。

可以通过将2元组作为文件参数传递来增强此基本行为。元组的第一个元素应指定应安装其余文件序列的相对路径（在软件包安装目录下）（它与源代码分发中的文件名无关）。元组的第二个元素是应该安装的文件序列。此序列中的文件可以是文件名，相对路径或绝对路径。对于绝对路径，文件将安装在顶级包安装目录中（无论第一个参数如何）。文件名和相对路径名将安装在作为元组的第一个元素给出的路径名下的包安装目录中。

安装路径规则：

1. file.txt -> (., file.txt)-> parent/file.txt
1. foo/file.txt -> (foo, foo/file.txt) -> parent/foo/file.txt
1. /foo/bar/file.txt -> (., /foo/bar/file.txt) -> parent/file.txt
1. *.txt -> parent/a.txt, parent/b.txt
1. foo/*.txt\`\` -> parent/foo/a.txt, parent/foo/b.txt
1. */*.txt -> (*, */*.txt) -> parent/c/a.txt, parent/d/b.txt
1. (sun, file.txt) -> parent/sun/file.txt
1. (sun, bar/file.txt) -> parent/sun/file.txt
1. (sun, /foo/bar/file.txt) -> parent/sun/file.txt
1. (sun, *.txt) -> parent/sun/a.txt, parent/sun/b.txt
1. (sun, bar/*.txt) -> parent/sun/a.txt, parent/sun/b.txt
1. (sun/*, */*.txt) -> parent/sun/c/a.txt, parent/d/b.txt

另一个特性是数据文件的路径实际上可以是不带参数的函数，并将实际路径返回给数据文件。 在构建程序包时生成数据文件时，这很有用。

#### 例子

将文件添加到要包含在包中的data_files列表中。

```python
>>>
>>> self.add_data_files('foo.dat',
...     ('fun', ['gun.dat', 'nun/pun.dat', '/tmp/sun.dat']),
...     'bar/cat.dat',
...     '/full/path/to/can.dat') 
```

将这些数据文件安装到：

```
<package install directory>/
 foo.dat
 fun/
   gun.dat
   nun/
     pun.dat
 sun.dat
 bar/
   car.dat
 can.dat
```

其中 < package install directory > 是包（或子包）目录，例如 ‘/usr/lib/python2.4/site-packages/mypackage’ (‘C: Python2.4 Lib site-packages mypackage’) or ‘/usr/lib/python2.4/site- packages/mypackage/mysubpackage’ (‘C: Python2.4 Lib site-packages mypackage mysubpackage’).

### 方法add_data_dir()

#### 参数概览

```python
add_data_dir(data_path)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1052-L1171)

#### 简介

递归地将data_path下的文件添加到data_files列表中。

递归地将data_path下的文件添加到要安装（和分发）的data_files列表中。 data_path可以是相对路径名，也可以是绝对路径名，也可以是2元组，其中第一个参数显示安装目录中应安装数据目录的位置。

#### 参数说明

- **data_path** : 序列 或 字符串
    参数可以是：
    - 2元序列（< datadir suffix >，<数据目录的路径>）
    - 数据目录的路径，其中python datadir后缀默认为package dir。

#### 提示

安装路径规则：

```
foo/bar -> (foo/bar, foo/bar) -> parent/foo/bar
(gun, foo/bar) -> parent/gun
foo/* -> (foo/a, foo/a), (foo/b, foo/b) -> parent/foo/a, parent/foo/b
(gun, foo/*) -> (gun, foo/a), (gun, foo/b) -> gun
(gun/*, foo/*) -> parent/gun/a, parent/gun/b
/foo/bar -> (bar, /foo/bar) -> parent/bar
(gun, /foo/bar) -> parent/gun
(fun/*/gun/*, sun/foo/bar) -> parent/fun/foo/gun/bar
```

#### 例子

例如，假设源目录包含fun/foo.dat 和 fun/bar/car.dat：

```python
>>> self.add_data_dir('fun')                       
>>> self.add_data_dir(('sun', 'fun'))              
>>> self.add_data_dir(('gun', '/full/path/to/fun'))
```

将数据文件安装到位置：

```
<package install directory>/
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
    car.dat
```

### 方法add_include_dirs()

#### 参数概览

```python
add_include_dirs(*paths)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1351-L1365)

#### 简介

添加配置包含目录的路径。

将给定的路径序列添加到include_dirs列表的开头。 此列表将对当前包的所有扩展模块可见。

### 方法add_headers()

#### 参数概览

```python
add_headers(*files)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1367-L1399)

将可安装标头添加到配置中。

将给定的文件序列添加到标题列表的开头。 默认情况下，标题将安装在< python-include > / < self.name.replace（'。'，'/'）> /目录下。 如果文件项是元组，则其第一个参数指定相对于< python-include >路径的实际安装位置。

#### 参数说明

- **files** : 字符串 或 序列
    参数可以是：
    - 2元序列（< includedir后缀 >，< 头文件的路径 >）
    - 路径（s）到头文件，其中python includedir后缀将默认为包名称。

### 方法add_extension()

#### 参数概览

```python
add_extension(name, sources, **kw)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1424-L1525)

#### 简介

添加扩展到配置。

创建Extension实例并将其添加到ext_modules列表中。 此方法还采用传递给Extension构造函数的以下可选关键字参数。

#### 参数说明

- **name** : 字符串
    扩展名
- **sources** : 序列
    来源清单。 源列表可能包含函数（称为源生成器），它们必须将扩展实例和构建目录作为输入并返回源文件或源文件列表或None。 如果返回None，则不生成任何源。 如果Extension实例在处理完所有源生成器后没有源，则不会构建任何扩展模块。
- **include_dirs** : （译者注：官方无介绍）
- **define_macros** :（译者注：官方无介绍）
- **undef_macros** :（译者注：官方无介绍）
- **library_dirs** :（译者注：官方无介绍）
- **libraries** :（译者注：官方无介绍）
- **runtime_library_dirs** :（译者注：官方无介绍）
- **extra_objects** :（译者注：官方无介绍）
- **extra_compile_args** :（译者注：官方无介绍）
- **extra_link_args** :（译者注：官方无介绍）
- **extra_f77_compile_args** :（译者注：官方无介绍）
- **extra_f90_compile_args** :（译者注：官方无介绍）
- **export_symbols** :（译者注：官方无介绍）
- **swig_opts** :（译者注：官方无介绍）
- **depends** :（译者注：官方无介绍）
    depends列表包含扩展模块的源所依赖的文件或目录的路径。 如果依赖列表中的任何路径比扩展模块更新，则将重建该模块。
- **language** :（译者注：官方无介绍）
- **f2py_options** :（译者注：官方无介绍）
- **module_dirs** :（译者注：官方无介绍）
- **extra_info** : dict or list
    要附加到关键字的关键字的字典或列表。

#### 提示

self.paths（...）方法应用于可能包含路径的所有列表。

### 方法add_library()

#### 参数概览

```python
add_library(name, sources, **build_info)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1527-L1561)

#### 简介

将库添加到配置中。

#### 参数说明

- **name** : 字符串
    扩展名。
- **sources** : 序列
    来源清单。 源列表可能包含函数（称为源生成器），它们必须将扩展实例和构建目录作为输入并返回源文件或源文件列表或None。 如果返回None，则不生成任何源。 如果Extension实例在处理完所有源生成器后没有源，则不会构建任何扩展模块。
- **build_info** : 字典，可选
    - 允许以下键：
        - depends
        - macros
        - include_dirs
        - extra_compiler_args
        - extra_f77_compile_args
        - extra_f90_compile_args
        - f2py_options
        - language

### 方法add_scripts()

#### 参数概览

```python
add_scripts(*files)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1704-L1718)

#### 简介

添加脚本到配置。

将文件序列添加到脚本列表的开头。 脚本将安装在< prefix >/bin/目录下。

### 方法add_installed_library()

#### 参数概览

```python
add_installed_library(name, sources, install_dir, build_info=None)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1580-L1629)

#### 简介

与add_library类似，但安装了指定的库。

与distutils一起使用的大多数C库仅用于构建python扩展，但是将安装通过此方法构建的库，以便第三方包可以重用它们。

#### 参数说明
- **name** : 字符串
    已安装库的名称。
- **sources** : 序列
    库的源文件列表。 有关详细信息，请参阅add_library。
- **install_dir** : 字符串
    相对于当前子包安装库的路径。
- **build_info** : 字典, 可选
    - 允许以下键：
        - depends
        - macros
        - include_dirs
        - extra_compiler_args
        - extra_f77_compile_args
        - extra_f90_compile_args
        - f2py_options
        - language

#### 返回值

```
None
```

另见：
> add_library, add_npy_pkg_config, get_info

#### 提示

编码链接指定C库所需选项的最佳方法是使用“libname.ini”文件，并使用get_info检索所需的选项（有关更多信息，请参阅add_npy_pkg_config）。

### 方法add_npy_pkg_config()

#### 参数概览

```python
add_npy_pkg_config(template, install_dir, subst_dict=None)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1631-L1701)

#### 简介

从模板生成并安装npy-pkg配置文件。

从模板生成的配置文件安装在给定的安装目录中，使用subst_dict进行变量替换。

#### 参数说明
- **template** : 字符串
    模板的路径，相对于当前包路径。
- **install_dir** : 字符串
    在哪里安装npy-pkg配置文件，相对于当前包路径。
- **subst_dict** : 字典, 可选
    如果给定，@key@形式的任何字符串将在安装时由模板文件中的 subst_dict[key] 替换。 安装前缀始终可以通过变量@prefix @获得，因为安装前缀不容易从setup.py中可靠地获得。

另见：
> add_installed_library, get_info

#### 提示

这适用于标准安装和就地构建，即@prefix@引用就地构建的源目录。

#### 例子

```python
config.add_npy_pkg_config('foo.ini.in', 'lib', {'foo': bar})
```

假设foo.ini.in文件具有以下内容：

```
[meta]
Name=@foo@
Version=1.0
Description=dummy description

[default]
Cflags=-I@prefix@/include
Libs=
```

The generated file will have the following content:

```
[meta]
Name=bar
Version=1.0
Description=dummy description

[default]
Cflags=-Iprefix_dir/include
Libs=
```

并将作为foo.ini安装在'lib'子路径中。

### 方法paths()

#### 参数概览

```python
paths(*paths, **kws)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1401-L1414)

#### 简介

将glob应用于路径并在需要时添加local_path。

将glob.glob（...）应用于序列中的每个路径（如果需要），并在需要时预先挂起local_path。 因为这是在所有源列表上调用的，所以这允许在扩展模块和库和脚本的源列表中指定通配符，并允许路径名相对于源目录。

### 方法get_config_cmd()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1758-L1770)

#### 简介

返回numpy.distutils config命令实例。

### 方法get_build_temp_dir()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1772-L1779)

#### 简介

返回应该放置临时文件的临时目录的路径。

### 方法have_f77c()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1781-L1798)

#### 简介

检查Fortran 77编译器的可用性。

在源生成函数中使用它以确保已初始化安装程序分发实例。

#### 提示

如果Fortran 77编译器可用，则为True（因为能够成功编译简单的Fortran 77代码）。

### 方法have_f90c()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1800-L1817)

#### 简介

检查Fortran 90编译器的可用性。

在源生成函数中使用它以确保已初始化安装程序分发实例。

#### 提示

如果Fortran 90编译器可用，则为True（因为能够成功编译简单的Fortran 90代码）

### 方法get_version()

#### 参数概览

```python
get_version(version_file=None, version_variable=None)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1913-L1981)

#### 简介

尝试获取包的版本字符串。

如果无法检测到版本信息，则返回当前包的版本字符串或None。

#### 提示

此方法扫描名为__version__.py，< packagename > _version.py，version.py和__svn_version__.py的文件，以查找字符串变量version，__ version__和< packagename > _version，直到找到版本号。

### 方法make_svn_version_py()

#### 参数概览

```python
make_svn_version_py(delete=True)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1983-L2024)

#### 简介

将数据函数附加到data_files列表，该列表将生成__svn_version__.py文件到当前包目录。

从SVN版本号生成包__svn_version__.py文件，它将在python退出后删除，但在执行sdist等命令时可用。

#### 提示

如果之前存在 __svn_version__.py，则不执行任何操作。

这适用于处理SVN存储库中的源目录。

### 方法make_config_py()

#### 参数概览

```python
make_config_py(name='__config__')
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L2068-L2076)

#### 简介

生成包__config__.py文件，其中包含构建程序包时使用的system_info信息。

此文件安装在软件包安装目录中。

### 方法get_info()

#### 参数概览

```python
get_info(*names)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L2078-L2088)

#### 简介

获取资源信息。

返回单个字典中参数列表中所有名称的信息（来自system_info.get_info）。

## 其他模块

- system_info.get_info(name[, notfound_action])	notfound_action：0 - 什么也不做1 - 显示警告信息2 - 引发错误
- system_info.get_standard_file(fname)	从1）返回名为'fname'的文件列表。系统范围的目录（该模块的目录位置）2）用户HOME目录（os.environ ['HOME']）3）本地目录
- cpuinfo.cpu	
- log.set_verbosity(v[, force])	
- exec_command	exec_command