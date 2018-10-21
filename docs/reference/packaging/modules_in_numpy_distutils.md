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

Add paths to configuration include directories.

Add the given sequence of paths to the beginning of the include_dirs list. This list will be visible to all extension modules of the current package.

### 方法add_headers()

#### 参数概览

```python
add_headers(*files)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1367-L1399)

Add installable headers to configuration.

Add the given sequence of files to the beginning of the headers list. By default, headers will be installed under < python- include >/< self.name.replace(‘.’,’/’) >/ directory. If an item of files is a tuple, then its first argument specifies the actual installation location relative to the < python-include > path.

#### 参数说明

- **files** : str or seq
    Argument(s) can be either:
    - 2-sequence (< includedir suffix >,< path to header file(s) >)
    - path(s) to header file(s) where python includedir suffix will default to package name.

### 方法add_extension()

#### 参数概览

```python
add_extension(name, sources, **kw)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1424-L1525)

#### 简介

Add extension to configuration.

Create and add an Extension instance to the ext_modules list. This method also takes the following optional keyword arguments that are passed on to the Extension constructor.

#### 参数说明

- **name** : str
    name of the extension
- **sources** : seq
    list of the sources. The list of sources may contain functions (called source generators) which must take an extension instance and a build directory as inputs and return a source file or list of source files or None. If None is returned then no sources are generated. If the Extension instance has no sources after processing all source generators, then no extension module is built.
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
    The depends list contains paths to files or directories that the sources of the extension module depend on. If any path in the depends list is newer than the extension module, then the module will be rebuilt.
- **language** :（译者注：官方无介绍）
- **f2py_options** :（译者注：官方无介绍）
- **module_dirs** :（译者注：官方无介绍）
- **extra_info** : dict or list
    dict or list of dict of keywords to be appended to keywords.

#### 提示

The self.paths(…) method is applied to all lists that may contain paths.

### 方法add_library()

#### 参数概览

```python
add_library(name, sources, **build_info)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1527-L1561)

#### 简介

Add library to configuration.

#### 参数说明

- **name** : str
    Name of the extension.
- **sources** : sequence
    List of the sources. The list of sources may contain functions (called source generators) which must take an extension instance and a build directory as inputs and return a source file or list of source files or None. If None is returned then no sources are generated. If the Extension instance has no sources after processing all source generators, then no extension module is built.
- **build_info** : dict, optional
    - The following keys are allowed:
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

Add scripts to configuration.

Add the sequence of files to the beginning of the scripts list. Scripts will be installed under the <prefix>/bin/ directory.

### 方法add_installed_library()

#### 参数概览

```python
add_installed_library(name, sources, install_dir, build_info=None)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1580-L1629)

#### 简介

Similar to add_library, but the specified library is installed.

Most C libraries used with distutils are only used to build python extensions, but libraries built through this method will be installed so that they can be reused by third-party packages.

#### 参数说明
- **name** : str
    Name of the installed library.
- **sources** : sequence
    List of the library’s source files. See add_library for details.
- **install_dir** : str
    Path to install the library, relative to the current sub-package.
- **build_info** : dict, optional
    - The following keys are allowed:
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

The best way to encode the options required to link against the specified C libraries is to use a “libname.ini” file, and use get_info to retrieve the required options (see add_npy_pkg_config for more information).

### 方法add_npy_pkg_config()

#### 参数概览

```python
add_npy_pkg_config(template, install_dir, subst_dict=None)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1631-L1701)

#### 简介

Generate and install a npy-pkg config file from a template.

The config file generated from template is installed in the given install directory, using subst_dict for variable substitution.

#### 参数说明
- **template** : str
    The path of the template, relatively to the current package path.
- **install_dir** : str
    Where to install the npy-pkg config file, relatively to the current package path.
- **subst_dict** : dict, optional
    If given, any string of the form @key@ will be replaced by subst_dict[key] in the template file when installed. The install prefix is always available through the variable @prefix@, since the install prefix is not easy to get reliably from setup.py.

另见：
> add_installed_library, get_info

#### 提示

This works for both standard installs and in-place builds, i.e. the @prefix@ refer to the source directory for in-place builds.

#### 例子

```python
config.add_npy_pkg_config('foo.ini.in', 'lib', {'foo': bar})
```

Assuming the foo.ini.in file has the following content:

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

and will be installed as foo.ini in the ‘lib’ subpath.

### 方法paths()

#### 参数概览

```python
paths(*paths, **kws)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1401-L1414)

#### 简介

Apply glob to paths and prepend local_path if needed.

Applies glob.glob(…) to each path in the sequence (if needed) and pre-pends the local_path if needed. Because this is called on all source lists, this allows wildcard characters to be specified in lists of sources for extension modules and libraries and scripts and allows path-names be relative to the source directory.

### 方法get_config_cmd()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1758-L1770)

#### 简介

Returns the numpy.distutils config command instance.

### 方法get_build_temp_dir()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1772-L1779)

#### 简介

Return a path to a temporary directory where temporary files should be placed.

### 方法have_f77c()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1781-L1798)

#### 简介

Check for availability of Fortran 77 compiler.

Use it inside source generating function to ensure that setup distribution instance has been initialized.

Notes

True if a Fortran 77 compiler is available (because a simple Fortran 77 code was able to be compiled successfully).

### 方法have_f90c()

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1800-L1817)

#### 简介

Check for availability of Fortran 90 compiler.

Use it inside source generating function to ensure that setup distribution instance has been initialized.

Notes

True if a Fortran 90 compiler is available (because a simple Fortran 90 code was able to be compiled successfully)

### 方法get_version()

#### 参数概览

```python
get_version(version_file=None, version_variable=None)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1913-L1981)

#### 简介

Try to get version string of a package.

Return a version string of the current package or None if the version information could not be detected.

Notes

This method scans files named __version__.py, < packagename >_version.py, version.py, and __svn_version__.py for string variables version, __version__, and < packagename >_version, until a version number is found.

### 方法make_svn_version_py()

#### 参数概览

```python
make_svn_version_py(delete=True)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L1983-L2024)

#### 简介

Appends a data function to the data_files list that will generate __svn_version__.py file to the current package directory.

Generate package __svn_version__.py file from SVN revision number, it will be removed after python exits but will be available when sdist, etc commands are executed.

Notes

If __svn_version__.py existed before, nothing is done.

This is intended for working with source directories that are in an SVN repository.

### 方法make_config_py()

#### 参数概览

```python
make_config_py(name='__config__')
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L2068-L2076)

#### 简介

Generate package __config__.py file containing system_info information used during building the package.

This file is installed to the package installation directory.

### 方法get_info()

#### 参数概览

```python
get_info(*names)
```

#### 源码

[点击查看源码](http://github.com/numpy/numpy/blob/v1.15.1/numpy/distutils/misc_util.py#L2078-L2088)

#### 简介

Get resources information.

Return information (from system_info.get_info) for all of the names in the argument list in a single dictionary.

## 其他模块

- system_info.get_info(name[, notfound_action])	notfound_action: 0 - do nothing 1 - display warning message 2 - raise error
- system_info.get_standard_file(fname)	Returns a list of files named ‘fname’ from 1) System-wide directory (directory-location of this module) 2) Users HOME directory (os.environ[‘HOME’]) 3) Local directory
- cpuinfo.cpu	
- log.set_verbosity(v[, force])	
- exec_command	exec_command