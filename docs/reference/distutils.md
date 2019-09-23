---
meta:
  - name: keywords
    content: NumPy 打包
  - name: description
    content: NumPy提供了增强的distutils功能， 使用 Fortran-compiled 库的子包、自动生成代码和扩展模块的生成和安装变得更加容易。 要使用NumPy distutils的功能，请使用...
---

# 打包（``numpy.distutils``）

NumPy提供了增强的distutils功能，
使用 Fortran-compiled 库的子包、自动生成代码和扩展模块的生成和安装变得更加容易。
要使用NumPy distutils的功能，请使用 ``numpy.distutils.core`` 中的 ``setup`` 命令。
numpy.distutils.misc_util中还提供了一个有用的 [``配置``](#numpy.distutils.misc_util.Configuration) 类，
它可以更容易地构造要传递给 setup 函数的关键字参数（通过传递从类的 todict() 方法获得的字典）。
有关详细信息，请参阅[NumPy Distutils - Users Guide](distutils_guide.html#distutils-user-guide)（NumPy Distutils-用户指南）。

## ``numpy.distutils`` 中的模块

### misc_util 模块

方法 | 描述
---|---
[get_numpy_include_dirs](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.get_numpy_include_dirs.html#numpy.distutils.misc_util.get_numpy_include_dirs)() | 
[dict_append](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.dict_append.html#numpy.distutils.misc_util.dict_append)(d, \*\*kws) | 
[appendpath](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.appendpath.html#numpy.distutils.misc_util.appendpath)(prefix, path) | 
[allpath](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.allpath.html#numpy.distutils.misc_util.allpath)(name) | 使用操作系统的路径分隔符 / 将分隔的路径名转换为路径名。
[dot_join](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.dot_join.html#numpy.distutils.misc_util.dot_join)(\*args) | 
[generate_config_py](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.generate_config_py.html#numpy.distutils.misc_util.generate_config_py)(target) |生成config.py文件，其中包含构建包期间使用的SYSTEM_INFO信息。
[get_cmd](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.get_cmd.html#numpy.distutils.misc_util.get_cmd)(cmdname[, _cache]) | 
[terminal_has_colors](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.terminal_has_colors.html#numpy.distutils.misc_util.terminal_has_colors)() | 
[red_text](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.red_text.html#numpy.distutils.misc_util.red_text)(s) | 
[green_text](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.green_text.html#numpy.distutils.misc_util.green_text)(s) | 
[yellow_text](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.yellow_text.html#numpy.distutils.misc_util.yellow_text)(s) | 
[blue_text](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.blue_text.html#numpy.distutils.misc_util.blue_text)(s) | 
[cyan_text](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.cyan_text.html#numpy.distutils.misc_util.cyan_text)(s) | 
[cyg2win32](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.cyg2win32.html#numpy.distutils.misc_util.cyg2win32)(path) | 
[all_strings](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.all_strings.html#numpy.distutils.misc_util.all_strings)(lst) | 如果lst中的所有项都是String对象，则返回True。
[has_f_sources](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.has_f_sources.html#numpy.distutils.misc_util.has_f_sources)(sources) | 如果源包含Fortran文件，则返回True。
[has_cxx_sources](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.has_cxx_sources.html#numpy.distutils.misc_util.has_cxx_sources)(sources) | 如果源包含C+文件，则返回True。
[filter_sources](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.filter_sources.html#numpy.distutils.misc_util.filter_sources)(sources) | 返回分别包含C、C+、Fortran和Fortran 90模块源的四个文件名列表。
[get_dependencies](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.get_dependencies.html#numpy.distutils.misc_util.get_dependencies)(sources) | 
[is_local_src_dir](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.is_local_src_dir.html#numpy.distutils.misc_util.is_local_src_dir)(directory) | 如果目录是本地目录，则返回True。
[get_ext_source_files](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.get_ext_source_files.html#numpy.distutils.misc_util.get_ext_source_files)(ext) | 
[get_script_files](https://numpy.org/devdocs/reference/generated/numpy.distutils.misc_util.get_script_files.html#numpy.distutils.misc_util.get_script_files)(scripts) | 


- *class* ``numpy.distutils.misc_util.``Configuration``(*package_name=None*, *parent_name=None*, *top_path=None*, *package_path=None*, \*\**attrs*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L731-L2070)

  Construct a configuration instance for the given package name. If
  *parent_name* is not None, then construct the package as a
  sub-package of the *parent_name* package. If *top_path* and
  *package_path* are None then they are assumed equal to
  the path of the file this instance was created in. The setup.py
  files in the numpy distribution are good examples of how to use
  the [``Configuration``](#numpy.distutils.misc_util.Configuration) instance.

  - ``todict``(*self*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L838-L855)

    返回与distutils setup函数的关键字参数兼容的字典。

    **示例：**

    ``` python
    >>> setup(**config.todict())                           #doctest: +SKIP
    ```

  - ``get_distribution``(*self*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L881-L884)

    Return the distutils distribution object for self.

  - ``get_subpackage``(*self*, *subpackage_name*, *subpackage_path=None*, *parent_name=None*, *caller_level=1*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L949-L1008)

    Return list of subpackage configurations. 

    **参数：**
    类型 | 描述
    --- | ---
    subpackage_name : str or None | Name of the subpackage to get the configuration. ‘*’ in subpackage_name is handled as a wildcard.
    subpackage_path : str | If None, then the path is assumed to be the local path plus the subpackage_name. If a setup.py file is not found in the subpackage_path, then a default configuration is used.
    parent_name : str |  Parent name.

  - ``add_subpackage``(*self*, *subpackage_name*, *subpackage_path=None*, *standalone=False*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1010-L1051)

    Add a sub-package to the current Configuration instance.

    This is useful in a setup.py script for adding sub-packages to a package.

    **参数：**

    类型 | 描述
    --- | ---
    subpackage_name : str | name of the subpackage
    subpackage_path : str | if given, the subpackage path such as the subpackage is in subpackage_path / subpackage_name. If None,the subpackage is assumed to be located in the local path / subpackage_name.
    standalone : bool | 

  - ``add_data_files``(*self*, **files*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1183-L1332)

    Add data files to configuration data_files.

    **参数：**

    类型 | 描述
    --- | ---
    files : sequence | Argument(s) can be either 1、2-sequence (``<datadir prefix>``,``<path to data file(s)>``) 2、paths to data files where python datadir prefix defaults to package dir.

    **注释：**

    The form of each element of the files sequence is very flexible
    allowing many combinations of where to get the files from the package
    and where they should ultimately be installed on the system. The most
    basic usage is for an element of the files argument sequence to be a
    simple filename. This will cause that file from the local path to be
    installed to the installation path of the self.name package (package
    path). The file argument can also be a relative path in which case the
    entire relative path will be installed into the package directory.
    Finally, the file can be an absolute path name in which case the file
    will be found at the absolute path name but installed to the package
    path.

    This basic behavior can be augmented by passing a 2-tuple in as the
    file argument. The first element of the tuple should specify the
    relative path (under the package install directory) where the
    remaining sequence of files should be installed to (it has nothing to
    do with the file-names in the source distribution). The second element
    of the tuple is the sequence of files that should be installed. The
    files in this sequence can be filenames, relative paths, or absolute
    paths. For absolute paths the file will be installed in the top-level
    package installation directory (regardless of the first argument).
    Filenames and relative path names will be installed in the package
    install directory under the path name given as the first element of
    the tuple.

    Rules for installation paths:

    1. file.txt -> (., file.txt)-> parent/file.txt
    1. foo/file.txt -> (foo, foo/file.txt) -> parent/foo/file.txt
    1. /foo/bar/file.txt -> (., /foo/bar/file.txt) -> parent/file.txt
    1. ``*``.txt -> parent/a.txt, parent/b.txt
    1. foo/``*``.txt`` -> parent/foo/a.txt, parent/foo/b.txt
    1. ``*/*.txt`` -> (``*``, ``*``/``*``.txt) -> parent/c/a.txt, parent/d/b.txt
    1. (sun, file.txt) -> parent/sun/file.txt
    1. (sun, bar/file.txt) -> parent/sun/file.txt
    1. (sun, /foo/bar/file.txt) -> parent/sun/file.txt
    1. (sun, ``*``.txt) -> parent/sun/a.txt, parent/sun/b.txt
    1. (sun, bar/``*``.txt) -> parent/sun/a.txt, parent/sun/b.txt
    1. (sun/``*``, ``*``/``*``.txt) -> parent/sun/c/a.txt, parent/d/b.txt

    An additional feature is that the path to a data-file can actually be
    a function that takes no arguments and returns the actual path(s) to
    the data-files. This is useful when the data files are generated while
    building the package.

    **Examples**

    Add files to the list of data_files to be included with the package.

    ``` python
    >>> self.add_data_files('foo.dat',
    ...     ('fun', ['gun.dat', 'nun/pun.dat', '/tmp/sun.dat']),
    ...     'bar/cat.dat',
    ...     '/full/path/to/can.dat')                   #doctest: +SKIP
    ```

    will install these data files to:

    ``` python
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

    where ``<package install directory>`` is the package (or sub-package)
    directory such as ‘/usr/lib/python2.4/site-packages/mypackage’ (‘C:
    Python2.4 Lib site-packages mypackage’) or
    ‘/usr/lib/python2.4/site- packages/mypackage/mysubpackage’ (‘C:
    Python2.4 Lib site-packages mypackage mysubpackage’).


  - ``add_data_dir``(*self*, *data_path*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1053-L1172)

    Recursively add files under data_path to data_files list.

    Recursively add files under data_path to the list of data_files to be
    installed (and distributed). The data_path can be either a relative
    path-name, or an absolute path-name, or a 2-tuple where the first
    argument shows where in the install directory the data directory
    should be installed to.

    **参数：**

    类型 | 描述
    --- | ---
    data_path : seq or str | Argument can be either 1、2-sequence (``<datadir suffix>``, ``<path to data directory>``) 2、path to data directory where python datadir suffix defaults to package dir.

    **注释：**

    Rules for installation paths:

    ``` python
    foo/bar -> (foo/bar, foo/bar) -> parent/foo/bar
    (gun, foo/bar) -> parent/gun
    foo/* -> (foo/a, foo/a), (foo/b, foo/b) -> parent/foo/a, parent/foo/b
    (gun, foo/*) -> (gun, foo/a), (gun, foo/b) -> gun
    (gun/*, foo/*) -> parent/gun/a, parent/gun/b
    /foo/bar -> (bar, /foo/bar) -> parent/bar
    (gun, /foo/bar) -> parent/gun
    (fun/*/gun/*, sun/foo/bar) -> parent/fun/foo/gun/bar
    ```

    Examples

    For example suppose the source directory contains fun/foo.dat and
    fun/bar/car.dat:

    ``` python
    >>> self.add_data_dir('fun')                       #doctest: +SKIP
    >>> self.add_data_dir(('sun', 'fun'))              #doctest: +SKIP
    >>> self.add_data_dir(('gun', '/full/path/to/fun'))#doctest: +SKIP
    ```

    Will install data-files to the locations:

    ``` python
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

  - ``add_include_dirs``(*self*, **paths*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1352-L1366)

    Add paths to configuration include directories.

    Add the given sequence of paths to the beginning of the include_dirs
    list. This list will be visible to all extension modules of the
    current package.

  - ``add_headers``(*self*, **files*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1368-L1400)

    Add installable headers to configuration.

    Add the given sequence of files to the beginning of the headers list.
    By default, headers will be installed under ``<python-include>/<self.name.replace(‘.’,’/’)>/`` directory. If an item of files
    is a tuple, then its first argument specifies the actual installation
    location relative to the ``<python-include>`` path.

    **参数：**

    类型 | 描述
    --- | ---
    files : str or seq | Argument(s) can be either:

    2-sequence (``<includedir suffix>``,``<path to header file(s)>``) path(s) to header file(s) where python includedir suffix will default to package name.

  - ``add_extension``(*self*, *name*, *sources*, ***kw*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1425-L1526)

    Add extension to configuration.

    Create and add an Extension instance to the ext_modules list. This
    method also takes the following optional keyword arguments that are
    passed on to the Extension constructor.

    **参数：**

    类型 | 描述
    --- | ---
    name : str | name of the extension
    sources : seq | list of the sources. The list of sources may contain functions (called source generators) which must take an extension instance and a build directory as inputs and return a source file or list of source files or None. If None is returned then no sources are generated. If the Extension instance has no sources after processing all source generators, then no extension module is built.
    include_dirs : |
    define_macros : |
    undef_macros : |
    library_dirs : |
    libraries : |
    runtime_library_dirs : |
    extra_objects : |
    extra_compile_args : |
    extra_link_args : |
    extra_f77_compile_args : |
    extra_f90_compile_args : |
    export_symbols : |
    swig_opts : |
    depends : | The depends list contains paths to files or directories that the sources of the extension module depend on. If any path in the depends list is newer than the extension module, then the module will be rebuilt.
    language : | 
    f2py_options : | 
    module_dirs : | 
    extra_info : dict or list | dict or list of dict of keywords to be appended to keywords.

    **注释：**
    
    The self.paths(…) method is applied to all lists that may contain paths.

  - ``add_library``(*self*, *name*, *sources*, ***build_info*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1528-L1562)

      Add library to configuration.

      **参数：**

      类型 | 描述
      --- | ---
      name : str | Name of the extension.
      sources : sequence | List of the sources. The list of sources may contain functions (called source generators) which must take an extension instance and a build directory as inputs and return a source file or list of source files or None. If None is returned then no sources are generated. If the Extension instance has no sources after processing all source generators, then no extension module is built.
      build_info : dict, optional | The following keys are allowed: 1、depends 2、macros 3、include_dirs 4、extra_compiler_args 5、extra_f77_compile_args 6、extra_f90_compile_args 7、f2py_options 8、language

  - ``add_scripts``(*self*, **files*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1703-L1717)

    Add scripts to configuration.

    Add the sequence of files to the beginning of the scripts list.
    Scripts will be installed under the ``<prefix>``/bin/ directory.

- ``add_installed_library``(*self*, *name*, *sources*, *install_dir*, *build_info=None*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1580-L1629)

    Similar to add_library, but the specified library is installed.

    Most C libraries used with [``distutils``](https://docs.python.org/dev/library/distutils.html#module-distutils) are only used to build python
    extensions, but libraries built through this method will be installed
    so that they can be reused by third-party packages.

    **参数：**

    类型 | 描述
    ---|---
    name : str | Name of the installed library.
    sources : sequence | List of the library’s source files. See [add_library](#numpy.distutils.misc_util.Configuration.add_library) for details.
    install_dir : str | Path to install the library, relative to the current sub-package.
    build_info : dict, optional | The following keys are allowed: 1、depends 2、macros 3、include_dirs 4、extra_compiler_args 5、extra_f77_compile_args 6、extra_f90_compile_args 7、f2py_options 8、language

    **返回：** None

    ::: tip 另见

    [``add_library``](#numpy.distutils.misc_util.Configuration.add_library), [``add_npy_pkg_config``](#numpy.distutils.misc_util.Configuration.add_npy_pkg_config), [``get_info``](#numpy.distutils.misc_util.Configuration.get_info)

    :::

    **注释**

    The best way to encode the options required to link against the specified
    C libraries is to use a “libname.ini” file, and use [``get_info``](#numpy.distutils.misc_util.Configuration.get_info) to
    retrieve the required options (see [``add_npy_pkg_config``](#numpy.distutils.misc_util.Configuration.add_npy_pkg_config) for more
    information).

  - ``add_npy_pkg_config``(*self*, *template*, *install_dir*, *subst_dict=None*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1631-L1700)

    Generate and install a npy-pkg config file from a template.

    The config file generated from *template* is installed in the
    given install directory, using *subst_dict* for variable substitution.

    **参数：**
    类型 | 描述
    ---|---
    template : str | The path of the template, relatively to the current package path.
    install_dir : str | Where to install the npy-pkg config file, relatively to the current package path.
    subst_dict : dict, optional | If given, any string of the form @key@ will be replaced by subst_dict[key] in the template file when installed. The install prefix is always available through the variable @prefix@, since the install prefix is not easy to get reliably from setup.py.

    ::: tip 另见

    [``add_installed_library``](#numpy.distutils.misc_util.Configuration.add_installed_library), [``get_info``](#numpy.distutils.misc_util.Configuration.get_info)

    :::

    **注释**

    This works for both standard installs and in-place builds, i.e. the
    ``@prefix@`` refer to the source directory for in-place builds.

    **示例：**

    ``` python
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

  - ``paths``(*self*, **paths*, ***kws*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1402-L1415)

    Apply glob to paths and prepend local_path if needed.

    Applies glob.glob(…) to each path in the sequence (if needed) and
    pre-pends the local_path if needed. Because this is called on all
    source lists, this allows wildcard characters to be specified in lists
    of sources for extension modules and libraries and scripts and allows
    path-names be relative to the source directory.

  - ``get_config_cmd``(*self*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1757-L1769)

    Returns the numpy.distutils config command instance.

  - ``get_build_temp_dir``(*self*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1771-L1778)

    Return a path to a temporary directory where temporary files should be placed.

  - ``have_f77c``(*self*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1780-L1797)

    Check for availability of Fortran 77 compiler.

    Use it inside source generating function to ensure that setup distribution instance has been initialized.

    **注释**

    True if a Fortran 77 compiler is available (because a simple Fortran 77
    code was able to be compiled successfully).

  - ``have_f90c``(*self*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1799-L1816)

    Check for availability of Fortran 90 compiler.

    Use it inside source generating function to ensure that
    setup distribution instance has been initialized.

    **注释**

    True if a Fortran 90 compiler is available (because a simple Fortran
    90 code was able to be compiled successfully)

  - ``get_version``(*self*, *version_file=None*, *version_variable=None*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1899-L1967)

    Try to get version string of a package.

    Return a version string of the current package or None if the version
    information could not be detected.

    **注释**

    This method scans files named
    \_\_version__.py, ``<packagename>``_version.py, version.py, and
    \_\_svn_version__.py for string variables version, \_\_version__, and
    ``<packagename>``_version, until a version number is found.

  - ``make_svn_version_py``(*self*, *delete=True*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L1969-L2008)

    Appends a data function to the data_files list that will generate
    __svn_version__.py file to the current package directory.

    Generate package __svn_version__.py file from SVN revision number,
    it will be removed after python exits but will be available
    when sdist, etc commands are executed.

    **注释**

    If __svn_version__.py existed before, nothing is done.

    This is
    intended for working with source directories that are in an SVN
    repository.

  - ``make_config_py``(*self*, *name='__config__'*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L2050-L2058)

    Generate package __config__.py file containing system_info
    information used during building the package.

    This file is installed to the
    package installation directory.

  - ``get_info``(*self*, **names*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/distutils/misc_util.py#L2060-L2070)

    Get resources information.

    Return information (from system_info.get_info) for all of the names in
    the argument list in a single dictionary.

### 其他模块

模块名 | 描述
---|---
[system_info.get_info](https://numpy.org/devdocs/reference/generated/numpy.distutils.system_info.get_info.html#numpy.distutils.system_info.get_info)(name[, notfound_action]) | notfound_action:
[system_info.get_standard_file](https://numpy.org/devdocs/reference/generated/numpy.distutils.system_info.get_standard_file.html#numpy.distutils.system_info.get_standard_file)(fname) | Returns a list of files named ‘fname’ from 1) System-wide directory (directory-location of this module) 2) Users HOME directory (os.environ[‘HOME’]) 3) Local directory
[cpuinfo.cpu](https://numpy.org/devdocs/reference/generated/numpy.distutils.cpuinfo.cpu.html#numpy.distutils.cpuinfo.cpu) | 
[log.set_verbosity](https://numpy.org/devdocs/reference/generated/numpy.distutils.log.set_verbosity.html#numpy.distutils.log.set_verbosity)(v[, force]) | 
[exec_command](https://numpy.org/devdocs/reference/generated/numpy.distutils.exec_command.html#module-numpy.distutils.exec_command) | exec_command

## 构建可安装的C库

Conventional C libraries (installed through *add_library*) are not installed, and
are just used during the build (they are statically linked).  An installable C
library is a pure C library, which does not depend on the python C runtime, and
is installed such that it may be used by third-party packages. To build and
install the C library, you just use the method *add_installed_library* instead of
*add_library*, which takes the same arguments except for an additional
``install_dir`` argument:

``` python
>>> config.add_installed_library('foo', sources=['foo.c'], install_dir='lib')
```

### npy-pkg-config 文件

To make the necessary build options available to third parties, you could use
the *npy-pkg-config* mechanism implemented in [``numpy.distutils``](#module-numpy.distutils). This mechanism is
based on a .ini file which contains all the options. A .ini file is very
similar to .pc files as used by the pkg-config unix utility:

```
[meta]
Name: foo
Version: 1.0
Description: foo library

[variables]
prefix = /home/user/local
libdir = ${prefix}/lib
includedir = ${prefix}/include

[default]
cflags = -I${includedir}
libs = -L${libdir} -lfoo
```

Generally, the file needs to be generated during the build, since it needs some
information known at build time only (e.g. prefix). This is mostly automatic if
one uses the *Configuration* method *add_npy_pkg_config*. Assuming we have a
template file foo.ini.in as follows:

```
[meta]
Name: foo
Version: @version@
Description: foo library

[variables]
prefix = @prefix@
libdir = ${prefix}/lib
includedir = ${prefix}/include

[default]
cflags = -I${includedir}
libs = -L${libdir} -lfoo
```

and the following code in setup.py:

``` python
>>> config.add_installed_library('foo', sources=['foo.c'], install_dir='lib')
>>> subst = {'version': '1.0'}
>>> config.add_npy_pkg_config('foo.ini.in', 'lib', subst_dict=subst)
```

This will install the file foo.ini into the directory package_dir/lib, and the
foo.ini file will be generated from foo.ini.in, where each ``@version@`` will be
replaced by ``subst_dict['version']``. The dictionary has an additional prefix
substitution rule automatically added, which contains the install prefix (since
this is not easy to get from setup.py).  npy-pkg-config files can also be
installed at the same location as used for numpy, using the path returned from
*get_npy_pkg_dir* function.

### 重用另一个包中的C库

可以从 [``numpy.distutils.misc_util``](#module-numpy.distutils.misc_util) 中的 *get_info* 函数轻松检索信息：

``` python
>>> info = get_info('npymath')
>>> config.add_extension('foo', sources=['foo.c'], extra_info=**info)
```

可以将用于查找 .ini 文件的其他路径列表提供给 *get_info*。

## ``.src``文件的转换

NumPy distutils支持名为 ``<somefile>``.src的源文件的自动转换。
这个工具可以用来维护非常相似的代码块，只需要在块之间进行简单的更改。
在安装程序的构建阶段，如果遇到名为 ``<somefile>`` .src的模板文件，
则会从该模板构建名为 ``<somefile>`` 的新文件，并将其放在要使用的构建目录中。
支持两种形式的模板转换。第一种形式出现在名为 ``<file>``.ext.src的文件中，
其中ext是公认的Fortran扩展名(f，f90，f95，f77，for，FTN，pyf)。
第二种形式用于所有其他情况。请参见使用[模板转换.src文件](distutils_guide.html#templating)。
