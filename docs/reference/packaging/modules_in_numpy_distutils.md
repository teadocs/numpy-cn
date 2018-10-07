# numpy.distutils中的模块

NumPy提供增强的distutils功能，以便更容易地构建和安装子包，自动生成代码以及使用Fortran编译库的扩展模块。 要使用NumPy distutils的功能，请使用numpy.distutils.core中的setup命令。 [numpy.distutils.misc_util](https://docs.scipy.org/doc/numpy/reference/distutils.html#module-numpy.distutils.misc_util) 中还提供了一个有用的 [Configuration](https://docs.scipy.org/doc/numpy/reference/distutils.html#numpy.distutils.misc_util.Configuration) 类，它可以更容易地构造关键字参数以传递给setup函数（通过传递从类的todict()方法获得的字典）。 有关详细信息，请参阅 ``<site-packages>/numpy/doc/DISTUTILS.txt`` 中的 NumPy Distutils用户指南。

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