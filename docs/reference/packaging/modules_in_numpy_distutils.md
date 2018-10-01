# numpy.distutils中的模块

NumPy提供增强的distutils功能，以便更容易地构建和安装子包，自动生成代码以及使用Fortran编译库的扩展模块。 要使用NumPy distutils的功能，请使用numpy.distutils.core中的setup命令。 numpy.distutils.misc_util中还提供了一个有用的Configuration类，它可以更容易地构造关键字参数以传递给setup函数（通过传递从类的todict（）方法获得的字典）。 有关详细信息，请参阅<site-packages> /numpy/doc/DISTUTILS.txt中的NumPy Distutils用户指南。

## distutils中的模块

### misc_util

- get_numpy_include_dirs()	
- dict_append(d, **kws)	
- appendpath(prefix, path)	
- allpath(name)	Convert a /-separated pathname to one using the OS’s path separator.
- dot_join(*args)	
- generate_config_py(target)	Generate config.py file containing system_info information used during building the package.
- get_cmd(cmdname[, _cache])	
- terminal_has_colors()	
- red_text(s)	
- green_text(s)	
- yellow_text(s)	
- blue_text(s)	
- cyan_text(s)	
- cyg2win32(path)	
- all_strings(lst)	Return True if all items in lst are string objects.
- has_f_sources(sources)	Return True if sources contains Fortran files
- has_cxx_sources(sources)	Return True if sources contains C++ files
- filter_sources(sources)	Return four lists of filenames containing C, C++, Fortran, and Fortran 90 module sources, respectively.
- get_dependencies(sources)	
- is_local_src_dir(directory)	Return true if directory is local directory.
- get_ext_source_files(ext)	
- get_script_files(scripts)	