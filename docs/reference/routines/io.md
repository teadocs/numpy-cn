# 输入和输出

## NumPy二进制文件（NPY, NPZ）

- load(file[, mmap_mode, allow_pickle, …])	从.npy，.npz或pickle文件加载数组或pickle对象。
- save(file, arr[, allow_pickle, fix_imports])	将数组保存为NumPy .npy格式的二进制文件。
- savez(file, *args, **kwds)	将多个数组以未压缩的.npz格式保存到单个文件中。
- savez_compressed(file, *args, **kwds)	将多个数组以压缩的.npz格式保存到单个文件中。

这些二进制文件类型的格式记录在[http://docs.scipy.org/doc/numpy/neps/npy-format.html](http://docs.scipy.org/doc/numpy/neps/npy-format.html)

## 文本文件

- loadtxt(fname[, dtype, comments, delimiter, …])	从文本文件加载数据。
- savetxt(fname, X[, fmt, delimiter, newline, …])	从文本文件加载数据。
- genfromtxt(fname[, dtype, comments, …])	从文本文件加载数据，并按指定处理缺失值。
- fromregex(file, regexp, dtype[, encoding])	使用来自文本文件构造数组

## 正则表达式解析。

- fromstring(string[, dtype, count, sep])	从字符串中的文本数据初始化的新1-D数组。
- ndarray.tofile(fid[, sep, format])	将数组作为文本或二进制写入文件（默认）。
- ndarray.tolist()	将数组作为（可能是嵌套的）列表返回。

## 原始二进制文件

- fromfile(file[, dtype, count, sep])	根据文本或二进制文件中的数据构造数组。
- ndarray.tofile(fid[, sep, format])	将数组作为文本或二进制写入文件（默认）。

## 字符串格式

- array2string(a[, max_line_width, precision, …])	返回数组的字符串表示形式。
- array_repr(arr[, max_line_width, precision, …])	返回数组的字符串表示形式。
- array_str(a[, max_line_width, precision, …])	返回数组中数据的字符串表示形式。
- format_float_positional(x[, precision, …])	在位置表示法中将浮点标量格式化为十进制字符串。
- format_float_scientific(x[, precision, …])	使用科学计数法将浮点标量格式化为十进制字符串。

## 内存映射文件

- memmap	为存储在磁盘上的二进制文件中的数组创建内存映射。

## 文本格式选项

- set_printoptions([precision, threshold, …])	设置打印选项。
- get_printoptions()	返回当前的打印选项。
- set_string_function(f[, repr])	设置一个Python函数，以便在相当打印数组时使用。

## Base-n 相关

- binary_repr(num[, width])	将输入数字的二进制表示形式返回为字符串。
- base_repr(number[, base, padding])	返回给定基本系统中数字的字符串表示形式。

## 数据源

- DataSource([destpath]) 通用数据源文件（文件，http，ftp，...）。