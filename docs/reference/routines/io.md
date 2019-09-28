# 输入和输出

## NumPy二进制文件（NPY，NPZ）

方法 | 描述
---|---
[load](https://numpy.org/devdocs/reference/generated/numpy.load.html#numpy.load)(file[, mmap_mode, allow_pickle, …]) | 从.npy、.npz或pickle文件加载阵列或pickle对象。
[save](https://numpy.org/devdocs/reference/generated/numpy.save.html#numpy.save)(file, arr[, allow_pickle, fix_imports]) | 将数组保存为NumPy.npy格式的二进制文件。
[savez](https://numpy.org/devdocs/reference/generated/numpy.savez.html#numpy.savez)(file, \*args, \*\*kwds) | 将几个数组以未压缩的.npz格式保存到单个文件中。
[savez_compressed](https://numpy.org/devdocs/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed)(file, \*args, \*\*kwds) | 以压缩的.npz格式将几个数组保存到单个文件中。

有关这些二进制文件类型的格式，请参阅
[``numpy.lib.format``](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#module-numpy.lib.format)

## 文本文件

方法 | 描述
---|---
[loadtxt](https://numpy.org/devdocs/reference/generated/numpy.loadtxt.html#numpy.loadtxt)(fname[, dtype, comments, delimiter, …]) | 从文本文件加载数据。
[savetxt](https://numpy.org/devdocs/reference/generated/numpy.savetxt.html#numpy.savetxt)(fname, X[, fmt, delimiter, newline, …]) | 将数组保存到文本文件。
[genfromtxt](https://numpy.org/devdocs/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt)(fname[, dtype, comments, …]) | 从文本文件加载数据，并按指定方式处理缺少的值。
[fromregex](https://numpy.org/devdocs/reference/generated/numpy.fromregex.html#numpy.fromregex)(file, regexp, dtype[, encoding]) | 使用正则表达式解析从文本文件构造数组。
[fromstring](https://numpy.org/devdocs/reference/generated/numpy.fromstring.html#numpy.fromstring)(string[, dtype, count, sep]) | 从字符串中的文本数据初始化的新一维数组。
[ndarray.tofile](https://numpy.org/devdocs/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile)(fid[, sep, format]) | 将数组以文本或二进制形式写入文件(默认)。
[ndarray.tolist](https://numpy.org/devdocs/reference/generated/numpy.ndarray.tolist.html#numpy.ndarray.tolist)() | 以Python标量的a.ndim级深嵌套列表的形式返回数组。

## 原始二进制文件

方法 | 描述
---|---
[fromfile](https://numpy.org/devdocs/reference/generated/numpy.fromfile.html#numpy.fromfile)(file[, dtype, count, sep, offset]) | 从文本或二进制文件中的数据构造数组。
[ndarray.tofile](https://numpy.org/devdocs/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile)(fid[, sep, format]) | 将数组以文本或二进制形式写入文件(默认)。

## 字符串格式

方法 | 描述
---|---
[array2string](https://numpy.org/devdocs/reference/generated/numpy.array2string.html#numpy.array2string)(a[, max_line_width, precision, …]) | 返回数组的字符串表示形式。
[array_repr](https://numpy.org/devdocs/reference/generated/numpy.array_repr.html#numpy.array_repr)(arr[, max_line_width, precision, …]) | 返回数组的字符串表示形式。
[array_str](https://numpy.org/devdocs/reference/generated/numpy.array_str.html#numpy.array_str)(a[, max_line_width, precision, …]) | 返回数组中数据的字符串表示形式。
[format_float_positional](https://numpy.org/devdocs/reference/generated/numpy.format_float_positional.html#numpy.format_float_positional)(x[, precision, …]) | 将浮点标量格式化为位置表示法中的十进制字符串。
[format_float_scientific](https://numpy.org/devdocs/reference/generated/numpy.format_float_scientific.html#numpy.format_float_scientific)(x[, precision, …]) | 将浮点标量格式化为科学记数法中的十进制字符串。

## 内存映射文件

方法 | 描述
---|---
[memmap](https://numpy.org/devdocs/reference/generated/numpy.memmap.html#numpy.memmap) | 创建存储在磁盘上二进制文件中的阵列的内存映射。

## 文本格式选项

方法 | 描述
---|---
[set_printoptions](https://numpy.org/devdocs/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions)([precision, threshold, …]) | 设置打印选项。
[get_printoptions](https://numpy.org/devdocs/reference/generated/numpy.get_printoptions.html#numpy.get_printoptions)() | 返回当前打印选项。
[set_string_function](https://numpy.org/devdocs/reference/generated/numpy.set_string_function.html#numpy.set_string_function)(f[, repr]) | 设置在更好的打印数组时要使用的Python函数。
[printoptions](https://numpy.org/devdocs/reference/generated/numpy.printoptions.html#numpy.printoptions)(\*args, \*\*kwargs) | 上下文管理器，用于设置打印选项。

## 基数n表示

方法 | 描述
---|---
[binary_repr](https://numpy.org/devdocs/reference/generated/numpy.binary_repr.html#numpy.binary_repr)(num[, width]) | 以字符串形式返回输入数字的二进制表示形式。
[base_repr](https://numpy.org/devdocs/reference/generated/numpy.base_repr.html#numpy.base_repr)(number[, base, padding]) | 返回给定基本系统中数字的字符串表示形式。

## 数据源

方法 | 描述
---|---
[DataSource](https://numpy.org/devdocs/reference/generated/numpy.DataSource.html#numpy.DataSource)([destpath]) | 通用数据源文件（file、http、ftp等）。

## 二进制格式描述

方法 | 描述
---|---
[lib.format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#module-numpy.lib.format) | 二进制序列化
