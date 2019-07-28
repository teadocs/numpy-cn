# 创建数组

另见：

> Array creation

## Ones 和 zeros 方法

- empty(shape[, dtype, order])	返回给定形状和类型的新数组，而不初始化条目。
- empty_like(a[, dtype, order, subok])	返回一个与给定数组具有相同形状和类型的新数组。
- eye(N[, M, k, dtype, order])	返回一个二维数组，其中对角线为1，零点为零。
- identity(n[, dtype])	返回标识数组。
- ones(shape[, dtype, order])	返回一个给定形状和类型的新数组，用一个填充。
- ones_like(a[, dtype, order, subok])	返回与给定数组具有相同形状和类型的数组。
- zeros(shape[, dtype, order])	返回给定形状和类型的新数组，用零填充。
- zeros_like(a[, dtype, order, subok])	返回与给定数组具有相同形状和类型的零数组。
- full(shape, fill_value[, dtype, order])	返回给定形状和类型的新数组，填充fill_value。
- full_like(a, fill_value[, dtype, order, subok])	返回与给定数组具有相同形状和类型的完整数组。

## 利用现有数据

- array(object[, dtype, copy, order, subok, ndmin])	创建一个新的数组。
- asarray(a[, dtype, order])	将输入的参数转换为数组。
- asanyarray(a[, dtype, order])	将输入转换为ndarray，但通过ndarray子类传递。
- ascontiguousarray(a[, dtype])	在内存中返回连续数组(C顺序)。
- asmatrix(data[, dtype])	将输入转换为矩阵。
- copy(a[, order])	返回给定对象的数组副本。
- frombuffer(buffer[, dtype, count, offset])	将缓冲区转换为一维数组。
- fromfile(file[, dtype, count, sep])	从文本或二进制文件中的数据构造数组。
- fromfunction(function, shape, **kwargs)	通过在每个坐标上执行函数来构造数组。
- fromiter(iterable, dtype[, count])	从可迭代对象创建一个新的一维数组。
- fromstring(string[, dtype, count, sep])	从字符串中的文本数据初始化的新的一维数组.
- loadtxt(fname[, dtype, comments, delimiter, …])	从文本文件加载数据。
- Creating record arrays (numpy.rec)

> **注意**
> ``numpy.rec`` 是numpy.core.records的首选别名。

- core.records.array(obj[, dtype, shape, …])	从各种各样的对象构造一个记录数组。
- core.records.fromarrays(arrayList[, dtype, …])	从数组的(平面)列表创建记录数组
- core.records.fromrecords(recList[, dtype, …])	 从文本形式的记录列表中创建一个重新数组
- core.records.fromstring(datastring[, dtype, …])	从字符串中包含的二进制数据创建(只读)记录数组。
core.records.fromfile(fd[, dtype, shape, …])	从二进制文件数据创建数组

## 创建字符数组(``numpy.charr``)

> **注意**
> ``numpy.char`` 是 ``numpy.core.defchararra`` 的首选别名。

- core.defchararray.array(obj[, itemsize, …])	创建一个字符数组。
- core.defchararray.asarray(obj[, itemsize, …])	将输入转换为字符数组，只在必要时复制数据。

## 数值范围

- arange([start,] stop[, step,][, dtype])	在给定的间隔内返回均匀间隔的值。
- linspace(start, stop[, num, endpoint, …])	在指定的间隔内返回均匀间隔的数字。
- logspace(start, stop[, num, endpoint, base, …])	返回数在对数刻度上均匀分布。
- geomspace(start, stop[, num, endpoint, dtype])	返回数在对数尺度上均匀分布(几何级数)。
- meshgrid(*xi, **kwargs)	从坐标向量返回坐标矩阵。
- mgrid nd_grid 实例，它返回一个密集的多维“meshgrid”。
- ogrid nd_grid 实例，它返回一个开放的多维“meshgrid”。

## 构建矩阵

- diag(v[, k])	提取对角线或构造对角线阵列。
- diagflat(v[, k])	使用展平输入创建二维数组作为对角线。
- tri(N[, M, k, dtype])	一个数组，其中包含给定对角线和低于给定对角线的数字，其他地方为零
- tril(m[, k])	数组的下三角形。
- triu(m[, k])	数组的上三角形。
- vander(x[, N, increasing])	生成Vandermonde矩阵。

## 矩阵类

- mat(data[, dtype])	将输入解释为矩阵。
- bmat(obj[, ldict, gdict])	 从字符串、嵌套序列或数组生成矩阵对象。