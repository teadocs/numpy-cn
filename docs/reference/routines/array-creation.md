# 创建数组

::: tip 另见

[数组创建](/user/basics/creation.html)

:::

## Ones 和 zeros 填充方式

方法 | 描述
---|---
[empty](https://numpy.org/devdocs/reference/generated/numpy.empty.html#numpy.empty)(shape[, dtype, order]) | 返回给定形状和类型的新数组，而无需初始化条目。
[empty_like](https://numpy.org/devdocs/reference/generated/numpy.empty_like.html#numpy.empty_like)(prototype[, dtype, order, subok, …]) | 返回形状和类型与给定数组相同的新数组。
[eye](https://numpy.org/devdocs/reference/generated/numpy.eye.html#numpy.eye)(N[, M, k, dtype, order]) | 返回一个二维数组，对角线上有一个，其他地方为零。
[identity](https://numpy.org/devdocs/reference/generated/numpy.identity.html#numpy.identity)(n[, dtype]) | 返回标识数组。
[ones](https://numpy.org/devdocs/reference/generated/numpy.ones.html#numpy.ones)(shape[, dtype, order]) | 返回给定形状和类型的新数组，并填充为1。
[ones_like](https://numpy.org/devdocs/reference/generated/numpy.ones_like.html#numpy.ones_like)(a[, dtype, order, subok, shape]) | 返回形状与类型与给定数组相同的数组。
[zeros](https://numpy.org/devdocs/reference/generated/numpy.zeros.html#numpy.zeros)(shape[, dtype, order]) | 返回给定形状和类型的新数组，并用零填充。
[zeros_like](https://numpy.org/devdocs/reference/generated/numpy.zeros_like.html#numpy.zeros_like)(a[, dtype, order, subok, shape]) | 返回形状与类型与给定数组相同的零数组。
[full](https://numpy.org/devdocs/reference/generated/numpy.full.html#numpy.full)(shape, fill_value[, dtype, order]) | 返回给定形状和类型的新数组，并用fill_value填充。
[full_like](https://numpy.org/devdocs/reference/generated/numpy.full_like.html#numpy.full_like)(a, fill_value[, dtype, order, …]) | 返回形状和类型与给定数组相同的完整数组。

## 从现有的数据创建

方法 | 描述
---|---
[array](https://numpy.org/devdocs/reference/generated/numpy.array.html#numpy.array)(object[, dtype, copy, order, subok, ndmin]) | 创建一个数组。
[asarray](https://numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray)(a[, dtype, order]) | 将输入转换为数组。
[asanyarray](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray)(a[, dtype, order]) | 将输入转换为ndarray，但通过ndarray子类。
[ascontiguousarray](https://numpy.org/devdocs/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray)(a[, dtype]) | 返回内存中的连续数组（ndim > = 1）（C顺序）。
[asmatrix](https://numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | 将输入解释为矩阵。
[copy](https://numpy.org/devdocs/reference/generated/numpy.copy.html#numpy.copy)(a[, order]) | 返回给定对象的数组副本。
[frombuffer](https://numpy.org/devdocs/reference/generated/numpy.frombuffer.html#numpy.frombuffer)(buffer[, dtype, count, offset]) | 将缓冲区解释为一维数组。
[fromfile](https://numpy.org/devdocs/reference/generated/numpy.fromfile.html#numpy.fromfile)(file[, dtype, count, sep, offset]) | 根据文本或二进制文件中的数据构造一个数组。
[fromfunction](https://numpy.org/devdocs/reference/generated/numpy.fromfunction.html#numpy.fromfunction)(function, shape, \*\*kwargs) | 通过在每个坐标上执行一个函数来构造一个数组。
[fromiter](https://numpy.org/devdocs/reference/generated/numpy.fromiter.html#numpy.fromiter)(iterable, dtype[, count]) | 从可迭代对象创建一个新的一维数组。
[fromstring](https://numpy.org/devdocs/reference/generated/numpy.fromstring.html#numpy.fromstring)(string[, dtype, count, sep]) | 从字符串中的文本数据初始化的新一维数组。
[loadtxt](https://numpy.org/devdocs/reference/generated/numpy.loadtxt.html#numpy.loadtxt)(fname[, dtype, comments, delimiter, …]) | 从文本文件加载数据。

## 创建记录数组（``numpy.rec``）

::: tip 注意

``numpy.rec`` 是的首选别名 ``numpy.core.records``。

:::

方法 | 描述
---|---
[core.records.array](https://numpy.org/devdocs/reference/generated/numpy.core.records.array.html#numpy.core.records.array)(obj[, dtype, shape, …]) | 从各种各样的对象构造一个记录数组。
[core.records.fromarrays](https://numpy.org/devdocs/reference/generated/numpy.core.records.fromarrays.html#numpy.core.records.fromarrays)(arrayList[, dtype, …]) | 从（平面）数组列表创建记录数组
[core.records.fromrecords](https://numpy.org/devdocs/reference/generated/numpy.core.records.fromrecords.html#numpy.core.records.fromrecords)(recList[, dtype, …]) | 从文本格式的记录列表创建一个rearray
[core.records.fromstring](https://numpy.org/devdocs/reference/generated/numpy.core.records.fromstring.html#numpy.core.records.fromstring)(datastring[, dtype, …])	 | 根据字符串中包含的二进制数据创建（只读）记录数组
[core.records.fromfile](https://numpy.org/devdocs/reference/generated/numpy.core.records.fromfile.html#numpy.core.records.fromfile)(fd[, dtype, shape, …]) | 根据二进制文件数据创建数组

## 创建字符数组（``numpy.char``）

::: tip 注意

[``numpy.char``](char.html)是的首选别名 ``numpy.core.defchararray``。

:::

方法 | 描述
---|---
[core.defchararray.array](https://numpy.org/devdocs/reference/generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array)(obj[, itemsize, …]) | 创建一个chararray。
[core.defchararray.asarray](https://numpy.org/devdocs/reference/generated/numpy.core.defchararray.asarray.html#numpy.core.defchararray.asarray)(obj[, itemsize, …]) | 将输入转换为chararray，仅在必要时复制数据。

## 数值范围

方法 | 描述
---|---
[arange](https://numpy.org/devdocs/reference/generated/numpy.arange.html#numpy.arange)([start,] stop[, step,][, dtype]) | 返回给定间隔内的均匀间隔的值。
[linspace](https://numpy.org/devdocs/reference/generated/numpy.linspace.html#numpy.linspace)(start, stop[, num, endpoint, …]) | 返回指定间隔内的等间隔数字。
[logspace](https://numpy.org/devdocs/reference/generated/numpy.logspace.html#numpy.logspace)(start, stop[, num, endpoint, base, …]) | 返回数以对数刻度均匀分布。
[geomspace](https://numpy.org/devdocs/reference/generated/numpy.geomspace.html#numpy.geomspace)(start, stop[, num, endpoint, …]) | 返回数字以对数刻度（几何级数）均匀分布。
[meshgrid](https://numpy.org/devdocs/reference/generated/numpy.meshgrid.html#numpy.meshgrid)(\*xi, \*\*kwargs) | 从坐标向量返回坐标矩阵。
[mgrid](https://numpy.org/devdocs/reference/generated/numpy.mgrid.html#numpy.mgrid) | nd_grid实例，它返回一个密集的多维“ meshgrid”。
[ogrid](https://numpy.org/devdocs/reference/generated/numpy.ogrid.html#numpy.ogrid) | nd_grid实例，它返回一个开放的多维“ meshgrid”。

## 创建矩阵

方法 | 描述
---|---
[diag](https://numpy.org/devdocs/reference/generated/numpy.diag.html#numpy.diag)(v[, k]) | 提取对角线或构造对角线数组。
[diagflat](https://numpy.org/devdocs/reference/generated/numpy.diagflat.html#numpy.diagflat)(v[, k]) | 使用展平的输入作为对角线创建二维数组。
[tri](https://numpy.org/devdocs/reference/generated/numpy.tri.html#numpy.tri)(N[, M, k, dtype]) | 在给定对角线处及以下且在其他位置为零的数组。
[tril](https://numpy.org/devdocs/reference/generated/numpy.tril.html#numpy.tril)(m[, k]) | 数组的下三角。
[triu](https://numpy.org/devdocs/reference/generated/numpy.triu.html#numpy.triu)(m[, k]) | 数组的上三角。
[vander](https://numpy.org/devdocs/reference/generated/numpy.vander.html#numpy.vander)(x[, N, increasing]) | 生成范德蒙矩阵。

## 矩阵类

方法 | 描述
---|---
[mat](https://numpy.org/devdocs/reference/generated/numpy.mat.html#numpy.mat)(data[, dtype]) | 将输入解释为矩阵。
[bmat](https://numpy.org/devdocs/reference/generated/numpy.bmat.html#numpy.bmat)(obj[, ldict, gdict])	 | 从字符串，嵌套序列或数组构建矩阵对象。
