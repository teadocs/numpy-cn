# 矩阵库 (``numpy.matlib``)

该模块包含 [``numpy``](index.html#module-numpy) 命名空间中的所有函数, 以下返回 [``矩阵``](https://numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix) 而不是 [``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)的替换函数。

也在numpy命名空间中的函数并返回矩阵

method | description
---|---
[mat](https://numpy.org/devdocs/reference/generated/numpy.mat.html#numpy.mat)(data[, dtype]) | 将输入解释为 [矩阵](https://numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix).
[matrix](https://numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix)(data[, dtype, copy]) | 注意：不再建议使用此类，即使对于线性
[asmatrix](https://numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | 将输入解释为矩阵。
[bmat](https://numpy.org/devdocs/reference/generated/numpy.bmat.html#numpy.bmat)(obj[, ldict, gdict]) | 从字符串，嵌套序列或数组构建矩阵对象。

 [``matlib``](#module-numpy.matlib)的替换函数

method | description
---|---
[empty](https://numpy.org/devdocs/reference/generated/numpy.matlib.empty.html#numpy.matlib.empty)(shape[, dtype, order]) | 返回给定形状和类型的新矩阵，而无需初始化条目。
[zeros](https://numpy.org/devdocs/reference/generated/numpy.matlib.zeros.html#numpy.matlib.zeros)(shape[, dtype, order]) | 返回给定形状和类型的矩阵，并用零填充。
[ones](https://numpy.org/devdocs/reference/generated/numpy.matlib.ones.html#numpy.matlib.ones)(shape[, dtype, order]) | 一个矩阵。
[eye](https://numpy.org/devdocs/reference/generated/numpy.matlib.eye.html#numpy.matlib.eye)(n[, M, k, dtype, order]) | 返回一个矩阵，在对角线上有一个，在其他地方为零。
[identity](https://numpy.org/devdocs/reference/generated/numpy.matlib.identity.html#numpy.matlib.identity)(n[, dtype]) | 返回给定大小的平方单位矩阵。
[repmat](https://numpy.org/devdocs/reference/generated/numpy.matlib.repmat.html#numpy.matlib.repmat)(a, m, n) | 重复从0D到2D数组或矩阵MxN次。
[rand](https://numpy.org/devdocs/reference/generated/numpy.matlib.rand.html#numpy.matlib.rand)(\*args) |返回具有给定形状的随机值矩阵。
[randn](https://numpy.org/devdocs/reference/generated/numpy.matlib.randn.html#numpy.matlib.randn)(\*args) | 返回一个随机矩阵，其中包含来自“标准正态”分布的数据。
