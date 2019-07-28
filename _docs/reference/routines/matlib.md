# 矩阵库

该模块包含``numpy``命名空间中的所有函数，以下替换函数返回``matrices``而不是``ndarrays``。

也在numpy命名空间和返回矩阵中的函数

- mat(data[, dtype])	将输入参数解释为矩阵。
- matrix(data[, dtype, copy])	从类数组对象或数据字符串返回矩阵。
- asmatrix(data[, dtype])	将输入参数解释为矩阵。
- bmat(obj[, ldict, gdict])	从字符串，嵌套序列或数组构建矩阵对象。

``matlab``替代函数

- empty(shape[, dtype, order])	返回给定形状和类型的新矩阵，而不初始化条目。
- zeros(shape[, dtype, order])	返回给定形状和类型的矩阵，用零填充。
- ones(shape[, dtype, order])	返回给定形状和类型的矩阵，用一填充。
- eye(n[, M, k, dtype, order])	返回一个矩阵，其中对角线为1，零点为零。
- identity(n[, dtype])	返回给定大小的方形单位矩阵。
- repmat(a, m, n)	重复0-D到2-D阵列或矩阵MxN次。
- rand(*args)	返回给定形状的随机值矩阵。
- randn(*args)	返回包含“标准正态”分布数据的随机矩阵。
