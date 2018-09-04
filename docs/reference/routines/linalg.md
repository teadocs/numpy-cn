# 线性代数

## 矩阵和矢量产品

- dot(a, b[, out])	两个数组的点积。
linalg.multi_dot(arrays)	在单个函数调用中计算两个或多个数组的点积，同时自动选择最快的求值顺序。
- vdot(a, b)	返回两个向量的点积。
- inner(a, b)	两个数组的内积。
- outer(a, b[, out])	计算两个向量的外积。
- matmul(a, b[, out])	两个数组的矩阵乘积。
- tensordot(a, b[, axes])	对于数组> = 1-D，沿指定轴计算张量点积。
- einsum(subscripts, *operands[, out, dtype, …])	评估操作数上的爱因斯坦求和约定。
- einsum_path(subscripts, *operands[, optimize])	通过考虑中间数组的创建，评估einsum表达式的最低成本收缩顺序。
- linalg.matrix_power(M, n)	将方阵提高到（整数）幂n。
- kron(a, b)	两个阵列的Kronecker产品。

## 分解

- linalg.cholesky(a)	Cholesky分解。
- linalg.qr(a[, mode])	计算矩阵的qr分解。
- linalg.svd(a[, full_matrices, compute_uv])	奇异值分解。

## 矩阵特征值

- linalg.eig(a)	计算正方形阵列的特征值和右特征向量。
- linalg.eigh(a[, UPLO])	返回Hermitian或对称矩阵的特征值和特征向量。
- linalg.eigvals(a)	计算一般矩阵的特征值。
- linalg.eigvalsh(a[, UPLO])	计算Hermitian或实对称矩阵的特征值。

## 规范和其他数字

- linalg.norm(x[, ord, axis, keepdims])	矩阵或矢量规范。
- linalg.cond(x[, p])	计算矩阵的条件数。
- linalg.det(a)	计算数组的行列式。
- linalg.matrix_rank(M[, tol, hermitian])	使用SVD方法返回阵列的矩阵等级
- linalg.slogdet(a)	Compute the sign and (natural) 数组行列式的对数。
- trace(a[, offset, axis1, axis2, dtype, out])	返回数组对角线的总和。

## 求解方程和反转矩阵

- linalg.solve(a, b)	求解线性矩阵方程或线性标量方程组。
- linalg.tensorsolve(a, b[, axes])	求解x的张量方程ax = b。
- linalg.lstsq(a, b[, rcond])	将最小二乘解返回到线性矩阵方程。
- linalg.inv(a)	计算矩阵的（乘法）逆。
- linalg.pinv(a[, rcond])	计算矩阵的（Moore-Penrose）伪逆。
- linalg.tensorinv(a[, ind])	计算N维数组的“逆”。

## 例外

- linalg.LinAlgError	由linalg函数引发的通用Python异常派生对象。

## 一次在几个矩阵上的线性代数

*版本1.8.0的新特性*.

如果将多个矩阵堆叠到同一个数组中，则上面列出的几个线性代数例程能够一次计算多个矩阵的结果。

这在文档中通过输入参数规范表示，例如``a:(..., M, M)array_like``。 这意味着如果给定一个输入数组``a.shape == (N, M,M)``，它被解释为N个矩阵的“堆栈”，每个矩阵的大小为M-by-M。 类似的规范适用于返回值，例如行列式具有``det : (...)``并且在这种情况下将返回一个形状为``det(a).shape == (N,)``的数组。 这推广到高维数组上的线性代数运算：多维数组的最后1或2维被解释为向量或矩阵，适用于每个操作。