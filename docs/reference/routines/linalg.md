# 线性代数（``numpy.linalg``）

NumPy线性代数函数依赖于BLAS和LAPACK来提供标准线性代数算法的高效低级实现。
这些库可以由NumPy本身使用其参考实现子集的C版本提供，
但如果可能，最好是利用专用处理器功能的高度优化的库。
这样的库的例子是[OpenBLAS](https://www.openblas.net/)、MKL(TM)和ATLAS。因为这些库是多线程和处理器相关的，
所以可能需要环境变量和外部包（如[threadpoolctl](https://github.com/joblib/threadpoolctl)）来控制线程数量或指定处理器体系结构。

## 矩阵和向量积

方法 | 描述
---|---
[dot](https://numpy.org/devdocs/reference/generated/numpy.dot.html#numpy.dot)(a, b[, out]) | 两个数组的点积。
[linalg.multi_dot](https://numpy.org/devdocs/reference/generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot)(arrays) | 在单个函数调用中计算两个或更多数组的点积，同时自动选择最快的求值顺序。
[vdot](https://numpy.org/devdocs/reference/generated/numpy.vdot.html#numpy.vdot)(a, b) | 返回两个向量的点积。
[inner](https://numpy.org/devdocs/reference/generated/numpy.inner.html#numpy.inner)(a, b) | 两个数组的内积。
[outer](https://numpy.org/devdocs/reference/generated/numpy.outer.html#numpy.outer)(a, b[, out]) | 计算两个向量的外积。
[matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul)(x1, x2, /[, out, casting, order, …]) | 两个数组的矩阵乘积。
[tensordot](https://numpy.org/devdocs/reference/generated/numpy.tensordot.html#numpy.tensordot)(a, b[, axes]) | 沿指定轴计算张量点积。
[einsum](https://numpy.org/devdocs/reference/generated/numpy.einsum.html#numpy.einsum)(subscripts, *operands[, out, dtype, …]) | 计算操作数上的爱因斯坦求和约定。
[einsum_path](https://numpy.org/devdocs/reference/generated/numpy.einsum_path.html#numpy.einsum_path)(subscripts, *operands[, optimize]) | 通过考虑中间数组的创建，计算einsum表达式的最低成本压缩顺序。
[linalg.matrix_power](https://numpy.org/devdocs/reference/generated/numpy.linalg.matrix_power.html#numpy.linalg.matrix_power)(a, n) | 将方阵提升为(整数)n次方。
[kron](https://numpy.org/devdocs/reference/generated/numpy.kron.html#numpy.kron)(a, b) | 两个数组的Kronecker乘积。

## 分解

方法 | 描述
---|---
[linalg.cholesky](https://numpy.org/devdocs/reference/generated/numpy.linalg.cholesky.html#numpy.linalg.cholesky)(a) | Cholesky分解
[linalg.qr](https://numpy.org/devdocs/reference/generated/numpy.linalg.qr.html#numpy.linalg.qr)(a[, mode]) | 计算矩阵的QR分解。
[linalg.svd](https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd)(a[, full_matrices, compute_uv, …]) | 奇异值分解

## 矩阵特征值

方法 | 描述
---|---
[linalg.eig](https://numpy.org/devdocs/reference/generated/numpy.linalg.eig.html#numpy.linalg.eig)(a) | 计算方阵的特征值和右特征向量。
[linalg.eigh](https://numpy.org/devdocs/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh)(a[, UPLO]) | 返回复数Hermitian（共轭对称）或实对称矩阵的特征值和特征向量。
[linalg.eigvals](https://numpy.org/devdocs/reference/generated/numpy.linalg.eigvals.html#numpy.linalg.eigvals)(a) | 计算通用矩阵的特征值。
[linalg.eigvalsh](https://numpy.org/devdocs/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh)(a[, UPLO]) | 计算复杂的Hermitian或实对称矩阵的特征值。

## 范数和其他数字

方法 | 描述
---|---
[linalg.norm](https://numpy.org/devdocs/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm)(x[, ord, axis, keepdims]) | 矩阵或向量范数。
[linalg.cond](https://numpy.org/devdocs/reference/generated/numpy.linalg.cond.html#numpy.linalg.cond)(x[, p]) | 计算矩阵的条件数。
[linalg.det](https://numpy.org/devdocs/reference/generated/numpy.linalg.det.html#numpy.linalg.det)(a) | 计算数组的行列式。
[linalg.matrix_rank](https://numpy.org/devdocs/reference/generated/numpy.linalg.matrix_rank.html#numpy.linalg.matrix_rank)(M[, tol, hermitian]) | 使用SVD方法返回数组的矩阵的rank
[linalg.slogdet](https://numpy.org/devdocs/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet)(a) | 计算数组行列式的符号和（自然）对数。
[trace](https://numpy.org/devdocs/reference/generated/numpy.trace.html#numpy.trace)(a[, offset, axis1, axis2, dtype, out]) | 返回数组对角线的和。

## 解方程和逆矩阵

方法 | 描述
---|---
[linalg.solve](https://numpy.org/devdocs/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve)(a, b) | 求解线性矩阵方程或线性标量方程组。
[linalg.tensorsolve](https://numpy.org/devdocs/reference/generated/numpy.linalg.tensorsolve.html#numpy.linalg.tensorsolve)(a, b[, axes]) | 对x求解张量方程a x = b。
[linalg.lstsq](https://numpy.org/devdocs/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq)(a, b[, rcond]) | 返回线性矩阵方程的最小二乘解。
[linalg.inv](https://numpy.org/devdocs/reference/generated/numpy.linalg.inv.html#numpy.linalg.inv)(a) | 计算矩阵的（乘法）逆。
[linalg.pinv](https://numpy.org/devdocs/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv)(a[, rcond, hermitian]) | 计算矩阵的（Moore-Penrose）伪逆。
[linalg.tensorinv](https://numpy.org/devdocs/reference/generated/numpy.linalg.tensorinv.html#numpy.linalg.tensorinv)(a[, ind]) | 计算N维数组的“逆”。

## 例外

方法 | 描述
---|---
[linalg.LinAlgError](https://numpy.org/devdocs/reference/generated/numpy.linalg.LinAlgError.html#numpy.linalg.LinAlgError) | 泛型Python-linalg函数引发的异常派生对象。

## 一次在多个矩阵上的线性代数

*1.8.0版中的新功能* 

上面列出的几个线性代数例程能够一次计算几个矩阵的结果，如果它们堆叠在同一数组中的话。

这在文档中通过输入参数规范（如 ``a : (..., M, M) array_like`` ）表示。
这意味着，例如，如果给定输入数组 ``a.shape == (N, M, M)`` ，则将其解释为N个矩阵的“堆栈”，
每个矩阵的大小为M×M。类似的规范也适用于返回值，
例如行列式 ``det : (...)`` 。并且在这种情况下将返回形状 ``det(a).shape == (N,)`` 的数组。
这推广到对高维数组的线性代数操作：多维数组的最后1或2维被解释为向量或矩阵，视每个操作而定。
