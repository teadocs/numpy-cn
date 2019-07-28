# 可选的Scipy加速支持

可以通过Scipy加速的函数的别名。

可以构建[Scipy](http://www.scipy.org/)以使用加速或其他改进的库来进行FFT，线性代数和特殊函数。 此模块允许开发人员在scipy可用时透明地支持这些加速功能，但仍支持仅安装了NumPy的用户

## 线性代数

- cholesky(a)	Cholesky分解。
- det(a)	计算数组的行列式。
- eig(a)	计算正方形阵列的特征值和右特征向量。
- eigh(a[, UPLO])	返回Hermitian或对称矩阵的特征值和特征向量。
- eigvals(a)	计算一般矩阵的特征值。
- eigvalsh(a[, UPLO])	计算Hermitian或实对称矩阵的特征值。
- inv(a)	计算矩阵的（乘法）逆。
- lstsq(a, b[, rcond]) 将最小二乘解返回到线性矩阵方程。
- norm(x[, ord, axis, keepdims])	矩阵或矢量规范。
- pinv(a[, rcond])	计算矩阵的（Moore-Penrose）伪逆。
- solve(a, b)	求解线性矩阵方程或线性标量方程组。
- svd(a[, full_matrices, compute_uv])	奇异值分解。

## FFT

- fft(a[, n, axis, norm])	计算一维离散傅立叶变换。
- fft2(a[, s, axes, norm])	计算二维离散傅立叶变换。
- fftn(a[, s, axes, norm])	计算N维离散傅立叶变换。
- ifft(a[, n, axis, norm])	计算一维离散傅里叶逆变换。
- ifft2(a[, s, axes, norm])	计算二维逆离散傅立叶变换。
- ifftn(a[, s, axes, norm])	计算N维逆离散傅立叶变换。

## Other

- i0(x) 修改了第一类贝塞尔函数，阶数为0。