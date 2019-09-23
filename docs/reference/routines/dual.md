# 可选的Scipy加速支持（``numpy.dual``）

可能由Scipy加速的函数的别名。

可以将[Scipy](https://www.scipy.org/)构建为使用FFT、线性代数和特殊函数的加速或其他改进的库。
这个模块允许开发人员在scipy可用时透明地支持这些加速功能，
但仍然支持只安装了NumPy的用户。

## 线性代数

方法 | 描述
---|---
[cholesky](generated/numpy.linalg.cholesky.html#numpy.linalg.cholesky)(a) | 胆固醇分解。
[det](generated/numpy.linalg.det.html#numpy.linalg.det)(a) | 计算数组的行列式。
[eig](generated/numpy.linalg.eig.html#numpy.linalg.eig)(a) | 计算方阵的特征值和右特征向量。
[eigh](generated/numpy.linalg.eigh.html#numpy.linalg.eigh)(a [, UPLO]) | 返回复数Hermitian（共轭对称）或实对称矩阵的特征值和特征向量。
[eigvals](generated/numpy.linalg.eigvals.html#numpy.linalg.eigvals)(a) | 计算通用矩阵的特征值。
[eigvalsh](generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh)(a [, UPLO]) | 计算复杂的Hermitian或实对称矩阵的特征值。
[inv](generated/numpy.linalg.inv.html#numpy.linalg.inv)(a) | 计算矩阵的（乘法）逆。
[lstsq](generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq)(a, b[, rcond]) | 将最小二乘解返回线性矩阵方程。
[norm](generated/numpy.linalg.norm.html#numpy.linalg.norm)(x [, ord，axis, keepdims]) | 矩阵或向量范数。
[pinv](generated/numpy.linalg.pinv.html#numpy.linalg.pinv)(a [, rcond, hermitian]) | 计算矩阵的（Moore-Penrose）伪逆。
[solve](generated/numpy.linalg.solve.html#numpy.linalg.solve)(a, b) | 求解线性矩阵方程或线性标量方程组。
[svd](generated/numpy.linalg.svd.html#numpy.linalg.svd)(a [, full_matrices, compute_uv, hermitian]) | 奇异值分解。

## FFT 

方法 | 描述
---|---
[fft](generated/numpy.fft.fft.html#numpy.fft.fft)(a[, n, axis, norm]) | 计算一维离散傅立叶变换。
[fft2](generated/numpy.fft.fft2.html#numpy.fft.fft2)(a[, s, axes, norm]) | 计算二维离散傅里叶变换
[fftn](generated/numpy.fft.fftn.html#numpy.fft.fftn)(a[, s, axes, norm]) | 计算N维离散傅里叶变换。
[ifft](generated/numpy.fft.ifft.html#numpy.fft.ifft)(a[, n, axis, norm]) | 计算一维逆离散傅立叶逆变换。
[ifft2](generated/numpy.fft.ifft2.html#numpy.fft.ifft2)(a[, s, axes, norm]) | 计算二维逆离散傅里叶逆变换。
[ifftn](generated/numpy.fft.ifftn.html#numpy.fft.ifftn)(a[, s, axes, norm]) | 计算N维离散傅里叶逆变换。

## 其他

方法 | 描述
---|---
[i0](generated/numpy.i0.html#numpy.i0)(X) | 第一种修改的Bessel函数，阶数为0。
