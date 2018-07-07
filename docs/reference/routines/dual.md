# 可选的Scipy加速支持

Aliases for functions which may be accelerated by Scipy.

[Scipy](http://www.scipy.org/) can be built to use accelerated or otherwise improved libraries for FFTs, linear algebra, and special functions. This module allows developers to transparently support these accelerated functions when scipy is available but still support users who have only installed NumPy.

## Linear algebra

- cholesky(a)	Cholesky decomposition.
- det(a)	Compute the determinant of an array.
- eig(a)	Compute the eigenvalues and right eigenvectors of a square array.
- eigh(a[, UPLO])	Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
- eigvals(a)	Compute the eigenvalues of a general matrix.
- eigvalsh(a[, UPLO])	Compute the eigenvalues of a Hermitian or real symmetric matrix.
- inv(a)	Compute the (multiplicative) inverse of a matrix.
- lstsq(a, b[, rcond])	Return the least-squares solution to a linear matrix equation.
- norm(x[, ord, axis, keepdims])	Matrix or vector norm.
- pinv(a[, rcond])	Compute the (Moore-Penrose) pseudo-inverse of a matrix.
- solve(a, b)	Solve a linear matrix equation, or system of linear scalar equations.
- svd(a[, full_matrices, compute_uv])	Singular Value Decomposition.

## FFT

- fft(a[, n, axis, norm])	Compute the one-dimensional discrete Fourier Transform.
- fft2(a[, s, axes, norm])	Compute the 2-dimensional discrete Fourier Transform
- fftn(a[, s, axes, norm])	Compute the N-dimensional discrete Fourier Transform.
- ifft(a[, n, axis, norm])	Compute the one-dimensional inverse discrete Fourier Transform.
- ifft2(a[, s, axes, norm])	Compute the 2-dimensional inverse discrete Fourier Transform.
- ifftn(a[, s, axes, norm])	Compute the N-dimensional inverse discrete Fourier Transform.

## Other

- i0(x)	Modified Bessel function of the first kind, order 0.