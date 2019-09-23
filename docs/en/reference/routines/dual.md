# Optionally Scipy-accelerated routines (``numpy.dual``)

Aliases for functions which may be accelerated by Scipy.

[Scipy](https://www.scipy.org) can be built to use accelerated or otherwise improved libraries
for FFTs, linear algebra, and special functions. This module allows
developers to transparently support these accelerated functions when
scipy is available but still support users who have only installed
NumPy.

## Linear algebra

method | description
---|---
[cholesky](https://numpy.org/devdocs/reference/generated/numpy.linalg.cholesky.html#numpy.linalg.cholesky)(a) | Cholesky decomposition.
[det](https://numpy.org/devdocs/reference/generated/numpy.linalg.det.html#numpy.linalg.det)(a) | Compute the determinant of an array.
[eig](https://numpy.org/devdocs/reference/generated/numpy.linalg.eig.html#numpy.linalg.eig)(a) | Compute the eigenvalues and right eigenvectors of a square array.
[eigh](https://numpy.org/devdocs/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh)(a[, UPLO]) | Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
[eigvals](https://numpy.org/devdocs/reference/generated/numpy.linalg.eigvals.html#numpy.linalg.eigvals)(a) | Compute the eigenvalues of a general matrix.
[eigvalsh](https://numpy.org/devdocs/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh)(a[, UPLO]) | Compute the eigenvalues of a complex Hermitian or real symmetric matrix.
[inv](https://numpy.org/devdocs/reference/generated/numpy.linalg.inv.html#numpy.linalg.inv)(a) | Compute the (multiplicative) inverse of a matrix.
[lstsq](https://numpy.org/devdocs/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq)(a, b[, rcond]) | Return the least-squares solution to a linear matrix equation.
[norm](https://numpy.org/devdocs/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm)(x[, ord, axis, keepdims]) | Matrix or vector norm.
[pinv](https://numpy.org/devdocs/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv)(a[, rcond, hermitian]) | Compute the (Moore-Penrose) pseudo-inverse of a matrix.
[solve](https://numpy.org/devdocs/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve)(a, b) | Solve a linear matrix equation, or system of linear scalar equations.
[svd](https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd)(a[, full_matrices, compute_uv, hermitian]) | Singular Value Decomposition.

## FFT

method | description
---|---
[fft](https://numpy.org/devdocs/reference/generated/numpy.fft.fft.html#numpy.fft.fft)(a[, n, axis, norm]) | Compute the one-dimensional discrete Fourier Transform.
[fft2](https://numpy.org/devdocs/reference/generated/numpy.fft.fft2.html#numpy.fft.fft2)(a[, s, axes, norm]) | Compute the 2-dimensional discrete Fourier Transform
[fftn](https://numpy.org/devdocs/reference/generated/numpy.fft.fftn.html#numpy.fft.fftn)(a[, s, axes, norm]) | Compute the N-dimensional discrete Fourier Transform.
[ifft](https://numpy.org/devdocs/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft)(a[, n, axis, norm]) | Compute the one-dimensional inverse discrete Fourier Transform.
[ifft2](https://numpy.org/devdocs/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2)(a[, s, axes, norm]) | Compute the 2-dimensional inverse discrete Fourier Transform.
[ifftn](https://numpy.org/devdocs/reference/generated/numpy.fft.ifftn.html#numpy.fft.ifftn)(a[, s, axes, norm]) | Compute the N-dimensional inverse discrete Fourier Transform.

## Other

method | description
---|---
[i0](https://numpy.org/devdocs/reference/generated/numpy.i0.html#numpy.i0)(x) | Modified Bessel function of the first kind, order 0.