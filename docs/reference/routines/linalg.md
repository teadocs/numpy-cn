# Linear algebra (``numpy.linalg``)

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient
low level implementations of standard linear algebra algorithms. Those
libraries may be provided by NumPy itself using C versions of a subset of their
reference implementations but, when possible, highly optimized libraries that
take advantage of specialized processor functionality are preferred. Examples
of such libraries are [OpenBLAS](https://www.openblas.net/), MKL (TM), and ATLAS. Because those libraries
are multithreaded and processor dependent, environmental variables and external
packages such as [threadpoolctl](https://github.com/joblib/threadpoolctl) may be needed to control the number of threads
or specify the processor architecture.

## Matrix and vector products

method | description
---|---
[dot](generated/numpy.dot.html#numpy.dot)(a, b[, out]) | Dot product of two arrays.
[linalg.multi_dot](generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot)(arrays) | Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.
[vdot](generated/numpy.vdot.html#numpy.vdot)(a, b) | Return the dot product of two vectors.
[inner](generated/numpy.inner.html#numpy.inner)(a, b) | Inner product of two arrays.
[outer](generated/numpy.outer.html#numpy.outer)(a, b[, out]) | Compute the outer product of two vectors.
[matmul](generated/numpy.matmul.html#numpy.matmul)(x1, x2, /[, out, casting, order, …]) | Matrix product of two arrays.
[tensordot](generated/numpy.tensordot.html#numpy.tensordot)(a, b[, axes]) | Compute tensor dot product along specified axes.
[einsum](generated/numpy.einsum.html#numpy.einsum)(subscripts, *operands[, out, dtype, …]) | Evaluates the Einstein summation convention on the operands.
[einsum_path](generated/numpy.einsum_path.html#numpy.einsum_path)(subscripts, *operands[, optimize]) | Evaluates the lowest cost contraction order for an einsum expression by considering the creation of intermediate arrays.
[linalg.matrix_power](generated/numpy.linalg.matrix_power.html#numpy.linalg.matrix_power)(a, n) | Raise a square matrix to the (integer) power n.
[kron](generated/numpy.kron.html#numpy.kron)(a, b) | Kronecker product of two arrays.

## Decompositions

method | description
---|---
[linalg.cholesky](generated/numpy.linalg.cholesky.html#numpy.linalg.cholesky)(a) | Cholesky decomposition.
[linalg.qr](generated/numpy.linalg.qr.html#numpy.linalg.qr)(a[, mode]) | Compute the qr factorization of a matrix.
[linalg.svd](generated/numpy.linalg.svd.html#numpy.linalg.svd)(a[, full_matrices, compute_uv, …]) | Singular Value Decomposition.

## Matrix eigenvalues

method | description
---|---
[linalg.eig](generated/numpy.linalg.eig.html#numpy.linalg.eig)(a) | Compute the eigenvalues and right eigenvectors of a square array.
[linalg.eigh](generated/numpy.linalg.eigh.html#numpy.linalg.eigh)(a[, UPLO]) | Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
[linalg.eigvals](generated/numpy.linalg.eigvals.html#numpy.linalg.eigvals)(a) | Compute the eigenvalues of a general matrix.
[linalg.eigvalsh](generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh)(a[, UPLO]) | Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

## Norms and other numbers

method | description
---|---
[linalg.norm](generated/numpy.linalg.norm.html#numpy.linalg.norm)(x[, ord, axis, keepdims]) | Matrix or vector norm.
[linalg.cond](generated/numpy.linalg.cond.html#numpy.linalg.cond)(x[, p]) | Compute the condition number of a matrix.
[linalg.det](generated/numpy.linalg.det.html#numpy.linalg.det)(a) | Compute the determinant of an array.
[linalg.matrix_rank](generated/numpy.linalg.matrix_rank.html#numpy.linalg.matrix_rank)(M[, tol, hermitian]) | Return matrix rank of array using SVD method
[linalg.slogdet](generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet)(a) | Compute the sign and (natural) logarithm of the determinant of an array.
[trace](generated/numpy.trace.html#numpy.trace)(a[, offset, axis1, axis2, dtype, out]) | Return the sum along diagonals of the array.

## Solving equations and inverting matrices

method | description
---|---
[linalg.solve](generated/numpy.linalg.solve.html#numpy.linalg.solve)(a, b) | Solve a linear matrix equation, or system of linear scalar equations.
[linalg.tensorsolve](generated/numpy.linalg.tensorsolve.html#numpy.linalg.tensorsolve)(a, b[, axes]) | Solve the tensor equation a x = b for x.
[linalg.lstsq](generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq)(a, b[, rcond]) | Return the least-squares solution to a linear matrix equation.
[linalg.inv](generated/numpy.linalg.inv.html#numpy.linalg.inv)(a) | Compute the (multiplicative) inverse of a matrix.
[linalg.pinv](generated/numpy.linalg.pinv.html#numpy.linalg.pinv)(a[, rcond, hermitian]) | Compute the (Moore-Penrose) pseudo-inverse of a matrix.
[linalg.tensorinv](generated/numpy.linalg.tensorinv.html#numpy.linalg.tensorinv)(a[, ind]) | Compute the ‘inverse’ of an N-dimensional array.

## Exceptions

method | description
---|---
[linalg.LinAlgError](generated/numpy.linalg.LinAlgError.html#numpy.linalg.LinAlgError) | Generic Python-exception-derived object raised by linalg functions.

## Linear algebra on several matrices at once

*New in version 1.8.0.* 

Several of the linear algebra routines listed above are able to
compute results for several matrices at once, if they are stacked into
the same array.

This is indicated in the documentation via input parameter
specifications such as ``a : (..., M, M) array_like``. This means that
if for instance given an input array ``a.shape == (N, M, M)``, it is
interpreted as a “stack” of N matrices, each of size M-by-M. Similar
specification applies to return values, for instance the determinant
has ``det : (...)`` and will in this case return an array of shape
``det(a).shape == (N,)``. This generalizes to linear algebra
operations on higher-dimensional arrays: the last 1 or 2 dimensions of
a multidimensional array are interpreted as vectors or matrices, as
appropriate for each operation.