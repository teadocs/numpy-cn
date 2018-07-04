# 线性代数

## Matrix and vector products¶

- dot(a, b[, out])	Dot product of two arrays.
linalg.multi_dot(arrays)	Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.
- vdot(a, b)	Return the dot product of two vectors.
- inner(a, b)	Inner product of two arrays.
- outer(a, b[, out])	Compute the outer product of two vectors.
- matmul(a, b[, out])	Matrix product of two arrays.
- tensordot(a, b[, axes])	Compute tensor dot product along specified axes for arrays >= 1-D.
- einsum(subscripts, *operands[, out, dtype, …])	Evaluates the Einstein summation convention on the operands.
- einsum_path(subscripts, *operands[, optimize])	Evaluates the lowest cost contraction order for an einsum expression by considering the creation of intermediate arrays.
- linalg.matrix_power(M, n)	Raise a square matrix to the (integer) power n.
- kron(a, b)	Kronecker product of two arrays.

## Decompositions

- linalg.cholesky(a)	Cholesky decomposition.
- linalg.qr(a[, mode])	Compute the qr factorization of a matrix.
- linalg.svd(a[, full_matrices, compute_uv])	Singular Value Decomposition.

## Matrix eigenvalues

- linalg.eig(a)	Compute the eigenvalues and right eigenvectors of a square array.
- linalg.eigh(a[, UPLO])	Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
- linalg.eigvals(a)	Compute the eigenvalues of a general matrix.
- linalg.eigvalsh(a[, UPLO])	Compute the eigenvalues of a Hermitian or real symmetric matrix.

## Norms and other numbers

- linalg.norm(x[, ord, axis, keepdims])	Matrix or vector norm.
- linalg.cond(x[, p])	Compute the condition number of a matrix.
- linalg.det(a)	Compute the determinant of an array.
- linalg.matrix_rank(M[, tol, hermitian])	Return matrix rank of array using SVD method
- linalg.slogdet(a)	Compute the sign and (natural) logarithm of the determinant of an array.
- trace(a[, offset, axis1, axis2, dtype, out])	Return the sum along diagonals of the array.

## Solving equations and inverting matrices

- linalg.solve(a, b)	Solve a linear matrix equation, or system of linear scalar equations.
- linalg.tensorsolve(a, b[, axes])	Solve the tensor equation a x = b for x.
- linalg.lstsq(a, b[, rcond])	Return the least-squares solution to a linear matrix equation.
- linalg.inv(a)	Compute the (multiplicative) inverse of a matrix.
- linalg.pinv(a[, rcond])	Compute the (Moore-Penrose) pseudo-inverse of a matrix.
- linalg.tensorinv(a[, ind])	Compute the ‘inverse’ of an N-dimensional array.

## Exceptions

- linalg.LinAlgError	Generic Python-exception-derived object raised by linalg functions.

## Linear algebra on several matrices at once

*New in version 1.8.0*.

Several of the linear algebra routines listed above are able to compute results for several matrices at once, if they are stacked into the same array.

This is indicated in the documentation via input parameter specifications such as ``a : (..., M, M) array_like``. This means that if for instance given an input array ``a.shape == (N, M, M)``, it is interpreted as a “stack” of N matrices, each of size M-by-M. Similar specification applies to return values, for instance the determinant has ``det : (...)`` and will in this case return an array of shape ``det(a).shape == (N,)``. This generalizes to linear algebra operations on higher-dimensional arrays: the last 1 or 2 dimensions of a multidimensional array are interpreted as vectors or matrices, as appropriate for each operation.