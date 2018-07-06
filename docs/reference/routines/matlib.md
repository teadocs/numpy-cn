# 矩阵库

This module contains all functions in the ``numpy`` namespace, with the following replacement functions that return ``matrices`` instead of ``ndarrays``.

Functions that are also in the numpy namespace and return matrices

- mat(data[, dtype])	Interpret the input as a matrix.
- matrix(data[, dtype, copy])	Returns a matrix from an array-like object, or from a string of data.
- asmatrix(data[, dtype])	Interpret the input as a matrix.
- bmat(obj[, ldict, gdict])	Build a matrix object from a string, nested sequence, or array.

Replacement functions in ``matlib``

- empty(shape[, dtype, order])	Return a new matrix of given shape and type, without initializing entries.
- zeros(shape[, dtype, order])	Return a matrix of given shape and type, filled with zeros.
- ones(shape[, dtype, order])	Matrix of ones.
- eye(n[, M, k, dtype, order])	Return a matrix with ones on the diagonal and zeros elsewhere.
- identity(n[, dtype])	Returns the square identity matrix of given size.
- repmat(a, m, n)	Repeat a 0-D to 2-D array or matrix MxN times.
- rand(*args)	Return a matrix of random values with given shape.
- randn(*args)	Return a random matrix with data from the “standard normal” distribution.
