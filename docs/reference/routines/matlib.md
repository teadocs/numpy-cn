# Matrix library (``numpy.matlib``)

This module contains all functions in the [``numpy``](index.html#module-numpy) namespace, with
the following replacement functions that return [``matrices``](generated/numpy.matrix.html#numpy.matrix) instead of [``ndarrays``](generated/numpy.ndarray.html#numpy.ndarray).

Functions that are also in the numpy namespace and return matrices

method | description
---|---
[mat](generated/numpy.mat.html#numpy.mat)(data[, dtype]) | Interpret the input as a [matrix](generated/numpy.matrix.html#numpy.matrix).
[matrix](generated/numpy.matrix.html#numpy.matrix)(data[, dtype, copy]) | Note: It is no longer recommended to use this class, even for linear
[asmatrix](generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | Interpret the input as a matrix.
[bmat](generated/numpy.bmat.html#numpy.bmat)(obj[, ldict, gdict]) | Build a matrix object from a string, nested sequence, or array.

Replacement functions in [``matlib``](#module-numpy.matlib)

method | description
---|---
[empty](generated/numpy.matlib.empty.html#numpy.matlib.empty)(shape[, dtype, order]) | Return a new matrix of given shape and type, without initializing entries.
[zeros](generated/numpy.matlib.zeros.html#numpy.matlib.zeros)(shape[, dtype, order]) | Return a matrix of given shape and type, filled with zeros.
[ones](generated/numpy.matlib.ones.html#numpy.matlib.ones)(shape[, dtype, order]) | Matrix of ones.
[eye](generated/numpy.matlib.eye.html#numpy.matlib.eye)(n[, M, k, dtype, order]) | Return a matrix with ones on the diagonal and zeros elsewhere.
[identity](generated/numpy.matlib.identity.html#numpy.matlib.identity)(n[, dtype]) | Returns the square identity matrix of given size.
[repmat](generated/numpy.matlib.repmat.html#numpy.matlib.repmat)(a, m, n) | Repeat a 0-D to 2-D array or matrix MxN times.
[rand](generated/numpy.matlib.rand.html#numpy.matlib.rand)(\*args) | Return a matrix of random values with given shape.
[randn](generated/numpy.matlib.randn.html#numpy.matlib.randn)(\*args) | Return a random matrix with data from the “standard normal” distribution.
