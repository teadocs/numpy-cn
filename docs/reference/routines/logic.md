# Logic functions

## Truth value testing

method | description
---|---
[all](https://numpy.org/devdocs/reference/generated/numpy.all.html#numpy.all)(a[, axis, out, keepdims]) | Test whether all array elements along a given axis evaluate to True.
[any](https://numpy.org/devdocs/reference/generated/numpy.any.html#numpy.any)(a[, axis, out, keepdims]) | Test whether any array element along a given axis evaluates to True.

## Array contents

method | description
---|---
[isfinite](https://numpy.org/devdocs/reference/generated/numpy.isfinite.html#numpy.isfinite)(x, /[, out, where, casting, order, …]) | Test element-wise for finiteness (not infinity or not Not a Number).
[isinf](https://numpy.org/devdocs/reference/generated/numpy.isinf.html#numpy.isinf)(x, /[, out, where, casting, order, …]) | Test element-wise for positive or negative infinity.
[isnan](https://numpy.org/devdocs/reference/generated/numpy.isnan.html#numpy.isnan)(x, /[, out, where, casting, order, …]) | Test element-wise for NaN and return result as a boolean array.
[isnat](https://numpy.org/devdocs/reference/generated/numpy.isnat.html#numpy.isnat)(x, /[, out, where, casting, order, …]) | Test element-wise for NaT (not a time) and return result as a boolean array.
[isneginf](https://numpy.org/devdocs/reference/generated/numpy.isneginf.html#numpy.isneginf)(x[, out]) | Test element-wise for negative infinity, return result as bool array.
[isposinf](https://numpy.org/devdocs/reference/generated/numpy.isposinf.html#numpy.isposinf)(x[, out]) | Test element-wise for positive infinity, return result as bool array.

## Array type testing

method | description
---|---
[iscomplex](https://numpy.org/devdocs/reference/generated/numpy.iscomplex.html#numpy.iscomplex)(x) | Returns a bool array, where True if input element is complex.
[iscomplexobj](https://numpy.org/devdocs/reference/generated/numpy.iscomplexobj.html#numpy.iscomplexobj)(x) | Check for a complex type or an array of complex numbers.
[isfortran](https://numpy.org/devdocs/reference/generated/numpy.isfortran.html#numpy.isfortran)(a) | Check if the array is Fortran contiguous but not C contiguous.
[isreal](https://numpy.org/devdocs/reference/generated/numpy.isreal.html#numpy.isreal)(x) | Returns a bool array, where True if input element is real.
[isrealobj](https://numpy.org/devdocs/reference/generated/numpy.isrealobj.html#numpy.isrealobj)(x) | Return True if x is a not complex type or an array of complex numbers.
[isscalar](https://numpy.org/devdocs/reference/generated/numpy.isscalar.html#numpy.isscalar)(num) | Returns True if the type of num is a scalar type.

## Logical operations

method | description
---|---
[logical_and](https://numpy.org/devdocs/reference/generated/numpy.logical_and.html#numpy.logical_and)(x1, x2, /[, out, where, …]) | Compute the truth value of x1 AND x2 element-wise.
[logical_or](https://numpy.org/devdocs/reference/generated/numpy.logical_or.html#numpy.logical_or)(x1, x2, /[, out, where, casting, …]) | Compute the truth value of x1 OR x2 element-wise.
[logical_not](https://numpy.org/devdocs/reference/generated/numpy.logical_not.html#numpy.logical_not)(x, /[, out, where, casting, …]) | Compute the truth value of NOT x element-wise.
[logical_xor](https://numpy.org/devdocs/reference/generated/numpy.logical_xor.html#numpy.logical_xor)(x1, x2, /[, out, where, …]) | Compute the truth value of x1 XOR x2, element-wise.

## Comparison

method | description
---|---
[allclose](https://numpy.org/devdocs/reference/generated/numpy.allclose.html#numpy.allclose)(a, b[, rtol, atol, equal_nan]) | Returns True if two arrays are element-wise equal within a tolerance.
[isclose](https://numpy.org/devdocs/reference/generated/numpy.isclose.html#numpy.isclose)(a, b[, rtol, atol, equal_nan]) | Returns a boolean array where two arrays are element-wise equal within a tolerance.
[array_equal](https://numpy.org/devdocs/reference/generated/numpy.array_equal.html#numpy.array_equal)(a1, a2) | True if two arrays have the same shape and elements, False otherwise.
[array_equiv](https://numpy.org/devdocs/reference/generated/numpy.array_equiv.html#numpy.array_equiv)(a1, a2) | Returns True if input arrays are shape consistent and all elements equal.
[greater](https://numpy.org/devdocs/reference/generated/numpy.greater.html#numpy.greater)(x1, x2, /[, out, where, casting, …]) | Return the truth value of (x1 > x2) element-wise.
[greater_equal](https://numpy.org/devdocs/reference/generated/numpy.greater_equal.html#numpy.greater_equal)(x1, x2, /[, out, where, …]) | Return the truth value of (x1 >= x2) element-wise.
[less](https://numpy.org/devdocs/reference/generated/numpy.less.html#numpy.less)(x1, x2, /[, out, where, casting, …]) | Return the truth value of (x1 < x2) element-wise.
[less_equal](https://numpy.org/devdocs/reference/generated/numpy.less_equal.html#numpy.less_equal)(x1, x2, /[, out, where, casting, …]) | Return the truth value of (x1 =< x2) element-wise.
[equal](https://numpy.org/devdocs/reference/generated/numpy.equal.html#numpy.equal)(x1, x2, /[, out, where, casting, …]) | Return (x1 == x2) element-wise.
[not_equal](https://numpy.org/devdocs/reference/generated/numpy.not_equal.html#numpy.not_equal)(x1, x2, /[, out, where, casting, …]) | Return (x1 != x2) element-wise.