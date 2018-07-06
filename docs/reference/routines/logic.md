# 逻辑函数

## Truth value testing

- all(a[, axis, out, keepdims])	Test whether all array elements along a given axis evaluate to True.
- any(a[, axis, out, keepdims])	Test whether any array element along a given axis evaluates to True.

## Array contents

- isfinite(x, /[, out, where, casting, order, …])	Test element-wise for finiteness (not infinity or not Not a Number).
- isinf(x, /[, out, where, casting, order, …])	Test element-wise for positive or negative infinity.
- isnan(x, /[, out, where, casting, order, …])	Test element-wise for NaN and return result as a boolean array.
- isnat(x, /[, out, where, casting, order, …])	Test element-wise for NaT (not a time) and return result as a boolean array.
- isneginf(x[, out])	Test element-wise for negative infinity, return result as bool array.
- isposinf(x[, out])	Test element-wise for positive infinity, return result as bool array.

## Array type testing

- iscomplex(x)	Returns a bool array, where True if input element is complex.
- iscomplexobj(x)	Check for a complex type or an array of complex numbers.
- isfortran(a)	Returns True if the array is Fortran contiguous but not C contiguous.
- isreal(x)	Returns a bool array, where True if input element is real.
- isrealobj(x)	Return True if x is a not complex type or an array of complex numbers.
- isscalar(num)	Returns True if the type of num is a scalar type.

## Logical operations

- logical_and(x1, x2, /[, out, where, …])	Compute the truth value of x1 AND x2 element-wise.
- logical_or(x1, x2, /[, out, where, casting, …])	Compute the truth value of x1 OR x2 element-wise.
- logical_not(x, /[, out, where, casting, …])	Compute the truth value of NOT x element-wise.
- logical_xor(x1, x2, /[, out, where, …])	Compute the truth value of x1 XOR x2, element-wise.

## Comparison

- allclose(a, b[, rtol, atol, equal_nan])	Returns True if two arrays are element-wise equal within a tolerance.
- isclose(a, b[, rtol, atol, equal_nan])	Returns a boolean array where two arrays are element-wise equal within a tolerance.
- array_equal(a1, a2)	True if two arrays have the same shape and elements, False otherwise.
- array_equiv(a1, a2)	Returns True if input arrays are shape consistent and all elements equal.
- greater(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 > x2) element-wise.
- greater_equal(x1, x2, /[, out, where, …])	Return the truth value of (x1 >= x2) element-wise.
- less(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 < x2) element-wise.
- less_equal(x1, x2, /[, out, where, casting, …])	Return the truth value of (x1 =< x2) element-wise.
- equal(x1, x2, /[, out, where, casting, …])	Return (x1 == x2) element-wise.
- not_equal(x1, x2, /[, out, where, casting, …])	Return (x1 != x2) element-wise.