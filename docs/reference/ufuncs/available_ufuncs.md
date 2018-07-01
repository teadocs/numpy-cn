# 可用的通用函数列表

There are currently more than 60 universal functions defined in ``numpy`` on one or more types, covering a wide variety of operations. Some of these ufuncs are called automatically on arrays when the relevant infix notation is used (e.g., ``add(a, b)`` is called internally when ``a + b`` is written and a or b is an ``ndarray``). Nevertheless, you may still want to use the ufunc call in order to use the optional output argument(s) to place the output(s) in an object (or objects) of your choice.

Recall that each ufunc operates element-by-element. Therefore, each scalar ufunc will be described as if acting on a set of scalar inputs to return a set of scalar outputs.

> **Note**
> The ufunc still returns its output(s) even if you use the optional output argument(s).

## Math operations

method | desc
---|---
add(x1, x2, /[, out, where, casting, order, …]) | Add arguments element-wise.
subtract(x1, x2, /[, out, where, casting, …]) | Subtract arguments, element-wise.
multiply(x1, x2, /[, out, where, casting, …]) | Multiply arguments element-wise.
divide(x1, x2, /[, out, where, casting, …]) | Returns a true division of the inputs, element-wise.
logaddexp(x1, x2, /[, out, where, casting, …]) | Logarithm of the sum of exponentiations of the inputs.
logaddexp2(x1, x2, /[, out, where, casting, …]) | Logarithm of the sum of exponentiations of the inputs in base-2.
true_divide(x1, x2, /[, out, where, …]) | Returns a true division of the inputs, element-wise.
floor_divide(x1, x2, /[, out, where, …]) | Return the largest integer smaller or equal to the division of the inputs.
negative(x, /[, out, where, casting, order, …]) | Numerical negative, element-wise.
positive(x, /[, out, where, casting, order, …]) | Numerical positive, element-wise.
power(x1, x2, /[, out, where, casting, …]) | First array elements raised to powers from second array, element-wise.
remainder(x1, x2, /[, out, where, casting, …]) | Return element-wise remainder of division.
mod(x1, x2, /[, out, where, casting, order, …]) | Return element-wise remainder of division.
fmod(x1, x2, /[, out, where, casting, …]) | Return the element-wise remainder of division.
divmod(x1, x2[, out1, out2], / [[, out, …]) | Return element-wise quotient and remainder simultaneously.
absolute(x, /[, out, where, casting, order, …]) | Calculate the absolute value element-wise.
fabs(x, /[, out, where, casting, order, …]) | Compute the absolute values element-wise.
rint(x, /[, out, where, casting, order, …]) | Round elements of the array to the nearest integer.
sign(x, /[, out, where, casting, order, …]) | Returns an element-wise indication of the sign of a number.
heaviside(x1, x2, /[, out, where, casting, …]) | Compute the Heaviside step function.
conj(x, /[, out, where, casting, order, …]) | Return the complex conjugate, element-wise.
exp(x, /[, out, where, casting, order, …]) | Calculate the exponential of all elements in the input array.
exp2(x, /[, out, where, casting, order, …]) | Calculate 2**p for all p in the input array.
log(x, /[, out, where, casting, order, …]) | Natural logarithm, element-wise.
log2(x, /[, out, where, casting, order, …]) | Base-2 logarithm of x.
log10(x, /[, out, where, casting, order, …]) | Return the base 10 logarithm of the input array, element-wise.
expm1(x, /[, out, where, casting, order, …]) | Calculate exp(x) - 1 for all elements in the array.
log1p(x, /[, out, where, casting, order, …]) | Return the natural logarithm of one plus the input array, element-wise.
sqrt(x, /[, out, where, casting, order, …]) | Return the positive square-root of an array, element-wise.
square(x, /[, out, where, casting, order, …]) | Return the element-wise square of the input.
cbrt(x, /[, out, where, casting, order, …]) | Return the cube-root of an array, element-wise.
reciprocal(x, /[, out, where, casting, …]) | Return the reciprocal of the argument, element-wise.

**Tip:**

The optional output arguments can be used to help you save memory for large calculations. If your arrays are large, complicated expressions can take longer than absolutely necessary due to the creation and (later) destruction of temporary calculation spaces. For example, the expression ``G = a * b + c`` is equivalent to ``t1 = A * B; G = T1 + C; del t1``. It will be more quickly executed as ``G = A * B; add(G, C, G)`` which is the same as ``G = A * B; G += C``.

## Trigonometric functions

All trigonometric functions use radians when an angle is called for. The ratio of degrees to radians is 180°/π.

method | desc
---|---
sin(x, /[, out, where, casting, order, …]) | Trigonometric sine, element-wise.
cos(x, /[, out, where, casting, order, …]) | Cosine element-wise.
tan(x, /[, out, where, casting, order, …]) | Compute tangent element-wise.
arcsin(x, /[, out, where, casting, order, …]) | Inverse sine, element-wise.
arccos(x, /[, out, where, casting, order, …]) | Trigonometric inverse cosine, element-wise.
arctan(x, /[, out, where, casting, order, …]) | Trigonometric inverse tangent, element-wise.
arctan2(x1, x2, /[, out, where, casting, …]) | Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
hypot(x1, x2, /[, out, where, casting, …]) | Given the “legs” of a right triangle, return its hypotenuse.
sinh(x, /[, out, where, casting, order, …]) | Hyperbolic sine, element-wise.
cosh(x, /[, out, where, casting, order, …]) | Hyperbolic cosine, element-wise.
tanh(x, /[, out, where, casting, order, …]) | Compute hyperbolic tangent element-wise.
arcsinh(x, /[, out, where, casting, order, …]) | Inverse hyperbolic sine element-wise.
arccosh(x, /[, out, where, casting, order, …]) | Inverse hyperbolic cosine, element-wise.
arctanh(x, /[, out, where, casting, order, …]) | Inverse hyperbolic tangent element-wise.
deg2rad(x, /[, out, where, casting, order, …]) | Convert angles from degrees to radians.
rad2deg(x, /[, out, where, casting, order, …]) | Convert angles from radians to degrees.

## Bit-twiddling functions

These function all require integer arguments and they manipulate the bit-pattern of those arguments.

method | desc
---|---
bitwise_and(x1, x2, /[, out, where, …]) | Compute the bit-wise AND of two arrays element-wise.
bitwise_or(x1, x2, /[, out, where, casting, …]) | Compute the bit-wise OR of two arrays element-wise.
bitwise_xor(x1, x2, /[, out, where, …]) | Compute the bit-wise XOR of two arrays element-wise.
invert(x, /[, out, where, casting, order, …]) | Compute bit-wise inversion, or bit-wise NOT, element-wise.
left_shift(x1, x2, /[, out, where, casting, …]) | Shift the bits of an integer to the left.
right_shift(x1, x2, /[, out, where, …]) | Shift the bits of an integer to the right.

## Comparison functions

method | desc
---|---
greater(x1, x2, /[, out, where, casting, …]) | Return the truth value of (x1 > x2) element-wise.
greater_equal(x1, x2, /[, out, where, …]) | Return the truth value of (x1 >= x2) element-wise.
less(x1, x2, /[, out, where, casting, …]) | Return the truth value of (x1 < x2) element-wise.
less_equal(x1, x2, /[, out, where, casting, …]) | Return the truth value of (x1 =< x2) element-wise.
not_equal(x1, x2, /[, out, where, casting, …]) | Return (x1 != x2) element-wise.
equal(x1, x2, /[, out, where, casting, …]) | Return (x1 == x2) element-wise.

<div class="warning-warp">
<b>Warning</b>
<p>Do not use the Python keywords and and or to combine logical array expressions. These keywords will test the truth value of the entire array (not element-by-element as you might expect). Use the bitwise operators & and | instead.</p>
</div>

method | desc
---|---
logical_and(x1, x2, /[, out, where, …]) | Compute the truth value of x1 AND x2 element-wise.
logical_or(x1, x2, /[, out, where, casting, …]) | Compute the truth value of x1 OR x2 element-wise.
logical_xor(x1, x2, /[, out, where, …]) | Compute the truth value of x1 XOR x2, element-wise.
logical_not(x, /[, out, where, casting, …]) | Compute the truth value of NOT x element-wise.

<div class="warning-warp">
<b>Warning</b>
<p>The bit-wise operators & and | are the proper way to perform element-by-element array comparisons. Be sure you understand the operator precedence: <code>(a &gt; 2) & (a &lt; 5)</code> is the proper syntax because <code>a &gt; 2 & a &lt; 5</code> will result in an error due to the fact that <code>2 & a</code> is evaluated first.</p>
</div>

method | desc
---|---
maximum(x1, x2, /[, out, where, casting, …]) | Element-wise maximum of array elements.

**Tip:**

The Python function ``max()`` will find the maximum over a one-dimensional array, but it will do so using a slower sequence interface. The reduce method of the maximum ufunc is much faster. Also, the ``max()`` method will not give answers you might expect for arrays with greater than one dimension. The reduce method of minimum also allows you to compute a total minimum over an array.

method | desc
---|---
minimum(x1, x2, /[, out, where, casting, …]) | Element-wise minimum of array elements.

<div class="warning-warp">
<b>Warning</b>
<p>the behavior of maximum(a, b) is different than that of max(a, b). As a ufunc, maximum(a, b) performs an element-by-element comparison of a and b and chooses each element of the result according to which element in the two arrays is larger. In contrast, max(a, b) treats the objects a and b as a whole, looks at the (total) truth value of a > b and uses it to return either a or b (as a whole). A similar difference exists between minimum(a, b) and min(a, b).
</p>
</div>

method | desc
---|---
fmax(x1, x2, /[, out, where, casting, …]) | Element-wise maximum of array elements.
fmin(x1, x2, /[, out, where, casting, …]) | Element-wise minimum of array elements.

## Floating functions

Recall that all of these functions work element-by-element over an array, returning an array output. The description details only a single operation.

method | desc
---|---
isfinite(x, /[, out, where, casting, order, …]) | Test element-wise for finiteness (not infinity or not Not a Number).
isinf(x, /[, out, where, casting, order, …]) | Test element-wise for positive or negative infinity.
isnan(x, /[, out, where, casting, order, …]) | Test element-wise for NaN and return result as a boolean array.
isnat(x, /[, out, where, casting, order, …]) | Test element-wise for NaT (not a time) and return result as a boolean array.
fabs(x, /[, out, where, casting, order, …]) | Compute the absolute values element-wise.
signbit(x, /[, out, where, casting, order, …]) | Returns element-wise True where signbit is set (less than zero).
copysign(x1, x2, /[, out, where, casting, …]) | Change the sign of x1 to that of x2, element-wise.
nextafter(x1, x2, /[, out, where, casting, …]) | Return the next floating-point value after x1 towards x2, element-wise.
spacing(x, /[, out, where, casting, order, …]) | Return the distance between x and the nearest adjacent number.
modf(x[, out1, out2], / [[, out, where, …]) | Return the fractional and integral parts of an array, element-wise.
ldexp(x1, x2, /[, out, where, casting, …]) | Returns x1 * 2**x2, element-wise.
frexp(x[, out1, out2], / [[, out, where, …]) | Decompose the elements of x into mantissa and twos exponent.
fmod(x1, x2, /[, out, where, casting, …]) | Return the element-wise remainder of division.
floor(x, /[, out, where, casting, order, …]) | Return the floor of the input, element-wise.
ceil(x, /[, out, where, casting, order, …]) | Return the ceiling of the input, element-wise.
trunc(x, /[, out, where, casting, order, …]) | Return the truncated value of the input, element-wise.