# 数学函数

## 三角函数

- sin(x, /[, out, where, casting, order, …])	逐个元素运算三角正弦函数。
- cos(x, /[, out, where, casting, order, …])	逐个元素运算三角余弦函数。
- tan(x, /[, out, where, casting, order, …])	逐个元素运算三角正切函数。
- arcsin(x, /[, out, where, casting, order, …])	逐个元素运算三角反正弦函数。
- arccos(x, /[, out, where, casting, order, …])	逐个元素运算三角反余弦函数。
- arctan(x, /[, out, where, casting, order, …])	逐个元素运算三角反正切函数。
- hypot(x1, x2, /[, out, where, casting, …])	给定直角三角形的“腿”，返回它的斜边。
- arctan2(x1, x2, /[, out, where, casting, …])	元素弧切线x1/x2正确选择象限。
- degrees(x, /[, out, where, casting, order, …])	将角度从弧度转换为度数。
- radians(x, /[, out, where, casting, order, …])	将角度从度数转换为弧度。
- unwrap(p[, discont, axis])	通过将值之间的差值更改为2*pi补码来展开。
- deg2rad(x, /[, out, where, casting, order, …])	将角度从度数转换为弧度。
- rad2deg(x, /[, out, where, casting, order, …])	将角度从弧度转换为度数。

## 双曲函数

- sinh(x, /[, out, where, casting, order, …])	逐个元素运算双曲正弦函数。
- cosh(x, /[, out, where, casting, order, …])	逐个元素运算双曲余弦函数。
- tanh(x, /[, out, where, casting, order, …])	逐个元素运算双曲正切函数。
- arcsinh(x, /[, out, where, casting, order, …])	逐个元素运算逆双曲正弦函数。
- arccosh(x, /[, out, where, casting, order, …])	逐个元素运算逆双曲正弦函数。
- arctanh(x, /[, out, where, casting, order, …])	逐个元素运算逆双曲正弦函数。

## 小数操作

- around(a[, decimals, out])	按给定的小数位数均匀地四舍五入。
- round_(a[, decimals, out])	将数组舍入到给定的小数位数。
- rint(x, /[, out, where, casting, order, …])	将数组的元素舍入为最接近的整数。
- fix(x[, out])	向零舍入到最接近的整数。
- floor(x, /[, out, where, casting, order, …])	逐个元素返回输入的下限。
- ceil(x, /[, out, where, casting, order, …])	逐个元素返回输入的上限。
- trunc(x, /[, out, where, casting, order, …])	逐个元素返回输入的截断值。

## 求总和, 求乘积, 求差异

- prod(a[, axis, dtype, out, keepdims])	返回给定轴上的数组元素的乘积。
- sum(a[, axis, dtype, out, keepdims])	给定轴上的数组元素的总和。
- nanprod(a[, axis, dtype, out, keepdims])	返回给定轴上的数组元素的乘积，将非数字（NaN）视为1。
- nansum(a[, axis, dtype, out, keepdims])	返回给定轴上的数组元素的总和，将非数字（NaN）视为零。
- cumprod(a[, axis, dtype, out])	返回给定轴上元素的累积乘积。
- cumsum(a[, axis, dtype, out])	返回给定轴上元素的累积和。
- nancumprod(a[, axis, dtype, out])	返回给定轴上的数组元素的累积乘积，将非数字（NaN）视为一个。
- nancumsum(a[, axis, dtype, out])	返回给定轴上的数组元素的累积和，将非数字（NaN）视为零。
- diff(a[, n, axis])	计算沿给定轴的第n个离散差。
- ediff1d(ary[, to_end, to_begin]) 数组的连续元素之间的差异。
- gradient(f, *varargs, **kwargs)	返回N维数组的渐变。
- cross(a, b[, axisa, axisb, axisc, axis])	返回两个（数组）向量的叉积。
- trapz(y[, x, dx, axis])	沿给定的轴积分使用复合梯形规则运算。

## 指数和对数

- exp(x, /[, out, where, casting, order, …])	计算输入数组中所有元素的指数。
- expm1(x, /[, out, where, casting, order, …])	计算数组中所有元素的exp(X)-1。
- exp2(x, /[, out, where, casting, order, …])	为输入数组中的所有p计算2**p。
- log(x, /[, out, where, casting, order, …])	逐个元素计算自然对数。
- log10(x, /[, out, where, casting, order, …])	逐个元素计算返回输入数组的以10为底的对数。
- log2(x, /[, out, where, casting, order, …])	以-2为底的对数。
- log1p(x, /[, out, where, casting, order, …])	逐个元素计算返回一个自然对数加上输入数组。
- logaddexp(x1, x2, /[, out, where, casting, …])	输入的指数之和的对数。
- logaddexp2(x1, x2, /[, out, where, casting, …])	以-2为基的输入的指数和的对数。

## 其他的特殊函数

- i0(x)	修正的第一类贝塞尔函数，0阶。
- sinc(x)	返回sinc函数。

## Floating point routines

- signbit(x, /[, out, where, casting, order, …])	Returns element-wise True where signbit is set (less than zero).
- copysign(x1, x2, /[, out, where, casting, …])	Change the sign of x1 to that of x2, element-wise.
- frexp(x[, out1, out2], / [[, out, where, …])	Decompose the elements of x into mantissa and twos exponent.
- ldexp(x1, x2, /[, out, where, casting, …])	Returns x1 * 2**x2, element-wise.
- nextafter(x1, x2, /[, out, where, casting, …])	Return the next floating-point value after x1 towards x2, element-wise.
- spacing(x, /[, out, where, casting, order, …])	Return the distance between x and the nearest adjacent number.

## Arithmetic operations

- add(x1, x2, /[, out, where, casting, order, …])	Add arguments element-wise.
- reciprocal(x, /[, out, where, casting, …])	Return the reciprocal of the argument, element-wise.
- positive(x, /[, out, where, casting, order, …])	Numerical positive, element-wise.
- negative(x, /[, out, where, casting, order, …])	Numerical negative, element-wise.
- multiply(x1, x2, /[, out, where, casting, …])	Multiply arguments element-wise.
- divide(x1, x2, /[, out, where, casting, …])	Returns a true division of the inputs, element-wise.
- power(x1, x2, /[, out, where, casting, …])	First array elements raised to powers from second array, element-wise.
- subtract(x1, x2, /[, out, where, casting, …])	Subtract arguments, element-wise.
- true_divide(x1, x2, /[, out, where, …])	Returns a true division of the inputs, element-wise.
- floor_divide(x1, x2, /[, out, where, …])	Return the largest integer smaller or equal to the division of the inputs.
- float_power(x1, x2, /[, out, where, …])	First array elements raised to powers from second array, element-wise.
- fmod(x1, x2, /[, out, where, casting, …])	Return the element-wise remainder of division.
- mod(x1, x2, /[, out, where, casting, order, …])	Return element-wise remainder of division.
- modf(x[, out1, out2], / [[, out, where, …])	Return the fractional and integral parts of an array, element-wise.
- remainder(x1, x2, /[, out, where, casting, …])	Return element-wise remainder of division.
- divmod(x1, x2[, out1, out2], / [[, out, …])	Return element-wise quotient and remainder simultaneously.
- Handling complex numbers
- angle(z[, deg])	Return the angle of the complex argument.
- real(val)	Return the real part of the complex argument.
- imag(val)	Return the imaginary part of the complex argument.
- conj(x, /[, out, where, casting, order, …])	Return the complex conjugate, element-wise.

## Miscellaneous

- convolve(a, v[, mode])	Returns the discrete, linear convolution of two one-dimensional sequences.
- clip(a, a_min, a_max[, out])	Clip (limit) the values in an array.
- sqrt(x, /[, out, where, casting, order, …])	Return the positive square-root of an array, element-wise.
- cbrt(x, /[, out, where, casting, order, …])	Return the cube-root of an array, element-wise.
- square(x, /[, out, where, casting, order, …])	Return the element-wise square of the input.
- absolute(x, /[, out, where, casting, order, …])	Calculate the absolute value element-wise.
- fabs(x, /[, out, where, casting, order, …])	Compute the absolute values element-wise.
- sign(x, /[, out, where, casting, order, …])	Returns an element-wise indication of the sign of a number.
- heaviside(x1, x2, /[, out, where, casting, …])	Compute the Heaviside step function.
- maximum(x1, x2, /[, out, where, casting, …])	Element-wise maximum of array elements.
- minimum(x1, x2, /[, out, where, casting, …])	Element-wise minimum of array elements.
- fmax(x1, x2, /[, out, where, casting, …])	Element-wise maximum of array elements.
- fmin(x1, x2, /[, out, where, casting, …])	Element-wise minimum of array elements.
- nan_to_num(x[, copy])	Replace nan with zero and inf with large finite numbers.
- real_if_close(a[, tol])	If complex input returns a real array if complex parts are close to zero.
- interp(x, xp, fp[, left, right, period])	One-dimensional linear interpolation.