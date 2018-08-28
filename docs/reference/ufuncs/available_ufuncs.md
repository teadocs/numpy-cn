# 可用的通用函数列表

目前在一种或多种类型的``numpy``中定义了60多个通用函数，涵盖了各种各样的操作。 当使用相关的中缀符号时，这些ufunc中的一些会在数组上自动调用（例如，当``a + b``写入时，``add（a，b）``在内部调用，而a或b是``ndarray``）。 不过，您可能仍希望使用ufunc调用以使用可选的输出参数将输出放置在您选择的对象（或多个对象）中。

回想一下，每个ufunc都是逐个元素运行的。 因此，每个标量ufunc将被描述为如果作用于一组标量输入以返回一组标量输出。

> **注意**
> 即使您使用可选的输出参数，ufunc仍会返回其输出。

## 数学运算

方法 | 描述
---|---
add(x1, x2, /[, out, where, casting, order, …]) | 按元素添加参数。
subtract(x1, x2, /[, out, where, casting, …]) | 从元素方面减去参数。
multiply(x1, x2, /[, out, where, casting, …]) | 按元素计算多个参数。
divide(x1, x2, /[, out, where, casting, …]) | 逐个元素方式返回输入的真正除法。
logaddexp(x1, x2, /[, out, where, casting, …]) | 输入的指数之和的对数。
logaddexp2(x1, x2, /[, out, where, casting, …]) | 以-2为基的输入的指数和的对数。
true_divide(x1, x2, /[, out, where, …]) | 以元素方式返回输入的真正除法。
floor_divide(x1, x2, /[, out, where, …]) | 返回小于或等于输入除法的最大整数。
negative(x, /[, out, where, casting, order, …]) | 数字否定，元素方面。
positive(x, /[, out, where, casting, order, …]) | 数字正面，元素方面。
power(x1, x2, /[, out, where, casting, …]) | 第一个数组元素从第二个数组提升到幂，逐个元素。
remainder(x1, x2, /[, out, where, casting, …]) | 返回除法元素的余数。
mod(x1, x2, /[, out, where, casting, order, …]) | 返回除法元素的余数。
fmod(x1, x2, /[, out, where, casting, …]) | 返回除法的元素余数。
divmod(x1, x2[, out1, out2], / [[, out, …]) | 同时返回逐元素的商和余数。
absolute(x, /[, out, where, casting, order, …]) | 逐个元素地计算绝对值。
fabs(x, /[, out, where, casting, order, …]) | 以元素方式计算绝对值。
rint(x, /[, out, where, casting, order, …]) | 将数组的元素舍入为最接近的整数。
sign(x, /[, out, where, casting, order, …]) | 返回数字符号的元素指示。
heaviside(x1, x2, /[, out, where, casting, …]) | 计算Heaviside阶跃函数。
conj(x, /[, out, where, casting, order, …]) | 以元素方式返回复共轭。
exp(x, /[, out, where, casting, order, …]) | 计算输入数组中所有元素的指数。
exp2(x, /[, out, where, casting, order, …]) | 计算输入数组中所有p的2**p。
log(x, /[, out, where, casting, order, …]) | 自然对数，元素方面。
log2(x, /[, out, where, casting, order, …]) | x的基数为2的对数。
log10(x, /[, out, where, casting, order, …]) | 以元素方式返回输入数组的基数10对数。
expm1(x, /[, out, where, casting, order, …]) | 计算数组中所有元素的exp(x)-1。
log1p(x, /[, out, where, casting, order, …]) | 返回一个加上输入数组的自然对数，逐个元素。
sqrt(x, /[, out, where, casting, order, …]) | 以元素方式返回数组的正平方根。
square(x, /[, out, where, casting, order, …]) | 返回输入的元素方块。
cbrt(x, /[, out, where, casting, order, …]) | 以元素方式返回数组的立方根。
reciprocal(x, /[, out, where, casting, …]) | 以元素为单位返回参数的倒数。

**小贴士:**

The optional output arguments can be used to help you save memory for large calculations. If your arrays are large, complicated expressions can take longer than absolutely necessary due to the creation and (later) destruction of temporary calculation spaces. For example, the expression ``G = a * b + c`` is equivalent to ``t1 = A * B; G = T1 + C; del t1``. It will be more quickly executed as ``G = A * B; add(G, C, G)`` which is the same as ``G = A * B; G += C``.

可选的输出参数可用于帮助您节省大型计算的内存。 如果您的数组很大，由于临时计算空间的创建和（稍后）破坏，复杂的表达式可能需要比绝对必要的时间更长的时间。 例如，表达式 ``G = a * b + c`` 相当于 ``t1 = A * B; G = T1 + C; del t1``。 它将更快地执行为``G = A * B; add（G，C，G）`` 与 ``G = A * B; G + = C`` 相同。

## 三角函数

当需要角度时，所有三角函数都使用弧度。 度与弧度之比为180°/π。

方法 | 描述
---|---
sin(x, /[, out, where, casting, order, …]) | 三角正弦，逐元素。
cos(x, /[, out, where, casting, order, …]) | 余弦元素。
tan(x, /[, out, where, casting, order, …]) | 计算切线元素。
arcsin(x, /[, out, where, casting, order, …]) | 反正弦，逐元素。
arccos(x, /[, out, where, casting, order, …]) | 三角反余弦，逐元素。
arctan(x, /[, out, where, casting, order, …]) | 三角反正切，逐元素。
arctan2(x1, x2, /[, out, where, casting, …]) | x1 / x2的逐元素反正切正确选择象限。
hypot(x1, x2, /[, out, where, casting, …]) | 给定直角三角形的“腿”，返回其斜边。
sinh(x, /[, out, where, casting, order, …]) | 双曲正弦，逐元素。
cosh(x, /[, out, where, casting, order, …]) | 双曲余弦，逐元素。
tanh(x, /[, out, where, casting, order, …]) | 计算双曲正切元素。
arcsinh(x, /[, out, where, casting, order, …]) | 逐元素逆双曲正弦。
arccosh(x, /[, out, where, casting, order, …]) | 逐元素反双曲余弦。
arctanh(x, /[, out, where, casting, order, …]) | 逐元素逆双曲正切。
deg2rad(x, /[, out, where, casting, order, …]) | 将角度从度数转换为弧度。
rad2deg(x, /[, out, where, casting, order, …]) | 将角度从弧度转换为度数。

## 比特功能

这些函数都需要整数参数，并且它们操纵这些参数的位模式。

方法 | 描述
---|---
bitwise_and(x1, x2, /[, out, where, …]) | 逐个元素地计算两个数组的逐位AND。
bitwise_or(x1, x2, /[, out, where, casting, …]) | 逐个元素地计算两个数组的逐位OR。
bitwise_xor(x1, x2, /[, out, where, …]) | 计算两个数组的逐位XOR元素。
invert(x, /[, out, where, casting, order, …]) | 按位元素计算逐位反转或逐位NOT。
left_shift(x1, x2, /[, out, where, casting, …]) | 将整数位移到左侧。
right_shift(x1, x2, /[, out, where, …]) | 将整数位移到右侧。

## 比较功能

方法 | 描述
---|---
greater(x1, x2, /[, out, where, casting, …]) | 逐元素方式返回（x1> x2）的真值。
greater_equal(x1, x2, /[, out, where, …]) | 逐元素方式返回（x1> = x2）的真值。
less(x1, x2, /[, out, where, casting, …]) | 逐元素方式返回（x1 <x2）的真值。
less_equal(x1, x2, /[, out, where, casting, …]) | 逐元素方式返回（x1 = <x2）的真值。
not_equal(x1, x2, /[, out, where, casting, …]) | 逐元素方式返回（x1！= x2）。
equal(x1, x2, /[, out, where, casting, …]) | 逐元素方式返回（x1 == x2）。

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