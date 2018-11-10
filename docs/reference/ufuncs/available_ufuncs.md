# 可用的通用函数列表

目前在一种或多种类型的``numpy``中定义了60多个通用函数，涵盖了各种各样的操作。 当使用相关的中缀符号时，这些ufunc中的一些会在数组上自动调用（例如，当``a + b``写入时，``add（a，b）``在内部调用，而a或b是``ndarray``）。 不过，你可能仍希望使用ufunc调用以使用可选的输出参数将输出放置在你选择的对象（或多个对象）中。

回想一下，每个ufunc都是逐个元素运行的。 因此，每个标量ufunc将被描述为如果作用于一组标量输入以返回一组标量输出。

> **注意**
> 即使你使用可选的输出参数，ufunc仍会返回其输出。

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

可选的输出参数可用于帮助你节省大型计算的内存。 如果你的数组很大，由于临时计算空间的创建和（稍后）破坏，复杂的表达式可能需要比绝对必要的时间更长的时间。 例如，表达式 ``G = a * b + c`` 相当于 ``t1 = A * B; G = T1 + C; del t1``。 它将更快地执行为``G = A * B; add（G，C，G）`` 与 ``G = A * B; G + = C`` 相同。

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
<b>警告</b>
<p>不要使用Python关键字和/或组合逻辑数组表达式。 这些关键字将测试整个数组的真值（不是你想象的逐个元素）。 使用按位运算符＆和| 代替。</p>
</div>

方法 | 描述
---|---
logical_and(x1, x2, /[, out, where, …]) | 计算x1和x2元素的真值。
logical_or(x1, x2, /[, out, where, casting, …]) | 计算x1 OR x2元素的真值。
logical_xor(x1, x2, /[, out, where, …]) | 以元素方式计算x1 XOR x2的真值。
logical_not(x, /[, out, where, casting, …]) | 计算NOT x元素的真值。

<div class="warning-warp">
<b>警告</b>
<p>
逐位运算符＆和| 是执行逐元素数组比较的正确方法。确保你理解运算符优先级：<code>(a &gt; 2 ) ＆(a &lt; 5 )</code>  是正确的语法，因为 <code>a &gt; 2 & a &lt; 5</code>将导致错误，因为首先计算<code>2 & a</code>。
</p>
</div>

方法 | 描述
---|---
maximum(x1, x2, /[, out, where, casting, …]) | 数组元素的元素最大值。

**提示：**

Python函数``max（）``将在一维数组中找到最大值，但它会使用较慢的序列接口。 最大ufunc的reduce方法要快得多。 此外，``max（）``方法不会给出具有多个维度的数组所期望的答案。 reduce的minimal方法还允许你计算数组的总最小值。

方法 | 描述
---|---
minimum(x1, x2, /[, out, where, casting, …]) | 元素最小的数组元素。

<div class="warning-warp">
<b>警告：</b>
<p>最大值 (a, b) 的行为与 max(a, b)的行为不同。作为ufunc，maximum(a, b)执行a和b的逐元素比较，并根据两个数组中的哪个元素更大来选择结果的每个元素。 相反，max(a, b)将对象a和b视为一个整体，查看 a > b的（总）真值，并使用它返回a或b（作为一个整体）。 minimum(a, b) 和 min(a, b) 之间存在类似的差异。
</p>
</div>

方法 | 描述
---|---
fmax(x1, x2, /[, out, where, casting, …]) | 数组元素的逐个元素运算取得最大值。
fmin(x1, x2, /[, out, where, casting, …]) | 数组元素的逐个元素运算取得最小值。

## 浮动函数

回想一下，所有这些函数在一个数组上逐个元素运算，返回输出一个数组对象。下面的描述只详细说明了其中的一个操作。

方法 | 描述
---|---
isfinite(x, /[, out, where, casting, order, …]) | 对有限性(不是无限或不是数字)的测试元素。
isinf(x, /[, out, where, casting, order, …]) | 测试元件-对于正无穷大或负无穷大。
isnan(x, /[, out, where, casting, order, …]) | 对NaN进行元素测试，并将结果作为布尔数组返回。
isnat(x, /[, out, where, casting, order, …]) | 测试元素的NAT(不是时间)，并返回结果作为一个布尔数组。
fabs(x, /[, out, where, casting, order, …]) | 逐个元素计算绝对值。
signbit(x, /[, out, where, casting, order, …]) | 返回设置了符号位(小于零)的元素级True。
copysign(x1, x2, /[, out, where, casting, …]) | 将x1的符号改为x2的符号，就元素而言。
nextafter(x1, x2, /[, out, where, casting, …]) | 返回x1后的下一个浮点值到x2，元素级。
spacing(x, /[, out, where, casting, order, …]) | 返回x与最近邻数之间的距离。
modf(x[, out1, out2], / [[, out, where, …]) | 按元素返回数组的分数和整数部分.
ldexp(x1, x2, /[, out, where, casting, …]) | 逐个元素返回 ``x1*2*x2``。
frexp(x[, out1, out2], / [[, out, where, …]) | 将x的元素分解成尾数和TWOS指数。
fmod(x1, x2, /[, out, where, casting, …]) | 返回除法的元素余数。
floor(x, /[, out, where, casting, order, …]) | 返回输入的底面，按元素划分。
ceil(x, /[, out, where, casting, order, …]) | 逐个元素方式返回输入的上限。
trunc(x, /[, out, where, casting, order, …]) | 逐个元素方式返回输入的截断值。