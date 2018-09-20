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

## 浮点数操作

- signbit(x, /[, out, where, casting, order, …])	返回元素为True设置signbit（小于零）。
- copysign(x1, x2, /[, out, where, casting, …])	逐个元素将x1的符号改为x2的符号。
- frexp(x[, out1, out2], / [[, out, where, …])	将x的元素分解成尾数和TWOS指数。
- ldexp(x1, x2, /[, out, where, casting, …])	逐个元素返回x1\*2*x2。
- nextafter(x1, x2, /[, out, where, casting, …]) 逐个元素返回x1后的下一个浮点值到x2。
- spacing(x, /[, out, where, casting, order, …])	返回x与最近邻数之间的距离。

## 算术运算

- add(x1, x2, /[, out, where, casting, order, …])	按元素添加参数。
- reciprocal(x, /[, out, where, casting, …])	逐元素计算返回参数的倒数。
- positive(x, /[, out, where, casting, order, …])	逐元素正数计算
- negative(x, /[, out, where, casting, order, …])	逐元素负数计算
- multiply(x1, x2, /[, out, where, casting, …])	逐元素参数相乘
- divide(x1, x2, /[, out, where, casting, …])	逐元素方式返回输入的真正除法。
- power(x1, x2, /[, out, where, casting, …])	逐元素将第一个数组元素从第二个数组提升到幂。
- subtract(x1, x2, /[, out, where, casting, …])	逐元素参数相减。
- true_divide(x1, x2, /[, out, where, …])	按元素返回输入的真实相除。
- floor_divide(x1, x2, /[, out, where, …])	返回小于或等于输入除法的最大整数。
- float_power(x1, x2, /[, out, where, …])	逐元素将第一个数组元素从第二个数组提升到幂。
- fmod(x1, x2, /[, out, where, casting, …])	返回除法的元素余数。
- mod(x1, x2, /[, out, where, casting, order, …])	返回除法元素的余数。
- modf(x[, out1, out2], / [[, out, where, …])	以元素方式返回数组的分数和整数部分。
- remainder(x1, x2, /[, out, where, casting, …])	返回除法元素的余数。
- divmod(x1, x2[, out1, out2], / [[, out, …])	同时返回逐元素的商和余数。

## Handling complex numbers

- angle(z[, deg])	返回复杂参数的角度。
- real(val) 返回复杂参数的实部。
- imag(val)	返回复杂参数的虚部。
- conj(x, /[, out, where, casting, order, …])	以元素方式返回复共轭。

## 杂项

- convolve(a, v[, mode])	返回两个一维序列的离散线性卷积。
- clip(a, a_min, a_max[, out])	剪辑（限制）数组中的值。
- sqrt(x, /[, out, where, casting, order, …])	逐个元素返回数组的正平方根。
- cbrt(x, /[, out, where, casting, order, …])	逐个元素返回数组的立方根。
- square(x, /[, out, where, casting, order, …])	返回输入的元素方块。
- absolute(x, /[, out, where, casting, order, …])	逐个元素地计算绝对值。
- fabs(x, /[, out, where, casting, order, …])	逐个元素计算绝对值。
- sign(x, /[, out, where, casting, order, …])	返回数字符号的元素指示。
- heaviside(x1, x2, /[, out, where, casting, …])	计算Heaviside阶跃函数。
- maximum(x1, x2, /[, out, where, casting, …])	取数组中最大的元素。
- minimum(x1, x2, /[, out, where, casting, …])	取数组中最小的元素。
- fmax(x1, x2, /[, out, where, casting, …])	 取数组中最大的元素。
- fmin(x1, x2, /[, out, where, casting, …])	取数组中最小的元素。
- nan_to_num(x[, copy])	将nan替换为零，使用大的有限数替换inf。
- real_if_close(a[, tol])	如果复杂输入返回真实数组，如果复杂零件接近于零。
- interp(x, xp, fp[, left, right, period])	一维线性插值。