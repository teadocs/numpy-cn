# 逻辑函数

## 真值测试

- all(a[, axis, out, keepdims]) 测试沿给定轴的所有数组元素是否都计算为True。
- any(a[, axis, out, keepdims])	测试给定轴上的任何数组元素是否为True。

## 数组内容

- isfinite(x, /[, out, where, casting, order, …])	测试元素的有限性（不是无穷大或不是数字）。
- isinf(x, /[, out, where, casting, order, …])	对正或负无穷大进行元素级别测试。
- isnan(x, /[, out, where, casting, order, …])	元素级别的为NaN测试并将结果作为布尔数组返回。
- isnat(x, /[, out, where, casting, order, …])	元素级别为NaT（不是时间）测试并将结果作为布尔数组返回。
- isneginf(x[, out])	元素级别测试负无穷大的元素，返回结果为bool数组。
- isposinf(x[, out])	元素级别测试元素为正无穷大，返回结果为bool数组。

## 数组类型测试

- iscomplex(x)	返回一个bool数组，如果输入元素很复杂，则返回True。
- iscomplexobj(x)	检查复杂类型或复数数组。
- isfortran(a)	如果数组是Fortran连续但不是C连续，则返回True。
- isreal(x)	返回一个bool数组，如果输入元素是实数，则返回True。
- isrealobj(x)	如果x是非复数类型或复数数组，则返回True。
- isscalar(num)	如果num的类型是标量类型，则返回True。

## 逻辑运算

- logical_and(x1, x2, /[, out, where, …])	逐个元素计算x1和x2的真值。
- logical_or(x1, x2, /[, out, where, casting, …])	逐个元素计算x1 OR x2的真值。
- logical_not(x, /[, out, where, casting, …])	逐个元素计算NOT x元素的真值。
- logical_xor(x1, x2, /[, out, where, …])	逐个元素计算x1 XOR x2的真值。

## 比较

- allclose(a, b[, rtol, atol, equal_nan])	如果两个数组在容差范围内在元素方面相等，则返回True。
- isclose(a, b[, rtol, atol, equal_nan])	返回一个布尔数组，其中两个数组在容差范围内是元素相等的。
- array_equal(a1, a2)	如果两个数组具有相同的形状和元素，则为真，否则为False。
- array_equiv(a1, a2)	如果输入数组的形状一致且所有元素相等，则返回True。
- greater(x1, x2, /[, out, where, casting, …])	逐个元素方式返回（x1> x2）的真值。
- greater_equal(x1, x2, /[, out, where, …])	逐个元素方式返回（x1> = x2）的真值。
- less(x1, x2, /[, out, where, casting, …])	逐个元素方式返回（x1 <x2）的真值。
- less_equal(x1, x2, /[, out, where, casting, …])	逐个元素方式返回（x1 = <x2）的真值。
- equal(x1, x2, /[, out, where, casting, …])	逐个元素返回（x1 == x2）。
- not_equal(x1, x2, /[, out, where, casting, …])	逐个元素返回 Return (x1 != x2)。