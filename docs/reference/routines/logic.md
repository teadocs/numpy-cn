# 逻辑函数

## 真值测试

方法 | 描述
---|---
[all](https://numpy.org/devdocs/reference/generated/numpy.all.html#numpy.all)(a[, axis, out, keepdims]) | 测试是否沿给定轴的所有数组元素求值为True。
[any](https://numpy.org/devdocs/reference/generated/numpy.any.html#numpy.any)(a[, axis, out, keepdims]) | 测试沿给定轴的任何数组元素的求值是否为True。

## 数组内容

方法 | 描述
---|---
[isfinite](https://numpy.org/devdocs/reference/generated/numpy.isfinite.html#numpy.isfinite)(x, /[, out, where, casting, order, …]) | 逐一测试有限性（不是无穷大还是不是数字）。
[isinf](https://numpy.org/devdocs/reference/generated/numpy.isinf.html#numpy.isinf)(x, /[, out, where, casting, order, …]) | 逐元素测试正无穷大或负无穷大。
[isnan](https://numpy.org/devdocs/reference/generated/numpy.isnan.html#numpy.isnan)(x, /[, out, where, casting, order, …]) | 对NaN逐个元素进行测试，并将结果作为布尔数组返回。
[isnat](https://numpy.org/devdocs/reference/generated/numpy.isnat.html#numpy.isnat)(x, /[, out, where, casting, order, …]) | 对NaT（不是时间）逐个元素进行测试，然后将结果作为布尔数组返回。
[isneginf](https://numpy.org/devdocs/reference/generated/numpy.isneginf.html#numpy.isneginf)(x[, out]) | 逐项测试负无穷大，将结果作为布尔数组返回。
[isposinf](https://numpy.org/devdocs/reference/generated/numpy.isposinf.html#numpy.isposinf)(x[, out]) | 对正无穷大逐个元素进行测试，以布尔数组形式返回结果。

## 数组类型测试

方法 | 描述
---|---
[iscomplex](https://numpy.org/devdocs/reference/generated/numpy.iscomplex.html#numpy.iscomplex)(x) | 返回布尔数组，如果输入元素复杂，则返回True。
[iscomplexobj](https://numpy.org/devdocs/reference/generated/numpy.iscomplexobj.html#numpy.iscomplexobj)(x) | 检查复杂类型或复杂数字数组。
[isfortran](https://numpy.org/devdocs/reference/generated/numpy.isfortran.html#numpy.isfortran)(a) | 检查数组是否是Fortran连续的，而不是C连续的。
[isreal](https://numpy.org/devdocs/reference/generated/numpy.isreal.html#numpy.isreal)(x) | 返回布尔数组，如果输入元素为实，则返回True。
[isrealobj](https://numpy.org/devdocs/reference/generated/numpy.isrealobj.html#numpy.isrealobj)(x) | 如果x是非复数类型或复数数组，则返回True。
[isscalar](https://numpy.org/devdocs/reference/generated/numpy.isscalar.html#numpy.isscalar)(num) | 如果num的类型是标量类型，则返回True。

## 逻辑运算

方法 | 描述
---|---
[logical_and](https://numpy.org/devdocs/reference/generated/numpy.logical_and.html#numpy.logical_and)(x1, x2, /[, out, where, …]) | 按元素计算x1和x2的真值。
[logical_or](https://numpy.org/devdocs/reference/generated/numpy.logical_or.html#numpy.logical_or)(x1, x2, /[, out, where, casting, …]) | 按元素计算x1或x2的真值。
[logical_not](https://numpy.org/devdocs/reference/generated/numpy.logical_not.html#numpy.logical_not)(x, /[, out, where, casting, …]) | 计算非x元素的真值。
[logical_xor](https://numpy.org/devdocs/reference/generated/numpy.logical_xor.html#numpy.logical_xor)(x1, x2, /[, out, where, …]) | 按元素计算x1 XOR x2的真值。

## 对照

方法 | 描述
---|---
[allclose](https://numpy.org/devdocs/reference/generated/numpy.allclose.html#numpy.allclose)(a, b[, rtol, atol, equal_nan]) | 如果两个数组在公差范围内按元素方式相等，则返回True。
[isclose](https://numpy.org/devdocs/reference/generated/numpy.isclose.html#numpy.isclose)(a, b[, rtol, atol, equal_nan]) | 返回一个布尔数组，其中两个数组在公差范围内按元素方式相等。
[array_equal](https://numpy.org/devdocs/reference/generated/numpy.array_equal.html#numpy.array_equal)(a1, a2) | 如果两个数组具有相同的形状和元素，则为True，否则为False。
[array_equiv](https://numpy.org/devdocs/reference/generated/numpy.array_equiv.html#numpy.array_equiv)(a1, a2) | 如果输入数组的形状一致且所有元素均相等，则返回True。
[greater](https://numpy.org/devdocs/reference/generated/numpy.greater.html#numpy.greater)(x1, x2, /[, out, where, casting, …]) | 按元素返回（x1 > x2）的真值。
[greater_equal](https://numpy.org/devdocs/reference/generated/numpy.greater_equal.html#numpy.greater_equal)(x1, x2, /[, out, where, …]) | 按元素返回（x1 >= x2）的真值。
[less](https://numpy.org/devdocs/reference/generated/numpy.less.html#numpy.less)(x1, x2, /[, out, where, casting, …]) | 按元素返回（x1 < x2）的真值。
[less_equal](https://numpy.org/devdocs/reference/generated/numpy.less_equal.html#numpy.less_equal)(x1, x2, /[, out, where, casting, …]) | 按元素返回（x1 =< x2）的真值。
[equal](https://numpy.org/devdocs/reference/generated/numpy.equal.html#numpy.equal)(x1, x2, /[, out, where, casting, …]) | 按元素返回（x1 == x2）。
[not_equal](https://numpy.org/devdocs/reference/generated/numpy.not_equal.html#numpy.not_equal)(x1, x2, /[, out, where, casting, …]) | 按元素返回（x1 != x2）。