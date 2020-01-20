# 操作集合(`Set routines`)

## 进行适当的sets(`Making proper sets`)

method | description
---|---
[unique](https://numpy.org/devdocs/reference/generated/numpy.unique.html#numpy.unique)(ar[, return_index, return_inverse, …]) | 查找数组的唯一元素。

## 布尔运算(`Boolean operations`)

method | description
---|---
[in1d](https://numpy.org/devdocs/reference/generated/numpy.in1d.html#numpy.in1d)(ar1, ar2[, assume_unique, invert]) | 测试一维数组的每个元素是否也存在于第二个数组中。
[intersect1d](https://numpy.org/devdocs/reference/generated/numpy.intersect1d.html#numpy.intersect1d)(ar1, ar2[, assume_unique, …]) | 找到两个数组的交集。
[isin](https://numpy.org/devdocs/reference/generated/numpy.isin.html#numpy.isin)(element, test_elements[, …]) | 计算test_elements中的元素，仅在element上广播。
[setdiff1d](https://numpy.org/devdocs/reference/generated/numpy.setdiff1d.html#numpy.setdiff1d)(ar1, ar2[, assume_unique]) | 找到两个数组的集合差。
[setxor1d](https://numpy.org/devdocs/reference/generated/numpy.setxor1d.html#numpy.setxor1d)(ar1, ar2[, assume_unique]) | 找到两个数组的集合异或。
[union1d](https://numpy.org/devdocs/reference/generated/numpy.union1d.html#numpy.union1d)(ar1, ar2) | 找到两个数组的并集。
