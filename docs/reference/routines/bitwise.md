# 二进制运算

## 逐元素位操作

方法 | 描述
---|---
[bitwise_and](https://numpy.org/devdocs/reference/generated/numpy.bitwise_and.html#numpy.bitwise_and)(x1, x2, /[, out, where, …]) | 按元素计算两个数组的按位与。
[bitwise_or](https://numpy.org/devdocs/reference/generated/numpy.bitwise_or.html#numpy.bitwise_or)(x1, x2, /[, out, where, casting, …]) | 按元素计算两个数组的按位或。
[bitwise_xor](https://numpy.org/devdocs/reference/generated/numpy.bitwise_xor.html#numpy.bitwise_xor)(x1, x2, /[, out, where, …]) | 按元素计算两个数组的按位XOR。
[invert](https://numpy.org/devdocs/reference/generated/numpy.invert.html#numpy.invert)(x, /[, out, where, casting, order, …]) | 按元素计算按位求逆，或按位求非。
[left_shift](https://numpy.org/devdocs/reference/generated/numpy.left_shift.html#numpy.left_shift)(x1, x2, /[, out, where, casting, …]) | 将整数的位向左移动。
[right_shift](https://numpy.org/devdocs/reference/generated/numpy.right_shift.html#numpy.right_shift)(x1, x2, /[, out, where, …]) | 向右移整数的位。

## 打包二进制

方法 | 描述
---|---
[packbits](https://numpy.org/devdocs/reference/generated/numpy.packbits.html#numpy.packbits)(a[, axis, bitorder]) | 将二进制值数组的元素打包为uint8数组中的位。
[unpackbits](https://numpy.org/devdocs/reference/generated/numpy.unpackbits.html#numpy.unpackbits)(a[, axis, count, bitorder]) | 将uint8数组的元素解压缩为二进制值输出数组。

## 输出格式

方法 | 描述
---|---
[binary_repr](https://numpy.org/devdocs/reference/generated/numpy.binary_repr.html#numpy.binary_repr)(num[, width]) | 以字符串形式返回输入数字的二进制表示形式。