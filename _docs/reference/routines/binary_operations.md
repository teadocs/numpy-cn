# 二进制操作

## 单元位运算

- bitwise_and(x1, x2, /[, out, where, …])	计算位和两个数组的元素。
- bitwise_or(x1, x2, /[, out, where, casting, …])	计算两个数组元素的位或。
- bitwise_xor(x1, x2, /[, out, where, …])	计算两个数组元素的位异或。
- invert(x, /[, out, where, casting, order, …])	按位计算求逆，或按位求逆，按元素计算。
- left_shift(x1, x2, /[, out, where, casting, …])	将整数的位向左移。
- right_shift(x1, x2, /[, out, where, …])	将整数的位向右移。

## Bit位打包

- packbits(myarray[, axis])	将二进制值数组的元素打包到uint8数组中的位中。
- unpackbits(myarray[, axis])	将uint8数组的元素解包为二进制值输出数组。

## 输出格式

- binary_repr(num[, width])	将输入数字的二进制表示形式返回为字符串。