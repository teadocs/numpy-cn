# 杂项API

## Buffer 对象

- getbuffer	（获取buffer）
- newbuffer	（新建buffer）

## 性能调优

- setbufsize(size)	设置ufuns中使用的缓冲区的大小。
- getbufsize()	返回ufuns中使用的缓冲区的大小。

## 内存范围

- shares_memory(a, b[, max_work])	确定两个数组是否共享内存
- may_share_memory(a, b[, max_work])	确定两个数组是否可以共享内存

## 数组混合器

- lib.mixins.NDArrayOperatorsMixin	Mixin使用 __array_ufunc__ 定义所有运算符特殊方法。

## NumPy版本比较

- lib.NumpyVersion(vstring)	解析并比较numpy版本字符串。