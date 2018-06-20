# 深拷贝

``copy`` 方法生成数组及其数据的完整拷贝。

```python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

## 函数和方法概述

这里列出了一些根据类别排列的有用的NumPy函数和方法名称。完整列表见Routines。

1. 数组创建
    > arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like
1. 转换 
    > ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat
1. 手法
    > array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack
1. 问题
    > all, any, nonzero, where
1. 顺序
    > argmax, argmin, argsort, max, min, ptp, searchsorted, sort
1. 操作
    > choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum
1. 基本统计
    > cov, mean, std, var
1. 基本线性代数
    > cross, dot, outer, linalg.svd, vdot
