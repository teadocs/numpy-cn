# 广播(Broadcasting)

另见：

> numpy.broadcast

术语broadcasting描述numpy在算术运算期间如何处理具有不同形状的数组。受限于某些约束，较小的数组依据较大数组“broadcasting”，使得它们具有兼容的形状。Broadcasting提供了一种矢量化数组操作的方法，使得循环发生在C而不是Python。它做到这一点且不用不必要的数据拷贝，通常导致高效的算法实现。然而，有些情况下，broadcasting是一个坏主意，因为它导致低效的内存使用并减慢计算。

NumPy操作通常是在逐个元素的基础上在数组对上完成的。在最简单的情况下，两个数组必须具有完全相同的形状，如下例所示：

```python
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([2.0, 2.0, 2.0])
>>> a * b
array([ 2.,  4.,  6.])
```

当数组的形状满足一定的条件时，NumPy的broadcasting规则可以放宽这个限制。最简单的broadcasting示例发生在一个操作包含数组和标量值的时候：

```python
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = 2.0
>>> a * b
array([ 2.,  4.,  6.])
```

结果等同于前面的例子，其中``b``是一个数组。在算术运算期间，我们可以认为标量``b``被拉伸了，形成与``a``相同形状的数组。``b``中的新元素是原始标量简单的拷贝。拉伸这个比喻只是概念性的。NumPy足够聪明，它使用原始的标量值而不会真正拷贝，使broadcasting操作尽可能的内存和计算高效。

第二个例子中的代码比第一个例子中的代码更有效，因为broadcasting在乘法期间移动较少的内存（``b``是标量而不是数组）。

## Broadcasting的一般规则

当在两个数组上操作时，NumPy在元素级别比较它们的形状。它从尾随的维度开始，并朝着前进的方向前进。两个维度兼容，当

1. 他们是平等的，或者
1. 其中之一是1

如果不满足这些条件，则抛出``ValueError: frames are not aligned``异常，指示数组具有不兼容的形状。结果数组的大小是沿着输入数组的每个维度的最大大小。

数组不需要具有相同维度的数目。例如，如果你有一个``256x256x3``数值的RGB值，并且你想要通过一个不同的值缩放图像中的每个颜色，你可以将图像乘以一个具有3个值的一维数组。根据broadcast规则排列这些数组的最后一个轴的大小，表明它们是兼容的：

```
Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3
```

当比较的任何一个维度为1时，则使用另一个。换句话说，大小为1的维被拉伸或“复制”以匹配另一维。

在以下示例中，A和B数组都具有长度为1的轴，在broadcast操作期间将其扩展为更大的大小：

```
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

这里有一些例子：

```
A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4

A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5
```

以下是不broadcast的形状示例：

```
A      (1d array):  3
B      (1d array):  4 # trailing dimensions do not match

A      (2d array):      2 x 1
B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched
```

broadcasting在实践中的一个例子：

```python
>>> x = np.arange(4)
>>> xx = x.reshape(4,1)
>>> y = np.ones(5)
>>> z = np.ones((3,4))

>>> x.shape
(4,)

>>> y.shape
(5,)

>>> x + y
<type 'exceptions.ValueError'>: shape mismatch: objects cannot be broadcast to a single shape

>>> xx.shape
(4, 1)

>>> y.shape
(5,)

>>> (xx + y).shape
(4, 5)

>>> xx + y
array([[ 1.,  1.,  1.,  1.,  1.],
       [ 2.,  2.,  2.,  2.,  2.],
       [ 3.,  3.,  3.,  3.,  3.],
       [ 4.,  4.,  4.,  4.,  4.]])

>>> x.shape
(4,)

>>> z.shape
(3, 4)

>>> (x + z).shape
(3, 4)

>>> x + z
array([[ 1.,  2.,  3.,  4.],
       [ 1.,  2.,  3.,  4.],
       [ 1.,  2.,  3.,  4.]])
```

Broadcasting提供了获取两个数组的外积（或任何其他outer操作）的方便方式。以下示例显示了两个1-d数组的外积操作：

```python
>>> a = np.array([0.0, 10.0, 20.0, 30.0])
>>> b = np.array([1.0, 2.0, 3.0])
>>> a[:, np.newaxis] + b
array([[  1.,   2.,   3.],
       [ 11.,  12.,  13.],
       [ 21.,  22.,  23.],
       [ 31.,  32.,  33.]])
```

这里``newaxis``索引操作符将一个新轴插入到``a``中，使其成为一个二维``4x1``数组。将``4x1``数组与形状为``(3,)``的``b``组合，产生一个``4x3``数组。

有关broadcasting概念的图解，请参阅[本文](#)。