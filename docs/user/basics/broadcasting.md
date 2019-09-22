# 广播（Broadcasting）

::: tip 另见

- [``numpy.broadcast``](https://numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast)
- [Numpy中的数组广播](https://numpy.org/devdocs/user/theory.broadcasting.html#array-broadcasting-in-numpy)

:::

::: tip 注意

有关广播概念的说明，请参阅[此文章](https://numpy.org/devdocs/user/theory.broadcasting.html)。

:::

术语广播（Broadcasting）描述了 numpy 如何在算术运算期间处理具有不同形状的数组。受某些约束的影响，较小的数组在较大的数组上“广播”，以便它们具有兼容的形状。广播提供了一种矢量化数组操作的方法，以便在C而不是Python中进行循环。它可以在不制作不必要的数据副本的情况下实现这一点，通常导致高效的算法实现。然而，有些情况下广播是一个坏主意，因为它会导致内存使用效率低下，从而减慢计算速度。

NumPy 操作通常在逐个元素的基础上在数组对上完成。在最简单的情况下，两个数组必须具有完全相同的形状，如下例所示：

``` python
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([2.0, 2.0, 2.0])
>>> a * b
array([ 2.,  4.,  6.])
```

当数组的形状满足某些约束时，NumPy的广播规则放宽了这种约束。当一个数组和一个标量值在一个操作中组合时，会发生最简单的广播示例：

``` python
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = 2.0
>>> a * b
array([ 2.,  4.,  6.])
```

结果等同于前面的示例，其中``b``是数组。我们可以将在算术运算期间``b``被 *拉伸* 的标量想象成具有相同形状的数组``a``。新元素
 ``b``只是原始标量的副本。拉伸类比只是概念性的。NumPy足够聪明，可以使用原始标量值而无需实际制作副本，因此广播操作尽可能具有内存和计算效率。

第二个示例中的代码比第一个示例中的代码更有效，因为广播在乘法期间移动的内存较少（``b``是标量而不是数组）。

## 一般广播规则

在两个数组上运行时，NumPy会逐元素地比较它们的形状。它从尾随尺寸开始，并向前发展。两个尺寸兼容时

1. 他们是平等的，或者
1. 其中一个是1

如果不满足这些条件，则抛出 ``ValueError: operands could not be broadcast together`` 异常，指示数组具有不兼容的形状。结果数组的大小是沿输入的每个轴不是1的大小。

数组不需要具有相同 *数量* 的维度。例如，如果您有一个``256x256x3``RGB值数组，并且希望将图像中的每种颜色缩放不同的值，则可以将图像乘以具有3个值的一维数组。根据广播规则排列这些数组的尾轴的大小，表明它们是兼容的：

``` python
Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3
```

当比较的任何一个尺寸为1时，使用另一个尺寸。换句话说，尺寸为1的尺寸被拉伸或“复制”以匹配另一个尺寸。

在以下示例中，``A``和``B``数组都具有长度为1的轴，在广播操作期间会扩展为更大的大小：

``` python
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

以下是一些例子：

``` python
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

以下是不广播的形状示例：

``` python
A      (1d array):  3
B      (1d array):  4 # trailing dimensions do not match

A      (2d array):      2 x 1
B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched
```

实践中广播的一个例子：

``` python
>>> x = np.arange(4)
>>> xx = x.reshape(4,1)
>>> y = np.ones(5)
>>> z = np.ones((3,4))

>>> x.shape
(4,)

>>> y.shape
(5,)

>>> x + y
ValueError: operands could not be broadcast together with shapes (4,) (5,)

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

广播提供了一种方便的方式来获取两个数组的外积（或任何其他外部操作）。以下示例显示了两个1-d数组的外积操作：

``` python
>>> a = np.array([0.0, 10.0, 20.0, 30.0])
>>> b = np.array([1.0, 2.0, 3.0])
>>> a[:, np.newaxis] + b
array([[  1.,   2.,   3.],
       [ 11.,  12.,  13.],
       [ 21.,  22.,  23.],
       [ 31.,  32.,  33.]])
```

这里 ``newaxis`` 索引操作符插入一个新轴 ``a`` ，使其成为一个二维 ``4x1`` 数组。将 ``4x1`` 数组与形状为 ``(3,)`` 的 ``b`` 组合，产生一个``4x3``数组。

