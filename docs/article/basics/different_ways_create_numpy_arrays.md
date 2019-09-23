---
meta:
  - name: keywords
    content: 创建 Numpy 数组的不同方式
  - name: description
    content: Numpy库的核心是数组对象或ndarray对象（n维数组）。你将使用Numpy数组执行逻辑，统计和傅里叶变换等运算。作为...
---

# 创建 Numpy 数组的不同方式

Numpy库的核心是数组对象或ndarray对象（n维数组）。你将使用Numpy数组执行逻辑，统计和傅里叶变换等运算。作为使用Numpy的一部分，你要做的第一件事就是创建Numpy数组。本指南的主要目的是帮助数据科学爱好者了解可用于创建Numpy数组的不同方式。

创建Numpy数组有三种不同的方法：

1. 使用Numpy内部功能函数
1. 从列表等其他Python的结构进行转换
1. 使用特殊的库函数

## 使用Numpy内部功能函数

Numpy具有用于创建数组的内置函数。 我们将在本指南中介绍其中一些内容。

### 创建一个一维的数组

首先，让我们创建一维数组或rank为1的数组。``arange``是一种广泛使用的函数，用于快速创建数组。将值20传递给``arange``函数会创建一个值范围为0到19的数组。

```python
import Numpy as np
array = np.arange(20)
array
```

输出：

```python
array([0,  1,  2,  3,  4,
       5,  6,  7,  8,  9,
       10, 11, 12, 13, 14,
       15, 16, 17, 18, 19])
```

要验证此数组的维度，请使用shape属性。

```python
array.shape
```

输出：

```python
(20,)
```

由于逗号后面没有值，因此这是一维数组。 要访问此数组中的值，请指定非负索引。 与其他编程语言一样，索引从零开始。 因此，要访问数组中的第四个元素，请使用索引3。

```python
array[3]
```

输出：

```python
3
```

Numpy的数组是可变的，这意味着你可以在初始化数组后更改数组中元素的值。 使用print函数查看数组的内容。

```python
array[3] = 100
print(array)
```

输出：

```python
[  0   1   2 100
   4   5   6   7
   8   9  10  11
   12  13  14  15
   16  17  18  19]
```

与Python列表不同，Numpy数组的内容是同质的。 因此，如果你尝试将字符串值分配给数组中的元素，其数据类型为int，则会出现错误。

```python
array[3] ='Numpy'
```

输出：

```python
ValueError: invalid literal for int() with base 10: 'Numpy'
```

### 创建一个二维数组

我们来谈谈创建一个二维数组。 如果只使用arange函数，它将输出一维数组。 要使其成为二维数组，请使用reshape函数链接其输出。

```python
array = np.arange(20).reshape(4,5)
array
```

输出：

```python
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])
```

首先，将创建20个整数，然后将数组转换为具有4行和5列的二维数组。 我们来检查一下这个数组的维数。

```python
(4, 5)
```

由于我们得到两个值，这是一个二维数组。 要访问二维数组中的元素，需要为行和列指定索引。

```python
array[3][4]
```

输出：

```python
19
```

### 创建三维数组及更多维度

要创建三维数组，请为重塑形状函数指定3个参数。

```python
array = np.arange(27).reshape(3,3,3)
array
```

输出：

```python
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
```

需要注意的是：数组中元素的数量（27）必须是其尺寸（3 * 3 * 3）的乘积。 要交叉检查它是否是三维数组，可以使用shape属性。

```python
array.shape
```

输出：

```python
(3, 3, 3)
```

此外，使用``arange``函数，你可以创建一个在定义的起始值和结束值之间具有特定序列的数组。

```python
np.arange(10, 35, 3)
```

输出：

```python
array([10, 13, 16, 19, 22, 25, 28, 31, 34])
```

### 使用其他Numpy函数


除了arange函数之外，你还可以使用其他有用的函数（如 ``zeros`` 和 ``ones``）来快速创建和填充数组。

使用``zeros``函数创建一个填充零的数组。函数的参数表示行数和列数（或其维数）。

```python
np.zeros((2,4))
```

输出：

```python
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.]])
```

使用``ones``函数创建一个填充了1的数组。

```python
np.ones((3,4))
```

输出：

```python
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
```

``empty``函数创建一个数组。它的初始内容是随机的，取决于内存的状态。

```python
np.empty((2,3))
```

输出：

```python
array([[0.65670626, 0.52097334, 0.99831087],
       [0.07280136, 0.4416958 , 0.06185705]])
```

``full``函数创建一个填充给定值的n * n数组。

```python
np.full((2,2), 3)
```

输出：

```python
array([[3, 3],
       [3, 3]])
```

``eye``函数可以创建一个n * n矩阵，对角线为1s，其他为0。

```python
np.eye(3,3)
```

输出：

```python
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```

函数``linspace``在指定的时间间隔内返回均匀间隔的数字。 例如，下面的函数返回0到10之间的四个等间距数字。

```python
np.linspace(0, 10, num=4)
```

输出：

```python
array([ 0., 3.33333333, 6.66666667, 10.])
```

## 从Python列表转换

除了使用Numpy函数之外，你还可以直接从Python列表创建数组。将Python列表传递给数组函数以创建Numpy数组：

```python
array = np.array([4,5,6])
array
```

输出：

```python
array([4, 5, 6])
```

你还可以创建Python列表并传递其变量名以创建Numpy数组。

```python
list = [4,5,6]
list
```

输出：

```python
[4, 5, 6]
```

```python
array = np.array(list)
array
```

输出：

```python
array([4, 5, 6])
```

你可以确认变量``array``和``list``分别是Python列表和Numpy数组。

```python
type(list)
```

> list

```python
type(array)
```

> Numpy.ndarray

要创建二维数组，请将一系列列表传递给数组函数。

```python
array = np.array([(1,2,3), (4,5,6)])
array
```

输出：

```python
array([[1, 2, 3],
       [4, 5, 6]])
```

```python
array.shape
```

输出：

```
(2, 3)
```

## 使用特殊的库函数

你还可以使用特殊库函数来创建数组。例如，要创建一个填充0到1之间随机值的数组，请使用``random``函数。这对于需要随机状态才能开始的问题特别有用。

```python
np.random.random((2,2))
```

输出：

```python
array([[0.1632794 , 0.34567049],
       [0.03463241, 0.70687903]])
```

## 总结 

创建和填充Numpy数组是使用Numpy执行快速数值数组计算的第一步。使用不同的方式创建数组，你现在可以很好地执行基本的数组操作。

## 文章出处

由NumPy中文文档翻译，原作者为 Ravikiran Srinivasulu，翻译至：[https://www.pluralsight.com/guides/different-ways-create-numpy-arrays](https://www.pluralsight.com/guides/different-ways-create-numpy-arrays)