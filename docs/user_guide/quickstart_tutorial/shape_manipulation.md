# 形状操作

## 更改数组的形状

一个数组具有由每个轴上的元素数量给出的形状：

```python
>>> a = np.floor(10*np.random.random((3,4)))
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.shape
(3, 4)
```

数组的形状可以通过各种命令进行更改。请注意，以下三个命令都返回一个修改后的数组，但不要更改原始数组：

```python
>>> a.ravel()  # returns the array, flattened
array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])
>>> a.reshape(6,2)  # returns the array with a modified shape
array([[ 2.,  8.],
       [ 0.,  6.],
       [ 4.,  5.],
       [ 1.,  1.],
       [ 8.,  9.],
       [ 3.,  6.]])
>>> a.T  # returns the array, transposed
array([[ 2.,  4.,  8.],
       [ 8.,  5.,  9.],
       [ 0.,  1.,  3.],
       [ 6.,  1.,  6.]])
>>> a.T.shape
(4, 3)
>>> a.shape
(3, 4)
```

由ravel()产生的数组中元素的顺序通常是“C风格”，也就是说，最右边的索引“改变最快”，所以[0,0]之后的元素是[0,1] 。如果数组被重新塑造成其他形状，数组又被视为“C-style”。NumPy通常创建按此顺序存储的数组，因此ravel()通常不需要复制其参数，但如果数组是通过切片另一个数组或使用不寻常选项创建的，则可能需要复制它。函数ravel()和reshape()也可以通过使用可选参数来指示使用FORTRAN风格的数组，其中最左侧的索引更改速度最快。

``reshape`` 函数返回具有修改形状的参数，而 ``ndarray.resize`` 方法修改数组本身：

```python
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.resize((2,6))
>>> a
array([[ 2.,  8.,  0.,  6.,  4.,  5.],
       [ 1.,  1.,  8.,  9.,  3.,  6.]])
```

如果在reshape操作中将维度指定为-1，则会自动计算其他维度：

```python
>>> a.reshape(3,-1)
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
```

另见：

> ndarray.shape, reshape, resize, ravel

## 将不同数组堆叠在一起

几个数组可以沿不同的轴堆叠在一起：

```python
>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
>>> np.vstack((a,b))
array([[ 8.,  8.],
       [ 0.,  0.],
       [ 1.,  8.],
       [ 0.,  4.]])
>>> np.hstack((a,b))
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
```

函数 ``column_stack`` 将1D数组作为列叠加到2D数组中。它相当于仅用于二维数组的 ``hstack``：

```python
>>> from numpy import newaxis
>>> np.column_stack((a,b))     # with 2D arrays
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
>>> a = np.array([4.,2.])
>>> b = np.array([3.,8.])
>>> np.column_stack((a,b))     # returns a 2D array
array([[ 4., 3.],
       [ 2., 8.]])
>>> np.hstack((a,b))           # the result is different
array([ 4., 2., 3., 8.])
>>> a[:,newaxis]               # this allows to have a 2D columns vector
array([[ 4.],
       [ 2.]])
>>> np.column_stack((a[:,newaxis],b[:,newaxis]))
array([[ 4.,  3.],
       [ 2.,  8.]])
>>> np.hstack((a[:,newaxis],b[:,newaxis]))   # the result is the same
array([[ 4.,  3.],
       [ 2.,  8.]])
```

另一方面，对于任何输入数组，函数 ``row_stack`` 相当于 ``vstack``。一般来说，对于具有两个以上维度的数组，``hstack`` 沿第二轴堆叠，``vstack`` 沿第一轴堆叠，``concatenate`` 允许一个可选参数，给出串接应该发生的轴。

**请注意**

在复杂情况下，``r_`` 和 ``c_`` 可用于通过沿一个轴叠加数字来创建数组。它们允许使用范围字面量（“：”）

```python
>>> np.r_[1:4,0,4]
array([1, 2, 3, 0, 4])
```

当以数组作为参数使用时，``r_`` 和 ``c_`` 类似于其默认行为中的 ``vstack`` 和 ``hstack`` ，但是允许一个可选参数给出要沿其连接的轴的编号。

另见：

> hstack, vstack, column_stack, concatenate, c\_, r\_

## 将一个数组分成几个较小的数组

使用 ``hsplit`` ，可以沿其水平轴拆分数组，通过指定要返回的均匀划分的数组数量，或通过指定要在其后进行划分的列：

```python
>>> a = np.floor(10*np.random.random((2,12)))
>>> a
array([[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])
>>> np.hsplit(a,3)   # Split a into 3
[array([[ 9.,  5.,  6.,  3.],
       [ 1.,  4.,  9.,  2.]]), array([[ 6.,  8.,  0.,  7.],
       [ 2.,  1.,  0.,  6.]]), array([[ 9.,  7.,  2.,  7.],
       [ 2.,  2.,  4.,  0.]])]
>>> np.hsplit(a,(3,4))   # Split a after the third and the fourth column
[array([[ 9.,  5.,  6.],
       [ 1.,  4.,  9.]]), array([[ 3.],
       [ 2.]]), array([[ 6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])]
```

``vsplit`` 沿纵轴分割，并且 ``array_split`` 允许指定沿哪个轴分割。
