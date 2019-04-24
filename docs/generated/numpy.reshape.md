# numpy.all

## 调用概览

```python
numpy.reshape(a, newshape, order='C')
```

## 方法说明

为数组提供新形状而不更改其数据。

- **参数**: 
    - **a : 类数组类型**
        将要进行重塑的数组。
    - **newshape : int或tuple of ints**
        新形状应该与原始形状兼容。如果是一个整数，结果会返回一个对应长度的一维数组。另外，某个形状的维度值可以是-1。这种情况下，该值是从数组长度和剩余维度推断出来的。
    - **order : {'C', 'F', 'A'}, 可选**
        使用此索引顺序读取a的元素，并使用此索引顺序将元素放入重塑的数组中。'C'表示使用类似C的索引顺序读/写元素，首先读/写最后一个轴上的元素，其次是倒数第二个轴，最后是第一个轴上的元素。'F'表示使用类似Fortran的索引顺序读/写元素，首先读/写第一个轴上的元素，接下来是第二个轴上的元素，最后是最后一个轴上的元素。注意，'C'和'F'选项不考虑底层数组的内存布局，只涉及索引的顺序。'A'表示如果a的内存布局是Fortran连续的，那么就会以类似Fortran的索引顺序读/写元素，否则以类似于C的索引顺序读/写元素。
- **返回**:
    - **reshaped_array : ndarray**
        如果可能，它将返回一个新的视图对象; 否则，它返回一个原数组的拷贝。注意，该函数不保证返回数组的内存布局（C或Fortran连续）。

## 另见

- [ndarray.reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.reshape.html#numpy.ndarray.reshape) 等效方法

## 注意

我们并不总是可以在不复制数据的情况下更改数组的形状。如果要在复制数据时抛出错误，则应将新形状赋值给数组的shape属性：

```python
>>> a = np.zeros((10, 2))
# A transpose makes the array non-contiguous
>>> b = a.T
# Taking a view makes it possible to modify the shape without modifying
# the initial object.
>>> c = b.view()
>>> c.shape = (20)
AttributeError: incompatible shape for a non-contiguous array
```

*order*关键字提供索引顺序，用于从a中读取值以及将值写入到输出数组中。例如，假设有一个数组：

```python
>>> a = np.arange(6).reshape((3, 2))
>>> a
array([[0, 1],
       [2, 3],
       [4, 5]])
```

我们可以将重塑操作视为首先使用给定的索引顺序对数组进行拉直（ravel）操作，然后使用相同的索引顺序将拉直后数组中的元素插入到新数组中。

```python
>>> np.reshape(a, (2, 3)) # C-like index ordering
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
array([[0, 4, 3],
       [2, 1, 5]])
>>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
array([[0, 4, 3],
       [2, 1, 5]])
```

## 例子

```python
>>> a = np.array([[1,2,3], [4,5,6]])
>>> np.reshape(a, 6)
array([1, 2, 3, 4, 5, 6])
>>> np.reshape(a, 6, order='F')
array([1, 4, 2, 5, 3, 6])
```

```python
>>> np.reshape(a, (3,-1))
# the unspecified value is inferred to be 2
array([[1, 2],
       [3, 4],
       [5, 6]])
```