# 索引

``ndarrays``可以使用标准Python ``x[obj]` `语法编制索引，其中x是数组，obj是选择。有三种索引可用：字段访问，基本切片，高级索引。哪一个发生取决于obj。

> **注意**
> 在Python中，``x[（exp1，exp2，...，expN）]`` 相当于 ``x [exp1，exp2，...，expN]`` ; 后者只是前者的语法糖。

## 基本切片和索引

基本切片将Python的切片基本概念扩展到N维。 当obj是切片对象（由start：stop：括号内的步骤符号构造），整数或切片对象和整数的元组时，会发生基本切片。 省略号和newaxis对象也可以穿插其中。 为了保持向后兼容Numeric中的常见用法，如果选择对象是包含切片对象，Ellipsis对象或newaxis对象的任何非nararray序列（例如列表），则也会启动基本切片，但不是 整数数组或其他嵌入序列。

使用N个整数进行索引的最简单情况是返回表示相应项的数组标量。 与在Python中一样，所有索引都是从零开始的：对于第i个索引n_i，有效范围是 0\le n_i <d_i，其中d_i是数组形状的第i个元素。 负指数被解释为从数组的末尾开始计数（即，如果n_i <0，则表示n_i + d_i）。

通过基本切片生成的所有数组始终是原始数组的视图。

序列切片的标准规则适用于基于每维的基本切片（包括使用步骤索引）。 要记住的一些有用的概念包括：

- 基本切片语法是 i:j:k 其中i是起始索引，j是停止索引，k是步骤（k\neq0）。 这选择了具有索引值i，i + k，...，i +（m-1）k的m个元素（在相应的维度中），其中m = q +（r\neq0）并且q和r是获得的商和余数 通过将j-i除以k：j -i = qk + r，使得i +（m-1）k <j。

    **例子**

    ```python
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> x[1:7:2]
    array([1, 3, 5])
    ```

- 负i和j被解释为n + i和n + j，其中n是相应维度中的元素数量。 负k使得踩踏指向更小的指数。

    **例子**

    ```python
    >>> x[-2:10]
    array([8, 9])
    >>> x[-3:3:-1]
    array([7, 6, 5, 4])
    ```

- 假设n是要切片的维度中的元素数。然后，如果没有给出i，则k > 0时默认为 0, k < 0 时n = 1。 如果没有给出j，则对于k > 0，默认为n;对于k < 0，默认为-n-1。 如果没有给出k，则默认为1.注意::与:相同:并且表示沿该轴选择所有索引。

    **例子**

    ```python
    >>> x[5:]
    array([5, 6, 7, 8, 9])
    ```

- 如果选择元组中的对象数小于N，则对任何后续维度假定：。

    **例子**

    ```python
    >>> x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
    >>> x.shape
    (2, 3, 1)
    >>> x[1:2]
    array([[[4],
            [5],
            [6]]])
    ```

- ``Ellipsis`` 扩展为：x.ndim生成与长度相同的选择元组所需的对象数。可能只存在一个省略号。

    **例子**

    ```python
    >>> x[...,0]
    array([[1, 2, 3],
        [4, 5, 6]])
    ```
       
- 选择元组中的每个``newaxis``对象用于将结果选择的维度扩展一个单位长度维度。 添加的维度是选择元组中``newaxis``对象的位置。

    **例子**

    ```python
    >>> x[:,np.newaxis,:,:].shape
    (2, 1, 3, 1)
    ```

- 整数i返回与i：i + 1相同的值，除了返回对象的维数减少1.特别是，具有第p个元素的整数（和所有其他条目:)的选择元组返回 对应的子数组，其维数为N - 1.如果N = 1，则返回的对象是数组标量。 Scalars中解释了这些对象。

- 如果选择元组具有所有条目：除了作为切片对象i：j：k的第p个条目之外，则返回的数组具有通过连接由元素i，i + k的整数索引返回的子数组而形成的维N. ，...，i +（m - 1）k <j，

- 在切片元组中使用多个非：条目进行基本切片，就像使用单个非：条目重复应用切片一样，其中连续采用非：条目（所有其他非：条目替换为:)。 因此，``x[ind1, ..., ind2, :]``在基本切片下就像``x [ind1][...,ind2,:]``。

<div class="warning-warp">
<b>警告</b>

<p>对于高级索引，上述情况并非如此。</p>
</div>

- 你可以使用切片来设置数组中的值，但是（与列表不同）您永远不会增长数组。 要在x [obj] = value中设置的值的大小必须（可广播）为与x [obj]相同的形状。

> **注意**
> 请记住，切片元组始终可以构造为obj并在x [obj]表示法中使用。 可以在构造中使用切片对象代替[start:stop:step]表示法。 例如，x[1:10:5,:: - 1]也可以实现为obj=(slice(1,10,5)，slice(None，None, -1)); x[obj]。这对于构造适用于任意维数组的通用代码非常有用。

### ``numpy.newaxis``

可以在所有切片操作中使用``newaxis``对象来创建长度为1的轴。 ``newaxis``是'None'的别名，'None'可以代替它使用相同的结果。

## 高级索引

当选择对象obj是非元组序列对象，ndarray（数据类型为integer或bool）或具有至少一个序列对象或ndarray（数据类型为integer或bool）的元组时，将触发高级索引。 高级索引有两种类型：整数和布尔值。

高级索引始终返回数据的副本（与返回视图的基本切片形成对比）。

<div class="warning-warp">
<b>错误</b>

<p>高级索引的定义意味着x[(1, 2, 3), ] 与 x[(1, 2, 3)]根本不同。 后者相当于 x[1,2,3]，它将触发基本选择，而前者将触发高级索引。一定要明白为什么会这样。

还要认识到 x[[1,2,3]] 将触发高级索引，而 x[[1,2，slice（None）]] 将触发基本切片。</p>
</div>

### 整数数组索引

整数数组索引允许基于其N维索引选择数组中的任意项。 每个整数数组表示该维度的多个索引。

#### 纯整数数组索引

当索引由被索引的数组具有维度的整数数组组成时，索引是直接的，但与切片不同。

高级索引始终作为一个广播和迭代：

```python
result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
                           ..., ind_N[i_1, ..., i_M]]
```

请注意，结果形状与（广播）索引数组形状ind_1, ..., ind_N相同。

**例子**

从每一行开始，应选择一个特定元素。 行索引只是 ``[0,1,2]`` 而列索引指定了为相应行选择的元素，这里是 ``[0,1,0]``。 将两者结合使用可以使用高级索引解决任务：

```python
>>> x = np.array([[1, 2], [3, 4], [5, 6]])
>>> x[[0, 1, 2], [0, 1, 0]]
array([1, 4, 5])
```

为了实现类似于上面的基本切片的行为，可以使用广播。 函数ix_可以帮助这个广播。 通过示例可以最好地理解这一点。

**例子**

从4x3阵列中，应使用高级索引选择角元素。 因此，需要选择列为``[0,2]``之一且行为``[0,3]``之一的所有元素。 要使用高级索引，需要明确选择所有元素。 使用前面解释的方法可以写：

```python
>>> x = array([[ 0,  1,  2],
...            [ 3,  4,  5],
...            [ 6,  7,  8],
...            [ 9, 10, 11]])
>>> rows = np.array([[0, 0],
...                  [3, 3]], dtype=np.intp)
>>> columns = np.array([[0, 2],
...                     [0, 2]], dtype=np.intp)
>>> x[rows, columns]
array([[ 0,  2],
       [ 9, 11]])
```

但是，由于上面的索引数组只是重复自己，所以可以使用广播（比较诸如``rows [：，np.newaxis] + columns）``之类的操作来简化：

```python
>>> rows = np.array([0, 3], dtype=np.intp)
>>> columns = np.array([0, 2], dtype=np.intp)
>>> rows[:, np.newaxis]
array([[0],
       [3]])
>>> x[rows[:, np.newaxis], columns]
array([[ 0,  2],
       [ 9, 11]])
```

使用函数``ix_``也可以实现这种广播：

```python
>>> x[np.ix_(rows, columns)]
array([[ 0,  2],
       [ 9, 11]])
```

注意，如果没有``np.ix_``调用，只会选择对角线元素，就像前面的例子中所使用的那样。 对于使用多个高级索引进行索引，这个差异是最重要的。

#### 结合高级和基本索引

When there is at least one slice (``:``), ellipsis (``...``) or ``np.newaxis`` in the index (or the array has more dimensions than there are advanced indexes), then the behaviour can be more complicated. It is like concatenating the indexing result for each advanced index element

当索引中至少有一个切片（``:``），省略号（``...``）或``np.newaxis``时（或者数组的维度比高级索引多）， 然后行为可能会更复杂。 这就像连接每个高级索引元素的索引结果一样

在最简单的情况下，只有一个高级索引。 单个高级索引可以例如替换切片，并且结果数组将是相同的，但是，它是副本并且可以具有不同的存储器布局。 当可能时，切片是优选的。

**例子**

```python
>>> x[1:2, 1:3]
array([[4, 5]])
>>> x[1:2, [1, 2]]
array([[4, 5]])
```

理解这种情况的最简单方法可能是根据结果形状进行思考。 索引操作分为两部分，即由基本索引（不包括整数）定义的子空间和来自高级索引部分的子空间。 需要区分两种索引组合：

- 高级索引由切片，省略号或新轴分隔。 例如``x [arr1, :, arr2]``。
- 高级索引彼此相邻。 例如``x[..., arr1, arr2, :]``但不是``x[arr1, :, 1]``因为``1``在这方面是一个高级索引。

在第一种情况下，高级索引操作产生的维度首先出现在结果数组中，然后是子空间维度。 在第二种情况下，高级索引操作的维度将插入到结果数组中与初始数组中相同的位置（后一种逻辑使简单的高级索引行为就像切片一样）。

**例子**

假设``x.shape``是(10, 20, 30)而ind是(2, 3, 4)形索引的intp数组，那么result=x[..., ind, :]有shape (10, 2, 3, 4, 30) 因为 (20, )-shaped 子空间已被 (2, 3, 4) 形广播索引子空间所取代。如果我们让i，j，k在（2,3,4）形子空间上循环，那么结果[..., i, j, k, :] = x[...，ind[i, j, k], :]。 此示例产生与x.take(ind, axis = -2)相同的结果。

**例子**

Let ``x.shape`` be (10,20,30,40,50) and suppose ind_1 and ind_2 can be broadcast to the shape (2,3,4). Then x[:,ind_1,ind_2] has shape (10,2,3,4,40,50) because the (20,30)-shaped subspace from X has been replaced with the (2,3,4) subspace from the indices. However, x[:,ind_1,:,ind_2] has shape (2,3,4,10,30,50) because there is no unambiguous place to drop in the indexing subspace, thus it is tacked-on to the beginning. It is always possible to use .transpose() to move the subspace anywhere desired. Note that this example cannot be replicated using take.

设``x.shape``为(10, 20, 30, 40, 50)并假设ind_1和ind_2可以广播到形状(2, 3, 4)。然后x[:, ind_1，ind_2] 具有形状 (10, 2, 3, 4, 40, 50)，因为来自X的(20, 30)-shaped 子空间已被替换为来自 (2,3,4)的子空间 指数。但是，x[:, ind_1, :, ind_2]具有形状(2, 3, 4, 10, 30, 50)，因为在索引子空间中没有明确的位置，因此它被添加到开头。始终可以使用.transpose()在任何需要的位置移动子空间。请注意，此示例无法使用take进行复制。

### 布尔数组索引

This advanced indexing occurs when obj is an array object of Boolean type, such as may be returned from comparison operators. A single boolean index array is practically identical to ``x[obj.nonzero()]`` where, as described above, obj.nonzero() returns a tuple (of length obj.ndim) of integer index arrays showing the True elements of obj. However, it is faster when ``obj.shape == x.shape``.

If obj.ndim == x.ndim, x[obj] returns a 1-dimensional array filled with the elements of x corresponding to the True values of obj. The search order will be row-major, C-style. If obj has True values at entries that are outside of the bounds of x, then an index error will be raised. If obj is smaller than x it is identical to filling it with False.

**Example**

A common use case for this is filtering for desired element values. For example one may wish to select all entries from an array which are not NaN:

```python
>>> x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
>>> x[~np.isnan(x)]
array([ 1.,  2.,  3.])
```

Or wish to add a constant to all negative elements:

```python
>>> x = np.array([1., -1., -2., 3])
>>> x[x < 0] += 20
>>> x
array([  1.,  19.,  18.,   3.])
```

In general if an index includes a Boolean array, the result will be identical to inserting obj.nonzero() into the same position and using the integer array indexing mechanism described above. x[ind_1, boolean_array, ind_2] is equivalent to ``x[(ind_1,) + boolean_array.nonzero() + (ind_2,)]``.

If there is only one Boolean array and no integer indexing array present, this is straight forward. Care must only be taken to make sure that the boolean index has exactly as many dimensions as it is supposed to work with.

**Example**

From an array, select all rows which sum up to less or equal two:

```python
>>> x = np.array([[0, 1], [1, 1], [2, 2]])
>>> rowsum = x.sum(-1)
>>> x[rowsum <= 2, :]
array([[0, 1],
       [1, 1]])
```

But if ``rowsum`` would have two dimensions as well:

```python
>>> rowsum = x.sum(-1, keepdims=True)
>>> rowsum.shape
(3, 1)
>>> x[rowsum <= 2, :]    # fails
IndexError: too many indices
>>> x[rowsum <= 2]
array([0, 1])
```

The last one giving only the first elements because of the extra dimension. Compare ``rowsum.nonzero()`` to understand this example.

Combining multiple Boolean indexing arrays or a Boolean with an integer indexing array can best be understood with the ``obj.nonzero()``analogy. The function ``ix_`` also supports boolean arrays and will work without any surprises.

**Example**

Use boolean indexing to select all rows adding up to an even number. At the same time columns 0 and 2 should be selected with an advanced integer index. Using the ``ix_`` function this can be done with:

```python
>>> x = array([[ 0,  1,  2],
...            [ 3,  4,  5],
...            [ 6,  7,  8],
...            [ 9, 10, 11]])
>>> rows = (x.sum(-1) % 2) == 0
>>> rows
array([False,  True, False,  True])
>>> columns = [0, 2]
>>> x[np.ix_(rows, columns)]
array([[ 3,  5],
       [ 9, 11]])
```

Without the ``np.ix_`` call or only the diagonal elements would be selected.

Or without ``np.ix_`` (compare the integer array examples):

```python
>>> rows = rows.nonzero()[0]
>>> x[rows[:, np.newaxis], columns]
array([[ 3,  5],
       [ 9, 11]])
```

## Detailed notes

These are some detailed notes, which are not of importance for day to day indexing (in no particular order):

- The native NumPy indexing type is ``intp`` and may differ from the default integer array type. ``intp`` is the smallest data type sufficient to safely index any array; for advanced indexing it may be faster than other types.

- For advanced assignments, there is in general no guarantee for the iteration order. This means that if an element is set more than once, it is not possible to predict the final result.

- An empty (tuple) index is a full scalar index into a zero dimensional array. x[()] returns a scalar if x is zero dimensional and a view otherwise. On the other hand x[...] always returns a view.

- If a zero dimensional array is present in the index and it is a full integer index the result will be a scalar and not a zero dimensional array. (Advanced indexing is not triggered.)
When an ellipsis (...) is present but has no size (i.e. replaces zero :) the result will still always be an array. A view if no advanced index is present, otherwise a copy.

- the ``nonzero`` equivalence for Boolean arrays does not hold for zero dimensional boolean arrays.

- When the result of an advanced indexing operation has no elements but an individual index is out of bounds, whether or not an IndexError is raised is undefined (e.g. x[[], [123]] with 123 being out of bounds).
When a casting error occurs during assignment (for example updating a numerical array using a sequence of strings), the array being assigned to may end up in an unpredictable partially updated state. However, if any other error (such as an out of bounds index) occurs, the array will remain unchanged.

- The memory layout of an advanced indexing result is optimized for each indexing operation and no particular memory order can be assumed.

- When using a subclass (especially one which manipulates its shape), the default ndarray.__setitem__ behaviour will call __getitem__ for basic indexing but not for advanced indexing. For such a subclass it may be preferable to call ndarray.__setitem__ with a base class ndarray view on the data. This must be done if the subclasses __getitem__ does not return views.

## Field Access

另见：

> Data type objects (dtype), Scalars

If the ``ndarray`` object is a structured array the fields of the array can be accessed by indexing the array with strings, dictionary-like.

Indexing ``x['field-name']`` returns a new view to the array, which is of the same shape as x (except when the field is a sub-array) but of data type ``x.dtype['field-name']`` and contains only the part of the data in the specified field. Also record array scalars can be “indexed” this way.

Indexing into a structured array can also be done with a list of field names, e.g. ``x[['field-name1','field-name2']]``. Currently this returns a new array containing a copy of the values in the fields specified in the list. As of NumPy 1.7, returning a copy is being deprecated in favor of returning a view. A copy will continue to be returned for now, but a FutureWarning will be issued when writing to the copy. If you depend on the current behavior, then we suggest copying the returned array explicitly, i.e. use x[[‘field-name1’,’field-name2’]].copy(). This will work with both past and future versions of NumPy.

If the accessed field is a sub-array, the dimensions of the sub-array are appended to the shape of the result.

**Example**

```python
>>> x = np.zeros((2,2), dtype=[('a', np.int32), ('b', np.float64, (3,3))])
>>> x['a'].shape
(2, 2)
>>> x['a'].dtype
dtype('int32')
>>> x['b'].shape
(2, 2, 3, 3)
>>> x['b'].dtype
dtype('float64')
```

## Flat Iterator indexing

``x.flat`` returns an iterator that will iterate over the entire array (in C-contiguous style with the last index varying the fastest). This iterator object can also be indexed using basic slicing or advanced indexing as long as the selection object is not a tuple. This should be clear from the fact that ``x.flat`` is a 1-dimensional view. It can be used for integer indexing with 1-dimensional C-style-flat indices. The shape of any returned array is therefore the shape of the integer indexing object.