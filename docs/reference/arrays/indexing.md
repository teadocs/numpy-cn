# 索引

::: tip 另见

[索引基础知识](/user/basics/indexing.html)

:::

[``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)可以使用标准Python ``x[obj]``语法对其进行索引
 ，其中 *x* 是数组， *obj* 是选择。有三种可用的索引：字段访问，基本切片，高级索引。哪一个发生取决于 *obj* 。

::: tip 注意

在 Python 中，``x[(exp1，exp2，.，EXPN)]`` 等同于 ``x[exp1，exp2，.，EXPN]`` ；后者只是前者的语法糖。

:::

## 基本切片和索引

基本切片将Python的切片基本概念扩展到N维。
当 *obj* 是一个[``slice``](https://docs.python.org/dev/library/functions.html#slice)
对象（由``start:stop:step``括号内的符号构造），
整数或切片对象和整数的元组时，会发生基本切片。``Ellipsis``
和[``newaxis``](/reference/constants.html#numpy.newaxis)物体也可以穿插其中。

*从版本1.15.0开始不推荐使用：* 为了保持向后兼容Numeric中的常见用法，
如果选择对象是包含对象的任何非nararray和非元组序列（例如a [``list``](https://docs.python.org/dev/library/stdtypes.html#list)）
 [``slice``](https://docs.python.org/dev/library/functions.html#slice)，
 则也会启动基本切片``Ellipsis``，或者[``newaxis``](constants.html#numpy.newaxis)
对象，但不是整数数组或其他嵌入序列。

使用N个整数进行索引的最简单情况返回表示相应项的[数组标量](scalars.html)。
与Python一样，所有索引都是从零开始的：对于第i个索引 <img class="math" src="/static/images/math/4fbb0705523f8fdd2c3c5f55e4f8c4a9a46f18a6.svg" alt="n_i">，
有效范围为 <img class="math" src="/static/images/math/e2ed5018d325d2eed8f60b87169eef062d34e19e.svg" alt="0 \le n_i < d_i">，
其中 <img class="math" src="/static/images/math/93f9b5333c1aa73b70b1a61bf3bfbb88be44ecf2.svg" alt="d_i"> 是数组形状的第i个元素。
负索引被解释为从数组的末尾开始计数(即，如果 <img class="math" src="/static/images/math/c51e6b77da6dcf80b0afd223631451edab2ffcba.svg" alt="n_i < 0"> ，则表示 <img class="math" src="/static/images/math/d3f3fd2a242879a858674e0a3983f119d6265236.svg" alt="n_i + d_i">)。

基本切片生成的所有数组始终
是原始数组的[视图](https://numpy.org/devdocs/glossary.html#term-view)。

::: tip 注意

NumPy切片创建[视图](https://numpy.org/devdocs/glossary.html#term-view)而不是复制，就像内置Python序列（如string，tuple和list）一样。从大数组中提取一小部分时必须小心，这在提取后变得无用，因为提取的小部分包含对大原始数组的引用，其内存将不会被释放，直到从其派生的所有数组被垃圾收集。在这种情况下，``copy()``建议使用明确的。

:::

序列切片的标准规则适用于基于每维的基本切片（包括使用步骤索引）。要记住的一些有用的概念包括：

- 基本切片语法是 ``i:j:k``，其中 *i* 是起始索引，*j* 是停止索引，*k* 是步骤（<img class="math" src="/static/images/math/3bf28764fb6d6c43a60af989993514e85c51a308.svg" alt="k\neq0">）。这将选择具有索引值（在相应的维度中）*i, i+k, ..., i+(m-1) k* 的 *m* 个元素，其中 <img class="math" src="/static/images/math/a12ce497cac8b44aac74086b97925bdf76d51292.svg" alt="m = q + (r\neq0)">，*q* 和 *r* 是 *j-i* 除以 *k* 得到的商和余数：*j - i = q k + r*，使得*i + ( m - 1 ) k < j*。

    **示例：**

    ``` python
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> x[1:7:2]
    array([1, 3, 5])
    ```

- 负 *i* 和 *j* 被解释为 *n + i* 和 *n + j* ，其中
  *n* 是相应维度中的元素数量。负 *k* 使得踩踏指向更小的指数。

    **示例：**

    ``` python
    >>> x[-2:10]
    array([8, 9])
    >>> x[-3:3:-1]
    array([7, 6, 5, 4])
    ```

- 假设n是被切片的维度中的元素的数量。
  然后，如果没有给出 *i*，则对于 *k* > *0* ，它默认为 *0*，对于 *k* < *0*，它默认为 *n* - *1*。
  如果没有给出*j*，则对于*k* > *0*，它默认为*n*，对于*k* < *0*，则默认为 - *n* - *1*。
  如果没有给定*k*，则默认为*1*。请注意 ``::`` 与 ``:`` 相同，表示选择沿此轴的所有索引。

    **示例：**

    ``` python
    >>> x[5:]
    array([5, 6, 7, 8, 9])
    ```

- 如果选择元组中的对象数小于
  *N* ，则 ``:`` 假定任何后续维。

    **示例：**

    ``` python
    >>> x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
    >>> x.shape
    (2, 3, 1)
    >>> x[1:2]
    array([[[4],
            [5]
            [6]]])
    ```
 
- ``Ellipsis``扩展``:``为选择元组索引所有维度所需的对象数。在大多数情况下，这意味着扩展选择元组的长度是``x.ndim``。可能只存在一个省略号。

    **示例：**

    ``` python
    >>> x[...,0]
    array([[1, 2, 3],
          [4, 5, 6]])
    ```

- [``newaxis``](constants.html#numpy.newaxis)选择元组中的每个对象用于将所得选择的维度扩展一个单位长度维度。添加的维度是[``newaxis``](constants.html#numpy.newaxis)
对象在选择元组中的位置。

    **示例：**

    ``` python
    >>> x[:,np.newaxis,:,:].shape
    (2, 1, 3, 1)
    ```

- 整数 *i* 返回相同的值，``i:i+1`` **除了**返回的对象的维度减少1.特别是，具有第 *p* 个元素的整数（和所有其他条目``:``）的选择元组返回具有维度的相应子数组 *N  -  1* 。如果 *N = 1，*  
则返回的对象是数组标量。[Scalars](scalars.html)中解释了这些对象。
- 如果选择元组具有所有条目 ``:`` 除了第p个条目是切片对象 ``i:j:k``，那么返回的数组具有通过连接通过元素i，i+k，… 的整数索引返回的子数组而形成的维数N。i + ( m  - 1 ) k < j 。
- 在切片元组中具有多于一个非``:``条目的基本切片的作用类似于使用单个非``:``条目重复应用切片，其中连续获取非``:``条目(所有其他非``:``条目被替换为``:``)。因此，``x[ind1,...,ind2,:]`` 在基本切片下的作用类似于 ``x[ind1][...,ind2,:]``。

::: danger 警告

对于高级索引，上述情况**并非**如此。

:::

- 您可以使用切片来设置数组中的值，但(与列表不同)您永远不能扩大数组。要在 ``x[obj] = value`` 中设置的值的大小必须(可广播)为与 ``x[obj]`` 相同的形状。

::: tip 注意

请记住，切片元组始终可以构造为obj并在 ``x[obj]`` 表示法中使用。
可以在构造中使用切片对象来代替 ``[START：STOP：STEP]`` 表示法。
例如，``x[1：10：5，：-1]`` 也可以实现为 ``obj = (Slice(1，10，5)，Slice(None，-1)；x[obj]``。
这对于构造对任意维数的数组起作用的泛型代码很有用。

:::

- ``numpy.newaxis``

    该[``newaxis``](constants.html#numpy.newaxis)对象可用于所有切片操作，以创建长度为1的轴。[``newaxis``](constants.html#numpy.newaxis)是'None'的别名，'None'可以用来代替相同的结果。

## 高级索引

当选择对象 *obj* 是非元组序列对象，[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)（数据类型为整数或bool）或具有至少一个序列对象或ndarray（数据类型为integer或bool）的元组时，将触发高级索引。高级索引有两种类型：整数和布尔值。

高级索引始终返回数据的 *副本* （与返回[视图的](https://numpy.org/devdocs/glossary.html#term-view)基本切片形成对比）。

::: danger 警告

高级索引的定义意味着``x[(1,2,3),]``根本不同于``x[(1,2,3)]``。后者相当于``x[1,2,3]``触发基本选择，而前者将触发高级索引。一定要明白为什么会这样。

同时认识到``x[[1,2,3]]``将触发高级索引，而由于上面提到的不推荐的数字兼容性，
 ``x[[1,2,slice(None)]]``将触发基本切片。

:::

### 整数数组索引

整数数组索引允许根据数组的 *N* 维索引选择数组中的任意项。每个整数数组表示该维度的许多索引。

#### 纯整数数组索引

当索引包含尽可能多的整数数组时，索引的数组具有维度，索引是直接的，但与切片不同。

高级索引始终作为 *一个* 整体进行 [广播](/reference/ufuncs.html#广播) 和迭代：

``` python
result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
                           ..., ind_N[i_1, ..., i_M]]
```

请注意，结果形状与（广播）索引数组形状 ``ind_1, ..., ind_N`` 相同。

**示例：**

应从每一行中选择特定的元素。
行索引只是``[0，1，2]``，列索引指定要为相应行选择的元素，这里是``[0，1，0]``。
将两者结合使用，可以使用高级索引解决任务：

``` python
>>> x = np.array([[1, 2], [3, 4], [5, 6]])
>>> x[[0, 1, 2], [0, 1, 0]]
array([1, 4, 5])
```

为了实现类似于上面的基本切片的行为，可以使用广播。该功能[``ix_``](https://numpy.org/devdocs/reference/generated/numpy.ix_.html#numpy.ix_)可以帮助这种广播。通过示例可以最好地理解这一点。

**示例：**

应使用高级索引从4x3数组中选择角元素。
因此，需要选择列是 ``[0，2]`` 中的一个，行是 ``[0，3]`` 中的一个的所有元素。
要使用高级索引，需要*显式*  选择所有元素。使用前面解释的方法，可以写：

``` python
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

然而，由于上面的索引数组只是重复它们自己，所以可以使用广播（比较操作，如 ``rows[:, np.newaxis] + columns``）来简化这一点：

``` python
>>> rows = np.array([0, 3], dtype=np.intp)
>>> columns = np.array([0, 2], dtype=np.intp)
>>> rows[:, np.newaxis]
array([[0],
       [3]])
>>> x[rows[:, np.newaxis], columns]
array([[ 0,  2],
       [ 9, 11]])
```

这种广播也可以使用 [``ix_``](https://numpy.org/devdocs/reference/generated/numpy.ix_.html#numpy.ix_)： 功能来实现。

``` python
>>> x[np.ix_(rows, columns)]
array([[ 0,  2],
       [ 9, 11]])
```

请注意，如果没有``np.ix_``调用，只会选择对角线元素，如上例所示。对于使用多个高级索引进行索引，这个差异是最重要的。

#### 结合高级索引和基本索引

当至少有一个slice（``:``），省略号（``...``）或[``newaxis``](/reference/constants.html#numpy.newaxis)
索引（或者数组的维度多于高级索引）时，行为可能会更复杂。这就像连接每个高级索引元素的索引结果一样

在最简单的情况下，只有一个 *单一的* 指标先进。单个高级索引可以例如替换切片，并且结果数组将是相同的，但是，它是副本并且可以具有不同的存储器布局。当可能时，切片是优选的。

**示例：**

``` python
>>> x[1:2, 1:3]
array([[4, 5]])
>>> x[1:2, [1, 2]]
array([[4, 5]])
```

了解情况的最简单方法可能是考虑结果形状。索引操作分为两部分，即由基本索引（不包括整数）定义的子空间和来自高级索引部分的子空间。需要区分两种索引组合：

- 高级索引由切片分隔，``Ellipsis``或[``newaxis``](/reference/constants.html#numpy.newaxis)。例如。``x[arr1, :, arr2]``
- 高级索引都是相邻的。例如 ``x[..., arr1, arr2, :]``，但不是 ``x[arr1, :, 1]``，因为 ``1`` 是这方面的高级索引。

在第一种情况下，高级索引操作产生的维度首先出现在结果数组中，然后是子空间维度。
在第二种情况下，高级索引操作的维度将插入到结果数组中与初始数组中相同的位置（后一种逻辑使简单的高级索引行为就像切片一样）。

**示例：**

假设 ``x.shape`` 是(10，20，30)，
并且 ``ind`` 是（2,3,4）形状的索引 ``intp`` 数组，
那么 ``result = x[..., ind, : ]`` 具有形状(10，2，3，4，30)，
因为(20，)形状子空间已经被(2，3，4)形状的广播索引子空间所取代。
如果我们让i，j，k在(2，3，4)形状子空间上循环，则结果 ``result[...,i,j,k,:] = x[...,ind[i,j,k],:]`` 。
本例产生的结果与 ``x.take(ind，axis=-2)`` 相同。

设``x.shape``（10,20,30,40,50）并假设``ind_1``
并``ind_2``可以广播到形状（2,3,4）。然后
 ``x[:,ind_1,ind_2]``具有形状（10,2,3,4,40,50），因为来自X的（20,30）形子空间已经被索引的（2,3,4）子空间替换。但是，它
 ``x[:,ind_1,:,ind_2]``具有形状（2,3,4,10,30,50），因为在索引子空间中没有明确的位置，所以它在开头就被添加了。始终可以使用
 [``.transpose()``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.transpose.html#numpy.ndarray.transpose)在任何需要的位置移动子空间。
 请注意，此示例无法使用复制[``take``](https://numpy.org/devdocs/reference/generated/numpy.take.html#numpy.take)。

### 布尔数组索引

当obj是Boolean类型的数组对象(例如可以从比较运算符返回)时，
会发生这种高级索引。
单个布尔索引数组实际上与 ``x[obj.nonzero()]`` 相同，
其中，如上所述，``obj.nonzero()`` 返回整数索引数组的元组(长度为 ``obj.ndim``)，
显示obj的 ``True`` 元素。但是，当 ``obj.shape == x.shape`` 时，它会更快。

如果 ``obj.ndim == x.ndim``，``x[obj]`` 返回一个1维数组，
其中填充了与obj的 ``True`` 值对应的x的元素。
搜索顺序为[ row-major](https://numpy.org/devdocs/glossary.html#term-row-major)，
C样式。如果 *obj* 在 *x* 的界限之外的条目上有True值，
则会引发索引错误。如果 *obj* 小于 *x* ，则等同于用**False**填充它。

**示例：**

一个常见的用例是过滤所需的元素值。例如，可能希望从数组中选择非NaN的所有条目：

``` python
>>> x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
>>> x[~np.isnan(x)]
array([ 1.,  2.,  3.])
```

或者希望为所有负面元素添加常量：

``` python
>>> x = np.array([1., -1., -2., 3])
>>> x[x < 0] += 20
>>> x
array([  1.,  19.,  18.,   3.])
```

通常，如果索引包含布尔数组，则结果将与将 ``obj.nonzero()`` 插入到相同位置并使用上述整数组索引机制相同。
``x[ind_1，boolean_array，ind_2]`` 等价于 ``x[(ind_1，)+boolean_array.nonzero()+(ind_2，)]``。

如果只有一个布尔数组且没有整数索引数组，则这是直截了当的。必须注意确保布尔索引具有与其应该使用的维度 *完全相同的* 维度。

**示例：**

从数组中，选择总和小于或等于2的所有行：

``` python
>>> x = np.array([[0, 1], [1, 1], [2, 2]])
>>> rowsum = x.sum(-1)
>>> x[rowsum <= 2, :]
array([[0, 1],
       [1, 1]])
```

但如果``rowsum``还有两个维度：

``` python
>>> rowsum = x.sum(-1, keepdims=True)
>>> rowsum.shape
(3, 1)
>>> x[rowsum <= 2, :]    # fails
IndexError: too many indices
>>> x[rowsum <= 2]
array([0, 1])
```

由于额外的维度，最后一个只给出了第一个元素。比较``rowsum.nonzero()``以了解此示例。

通过[``obj.nonzero()``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.nonzero.html#numpy.ndarray.nonzero)类比可以最好地理解组合多个布尔索引数组或布尔与整数索引数组
 。该函数[``ix_``](https://numpy.org/devdocs/reference/generated/numpy.ix_.html#numpy.ix_)
还支持布尔数组，并且可以毫无意外地工作。

**示例：**

使用布尔索引选择加起来为偶数的所有行。同时，应使用高级整数索引选择列0和2。使用该[``ix_``](https://numpy.org/devdocs/reference/generated/numpy.ix_.html#numpy.ix_)功能可以通过以下方式完成：

``` python
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

没有``np.ix_``呼叫或只选择对角线元素。

或者没有``np.ix_``（比较整数数组示例）：

``` python
>>> rows = rows.nonzero()[0]
>>> x[rows[:, np.newaxis], columns]
array([[ 3,  5],
       [ 9, 11]])
```

## 详细说明

这些是一些详细的注释，对于日常索引（无特定顺序）并不重要：

- 本机NumPy索引类型``intp``可能与默认的整数数组类型不同。``intp``是足以安全索引任何数组的最小数据类型; 对于高级索引，它可能比其他类型更快。
- 对于高级分配，通常不保证迭代顺序。这意味着如果元素设置不止一次，则无法预测最终结果。
- 空（元组）索引是零维数组的完整标量索引。
如果是零维则``x[()]``返回 *标量，* 否则返回``x``视图。另一方面，``x[...]``总是返回一个视图。
- 如果索引中存在零维数组 *并且* 它是完整的整数索引，则结果将是 *标量* 而不是零维数组。（不会触发高级索引。）
- 当省略号（``...``）存在但没有大小（即替换为零
 ``:``）时，结果仍将始终为数组。如果没有高级索引，则为视图，否则为副本。
- ``nonzero``布尔数组的等价性不适用于零维布尔数组。
- 当高级索引操作的结果没有元素但单个索引超出界限时，是否引发 ``IndexError`` 是未定义的(例如，``x[[], [123]]`` 中的 ``123`` 超出界限)。
- 当在赋值期间发生 *转换* 错误时（例如，使用字符串序列更新数值数组），被分配的数组可能最终处于不可预测的部分更新状态。但是，如果发生任何其他错误（例如超出范围索引），则数组将保持不变。
- 高级索引结果的内存布局针对每个索引操作进行了优化，并且不能假设特定的内存顺序。
- 当使用一个子类（尤其是其操纵它的形状），默认``ndarray.__setitem__``行为会调用``__getitem__``的
  *基本* 索引而不是 *先进的* 索引。对于这样的子类，最好``ndarray.__setitem__``使用 *基类*  ndarray视图调用数据。如果子类不返回视图，则 *必须* 执行此操作``__getitem__``。

## 字段形式访问

::: tip 另见

[数据类型对象（dtype）](dtypes.html)、
[标量](scalars.html)

:::

如果[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)对象是结构化数组
，则可以通过使用字符串索引数组来访问数组的[字段](https://numpy.org/devdocs/glossary.html#term-field)，类似于字典。

索引``x['field-name']``返回数组的新[视图](https://numpy.org/devdocs/glossary.html#term-view)，该[视图](https://numpy.org/devdocs/glossary.html#term-view)与 *x* 具有相同的形状（当字段是子数组时除外）但是数据类型``x.dtype['field-name']``并且仅包含指定字段中的部分数据。还
 [记录数组](arrays.classes.html#arrays-classes-rec)标量可以被“索引”这种方式。

索引到结构化数组也可以使用字段名称列表来完成，
*例如*  ``x[['field-name1','field-name2']]``。
从NumPy 1.16开始，这将返回仅包含这些字段的视图。
在旧版本的numpy中，它返回了一个副本。
有关多字段索引的详细信息，
请参阅用户指南部分的[结构化数组](/user/basics/rec.html#结构化数组)。

如果访问的字段是子数组，则子数组的尺寸将附加到结果的形状。

``` python
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

## Flat Iterator索引

[``x.flat``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat)返回一个迭代器，它将遍历整个数组（以C-contiguous样式，最后一个索引变化最快）。只要选择对象不是元组，也可以使用基本切片或高级索引对此迭代器对象建立索引。这应该从[``x.flat``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat)一维视图的事实中清楚。它可以用于具有1维C风格平面索引的整数索引。因此，任何返回数组的形状都是整数索引对象的形状。
