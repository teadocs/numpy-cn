==================================
快速入门教程
==================================

----------------------------------
先决条件
----------------------------------

在阅读本教程之前，你应该了解一些Python的基础知识。如果你想复习一下，请回去看看Python教程。

如果你希望使用本教程中的示例，则还必须在计算机上安装一些软件。 有关说明，请参阅本指南的安装教程。

----------------------------------
基础知识
----------------------------------

NumPy的主要对象是同类型的多维数组。它是一张表，所有元素（通常是数字）的类型都相同，并通过正整数元组索引。在NumPy中，维度称为轴。轴的数目为rank。

例如，3D空间中的点的坐标 ``[1, 2, 1]`` 是rank为1的数组，因为它具有一个轴。该轴的长度为3。在下面的示例中，该数组有2个轴。
第一个轴（维度）的长度为2，第二个轴（维度）的长度为3。

.. code-block:: python

    [[ 1., 0., 0.],
    [ 0., 1., 2.]]

NumPy的数组类被称为ndarray。别名为 ``array``。 请注意，``numpy.array`` 与标准Python库类 ``array.array`` 不同，后者仅处理一维数组并提供较少的功能。 ``ndarray`` 对象则提供更关键的属性：

* **ndarray.ndim**：数组的轴（维度）的个数。在Python世界中，维度的数量被称为rank。
* **ndarray.shape**：数组的维度。这是一个整数的元组，表示每个维度中数组的大小。对于有n行和m列的矩阵，shape将是(n,m)。因此，``shape``元组的长度就是rank或维度的个数 ``ndim``。
* **ndarray.size**：数组元素的总数。这等于shape的元素的乘积。
* **ndarray.dtype**：一个描述数组中元素类型的对象。可以使用标准的Python类型创建或指定dtype。另外NumPy提供它自己的类型。例如numpy.int32、numpy.int16和numpy.float64。
* **ndarray.itemsize**：数组中每个元素的字节大小。例如，元素为 ``float64`` 类型的数组的 ``itemsize`` 为8（=64/8），而 ``complex32`` 类型的数组的 ``itemsize`` 为4（=32/8）。它等于 ``ndarray.dtype.itemsize`` 。
* **ndarray.data**：该缓冲区包含数组的实际元素。通常，我们不需要使用此属性，因为我们将使用索引访问数组中的元素。

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
一个典型的例子
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> import numpy as np
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> a.shape
    (3, 5)
    >>> a.ndim
    2
    >>> a.dtype.name
    'int64'
    >>> a.itemsize
    8
    >>> a.size
    15
    >>> type(a)
    <type 'numpy.ndarray'>
    >>> b = np.array([6, 7, 8])
    >>> b
    array([6, 7, 8])
    >>> type(b)
    <type 'numpy.ndarray'>

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
数组的创建
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有几种创建数组的方法。

例如，你可以使用array函数从常规Python列表或元组中创建数组。得到的数组的类型是从Python列表中元素的类型推导出来的。

.. code-block:: python

    >>> import numpy as np
    >>> a = np.array([2,3,4])
    >>> a
    array([2, 3, 4])
    >>> a.dtype
    dtype('int64')
    >>> b = np.array([1.2, 3.5, 5.1])
    >>> b.dtype
    dtype('float64')

一个常见的错误在于使用多个数值参数调用 ``array`` 函数，而不是提供一个数字列表（List）作为参数。

.. code-block:: python

    >>> a = np.array(1,2,3,4)    # WRONG
    >>> a = np.array([1,2,3,4])  # RIGHT

``array`` 将序列的序列转换成二维数组，将序列的序列的序列转换成三维数组，等等。

.. code-block:: python

    >>> b = np.array([(1.5,2,3), (4,5,6)])
    >>> b
    array([[ 1.5,  2. ,  3. ],
           [ 4. ,  5. ,  6. ]])

数组的类型也可以在创建时明确指定：

.. code-block:: python

    >>> c = np.array( [ [1,2], [3,4] ], dtype=complex )
    >>> c
    array([[ 1.+0.j,  2.+0.j],
           [ 3.+0.j,  4.+0.j]])

通常，数组的元素最初是未知的，但它的大小是已知的。因此，NumPy提供了几个函数来创建具有初始占位符内容的数组。这就减少了数组增长的必要，因为数组增长的操作花费很大。

函数 ``zeros`` 创建一个由0组成的数组，函数 ``ones`` 创建一个由1数组的数组，函数 ``empty`` 内容是随机的并且取决于存储器的状态。默认情况下，创建的数组的dtype是 ``float64``。

.. code-block:: python

    >>> np.zeros( (3,4) )
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])
    >>> np.ones( (2,3,4), dtype=np.int16 )                # dtype can also be specified
    array([[[ 1, 1, 1, 1],
            [ 1, 1, 1, 1],
            [ 1, 1, 1, 1]],
           [[ 1, 1, 1, 1],
            [ 1, 1, 1, 1],
            [ 1, 1, 1, 1]]], dtype=int16)
    >>> np.empty( (2,3) )                                 # uninitialized, output may vary
    array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
           [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
    
要创建数字序列，NumPy提供了一个类似于 ``range`` 的函数，该函数返回数组而不是列表。

.. code-block:: python

    >>> np.arange( 10, 30, 5 )
    array([10, 15, 20, 25])
    >>> np.arange( 0, 2, 0.3 )                 # it accepts float arguments
    array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])

当 ``arange`` 与浮点参数一起使用时，由于浮点数的精度是有限的，通常不可能预测获得的元素数量。出于这个原因，通常最好使用函数 ``linspace`` ，它接收我们想要的元素数量而不是步长作为参数：

.. code-block:: python

    >>> from numpy import pi
    >>> np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
    >>> x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
    >>> f = np.sin(x)


另见：

    `array <http://#>`_, zeros, zeros_like, ones, ones_like, empty, empty_like, arange, linspace, numpy.random.rand, numpy.random.randn, fromfunction, fromfile

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
打印数组
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当你打印数组时，NumPy以与嵌套列表类似的方式显示它，但是具有以下布局：

* 最后一个轴从左到右打印，
* 倒数第二个从上到下打印，
* 其余的也从上到下打印，每个切片与下一个用空行分开。

一维数组被打印为行、二维为矩阵和三维为矩阵列表。

.. code-block:: python

	>>> a = np.arange(6)                         # 1d array
	>>> print(a)
	[0 1 2 3 4 5]
	>>>
	>>> b = np.arange(12).reshape(4,3)           # 2d array
	>>> print(b)
	[[ 0  1  2]
	 [ 3  4  5]
	 [ 6  7  8]
	 [ 9 10 11]]
	>>>
	>>> c = np.arange(24).reshape(2,3,4)         # 3d array
	>>> print(c)
	[[[ 0  1  2  3]
	  [ 4  5  6  7]
	  [ 8  9 10 11]]
	 [[12 13 14 15]
	  [16 17 18 19]
	  [20 21 22 23]]]

有关 ``reshape`` 的详情，请参阅下文。

如果数组太大而无法打印，NumPy将自动跳过数组的中心部分并仅打印角点：

.. code-block:: python

    >>> print(np.arange(10000))
    [   0    1    2 ..., 9997 9998 9999]
    >>>
    >>> print(np.arange(10000).reshape(100,100))
    [[   0    1    2 ...,   97   98   99]
     [ 100  101  102 ...,  197  198  199]
     [ 200  201  202 ...,  297  298  299]
     ...,
     [9700 9701 9702 ..., 9797 9798 9799]
     [9800 9801 9802 ..., 9897 9898 9899]
     [9900 9901 9902 ..., 9997 9998 9999]]

要禁用此行为并强制NumPy打印整个数组，你可以使用 ``set_printoptions`` 更改打印选项。

.. code-block:: python

    >>> np.set_printoptions(threshold=np.nan)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
基本操作
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
数组上的算术运算符使用元素级别。一个新的数组被创建并填充结果。

.. code-block:: python

    >>> a = np.array( [20,30,40,50] )
    >>> b = np.arange( 4 )
    >>> b
    array([0, 1, 2, 3])
    >>> c = a-b
    >>> c
    array([20, 29, 38, 47])
    >>> b**2
    array([0, 1, 4, 9])
    >>> 10*np.sin(a)
    array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
    >>> a<35
    array([ True, True, False, False])

Unlike in many matrix languages, the product operator ``*`` operates elementwise in NumPy arrays. The matrix product can be performed using the ``dot`` function or method:

.. code-block:: python

    >>> A = np.array( [[1,1],
    ...             [0,1]] )
    >>> B = np.array( [[2,0],
    ...             [3,4]] )
    >>> A*B                         # elementwise product
    array([[2, 0],
           [0, 4]])
    >>> A.dot(B)                    # matrix product
    array([[5, 4],
           [3, 4]])
    >>> np.dot(A, B)                # another matrix product
    array([[5, 4],
           [3, 4]])

Some operations, such as += and \*=, act in place to modify an existing array rather than create a new one.

.. code-block:: python

    >>> a = np.ones((2,3), dtype=int)
    >>> b = np.random.random((2,3))
    >>> a *= 3
    >>> a
    array([[3, 3, 3],
           [3, 3, 3]])
    >>> b += a
    >>> b
    array([[ 3.417022  ,  3.72032449,  3.00011437],
           [ 3.30233257,  3.14675589,  3.09233859]])
    >>> a += b                  # b is not automatically converted to integer type
    Traceback (most recent call last):
      ...
    TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'

When operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one (a behavior known as upcasting).

.. code-block:: python

    >>> a = np.ones(3, dtype=np.int32)
    >>> b = np.linspace(0,pi,3)
    >>> b.dtype.name
    'float64'
    >>> c = a+b
    >>> c
    array([ 1.        ,  2.57079633,  4.14159265])
    >>> c.dtype.name
    'float64'
    >>> d = np.exp(c*1j)
    >>> d
    array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
           -0.54030231-0.84147098j])
    >>> d.dtype.name
    'complex128'

Many unary operations, such as computing the sum of all the elements in the array, are implemented as methods of the ``ndarray`` class.

.. code-block:: python

    >>> a = np.random.random((2,3))
    >>> a
    array([[ 0.18626021,  0.34556073,  0.39676747],
           [ 0.53881673,  0.41919451,  0.6852195 ]])
    >>> a.sum()
    2.5718191614547998
    >>> a.min()
    0.1862602113776709
    >>> a.max()
    0.6852195003967595

By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. However, by specifying the axis parameter you can apply an operation along the specified axis of an array:

.. code-block:: python

    >>> b = np.arange(12).reshape(3,4)
    >>> b
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>>
    >>> b.sum(axis=0)                            # sum of each column
    array([12, 15, 18, 21])
    >>>
    >>> b.min(axis=1)                            # min of each row
    array([0, 4, 8])
    >>>
    >>> b.cumsum(axis=1)                         # cumulative sum along each row
    array([[ 0,  1,  3,  6],
           [ 4,  9, 15, 22],
           [ 8, 17, 27, 38]])

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Universal Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called “universal functions”( ``ufunc`` ). Within NumPy, these functions operate elementwise on an array, producing an array as output.

.. code-block:: python

    >>> B = np.arange(3)
    >>> B
    array([0, 1, 2])
    >>> np.exp(B)
    array([ 1.        ,  2.71828183,  7.3890561 ])
    >>> np.sqrt(B)
    array([ 0.        ,  1.        ,  1.41421356])
    >>> C = np.array([2., -1., 4.])
    >>> np.add(B, C)
    array([ 2.,  0.,  6.])

另见：

    all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Indexing, Slicing and Iterating
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**One-dimensional** arrays can be indexed, sliced and iterated over, much like lists and other Python sequences.

.. code-block:: python

    >>> a = np.arange(10)**3
    >>> a
    array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
    >>> a[2]
    8
    >>> a[2:5]
    array([ 8, 27, 64])
    >>> a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
    >>> a
    array([-1000,     1, -1000,    27, -1000,   125,   216,   343,   512,   729])
    >>> a[ : :-1]                                 # reversed a
    array([  729,   512,   343,   216,   125, -1000,    27, -1000,     1, -1000])
    >>> for i in a:
    ...     print(i**(1/3.))
    ...
    nan
    1.0
    nan
    3.0
    nan
    5.0
    6.0
    7.0
    8.0
    9.0

**Multidimensional** arrays can have one index per axis. These indices are given in a tuple separated by commas:

.. code-block:: python

    >>> def f(x,y):
    ...     return 10*x+y
    ...
    >>> b = np.fromfunction(f,(5,4),dtype=int)
    >>> b
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23],
           [30, 31, 32, 33],
           [40, 41, 42, 43]])
    >>> b[2,3]
    23
    >>> b[0:5, 1]                       # each row in the second column of b
    array([ 1, 11, 21, 31, 41])
    >>> b[ : ,1]                        # equivalent to the previous example
    array([ 1, 11, 21, 31, 41])
    >>> b[1:3, : ]                      # each column in the second and third row of b
    array([[10, 11, 12, 13],
           [20, 21, 22, 23]])

When fewer indices are provided than the number of axes, the missing indices are considered complete slices ``:``

.. code-block:: python

    >>> b[-1]                                  # the last row. Equivalent to b[-1,:]
    array([40, 41, 42, 43])

The expression within brackets in ``b[i]`` is treated as an ``i`` followed by as many instances of ``:`` as needed to represent the remaining axes. NumPy also allows you to write this using dots as ``b[i,...]``.

The **dots** (``...``) represent as many colons as needed to produce a complete indexing tuple. For example, if ``x`` is an array with 5 axes, then

* ``x[1,2,...]`` is equivalent to ``x[1,2,:,:,:]``,
* ``x[...,3]`` to ``x[:,:,:,:,3]`` and
* ``x[4,...,5,:]`` to ``x[4,:,:,5,:]``.

.. code-block:: python

    >>> c = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
    ...                 [ 10, 12, 13]],
    ...                [[100,101,102],
    ...                 [110,112,113]]])
    >>> c.shape
    (2, 2, 3)
    >>> c[1,...]                                   # same as c[1,:,:] or c[1]
    array([[100, 101, 102],
           [110, 112, 113]])
    >>> c[...,2]                                   # same as c[:,:,2]
    array([[  2,  13],
           [102, 113]])

**Iterating** over multidimensional arrays is done with respect to the first axis:

.. code-block:: python

    >>> for row in b:
    ...     print(row)
    ...
    [0 1 2 3]
    [10 11 12 13]
    [20 21 22 23]
    [30 31 32 33]
    [40 41 42 43]

However, if one wants to perform an operation on each element in the array, one can use the flat attribute which is an iterator over all the elements of the array:

.. code-block:: python

    >>> for element in b.flat:
    ...     print(element)
    ...
    0
    1
    2
    3
    10
    11
    12
    13
    20
    21
    22
    23
    30
    31
    32
    33
    40
    41
    42
    43

另见：

    Indexing, Indexing (reference), newaxis, ndenumerate, indices

----------------------------------
Shape Manipulation
----------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changing the shape of an array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An array has a shape given by the number of elements along each axis:

.. code-block:: python

    >>> a = np.floor(10*np.random.random((3,4)))
    >>> a
    array([[ 2.,  8.,  0.,  6.],
           [ 4.,  5.,  1.,  1.],
           [ 8.,  9.,  3.,  6.]])
    >>> a.shape
    (3, 4)

The shape of an array can be changed with various commands. Note that the following three commands all return a modified array, but do not change the original array:

.. code-block:: python

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

The order of the elements in the array resulting from ravel() is normally “C-style”, that is, the rightmost index “changes the fastest”, so the element after a[0,0] is a[0,1]. If the array is reshaped to some other shape, again the array is treated as “C-style”. NumPy normally creates arrays stored in this order, so ravel() will usually not need to copy its argument, but if the array was made by taking slices of another array or created with unusual options, it may need to be copied. The functions ravel() and reshape() can also be instructed, using an optional argument, to use FORTRAN-style arrays, in which the leftmost index changes the fastest.

The ``reshape`` function returns its argument with a modified shape, whereas the ``ndarray.resize`` method modifies the array itself:

.. code-block:: python

    >>> a
    array([[ 2.,  8.,  0.,  6.],
           [ 4.,  5.,  1.,  1.],
           [ 8.,  9.,  3.,  6.]])
    >>> a.resize((2,6))
    >>> a
    array([[ 2.,  8.,  0.,  6.,  4.,  5.],
           [ 1.,  1.,  8.,  9.,  3.,  6.]])

If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated:>>> a.reshape(3,-1)

.. code-block:: python

    array([[ 2.,  8.,  0.,  6.],
           [ 4.,  5.,  1.,  1.],
           [ 8.,  9.,  3.,  6.]])

另见：

    ndarray.shape, reshape, resize, ravel

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Stacking together different arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several arrays can be stacked together along different axes:

.. code-block:: python

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

he function ``column_stack`` stacks 1D arrays as columns into a 2D array. It is equivalent to ``hstack`` only for 2D arrays:

.. code-block:: python

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

On the other hand, the function ``row_stack`` is equivalent to ``vstack`` for any input arrays. In general, for arrays of with more than two dimensions, ``hstack`` stacks along their second axes, ``vstack`` stacks along their first axes, and ``concatenate`` allows for an optional arguments giving the number of the axis along which the concatenation should happen.

**Note**

In complex cases, ``r_`` and ``c_`` are useful for creating arrays by stacking numbers along one axis. They allow the use of range literals (“:”)

.. code-block:: python

    >>> np.r_[1:4,0,4]
    array([1, 2, 3, 0, 4])

When used with arrays as arguments, ``r_`` and ``c_`` are similar to ``vstack`` and ``hstack`` in their default behavior, but allow for an optional argument giving the number of the axis along which to concatenate.

另见：

    hstack, vstack, column_stack, concatenate, c\_, r\_

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Splitting one array into several smaller ones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``hsplit``, you can split an array along its horizontal axis, either by specifying the number of equally shaped arrays to return, or by specifying the columns after which the division should occur:

.. code-block:: python

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

``vsplit`` splits along the vertical axis, and ``array_split`` allows one to specify along which axis to split.

----------------------------------
Copies and Views
----------------------------------

When operating and manipulating arrays, their data is sometimes copied into a new array and sometimes not. This is often a source of confusion for beginners. There are three cases:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
No Copy at All
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simple assignments make no copy of array objects or of their data.

.. code-block:: python

    >>> a = np.arange(12)
    >>> b = a            # no new object is created
    >>> b is a           # a and b are two names for the same ndarray object
    True
    >>> b.shape = 3,4    # changes the shape of a
    >>> a.shape
    (3, 4)

Python passes mutable objects as references, so function calls make no copy.

.. code-block:: python

    >>> def f(x):
    ...     print(id(x))
    ...
    >>> id(a)                           # id is a unique identifier of an object
    148293216
    >>> f(a)
    148293216

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
View or Shallow Copy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different array objects can share the same data. The ``view`` method creates a new array object that looks at the same data.

.. code-block:: python

    >>> c = a.view()
    >>> c is a
    False
    >>> c.base is a                        # c is a view of the data owned by a
    True
    >>> c.flags.owndata
    False
    >>>
    >>> c.shape = 2,6                      # a's shape doesn't change
    >>> a.shape
    (3, 4)
    >>> c[0,4] = 1234                      # a's data changes
    >>> a
    array([[   0,    1,    2,    3],
           [1234,    5,    6,    7],
           [   8,    9,   10,   11]])

Slicing an array returns a view of it:

.. code-block:: python

    >>> s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
    >>> s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
    >>> a
    array([[   0,   10,   10,    3],
           [1234,   10,   10,    7],
           [   8,   10,   10,   11]])

----------------------------------
Deep Copy
----------------------------------

The copy method makes a complete copy of the array and its data.

.. code-block:: python

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


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Functions and Methods Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a list of some useful NumPy functions and methods names ordered in categories. See Routines for the full list.

Array Creation
    arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like

Conversions
    ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat

anipulations
    array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack

Questions
    all, any, nonzero, where

Ordering
    argmax, argmin, argsort, max, min, ptp, searchsorted, sort

Operations
    choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum

Basic Statistics
    cov, mean, std, var

Basic Linear Algebra
    cross, dot, outer, linalg.svd, vdot

----------------------------------
Less Basic
----------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Broadcasting rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Broadcasting allows universal functions to deal in a meaningful way with inputs that do not have exactly the same shape.

The first rule of broadcasting is that if all input arrays do not have the same number of dimensions, a “1” will be repeatedly prepended to the shapes of the smaller arrays until all the arrays have the same number of dimensions.

The second rule of broadcasting ensures that arrays with a size of 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension. The value of the array element is assumed to be the same along that dimension for the “broadcast” array.

After application of the broadcasting rules, the sizes of all arrays must match. More details can be found in Broadcasting.

----------------------------------
Fancy indexing and index tricks
----------------------------------

NumPy offers more indexing facilities than regular Python sequences. In addition to indexing by integers and slices, as we saw before, arrays can be indexed by arrays of integers and arrays of booleans.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Indexing with Arrays of Indices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> a = np.arange(12)**2                       # the first 12 square numbers
    >>> i = np.array( [ 1,1,3,8,5 ] )              # an array of indices
    >>> a[i]                                       # the elements of a at the positions i
    array([ 1,  1,  9, 64, 25])
    >>>
    >>> j = np.array( [ [ 3, 4], [ 9, 7 ] ] )      # a bidimensional array of indices
    >>> a[j]                                       # the same shape as j
    array([[ 9, 16],
           [81, 49]])

When the indexed array ``a`` is multidimensional, a single array of indices refers to the first dimension of ``a``. The following example shows this behavior by converting an image of labels into a color image using a palette.

.. code-block:: python

    >>> palette = np.array( [ [0,0,0],                # black
    ...                       [255,0,0],              # red
    ...                       [0,255,0],              # green
    ...                       [0,0,255],              # blue
    ...                       [255,255,255] ] )       # white
    >>> image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
    ...                     [ 0, 3, 4, 0 ]  ] )
    >>> palette[image]                            # the (2,4,3) color image
    array([[[  0,   0,   0],
            [255,   0,   0],
            [  0, 255,   0],
            [  0,   0,   0]],
           [[  0,   0,   0],
            [  0,   0, 255],
            [255, 255, 255],
            [  0,   0,   0]]])

We can also give indexes for more than one dimension. The arrays of indices for each dimension must have the same shape.

