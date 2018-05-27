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

要禁用此行为并强制NumPy打印整个数组，你可以使用set_printoptions更改打印选项。

.. code-block:: python

    >>> np.set_printoptions(threshold='nan')

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
基本操作
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
数组上的算术运算符使用元素级别。一个新的数组被创建并填充结果。