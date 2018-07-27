# N维数组(``ndarray``)

``ndarray``是具有相同类型和大小的项目的（通常是固定大小的）多维容器。数组中的维和项的数量由其``shape``（形状）定义，该形状是指定每个维的大小的N个正整数的 ``元组`` 数组中的项类型由单独的数据类型对象（dtype）指定，其中一个对象与每个ndarray关联。

与Python中的其他容器对象一样，可以通过对数组进行索引或切片(例如，使用整数n)，以及通过 ``ndarray`` 的方法和属性来访问和修改 ``ndarray`` 的内容。

不同的``ndarrays``可以共享相同的数据，因此在一个``ndarray``中所做的更改可能在另一个中可见。 也就是说，ndarray可以是另一个ndarray的“视图”，它所指的数据由“base”ndarray处理。 ndarrays也可以是Python``chace``所拥有的内存视图或实现``buffer``或数组接口的对象

**例子**

大小为2 x 3的二维数组，由4字节整数元素组成：

```python
>>> x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
>>> type(x)
<type 'numpy.ndarray'>
>>> x.shape
(2, 3)
>>> x.dtype
dtype('int32')
```

可以使用类似Python容器的语法对数组建立索引：

```python
>>> # The element of x in the *second* row, *third* column, namely, 6.
>>> x[1, 2]
```

例如，切片可以生成数组的视图：

```python
>>> y = x[:,1]
>>> y
array([2, 5])
>>> y[0] = 9 # this also changes the corresponding element in x
>>> y
array([9, 5])
>>> x
array([[1, 9, 3],
       [4, 5, 6]])
```

## 构造数组

可以参考Array创建API中详细介绍的API以及使用低级ndarray构造函数构建新数组：

ndarray(shape[, dtype, buffer, offset, …]) 数组对象表示一个多维的、同构的固定大小的项。

## 数组的索引

数组可以使用扩展的Python切片语法-array[selection（查询选择语法）] 来索引。类似的语法也用于访问结构化数组中的字段。

另见

> Array Indexing.

## ndarray在内存中的设计原理

类ndarray的实例由一个连续的一维计算机内存段(由数组或其他对象拥有)和一个索引方案组合而成，该索引方案将N个整数映射到块中一个项的位置。索引可以变化的范围由数组的形状指定。每个项需要多少字节以及如何解释这些字节是由与数组关联的数据类型对象定义的。

内存段本质上是一维的，在一维块中排列N维数组中的项有许多不同的方案。NumPy非常灵活，ndarray对象可以适应任何跨步索引方案。在跨步方案中，N维索引（n_0，n_1，...，n_ {N-1}）对应于偏移量（以字节为单位）：

![公式](/static/images/1388948b609ce9a1d9ae0380d361628d6b385812.svg)

从与数组关联的内存块的开头。 这里，s_k是指定数组步长的整数。 列主要顺序（例如，在Fortran语言和Matlab中使用）和行主要顺序（在C中使用）方案只是特定种类的跨步方案，并且对应于可由步幅解决的内存：

![公式](/static/images/af328186eedd2e4200b34e0e6a31acae4dbc9d20.svg)

条件 d_j = self.shape[j].

C和Fortran顺序都是连续的，即单段存储器布局，其中存储器块的每个部分可以通过索引的某种组合来访问。

虽然具有相应标志集的C风格和Fortran风格的连续数组可以通过上述步骤来解决，但实际步幅可能不同。 这可能发生在两种情况：

> 1. 如果self.shape [k] == 1那么对于任何合法索引索引[k] == 0.这意味着在偏移量n_k = 0的公式中，因此s_k n_k = 0，s_k = self.strides的值 [k]是任意的。
> 1. 如果数组没有元素（self.size == 0），则没有合法索引，并且从不使用步幅。 任何没有元素的数组都可以被认为是C风格和Fortran风格的连续数组。

第一条中，表示 self 和 self.squeeze() 始终具有相同的连续性和对齐的标志值。这也意味着即使是高维数组也可能同时是C风格和Fortran风格的连续。

如果所有元素的内存偏移量和基本偏移量本身是self.itemsize的倍数，则认为数组是对齐的。

> **注意**
> 默认情况下尚未应用第一条和第二条. 从NumPy 1.8.0开始，只有在构建NumPy时定义了环境变量NPY_RELAXED_STRIDES_CHECKING = 1时才会一致地应用它们。逐步的会成为默认值。
> 
> 您可以通过查看np.ones((10,1), order = 'C').flags.f_contiguous 的值来检查在构建NumPy时是否启用了此选项。如果是True，那么你的NumPy就没有启用步幅检查的功能。

<div class="warning-warp">
<b>警告</b>
<p>对于C风格的连续数组，它通常不认为 <code>self.strides[-1] == self.itemsize</code> 的值是True，或者对于Fortran风格的连续数组，<code>self.strides [0] == self.itemsize</code> 的值是True。</p>
</div>

除非另有说明，否则新ndarray中的数据采用行主（C）顺序，但是，例如，基本数组切片通常会以不同的方案生成视图。

> **注意**
> NumPy中的几种算法适用于任意跨步数组。但是，某些算法需要单段数组。当将不规则跨越的阵列传递给这样的算法时，会自动进行复制。

## 数组属性

数组属性反映数组本身固有的信息。通常，通过数组的属性访问它，您可以获取并设置数组的内部属性，而无需创建新的数组。公开的属性是数组的核心部分，只有其中一些属性可以在不创建新数组的情况下进行有意义的重置。每个属性的信息如下。

### 内存相关的属性

以下属性包含有关数组内存的信息：

- ``ndarray.flags``	有关数组内存分布的信息。
- ``ndarray.shape``	数组维度的元组。
- ``ndarray.strides``	遍历数组时要在每个维度中执行的字节元组。
- ``ndarray.ndim``	数组维数。
- ``ndarray.data``	指向数组数据开始的Python缓冲区对象。
- ``ndarray.size``	数组中的元素数。
- ``ndarray.itemsize``	一个数组元素的长度(以字节为单位)。
- ``ndarray.nbytes``	数组元素消耗的总字节。
- ``ndarray.base``	如果内存来自其他对象，则为基本对象。

### 数据类型

另见：

> Data type objects

与数组关联的数据类型对象可以在 ``dtype`` 属性中找到：

``ndarray.dtype``数组元素的数据类型。

### 其他属性

- ``ndarray.T``	        与 self.transpose()相同，只是如果 self.ndim <2 则返回自己。
- ``ndarray.real``	数组的真实部分。
- ``ndarray.imag``	数组的虚部。
- ``ndarray.flat``	数组上的一维迭代器。
- ``ndarray.ctypes``	一个简化数组与ctypes模块交互的对象。

### 数组接口

另见：

> The Array Interface.

``__array_interface__``	数组接口的Python端。
``__array_struct__``	数组接口的C端。

### ``ctypes`` 外来函数接口

``ndarray.ctypes``	一个简化数组与ctypes模块交互的对象。

## 数组的方法

``ndarray`` 对象有许多方法以某种方式对数组进行操作或与数组一起操作，通常返回数组结果。 下面简要说明这些方法。（每个方法的文档都有更完整的描述。）

对于以下方法，numpy中还有相应的函数：all，any，argmax，argmin，argpartition，argsort，choose，clip，compress，copy，cumprod，cumsum，diagonal，imag，max，mean，min，nonzero，partition， prod，ptp，put，ravel，real，repeat，reshape，round，searchsorted，sort，squeeze，std，sum，swapaxes，take，trace，transpose，var。

### 数组转换

- ndarray.item(*args)	将数组元素复制到标准Python标量并返回它。
- ndarray.tolist()	将数组作为（可能是嵌套的）列表返回。
- ndarray.itemset(*args)	将标量插入数组（如果可能，将标量转换为数组的dtype）
- ndarray.tostring([order])	构造包含数组中原始数据字节的Python字节。
- ndarray.tobytes([order])	构造包含数组中原始数据字节的Python字节。
- ndarray.tofile(fid[, sep, format])	将数组作为文本或二进制写入文件（默认）。
- ndarray.dump(file)	将数组的pickle转储到指定的文件。
- ndarray.dumps()	以字符串形式返回数组的pickle。
- ndarray.astype(dtype[, order, casting, …])	数组的副本，强制转换为指定的类型。
- ndarray.byteswap([inplace])	交换数组元素的字节
- ndarray.copy([order])	返回数组的副本。
- ndarray.view([dtype, type])	具有相同数据的数组的新视图。
- ndarray.getfield(dtype[, offset])	返回给定数组的字段作为特定类型。
- ndarray.setflags([write, align, uic])	分别设置数组标志WRITEABLE，ALIGNED，（WRITEBACKIFCOPY和UPDATEIFCOPY）。
- ndarray.fill(value)	使用标量值填充数组。

### 项目选择和操作

对于采用axis关键字的数组方法，默认为None。 如果axis为None，则将数组视为1维数组。轴的任何其他值表示操作应该沿着的维度。

- ndarray.take(indices[, axis, out, mode])	Return an array formed from the elements of a at the given indices.
- ndarray.put(indices, values[, mode])	Set a.flat[n] = values[n] for all n in indices.
- ndarray.repeat(repeats[, axis])	Repeat elements of an array.
- ndarray.choose(choices[, out, mode])	Use an index array to construct a new array from a set of choices.
- ndarray.sort([axis, kind, order])	Sort an array, in-place.
- ndarray.argsort([axis, kind, order])	Returns the indices that would sort this array.
- ndarray.partition(kth[, axis, kind, order])	Rearranges the elements in the array in such a way that value of the element in kth position is in the position it would be in a sorted array.
- ndarray.argpartition(kth[, axis, kind, order])	Returns the indices that would partition this array.
- ndarray.searchsorted(v[, side, sorter])	Find indices where elements of v should be inserted in a to maintain order.
- ndarray.nonzero()	Return the indices of the elements that are non-zero.
- ndarray.compress(condition[, axis, out])	Return selected slices of this array along given axis.
- ndarray.diagonal([offset, axis1, axis2])	Return specified diagonals.

### Calculation

Many of these methods take an argument named axis. In such cases,

- If axis is None (the default), the array is treated as a 1-D array and the operation is performed over the entire array. This behavior is also the default if self is a 0-dimensional array or array scalar. (An array scalar is an instance of the types/classes float32, float64, etc., whereas a 0-dimensional array is an ndarray instance containing precisely one array scalar.)
- If axis is an integer, then the operation is done over the given axis (for each 1-D subarray that can be created along the given axis).

**Example of the axis argument**

A 3-dimensional array of size 3 x 3 x 3, summed over each of its three axes

```python
>>> x
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],
       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],
       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
>>> x.sum(axis=0)
array([[27, 30, 33],
       [36, 39, 42],
       [45, 48, 51]])
>>> # for sum, axis is the first keyword, so we may omit it,
>>> # specifying only its value
>>> x.sum(0), x.sum(1), x.sum(2)
(array([[27, 30, 33],
        [36, 39, 42],
        [45, 48, 51]]),
 array([[ 9, 12, 15],
        [36, 39, 42],
        [63, 66, 69]]),
 array([[ 3, 12, 21],
        [30, 39, 48],
        [57, 66, 75]]))
```

The parameter dtype specifies the data type over which a reduction operation (like summing) should take place. The default reduce data type is the same as the data type of self. To avoid overflow, it can be useful to perform the reduction using a larger data type.

For several methods, an optional out argument can also be provided and the result will be placed into the output array given. The out argument must be an ``ndarray`` and have the same number of elements. It can have a different data type in which case casting will be performed.

- ndarray.argmax([axis, out])	Return indices of the maximum values along the given axis.
- ndarray.min([axis, out, keepdims])	Return the minimum along a given axis.
- ndarray.argmin([axis, out])	Return indices of the minimum values along the given axis of a.
- ndarray.ptp([axis, out])	Peak to peak (maximum - minimum) value along a given axis.
- ndarray.clip([min, max, out])	Return an array whose values are limited to [min, max].
- ndarray.conj()	Complex-conjugate all elements.
- ndarray.round([decimals, out])	Return a with each element rounded to the given number of decimals.
- ndarray.trace([offset, axis1, axis2, dtype, out])	Return the sum along diagonals of the array.
- ndarray.sum([axis, dtype, out, keepdims])	Return the sum of the array elements over the given axis.
- ndarray.cumsum([axis, dtype, out])	Return the cumulative sum of the elements along the given axis.
- ndarray.mean([axis, dtype, out, keepdims])	Returns the average of the array elements along given axis.
- ndarray.var([axis, dtype, out, ddof, keepdims])	Returns the variance of the array elements, along given axis.
- ndarray.std([axis, dtype, out, ddof, keepdims])	Returns the standard deviation of the array elements along given axis.
- ndarray.prod([axis, dtype, out, keepdims])	Return the product of the array elements over the given axis
- ndarray.cumprod([axis, dtype, out])	Return the cumulative product of the elements along the given axis.
- ndarray.all([axis, out, keepdims])	Returns True if all elements evaluate to True.
- ndarray.any([axis, out, keepdims])	Returns True if any of the elements of a evaluate to True.

## Arithmetic, matrix multiplication, and comparison operations

Arithmetic and comparison operations on ndarrays are defined as element-wise operations, and generally yield ndarray objects as results.

Each of the arithmetic operations (+, -, *, /, //, %, divmod(), ** or pow(), <<, >>, &, ^, |, ~) and the comparisons (==, <, >, <=, >=, !=) is equivalent to the corresponding universal function (or ufunc for short) in NumPy. For more information, see the section on Universal Functions.

Comparison operators:

- ndarray.__lt__($self, value, /)	Return self<value.
- ndarray.__le__($self, value, /)	Return self<=value.
- ndarray.__gt__($self, value, /)	Return self>value.
- ndarray.__ge__($self, value, /)	Return self>=value.
- ndarray.__eq__($self, value, /)	Return self==value.
- ndarray.__ne__($self, value, /)	Return self!=value.

Truth value of an array (``bool``):

``ndarray.__nonzero__``

> **Note**
> Truth-value testing of an array invokes ndarray.__nonzero__, which raises an error if the number of elements in the array is larger than 1, because the truth value of such arrays is ambiguous. Use .any() and .all() instead to be clear about what is meant in such cases. (If the number of elements is 0, the array evaluates to False.)

Unary operations:

- ndarray.__neg__($self, /)	-self
- ndarray.__pos__($self, /)	+self
- ndarray.__abs__(self)	
- ndarray.__invert__($self, /)	~self

Arithmetic:

- ndarray.__add__($self, value, /)	Return self+value.
- ndarray.__sub__($self, value, /)	Return self-value.
- ndarray.__mul__($self, value, /)	Return self*value.
- ndarray.__div__	
- ndarray.__truediv__($self, value, /)	Return self/value.
- ndarray.__floordiv__($self, value, /)	Return self//value.
- ndarray.__mod__($self, value, /)	Return self%value.
- ndarray.__divmod__($self, value, /)	Return divmod(self, value).
- ndarray.__pow__($self, value[, mod])	Return pow(self, value, mod).
- ndarray.__lshift__($self, value, /)	Return self<<value.
- ndarray.__rshift__($self, value, /)	Return self>>value.
- ndarray.__and__($self, value, /)	Return self&value.
- ndarray.__or__($self, value, /)	Return self|value.
- ndarray.__xor__($self, value, /)	Return self^value.

> **Note**
> - Any third argument to pow is silently ignored, as the underlying ufunc takes only two arguments.
> - he three division operators are all defined; div is active by default, truediv is active when __future__ division is in effect.
> - Because ndarray is a built-in type (written in C), the __r{op}__ special methods are not directly defined.
> - The functions called to implement many arithmetic special methods for arrays can be modified using set_numeric_ops.

Arithmetic, in-place:

- ndarray.__iadd__($self, value, /)	Return self+=value.
- ndarray.__isub__($self, value, /)	Return self-=value.
- ndarray.__imul__($self, value, /)	Return self*=value.
- ndarray.__idiv__	
- ndarray.__itruediv__($self, value, /)	Return self/=value.
- ndarray.__ifloordiv__($self, value, /)	Return self//=value.
- ndarray.__imod__($self, value, /)	Return self%=value.
- ndarray.__ipow__($self, value, /)	Return self**=value.
- ndarray.__ilshift__($self, value, /)	Return self<<=value.
- ndarray.__irshift__($self, value, /)	Return self>>=value.
- ndarray.__iand__($self, value, /)	Return self&=value.
- ndarray.__ior__($self, value, /)	Return self|=value.
- ndarray.__ixor__($self, value, /)	Return self^=value.

<div class="warning-warp">
<b>Warning</b>

<p>In place operations will perform the calculation using the precision decided by the data type of the two operands, but will silently downcast the result (if necessary) so it can fit back into the array. Therefore, for mixed precision calculations, A {op}= B can be different than A = A {op} B. For example, suppose a = ones((3,3)). Then, a += 3j is different than a = a + 3j: while they both perform the same computation, a += 3 casts the result to fit back in a, whereas a = a + 3j re-binds the name a to the result.</p>

</div>

Matrix Multiplication:

``ndarray.__matmul__``($self, value, /)	Return self@value.

> **Note**
> Matrix operators @ and @= were introduced in Python 3.5 following PEP465. NumPy 1.10.0 has a preliminary implementation of @ for testing purposes. Further documentation can be found in the matmul documentation.

## Special methods

For standard library functions:

- ndarray.__copy__()	Used if copy.copy is called on an array.
- ndarray.__deepcopy__(memo, /)	Used if copy.deepcopy is called on an array.
- ndarray.__reduce__()	For pickling.
- ndarray.__setstate__(state, /)	For unpickling.

Basic customization:

- ndarray.__new__($type, *args, **kwargs)	Create and return a new object.
- ndarray.__array__(|dtype)	Returns either a new reference to self if dtype is not given or a new array of provided data type if dtype is different from the current dtype of the array.
- ndarray.__array_wrap__(obj)	

Container customization: (see Indexing)

- ndarray.__len__($self, /)	Return len(self).
- ndarray.__getitem__($self, key, /)	Return self[key].
- ndarray.__setitem__($self, key, value, /)	Set self[key] to value.
- ndarray.__contains__($self, key, /)	Return key in self.

Conversion; the operations complex, int, long, float, oct, and hex. They work only on arrays that have one element in them and return the appropriate scalar.

- ndarray.__int__(self)	
- ndarray.__long__	
- ndarray.__float__(self)	
- ndarray.__oct__	
- ndarray.__hex__	

String representations:

- ndarray.__str__($self, /)	Return str(self).
- ndarray.__repr__($self, /)	Return repr(self).