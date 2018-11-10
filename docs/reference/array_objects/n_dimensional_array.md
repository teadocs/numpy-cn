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
> 你可以通过查看np.ones((10,1), order = 'C').flags.f_contiguous 的值来检查在构建NumPy时是否启用了此选项。如果是True，那么你的NumPy就没有启用步幅检查的功能。

<div class="warning-warp">
<b>警告</b>
<p>对于C风格的连续数组，它通常不认为 <code>self.strides[-1] == self.itemsize</code> 的值是True，或者对于Fortran风格的连续数组，<code>self.strides [0] == self.itemsize</code> 的值是True。</p>
</div>

除非另有说明，否则新ndarray中的数据采用行主（C）顺序，但是，例如，基本数组切片通常会以不同的方案生成视图。

> **注意**
> NumPy中的几种算法适用于任意跨步数组。但是，某些算法需要单段数组。当将不规则跨越的阵列传递给这样的算法时，会自动进行复制。

## 数组属性

数组属性反映数组本身固有的信息。通常，通过数组的属性访问它，你可以获取并设置数组的内部属性，而无需创建新的数组。公开的属性是数组的核心部分，只有其中一些属性可以在不创建新数组的情况下进行有意义的重置。每个属性的信息如下。

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

- ndarray.take(indices[, axis, out, mode])	返回由给定索引处的a元素组成的数组。
- ndarray.put(indices, values[, mode])	为索引中的所有n设置 a.flat[n] = values[n]。
- ndarray.repeat(repeats[, axis])	重复数组的元素。
- ndarray.choose(choices[, out, mode])	使用索引数组从一组选项中构造新数组。
- ndarray.sort([axis, kind, order])	就地对数组进行排序。
- ndarray.argsort([axis, kind, order])	返回将对此数组进行排序的索引。
- ndarray.partition(kth[, axis, kind, order])	重新排列数组中的元素，使得第k个位置的元素值处于排序数组中的位置。
- ndarray.argpartition(kth[, axis, kind, order])	重新排列数组中的元素，使得第k个位置的元素值处于排序数组中的位置。
- ndarray.searchsorted(v[, side, sorter])	查找应在其中插入v的元素以维护顺序的索引。
- ndarray.nonzero()	返回非零元素的索引。
- ndarray.compress(condition[, axis, out])	沿给定轴返回此数组的选定切片。
- ndarray.diagonal([offset, axis1, axis2])	返回指定的对角线。

### 计算

在下面这种情况下，其中许多方法都采用名为axis的参数。 

- 如果axis为None（默认值），则将数组视为1-D数组，并对整个数组执行操作。 如果self是0维数组或数组标量，则此行为也是默认行为。 （数组标量是类型/类float32，float64等的实例，而0维数组是包含恰好一个数组标量的ndarray实例。）
- 如果axis是整数，则操作在给定轴上完成（对于可沿给定轴创建的每个1维的子阵列）

**轴参数的示例**

一个尺寸为3 x 3 x 3的三维阵列，在其三个轴中的每个轴上求和。

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

参数dtype指定应在其上进行缩减操作（如求和）的数据类型。 默认的reduce数据类型与self的数据类型相同。 为避免溢出，使用更大的数据类型执行缩减可能很有用。

对于多种方法，还可以提供可选的out参数，并将结果放入给定的输出数组中。 out参数必须是 ``ndarray`` 并且具有相同数量的元素。 它可以具有不同的数据类型，在这种情况下将执行转换。

- ndarray.argmax([axis, out])	返回给定轴的最大值索引。
- ndarray.min([axis, out, keepdims])	沿给定轴返回最小值。
- ndarray.argmin([axis, out])	沿a的给定轴返回最小值的索引。
- ndarray.ptp([axis, out])	沿给定轴的峰峰值（最大值 - 最小值）。
- ndarray.clip([min, max, out])	返回其值限制为 [min, max] 的数组。
- ndarray.conj()	复合共轭所有元素。
- ndarray.round([decimals, out])	返回a，每个元素四舍五入到给定的小数位数。
- ndarray.trace([offset, axis1, axis2, dtype, out])	返回数组对角线的总和。
- ndarray.sum([axis, dtype, out, keepdims])	返回给定轴上的数组元素的总和。
- ndarray.cumsum([axis, dtype, out])	返回给定轴上元素的累积和。
- ndarray.mean([axis, dtype, out, keepdims])	返回给定轴上数组元素的平均值。
- ndarray.var([axis, dtype, out, ddof, keepdims])	返回给定轴的数组元素的方差。
- ndarray.std([axis, dtype, out, ddof, keepdims])	返回给定轴的数组元素的标准偏差。
- ndarray.prod([axis, dtype, out, keepdims])	返回给定轴上的数组元素的乘积
- ndarray.cumprod([axis, dtype, out])	返回沿给定轴的元素的累积乘积。
- ndarray.all([axis, out, keepdims])	如果所有元素都计算为True，则返回True。
- ndarray.any([axis, out, keepdims])	如果求值的任何元素为True，则返回True。

## 算术，矩阵乘法和比较运算

ndarrays上的算术和比较操作被定义为元素操作，并且通常将ndarray对象作为结果产生。

每个算术运算（+， - ，*，/，//，％，divmod（），**或pow（），<<，>>，＆，^，|，〜）和比较运算符（== ，<，>，<=，> =，！=）相当于NumPy中相应的通用函数（或简称为ufunc）。 有关更多信息，请参阅[通用功能部分](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#ufuncs)。

比较运算符:

- ndarray.__lt__($self, value, /)	返回 self<value.
- ndarray.__le__($self, value, /)	返回 self<=value.
- ndarray.__gt__($self, value, /)	返回 self>value.
- ndarray.__ge__($self, value, /)	返回 self>=value.
- ndarray.__eq__($self, value, /)	返回 self==value.
- ndarray.__ne__($self, value, /)	返回 self!=value.

数组的真值(``bool``)：

``ndarray.__nonzero__``

> **注意**
> 数组的真值测试会调用 ndarray.__nonzero__，如果数组中的元素数大于1，则会引发错误，因为此类数组的真值是不明确的。使用.any()和.all()代替清楚这种情况下的含义。（如果元素数为0，则数组的计算结果为False。）

一元操作：

- ndarray.__neg__($self, /)	-self
- ndarray.__pos__($self, /)	+self
- ndarray.__abs__(self)	
- ndarray.__invert__($self, /)	~self

算术运算：

- ndarray.__add__($self, value, /)	返回 self+value.
- ndarray.__sub__($self, value, /)	返回 self-value.
- ndarray.__mul__($self, value, /)	返回 self*value.
- ndarray.__div__	
- ndarray.__truediv__($self, value, /)	返回 self/value.
- ndarray.__floordiv__($self, value, /)	返回 self//value.
- ndarray.__mod__($self, value, /)	返回 self%value.
- ndarray.__divmod__($self, value, /)	返回 divmod(self, value).
- ndarray.__pow__($self, value[, mod])	返回 pow(self, value, mod).
- ndarray.__lshift__($self, value, /)	返回 self<<value.
- ndarray.__rshift__($self, value, /)	返回 self>>value.
- ndarray.__and__($self, value, /)	返回 self&value.
- ndarray.__or__($self, value, /)	返回 self|value.
- ndarray.__xor__($self, value, /)	返回 self^value.

> **注意**
> - pow的任何第三个参数都会被默认忽略，因为底层的ufunc只接受两个参数。
> - 他划分了三个除法运算符; div默认处于活动状态，当__future__除法生效时，truediv处于活动状态。
> - 因为ndarray是内置类型（用C编写），所以不直接定义 __r{op}__ 特殊方法。
> - 可以使用set_numeric_ops修改为数组实现许多算术特殊方法的函数。

就地算数运算：

- ndarray.__iadd__($self, value, /)	返回 self+=value.
- ndarray.__isub__($self, value, /)	返回 self-=value.
- ndarray.__imul__($self, value, /)	返回 self*=value.
- ndarray.__idiv__	
- ndarray.__itruediv__($self, value, /)	返回 self/=value.
- ndarray.__ifloordiv__($self, value, /)	Return self//=value.
- ndarray.__imod__($self, value, /)	返回 self%=value.
- ndarray.__ipow__($self, value, /)	返回 self**=value.
- ndarray.__ilshift__($self, value, /)	返回 self<<=value.
- ndarray.__irshift__($self, value, /)	返回 self>>=value.
- ndarray.__iand__($self, value, /)	返回 self&=value.
- ndarray.__ior__($self, value, /)	返回 self|=value.
- ndarray.__ixor__($self, value, /)	返回 self^=value.

<div class="warning-warp">
<b>警告</b>

<p>就地操作将使用由两个操作数的数据类型决定的精度来执行计算，但是将静默地向下转换结果（如果需要），以便它可以适合回到数组中。 因此，对于混合精度计算，A {op} = B 可以与 A = A {op} B 不同。例如，假设 a = ones((3,3))。然后，a+= 3j 与 a = a + 3j 不同：虽然它们都执行相同的计算，但是 a += 3 将结果投射回适合a，而 a = a + 3j 将名称a重新绑定到 a 的结果。
</p>

</div>

矩阵乘法：

``ndarray.__matmul__``($self, value, /)	返回 self@value.

> **注意**
> 矩阵运算符@和@ =是在PEP465之后的Python 3.5中引入的。 NumPy 1.10.0初步实现了@用于测试目的。 更多文档可以在[matmul](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul)档中找到。

## 特殊方法

标准库函数：

- ndarray.__copy__()	如果在数组上调用copy.copy，则使用此方法。
- ndarray.__deepcopy__(memo, /)	如果在数组上调用copy.deepcopy，则使用此方法。
- ndarray.__reduce__() 用于腌制（译者注：很形象）。
- ndarray.__setstate__(state, /) 用于反腌制。

基本的定制：

- ndarray.__new__($type, *args, **kwargs)	创建并返回一个新对象。
- ndarray.__array__(|dtype)	如果没有给出dtype，则返回对self的新引用;如果dtype与数组的当前dtype不同，则返回提供的数据类型的新数组。
- ndarray.__array_wrap__(obj)	

容器的定制: (参见索引）

- ndarray.__len__($self, /)	返回 len(self).
- ndarray.__getitem__($self, key, /)	返回 self[key].
- ndarray.__setitem__($self, key, value, /)	给 self[key] 设置一个值。
- ndarray.__contains__($self, key, /)	返回 自身的关键索引。

转换;操作complex，int，long，float，oct和hex。它们位于数组中，其中包含一个元素并返回相应的标量。

- ndarray.__int__(self)	
- ndarray.__long__	
- ndarray.__float__(self)	
- ndarray.__oct__	
- ndarray.__hex__	

字符串表示：

- ndarray.__str__($self, /)	返回 str(self).
- ndarray.__repr__($self, /)	返回 repr(self).