<title>numpy结构化数组 - <%-__DOC_NAME__ %></title>
<meta name="keywords" content="numpy结构化数组" />

# 结构化数组

## 介绍

结构化数组其实就是ndarrays，其数据类型是由组成一系列命名字段的简单数据类型组成的。 例如：

```python
>>> x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
...              dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
>>> x
array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
      dtype=[('name', 'S10'), ('age', '<i4'), ('weight', '<f4')])
```

这里``x``是长度为2的一维数组，其数据类型是具有三个字段的结构：1、名为'name'的长度为10或更小的字符串。2、名为'age'的32位整数。3、名为'weight'的32位浮点数。

如果你的索引``x``是1，你会看到这样的结构：

```python
>>> x[1]
('Fido', 3, 27.0)
```

你可以通过使用字段名称进行索引来访问和修改结构化数组的各个字段的值：

```python
>>> x['age']
array([9, 3], dtype=int32)
>>> x['age'] = 5
>>> x
array([('Rex', 5, 81.0), ('Fido', 5, 27.0)],
      dtype=[('name', 'S10'), ('age', '<i4'), ('weight', '<f4')])
```

结构化数组设计用于结构化数据的底层操作，例如解释编译二进制数据块。结构化数据类型旨在模仿C语言中的 “structs”，使它们对于与C代码接口也很有用。 为了达到这些目的，numpy支持诸如子阵列和嵌套数据类型之类的特殊功能，并允许手动控制结构的内存布局。

如果你想进行表格数据的简单操作，那么其他 pydata 项目（例如pandas，xarray或DataArray）将为你提供更适合的更高级别的接口。 这些包也可以为表格数据分析提供更好的性能，因为NumPy中的结构化数组的类C结构内存布局会导致缓存行为不佳。

## 结构化数据类型

要使用结构化数组，首先需要定义结构化数据类型。

结构化数据类型可以被认为是一定长度的字节序列（结构的itemsize），它被解释为一个字段集合。 每个字段在结构中都有一个名称，一个数据类型和一个字节偏移量。 字段的数据类型可以是任何numpy数据类型，包括其他结构化数据类型，它也可以是一个子数组，其行为类似于指定形状的ndarray。 字段的偏移是任意的，并且字段甚至可以重叠。 这些偏移通常由numpy自动确定，但也可以指定。

### 结构化数据类型创建

结构化数据类型可以使用函数``numpy.dtype``创建。 有4种可选形式的规范，其灵活性和简洁性各不相同。 这些在[数据类型对象](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes-constructing)参考页面中都有进一步的记录，总之它们是：

1. **元组列表，每个字段一个元组**

    每个元组都有这些属性``（fieldname，datatype，shape）``，其中shape是可选的。 ``fieldname``是一个字符串（或元组，如果使用标题，请参阅下面的字段标题），``datatype``可以是任何可转换为数据类型的对象，``shape``是指定子阵列形状的整数元组。

    ```python
    >>> np.dtype([('x', 'f4'), ('y', np.float32), ('z', 'f4', (2,2))])
    dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4', (2, 2))])
    ```

    如果``fieldname``是空字符串``''``，那么该字段将被赋予一个默认名称形式``f#``，其中``#``是该字段的整数索引，从左边以0开始计数：

    ```python
    >>> np.dtype([('x', 'f4'),('', 'i4'),('z', 'i8')])
    dtype([('x', '<f4'), ('f1', '<i4'), ('z', '<i8')])
    ```
    
    结构中字段的字节偏移量和总体结构中元素的大小是自动确定的。

1. **一串用逗号分隔的dtype规范**

    在这种简写表示法中，任何 [string dtype specifications](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes-constructing) 都可以在字符串中使用逗号分隔，字段的元素大小（itemsize）和字节偏移量是自动确定的，并且字段名称被赋予默认名称如":“f0”、“f1” 等。 

    ```python
    >>> np.dtype('i8,f4,S3')
    dtype([('f0', '<i8'), ('f1', '<f4'), ('f2', 'S3')])
    >>> np.dtype('3int8, float32, (2,3)float64')
    dtype([('f0', 'i1', 3), ('f1', '<f4'), ('f2', '<f8', (2, 3))])
    ```

3. **字段参数数组的字典**

    这是最灵活的规范形式，因为它允许控制字段的字节偏移量和结构中的元素大小（itemsize）。

    字典有两个必需的键，'names' 和 'formats'，以及四个可选键，'offsets'，'itemsize'，'aligned' 和 'titles'。 'names' 和 'formats' 的值应该分别是长度相同的字段名称列表和dtype规范列表。 可选的 'offsets' 值应该是整数字节偏移量的列表，结构中的每个字段都有一个偏移量。 如果没有给出 'offsets'，则自动确定偏移量。 可选的 'itemsize' 值应该是一个描述dtype的总大小（以字节为单位）的整数，它必须足够大以包含所有字段。

    ```python
    >>> np.dtype({'names': ['col1', 'col2'], 'formats': ['i4','f4']})
    dtype([('col1', '<i4'), ('col2', '<f4')])
    >>> np.dtype({'names': ['col1', 'col2'],
    ...           'formats': ['i4','f4'],
    ...           'offsets': [0, 4],
    ...           'itemsize': 12})
    dtype({'names':['col1','col2'], 'formats':['<i4','<f4'], 'offsets':[0,4], 'itemsize':12})
    ```

    可以选择偏移使得字段重叠，但这意味着分配给一个字段可能破坏任何重叠字段的数据。 作为一个例外，``numpy.object``类型的字段不能与其他字段重叠，因为存在破坏内部对象指针然后解除引用的风险。
    
    可选的 “aligned” 值可以设置为True，以使自动偏移计算使用对齐的偏移（请参阅[自动字节偏移和对齐](https://docs.scipy.org/doc/numpy/user/basics.rec.html#offsets-and-alignment)），就好像numpy.dtype的'align'关键字参数已设置为True。

    可选的 “titles” 值应该是与 “names” 长度相同的标题列表，请参阅的[字段标题](https://docs.scipy.org/doc/numpy/user/basics.rec.html#titles)。

4. **字段名称的字典**

    不鼓励使用这种形式的规范，但也在此列出，因为较旧的numpy代码可能会使用它。字典的键是字段名称，值是指定类型和偏移量的元组：

    ```python
    >>> np.dtype=({'col1': ('i1',0), 'col2': ('f4',1)})
    dtype([(('col1'), 'i1'), (('col2'), '>f4')])
    ```
    
    不鼓励使用这种形式，因为Python的字典类型在Python3.6之前没有保留Python版本中的顺序，并且结构化dtype中字段的顺序具有意义。[字段标题](https://docs.scipy.org/doc/numpy/user/basics.rec.html#titles)可以通过使用3元组来指定，请参见下面的内容.

### 操作和显示结构化数据类型

可以在dtype对象的``names``属性中找到结构化数据类型的字段名称列表：

```python
>>> d = np.dtype([('x', 'i8'), ('y', 'f4')])
>>> d.names
('x', 'y')
```

可以通过使用相同长度的字符串序列分配 ``names`` 属性来修改字段名称。

dtype对象还有一个类似字典的属性fields，其键是字段名称（和[Field Titles](https://docs.scipy.org/doc/numpy/user/basics.rec.html#titles)，见下文），其值是包含每个字段的dtype和byte偏移量的元组。

```python
>>> d.fields
mappingproxy({'x': (dtype('int64'), 0), 'y': (dtype('float32'), 8)})
```

对于非结构化数组，``names``和``fields`` 属性都是 ``None``。

如果可能，结构化数据类型的字符串表示形式为“元组列表”的形式，否则numpy将回退到使用更通用的字典的形式。

### 自动字节偏移和对齐

Numpy使用两种方法中的一个来自动确定字节字节偏移量和结构化数据类型的整体项目大小，具体取决于是否将``align = True``指定为``numpy.dtype``的关键字参数。

默认情况下（``align = False``），numpy将字段打包在一起，使得每个字段从前一个字段结束的字节偏移开始，并且字段在内存中是连续的。

```python
>>> def print_offsets(d):
...     print("offsets:", [d.fields[name][1] for name in d.names])
...     print("itemsize:", d.itemsize)
>>> print_offsets(np.dtype('u1,u1,i4,u1,i8,u2'))
offsets: [0, 1, 2, 6, 7, 15]
itemsize: 17
```

如果设置``align = True``，numpy将以与许多C编译器填充C结构相同的方式填充结构。 在某些情况下，对齐结构可以提高性能，但代价是增加了数据的大小。 在字段之间插入填充字节，使得每个字段的字节偏移量将是该字段对齐的倍数，对于简单数据类型，通常等于字段的字节大小，请参阅“PyArray_Descr.alignment”。 该结构还将添加尾随填充，以使其itemsize是最大字段对齐的倍数。

```python
>>> print_offsets(np.dtype('u1,u1,i4,u1,i8,u2', align=True))
offsets: [0, 1, 4, 8, 16, 24]
itemsize: 32
```

请注意，尽管默认情况下几乎所有现代C编译器都以这种方式填充，但C结构中的填充依赖于C实现，因此不能保证此内存布局与C程序中相应结构的内容完全匹配。 为了获得确切的对应关系，可能需要在numpy或C这边进行一些工作。

如果在基于字典的dtype规范中使用可选的``offsets``键指定了偏移量，设置``align = True``将检查每个字段的偏移量是否为其大小的倍数，项大小是否为最大字段大小的倍数，如果不是，则引发异常。

如果结构化数组的字段和项目大小的偏移满足对齐条件，则数组将设置 ``ALIGNED`` 标志。

便捷函数``numpy.lib.recfunctions.repack_fields``将对齐的dtype或数组转换为已打包的dtype或数组，反之亦然。它接受dtype或结构化ndarray作为参数，并返回一个带有重新打包的字段的副本，无论是否有填充字节。

### 字段标题

除了字段名称之外，字段还可以具有关联的标题，备用名称，有时用作字段的附加说明或别名。 标题可用于索引数组，就像字段名一样。

要在使用dtype规范的list-of-tuples形式时添加标题，可以将字段名称指定为两个字符串的元组而不是单个字符串，它们分别是字段的标题和字段名称。 例如：

```python
>>> np.dtype([(('my title', 'name'), 'f4')])
```

当使用第一种形式的基于字典的规范时，``标题``可以作为额外的“标题”作为键提供，如上所述。 当使用第二个（不鼓励的）基于字典的规范时，可以通过提供3元素元组``（数据类型，偏移量，标题）``而不是通常的2元素元组来提供标题：

```python
>>> np.dtype({'name': ('i4', 0, 'my title')})
``` 

如果使用了标题，``dtype.field``字典将包含作为键的标题。这意味着具有标题的字段将在字段字典中表示两次。这些字段的元组值还有第三个元素，字段标题。因此，由于``name``属性保留了字段顺序，而``field``属性可能不能，建议使用dtype的``name``属性迭代dtype的字段，该属性不会列出标题，如下所示：

```python
>>> for name in d.names:
...     print(d.fields[name][:2])
```

### 联合类型

结构化数据类型在numpy中实现，默认情况下具有基类型``numpy.void``，但是可以使用数据类型对象中描述的dtype规范的``（base_dtype，dtype）``形式将其他numpy类型解释为结构化类型。 这里，``base_dtype``是所需的底层``dtype``，字段和标志将从dtype复制。 这个dtype类似于C中的'union'。

## 结构化数组的索引和分配

### 将数据分配给结构化数组

有许多方法可以为结构化数组赋值：使用python元组、使用标量值或使用其他结构化数组。

#### 使用Python原生类型(元组)来赋值

将值赋给结构化数组的最简单方法是使用python元组。 每个赋值应该是一个长度等于数组中字段数的元组，而不是列表或数组，因为它们将触发numpy的广播规则。 元组的元素从左到右分配给数组的连续字段：

```python
>>> x = np.array([(1,2,3),(4,5,6)], dtype='i8,f4,f8')
>>> x[1] = (7,8,9)
>>> x
array([(1, 2., 3.), (7, 8., 9.)],
     dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '<f8')])
```

#### 通过标量赋值

分配给结构化元素的标量将分配给所有字段。 将标量分配给结构化数组时，或者将非结构化数组分配给结构化数组时，会发生这种情况：

```python
>>> x = np.zeros(2, dtype='i8,f4,?,S1')
>>> x[:] = 3
>>> x
array([(3, 3.0, True, b'3'), (3, 3.0, True, b'3')],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])
>>> x[:] = np.arange(2)
>>> x
array([(0, 0.0, False, b'0'), (1, 1.0, True, b'1')],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])
```

结构化数组也可以分配给非结构化数组，但前提是结构化数据类型只有一个字段：

```python
>>> twofield = np.zeros(2, dtype=[('A', 'i4'), ('B', 'i4')])
>>> onefield = np.zeros(2, dtype=[('A', 'i4')])
>>> nostruct = np.zeros(2, dtype='i4')
>>> nostruct[:] = twofield
ValueError: Can't cast from structure to non-structure, except if the structure only has a single field.
>>> nostruct[:] = onefield
>>> nostruct
array([0, 0], dtype=int32)
```

#### 来自其他结构化数组的赋值

两个结构化数组之间的分配就像源元素已转换为元组然后分配给目标元素一样。 也就是说，源阵列的第一个字段分配给目标数组的第一个字段，第二个字段同样分配，依此类推，而不管字段名称如何。 具有不同数量的字段的结构化数组不能彼此分配。 未包含在任何字段中的目标结构的字节不受影响。

```python
>>> a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])
>>> b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])
>>> b[:] = a
>>> b
array([(0.0, b'0.0', b''), (0.0, b'0.0', b''), (0.0, b'0.0', b'')],
      dtype=[('x', '<f4'), ('y', 'S3'), ('z', 'O')])
```

#### 子阵列的赋值

分配给子阵列的字段时，首先将指定的值广播到子阵列的形状。

### 索引结构化数组

#### 访问单个字段

可以通过使用字段名称索引数组来访问和修改结构化数组的各个字段。

```python
>>> x = np.array([(1,2),(3,4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
>>> x['foo']
array([1, 3])
>>> x['foo'] = 10
>>> x
array([(10, 2.), (10, 4.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])
```

可以通过使用字段名称索引数组来访问和修改结构化数组的各个字段。

```python
>>> y = x['bar']
>>> y[:] = 10
>>> x
array([(10, 5.), (10, 5.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])
```

此视图与索引字段具有相同的dtype和itemsize，因此它通常是非结构化数组，但嵌套结构除外。

```python
>>> y.dtype, y.shape, y.strides
(dtype('float32'), (2,), (12,))
```

#### 访问多个字段

可以索引并分配具有多字段索引的结构化数组，其中索引是字段名称列表

<div class="warning-warp">
<b>警告</b>

<p>多字段索引的说明将从Numpy 1.14升级Numpy 1.15。</p>
</div>

在Numpy 1.15中，使用多字段索引进行索引的结果将是原始数组的视图，如下所示：

```python
>>> a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'f4')])
>>> a[['a', 'c']]
array([(0, 0.), (0, 0.), (0, 0.)],
     dtype={'names':['a','c'], 'formats':['<i4','<f4'], 'offsets':[0,8], 'itemsize':12})
```

对视图的赋值会修改原始数组。 视图的字段将按索引编号的顺序排列。 请注意，与单字段索引不同，视图的dtype与原始数组具有相同的项目大小，并且具有与原始数组中相同的偏移量的字段，并且仅缺少未编入索引的字段。

在Numpy 1.14中，索引具有多字段索引的数组将返回上述结果的副本(对于1.15)，但将字段打包在内存中，就好像通过了``numpy.lib.recFunctions.repack_field``。这是Numpy 1.7到1.13的行为。

<div class="warning-warp">
<b>警告</b>
<p>Numpy 1.15中的新行为导致在未索引字段的位置出现额外的“填充”字节。您将需要更新所有的代码，这取决于具有“打包”布局的数据。例如下面的代码：</p>
<pre class="prettyprint language-python">
<code class="hljs">>>> a[['a','c']].view('i8')  # will fail in Numpy 1.15
ValueError: When changing to a smaller dtype, its size must be a divisor of the size of original dtype</code>
</pre>
<p>需要升级。这段代码从Numpy 1.12开始引发了 <code>FutureWarning</code> 的错误 </p>
<p>以下是修复性建议，下面这段代码在Numpy 1.14和Numpy 1.15中的作用相同：</p>
<pre class="prettyprint language-python">
<code class="hljs">>>> from numpy.lib.recfunctions import repack_fields
>>> repack_fields(a[['a','c']]).view('i8')  # supported 1.14 and 1.15
array([0, 0, 0])</code>
</pre>
</div>

赋值给具有多字段索引的数组在Numpy 1.14和Numpy 1.15中的作用相同。在两个版本中，赋值都将修改原始数组：

```python
>>> a[['a', 'c']] = (2, 3)
>>> a
array([(2, 0, 3.0), (2, 0, 3.0), (2, 0, 3.0)],
      dtype=[('a', '<i8'), ('b', '<i4'), ('c', '<f8')])
```

这遵循上述结构化阵列赋值规则。 例如，这意味着可以使用适当的多字段索引交换两个字段的值：

```python
>>> a[['a', 'c']] = a[['c', 'a']]
```

#### 用整数索引获取结构化标量

索引结构化数组的单个元素(带有整数索引)返回结构化标量：

```python
>>> x = np.array([(1, 2., 3.)], dtype='i,f,f')
>>> scalar = x[0]
>>> scalar
(1, 2., 3.)
>>> type(scalar)
numpy.void
```

与其他数值标量不同的是，结构化标量是可变的，并且像原始数组中的视图一样，因此修改标量将修改原始数组。结构化标量还支持按字段名进行访问和赋值：

```python
>>> x = np.array([(1,2),(3,4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
>>> s = x[0]
>>> s['bar'] = 100
>>> x
array([(1, 100.), (3, 4.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])
```

与元组类似，结构化标量也可以用整数索引：

```python
>>> scalar = np.array([(1, 2., 3.)], dtype='i,f,f')[0]
>>> scalar[0]
1
>>> scalar[1] = 4
```

因此，元组可能被认为是原生Python中等同于numpy的结构化的类型，就像原生python整数相当于numpy的整数类型。 可以通过调用ndarray.item将结构化标量转换为元组：

```python
>>> scalar.item(), type(scalar.item())
((1, 2.0, 3.0), tuple)
```

### 查看包含对象的结构化数组

为了防止在```numpy.Object``类型的字段中阻塞对象指针，numpy目前不允许包含对象的结构化数组的视图。

### 结构比较

如果两个空结构数组的dtype相等，则测试数组的相等性将生成一个具有原始数组的维度的布尔数组，元素设置为True，其中相应结构的所有字段都相等。如果字段名称、dtype和标题相同，而忽略endianness，且字段的顺序相同，则结构化dtype是相等的：

```python
>>> a = np.zeros(2, dtype=[('a', 'i4'), ('b', 'i4')])
>>> b = np.ones(2, dtype=[('a', 'i4'), ('b', 'i4')])
>>> a == b
array([False, False])
```

目前，如果两个void结构化数组的dtypes不相等，则比较失败返回标量值“False”。 从numpy 1.10开始不推荐使用这种方式，并且这种方式将来会引发错误或执行元素比较。


The ``<`` and ``>`` operators always return ``False`` when comparing void structured arrays, and arithmetic and bitwise operations are not supported.

在比较空结构数组时，``<`` 和 ``>`` 操作符总是返回 ``False``，并且不支持算术和位运算符。

## 记录数组

作为一个可选的方便的选项，numpy提供了一个ndarray子类 ``numpy.recarray``，以及 ``numpy.rec``子模块中的相关帮助函数，它允许通过属性访问结构化数组的字段，而不仅仅是通过索引。记录数组还使用一种特殊的数据类型 ``numpy.Record``，它允许通过属性对从数组获得的结构化标量进行字段访问。

创建记录数组的最简单方法是使用 ``numpy.rec.array``, 像下面这样：

```python
>>> recordarr = np.rec.array([(1,2.,'Hello'),(2,3.,"World")],
...                    dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
>>> recordarr.bar
array([ 2.,  3.], dtype=float32)
>>> recordarr[1:2]
rec.array([(2, 3.0, 'World')],
      dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])
>>> recordarr[1:2].foo
array([2], dtype=int32)
>>> recordarr.foo[1:2]
array([2], dtype=int32)
>>> recordarr[1].baz
'World'
```

``numpy.rec.array`` 可以将各种参数转换为记录数组，包括结构化数组：

```python
>>> arr = array([(1,2.,'Hello'),(2,3.,"World")],
...             dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
>>> recordarr = np.rec.array(arr)
```

numpy.rec模块提供了许多其他便捷的函数来创建记录数组，[请参阅记录数组创建API]((https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html#routines-array-creation-rec))。

可以使用适当的视图获取结构化数组的记录数组表示：

```python
>>> arr = np.array([(1,2.,'Hello'),(2,3.,"World")],
...                dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'a10')])
>>> recordarr = arr.view(dtype=dtype((np.record, arr.dtype)),
...                      type=np.recarray)
```

为方便起见，查看类型为``np.recarray``的ndarray会自动转换为``np.record``数据类型，因此dtype可以不在视图之外：

```python
>>> recordarr = arr.view(np.recarray)
>>> recordarr.dtype
dtype((numpy.record, [('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')]))
```

要返回普通的ndarray，必须重置dtype和type。 以下视图是这样做的，考虑到recordarr不是结构化类型的异常情况：

```python
>>> arr2 = recordarr.view(recordarr.dtype.fields or recordarr.dtype, np.ndarray)
```

如果字段具有结构化类型，则返回由index或by属性访问的记录数组字段作为记录数组，否则返回普通ndarray。

```python
>>> recordarr = np.rec.array([('Hello', (1,2)),("World", (3,4))],
...                 dtype=[('foo', 'S6'),('bar', [('A', int), ('B', int)])])
>>> type(recordarr.foo)
<type 'numpy.ndarray'>
>>> type(recordarr.bar)
<class 'numpy.core.records.recarray'>
```

请注意，如果字段与ndarray属性具有相同的名称，则ndarray属性优先。 这些字段将无法通过属性访问，但仍可通过索引访问。