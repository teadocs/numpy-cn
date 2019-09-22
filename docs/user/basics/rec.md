# 结构化数组

## 介绍

结构化数组是ndarray，其数据类型是由一系列命名[字段](https://numpy.org/devdocs/glossary.html#term-field)组织的简单数据类型组成。例如：

``` python
>>> x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
...              dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
>>> x
array([('Rex', 9, 81.), ('Fido', 3, 27.)],
      dtype=[('name', 'U10'), ('age', '<i4'), ('weight', '<f4')])
```

``x`` 是一个长度为2的一维数组，其数据类型是一个包含三个字段的结构：

1. 长度为10或更少的字符串，名为“name”。
2. 一个32位整数，名为“age”。
3. 一个32位的名为'weight'的float类型。

如果您``x``在位置1处索引，则会得到一个结构：

``` python
>>> x[1]
('Fido', 3, 27.0)
```

您可以通过使用字段名称建立索引来访问和修改结构化数组的各个字段：

``` python
>>> x['age']
array([9, 3], dtype=int32)
>>> x['age'] = 5
>>> x
array([('Rex', 5, 81.), ('Fido', 5, 27.)],
      dtype=[('name', 'U10'), ('age', '<i4'), ('weight', '<f4')])
```

结构化数据类型旨在能够模仿C语言中的“结构”，并共享类似的内存布局。它们用于连接C代码和低级操作结构化缓冲区，例如用于解释二进制blob。出于这些目的，它们支持诸如子数组，嵌套数据类型和联合之类的专用功能，并允许控制结构的内存布局。

希望操纵表格数据的用户（例如存储在csv文件中）可能会发现其他更适合的pydata项目，例如xarray，pandas或DataArray。这些为表格数据分析提供了高级接口，并且针对该用途进行了更好的优化。例如，numpy中结构化数组的类似C-struct的内存布局可能导致较差的缓存行为。

## 结构化数据类型

结构化数据类型可以被认为是一定长度的字节序列（结构的项目[大小](https://numpy.org/devdocs/glossary.html#term-itemsize)），它被解释为字段集合。每个字段在结构中都有一个名称，一个数据类型和一个字节偏移量。字段的数据类型可以是包括其他结构化数据类型的任何numpy数据类型，也可以是[子行数据类型](https://numpy.org/devdocs/glossary.html#term-subarray-data-type)，其行为类似于指定形状的ndarray。字段的偏移是任意的，字段甚至可以重叠。这些偏移量通常由numpy自动确定，但也可以指定。

### 结构化数据类型创建

可以使用该函数创建结构化数据类型[``numpy.dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype)。有4种不同的规范形式，
其灵活性和简洁性各不相同。这些在 “[数据类型对象](/reference/arrays/dtypes.html)” 参考页面中进一步记录，总结如下：

1. 元组列表，每个字段一个元组

    每个元组都具有以下形式``（字段名称、数据类型、形状）``，其中Shape是可选的。
    ``fieldname`` 是字符串（如果使用标题，则为元组，请参见下面的[字段标题](#字段标题)），
    datatype 可以是任何可转换为数据类型的对象，而 ``shape`` 是指定子数组形状的整数元组。

    ``` python
    >>> np.dtype([('x', 'f4'), ('y', np.float32), ('z', 'f4', (2, 2))])
    dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4', (2, 2))])
    ```

    如果 ``fieldname`` 是空字符串 ``''`` ，则将为字段指定格式为 ``f#`` 的默认名称，
    其中 ``#`` 是字段的整数索引，从左侧开始从0开始计数：

    ``` python
    >>> np.dtype([('x', 'f4'), ('', 'i4'), ('z', 'i8')])
    dtype([('x', '<f4'), ('f1', '<i4'), ('z', '<i8')])
    ```

    自动确定结构内字段的字节偏移量和总结构项大小。
1. 逗号分隔的数据类型规范字符串

    在这个速记符号中，任何 [字符串dtype规范](/reference/arrays/dtypes.html#arrays-dtypes-constructing) 都可以在字符串中使用，
    并用逗号分隔。
    字段的项目大小和字节偏移是自动确定的，并且字段名称被赋予默认名称 ``f0``、``f1``等。

    ``` python
    >>> np.dtype('i8, f4, S3')
    dtype([('f0', '<i8'), ('f1', '<f4'), ('f2', 'S3')])
    >>> np.dtype('3int8, float32, (2, 3)float64')
    dtype([('f0', 'i1', (3,)), ('f1', '<f4'), ('f2', '<f8', (2, 3))])
    ```
1. 字段参数组字典

    这是最灵活的规范形式，因为它允许控制字段的字节偏移和结构的项目大小。

    字典有两个必需键 “names” 和 “format”，以及四个可选键 “offsets”、“itemsize”、“Aligned” 和 “title”。
    名称和格式的值应该分别是相同长度的字段名列表和dtype规范列表。
    可选的 “offsets” 值应该是整数字节偏移量的列表，结构中的每个字段都有一个偏移量。
    如果未给出 “Offsets” ，则自动确定偏移量。可选的 “itemsize” 值应该是一个整数，
    描述dtype的总大小（以字节为单位），它必须足够大以包含所有字段。

    ``` python
    >>> np.dtype({'names': ['col1', 'col2'], 'formats': ['i4', 'f4']})
    dtype([('col1', '<i4'), ('col2', '<f4')])
    >>> np.dtype({'names': ['col1', 'col2'],
    ...           'formats': ['i4', 'f4'],
    ...           'offsets': [0, 4],
    ...           'itemsize': 12})
    dtype({'names':['col1','col2'], 'formats':['<i4','<f4'], 'offsets':[0,4], 'itemsize':12})
    ```

    可以选择偏移量，使得字段重叠，尽管这将意味着分配给一个字段可能会破坏任何重叠字段的数据。
    作为一个例外，numpy.object类型的字段不能与其他字段重叠，因为存在破坏内部对象指针然后取消引用它的风险。

    可选的“Aligned”值可以设置为True，以使自动偏移计算使用对齐的偏移量（请参阅[自动字节偏移量和对齐](#自动字节偏移和对齐)），
    就好像numpy.dtype的“Align”关键字参数已设置为True一样。

    可选的 ‘titles’ 值应该是长度与 ‘names’ 相同的标题列表，请参阅下面的[字段标题](#字段标题)。
1. 字段名称字典
    不鼓励使用这种形式的规范，但这里有文档记录，因为较旧的numpy代码可能会使用它。
    字典的关键字是字段名称，值是指定类型和偏移量的元组：

    ``` python
    >>> np.dtype({'col1': ('i1', 0), 'col2': ('f4', 1)})
    dtype([('col1', 'i1'), ('col2', '<f4')])
    ```

    不鼓励使用这种形式，因为Python字典在Python 3.6之前的Python版本中不保留顺序，
    并且结构化dtype中字段的顺序有意义。[字段标题](#字段标题)可以通过使用3元组来指定，见下文。  

### 操作和显示结构化数据类型

可以``names``
在dtype对象的属性中找到结构化数据类型的字段名称列表：

``` python
>>> d = np.dtype([('x', 'i8'), ('y', 'f4')])
>>> d.names
('x', 'y')
```

可以通过``names``使用相同长度的字符串序列分配属性来修改字段名称。

dtype对象还具有类似字典的属性，``fields``其键是字段名称（和[字段标题](#字段标题)，见下文），
其值是包含每个字段的dtype和字节偏移量的元组。

``` python
>>> d.fields
mappingproxy({'x': (dtype('int64'), 0), 'y': (dtype('float32'), 8)})
```

对于非结构化数组，``names``和``fields``属性都相同``None``。
测试 *dtype* 是否结构化的推荐方法是， *如果dt.names不是None* 而不是 *dt.names* ，则考虑具有0字段的dtypes。

如果可能，结构化数据类型的字符串表示形式显示在“元组列表”表单中，否则numpy将回退到使用更通用的字典表单。

### 自动字节偏移和对齐

Numpy使用两种方法之一自动确定字段字节偏移量和结构化数据类型的总项目大小，具体取决于是否
 ``align=True``指定为关键字参数[``numpy.dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype)。

默认情况下（``align=False``），numpy将字段打包在一起，使得每个字段从前一个字段结束的字节偏移开始，并且字段在内存中是连续的。

``` python
>>> def print_offsets(d):
...     print("offsets:", [d.fields[name][1] for name in d.names])
...     print("itemsize:", d.itemsize)
>>> print_offsets(np.dtype('u1, u1, i4, u1, i8, u2'))
offsets: [0, 1, 2, 6, 7, 15]
itemsize: 17
```

如果``align=True``设置了，numpy将以与许多C编译器填充C结构相同的方式填充结构。在某些情况下，对齐结构可以提高性能，但代价是增加了数据类型的大小。在字段之间插入填充字节，使得每个字段的字节偏移量将是该字段对齐的倍数，对于简单数据类型，通常等于字段的字节大小，请参阅[``PyArray_Descr.alignment``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Descr.alignment)。该结构还将添加尾随填充，以使其itemsize是最大字段对齐的倍数。

``` python
>>> print_offsets(np.dtype('u1, u1, i4, u1, i8, u2', align=True))
offsets: [0, 1, 4, 8, 16, 24]
itemsize: 32
```

请注意，尽管默认情况下几乎所有现代C编译器都以这种方式填充，但C结构中的填充依赖于C实现，因此不能保证此内存布局与C程序中相应结构的内容完全匹配。为了获得确切的对应关系，可能需要在numpy侧或C侧进行一些工作。

如果使用``offsets``基于字典的dtype规范中的可选键指定了偏移量，则设置``align=True``将检查每个字段的偏移量是其大小的倍数，并且itemsize是最大字段大小的倍数，如果不是，则引发异常。

如果结构化数组的字段和项目大小的偏移满足对齐条件，则数组将具有该``ALIGNED`` [``flag``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flags.html#numpy.ndarray.flags)集合。

便捷函数[``numpy.lib.recfunctions.repack_fields``](#numpy.lib.recfunctions.repack_fields)将对齐的dtype或数组转换为打包的dtype或数组，反之亦然。它需要一个dtype或结构化的ndarray作为参数，并返回一个带有字段重新打包的副本，带或不带填充字节。

### 字段标题

除了字段名称之外，字段还可以具有关联的[标题](https://numpy.org/devdocs/glossary.html#term-title)，备用名称，有时用作字段的附加说明或别名。标题可用于索引数组，就像字段名一样。

要在使用dtype规范的list-of-tuples形式时添加标题，可以将字段名称指定为两个字符串的元组而不是单个字符串，它们分别是字段的标题和字段名称。例如：

``` python
>>> np.dtype([(('my title', 'name'), 'f4')])
dtype([(('my title', 'name'), '<f4')])
```

当使用第一种形式的基于字典的规范时，标题可以``'titles'``作为如上所述的额外密钥提供。当使用第二个（不鼓励的）基于字典的规范时，可以通过提供3元素元组而不是通常的2元素元组来提供标题：``(datatype, offset, title)``

``` python
>>> np.dtype({'name': ('i4', 0, 'my title')})
dtype([(('my title', 'name'), '<i4')])
```

该``dtype.fields``字典将包含标题作为键，如果使用任何头衔。这有效地表示具有标题的字段将在字典字典中表示两次。这些字段的元组值还将具有第三个元素，即字段标题。因此，并且因为``names``属性保留了字段顺序而``fields``
属性可能没有，所以建议使用dtype的``names``属性迭代dtype的字段，该属性不会列出标题，如：

``` python
>>> for name in d.names:
...     print(d.fields[name][:2])
(dtype('int64'), 0)
(dtype('float32'), 8)
```

### 联合类型

默认情况下，结构化数据类型在numpy中实现为基本类型 ``numpy.void``，
但是可以使用 [数据类型对象中](/reference/arrays/dtypes.html) 中描述的dtype规范的 ``(base_dtype, dtype)`` 
形式将其他 numpy 类型解释为结构化类型。
这里，``base_dtype`` 是所需的底层 ``dtype``，字段和标志将从dtype复制。此 dtype 类似于 C 中的“Union”。

## 索引和分配给结构化数组

### 将数据分配给结构化数组

有许多方法可以为结构化数组赋值：使用python元组，使用标量值或使用其他结构化数组。

#### 从Python本机类​​型（元组）分配

为结构化数组赋值的最简单方法是使用python元组。每个赋值应该是一个长度等于数组中字段数的元组，而不是列表或数组，因为它们将触发numpy的广播规则。元组的元素从左到右分配给数组的连续字段：

``` python
>>> x = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')
>>> x[1] = (7, 8, 9)
>>> x
array([(1, 2., 3.), (7, 8., 9.)],
     dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '<f8')])
```

#### Scalars的赋值

分配给结构化元素的标量将分配给所有字段。将标量分配给结构化数组时，或者将非结构化数组分配给结构化数组时，会发生这种情况：

``` python
>>> x = np.zeros(2, dtype='i8, f4, ?, S1')
>>> x[:] = 3
>>> x
array([(3, 3., True, b'3'), (3, 3., True, b'3')],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])
>>> x[:] = np.arange(2)
>>> x
array([(0, 0., False, b'0'), (1, 1., True, b'1')],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])
```

结构化数组也可以分配给非结构化数组，但前提是结构化数据类型只有一个字段：

``` python
>>> twofield = np.zeros(2, dtype=[('A', 'i4'), ('B', 'i4')])
>>> onefield = np.zeros(2, dtype=[('A', 'i4')])
>>> nostruct = np.zeros(2, dtype='i4')
>>> nostruct[:] = twofield
Traceback (most recent call last):
...
TypeError: Cannot cast scalar from dtype([('A', '<i4'), ('B', '<i4')]) to dtype('int32') according to the rule 'unsafe'
```

#### 来自其他结构化数组的赋值

两个结构化数组之间的分配就像源元素已转换为元组然后分配给目标元素一样。也就是说，源数组的第一个字段分配给目标数组的第一个字段，第二个字段同样分配，依此类推，而不管字段名称如何。具有不同数量的字段的结构化数组不能彼此分配。未包含在任何字段中的目标结构的字节不受影响。

``` python
>>> a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])
>>> b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])
>>> b[:] = a
>>> b
array([(0., b'0.0', b''), (0., b'0.0', b''), (0., b'0.0', b'')],
      dtype=[('x', '<f4'), ('y', 'S3'), ('z', 'O')])
```

#### 涉及子数组的分配

分配给子数组的字段时，首先将指定的值广播到子数组的形状。

### 索引结构化数组

#### 访问单个字段

可以通过使用字段名称索引数组来访问和修改结构化数组的各个字段。

``` python
>>> x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
>>> x['foo']
array([1, 3])
>>> x['foo'] = 10
>>> x
array([(10, 2.), (10, 4.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])
```

生成的数组是原始数组的视图。它共享相同的内存位置，写入视图将修改原始数组。

``` python
>>> y = x['bar']
>>> y[:] = 11
>>> x
array([(10, 11.), (10, 11.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])
```

此视图与索引字段具有相同的dtype和itemsize，因此它通常是非结构化数组，但嵌套结构除外。

``` python
>>> y.dtype, y.shape, y.strides
(dtype('float32'), (2,), (12,))
```

如果访问的字段是子数组，则子数组的维度将附加到结果的形状：

``` python
>>> x = np.zeros((2, 2), dtype=[('a', np.int32), ('b', np.float64, (3, 3))])
>>> x['a'].shape
(2, 2)
>>> x['b'].shape
(2, 2, 3, 3)
```

#### 访问多个字段

可以索引并分配具有多字段索引的结构化数组，其中索引是字段名称列表。

::: danger 警告

多字段索引的行为从Numpy 1.15变为Numpy 1.16。

:::

使用多字段索引进行索引的结果是原始数组的视图，如下所示：

``` python
>>> a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'f4')])
>>> a[['a', 'c']]
array([(0, 0.), (0, 0.), (0, 0.)],
     dtype={'names':['a','c'], 'formats':['<i4','<f4'], 'offsets':[0,8], 'itemsize':12})
```

对视图的赋值会修改原始数组。视图的字段将按其索引的顺序排列。请注意，与单字段索引不同，视图的dtype与原始数组具有相同的项目大小，并且具有与原始数组相同的偏移量的字段，并且仅缺少未编入索引的字段。

::: danger 警告

在Numpy 1.15中，使用多字段索引索引数组会返回上面结果的副本，但字段在内存中打包在一起，就像通过一样[``numpy.lib.recfunctions.repack_fields``](#numpy.lib.recfunctions.repack_fields)。

从Numpy 1.16开始的新行为导致在未编制索引的位置处的额外“填充”字节与1.15相比。您需要更新任何依赖于具有“打包”布局的数据的代码。例如代码如：

``` python
>>> a[['a', 'c']].view('i8')  # Fails in Numpy 1.16
Traceback (most recent call last):
   File "<stdin>", line 1, in <module>
ValueError: When changing to a smaller dtype, its size must be a divisor of the size of original dtype
```

需要改变。``FutureWarning``自从Numpy 1.12以来，这段代码已经提出了类似的代码，``FutureWarning``自1.7 以来也提出了类似的代码。

在1.16中，[``numpy.lib.recfunctions``](#module-numpy.lib.recfunctions)模块中引入了许多功能，
 以帮助用户解释此更改。这些是
 [``numpy.lib.recfunctions.repack_fields``](#numpy.lib.recfunctions.repack_fields)。
[``numpy.lib.recfunctions.structured_to_unstructured``](#numpy.lib.recfunctions.structured_to_unstructured)，
 [``numpy.lib.recfunctions.unstructured_to_structured``](#numpy.lib.recfunctions.unstructured_to_structured)，
 [``numpy.lib.recfunctions.apply_along_fields``](#numpy.lib.recfunctions.apply_along_fields)，
 [``numpy.lib.recfunctions.assign_fields_by_name``](#numpy.lib.recfunctions.assign_fields_by_name)，和
 [``numpy.lib.recfunctions.require_fields``](#numpy.lib.recfunctions.require_fields)。

该函数[``numpy.lib.recfunctions.repack_fields``](#numpy.lib.recfunctions.repack_fields)始终可用于重现旧行为，因为它将返回结构化数组的打包副本。例如，上面的代码可以替换为：

``` python
>>> from numpy.lib.recfunctions import repack_fields
>>> repack_fields(a[['a', 'c']]).view('i8')  # supported in 1.16
array([0, 0, 0])
```

此外，numpy现在提供了一个新功能
 [``numpy.lib.recfunctions.structured_to_unstructured``](#numpy.lib.recfunctions.structured_to_unstructured)，对于希望将结构化数组转换为非结构化数组的用户来说，这是一种更安全，更有效的替代方法，因为上面的视图通常不符合要求。此功能允许安全地转换为非结构化类型，并考虑填充，通常避免复制，并且还根据需要转换数据类型，这与视图不同。代码如：

``` python
>>> b = np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
>>> b[['x', 'z']].view('f4')
array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
```

可以通过替换为：更安全

``` python
>>> from numpy.lib.recfunctions import structured_to_unstructured
>>> structured_to_unstructured(b[['x', 'z']])
array([0, 0, 0])
```

:::

使用多字段索引分配数组会修改原始数组：

``` python
>>> a[['a', 'c']] = (2, 3)
>>> a
array([(2, 0, 3.), (2, 0, 3.), (2, 0, 3.)],
      dtype=[('a', '<i4'), ('b', '<i4'), ('c', '<f4')])
```

这遵循上述结构化数组分配规则。例如，这意味着可以使用适当的多字段索引交换两个字段的值：

``` python
>>> a[['a', 'c']] = a[['c', 'a']]
```

#### 使用整数进行索引以获得结构化标量

索引结构化数组的单个元素（带有整数索引）将返回结构化标量：

``` python
>>> x = np.array([(1, 2., 3.)], dtype='i, f, f')
>>> scalar = x[0]
>>> scalar
(1, 2., 3.)
>>> type(scalar)
<class 'numpy.void'>
```

与其他numpy标量不同，结构化标量是可变的，并且像原始数组中的视图一样，因此修改标量将修改原始数组。结构化标量还支持按字段名称进行访问和分配：

``` python
>>> x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
>>> s = x[0]
>>> s['bar'] = 100
>>> x
array([(1, 100.), (3, 4.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])
```

与元组类似，结构化标量也可以用整数索引：

``` python
>>> scalar = np.array([(1, 2., 3.)], dtype='i, f, f')[0]
>>> scalar[0]
1
>>> scalar[1] = 4
```

因此，元组可能被认为是本机Python等同于numpy的结构化类型，就像本机python整数相当于numpy的整数类型。结构化标量可以通过调用``ndarray.item``以下方式转换为元组：

``` python
>>> scalar.item(), type(scalar.item())
((1, 4.0, 3.0), <class 'tuple'>)
```

### 查看包含对象的结构化数组

为了防止``numpy.object``类型字段中的clobbering对象指针
 ，numpy当前不允许包含对象的结构化数组的视图。

### 结构比较

如果两个void结构化数组的dtypes相等，则测试数组的相等性将导致具有原始数组的维度的布尔数组，其中元素设置为``True``相应结构的所有字段相等的位置。如果字段名称，dtypes和标题相同，忽略字节顺序，并且字段的顺序相同，则结构化dtypes是相等的：

``` python
>>> a = np.zeros(2, dtype=[('a', 'i4'), ('b', 'i4')])
>>> b = np.ones(2, dtype=[('a', 'i4'), ('b', 'i4')])
>>> a == b
array([False, False])
```

目前，如果两个void结构化数组的dtypes不相等，则比较失败，返回标量值``False``。从numpy 1.10开始不推荐使用此行为，并且将来会引发错误或执行元素比较。

在``<``与``>``运营商总是返回``False``比较空洞结构数组时，与算术和位操作不被支持。

## 记录数组

作为一个可选的方便numpy [``numpy.recarray``](https://numpy.org/devdocs/reference/generated/numpy.recarray.html#numpy.recarray)在``numpy.rec``子模块中提供了一个ndarray子类，
 以及相关的辅助函数
 ，它允许按属性而不是仅通过索引访问结构化数组的字段。记录数组也使用特殊的数据类型，[``numpy.record``](https://numpy.org/devdocs/reference/generated/numpy.record.html#numpy.record)允许通过属性对从数组中获取的结构化标量进行字段访问。

创建记录数组的最简单方法是``numpy.rec.array``：

``` python
>>> recordarr = np.rec.array([(1, 2., 'Hello'), (2, 3., "World")],
...                    dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
>>> recordarr.bar
array([ 2.,  3.], dtype=float32)
>>> recordarr[1:2]
rec.array([(2, 3., b'World')],
      dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])
>>> recordarr[1:2].foo
array([2], dtype=int32)
>>> recordarr.foo[1:2]
array([2], dtype=int32)
>>> recordarr[1].baz
b'World'
```

``numpy.rec.array`` 可以将各种参数转换为记录数组，包括结构化数组：

``` python
>>> arr = np.array([(1, 2., 'Hello'), (2, 3., "World")],
...             dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
>>> recordarr = np.rec.array(arr)
```

该``numpy.rec``模块提供了许多其他便利函数来创建记录数组，请参阅[记录数组创建例程](/reference/routines/array-creation.html#routines-array-creation-rec)。

可以使用适当的视图获取结构化数组的记录数组表示：

``` python
>>> arr = np.array([(1, 2., 'Hello'), (2, 3., "World")],
...                dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'a10')])
>>> recordarr = arr.view(dtype=np.dtype((np.record, arr.dtype)),
...                      type=np.recarray)
```

为方便起见，将ndarray视为类型``np.recarray``将自动转换为``np.record``数据类型，因此dtype可以不在视图之外：

``` python
>>> recordarr = arr.view(np.recarray)
>>> recordarr.dtype
dtype((numpy.record, [('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')]))
```

要返回普通的ndarray，必须重置dtype和type。以下视图是这样做的，考虑到recordarr不是结构化类型的异常情况：

``` python
>>> arr2 = recordarr.view(recordarr.dtype.fields or recordarr.dtype, np.ndarray)
```

如果字段具有结构化类型，则返回由index或by属性访问的记录数组字段作为记录数组，否则返回普通ndarray。

``` python
>>> recordarr = np.rec.array([('Hello', (1, 2)), ("World", (3, 4))],
...                 dtype=[('foo', 'S6'),('bar', [('A', int), ('B', int)])])
>>> type(recordarr.foo)
<class 'numpy.ndarray'>
>>> type(recordarr.bar)
<class 'numpy.recarray'>
```

请注意，如果字段与ndarray属性具有相同的名称，则ndarray属性优先。这些字段将无法通过属性访问，但仍可通过索引访问。

## Recarray Helper 函数

用于操作结构化数组的实用程序的集合。

大多数这些功能最初由 John Hunter 为 matplotlib 实现。为方便起见，它们已被重写和扩展。

- numpy.lib.recfunctions.**append_fields**(*base*, *names*, *data*, *dtypes=None*, *fill_value=-1*, *usemask=True*, *asrecarray=False*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L674-L742)

  将新字段添加到现有数组。

  字段的名称使用 *names* 参数给出，相应的值使用 *data* 参数。如果追加单个字段，则 *names*、*data* 和 *dtypes* 不必是列表，只是值。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  base | array | 要扩展的输入数组。
  names | string, sequence | 对应于新字段名称的字符串或字符串序列。
  data | array or sequence of arrays | 存储要添加到基数的字段的数组或数组序列。
  dtypes | sequence of datatypes, optional | 数据类型或数据类型序列。如果没有填写，则从数据自动推断数据类型。
  fill_value | {float}, optional | 用于填充较短数组上缺少的数据的填充值。
  usemask | {False, True}, optional | 是否返回掩码数组。
  asrecarray | {False, True}, optional | 是否返回recarray(MaskedRecords)。

- numpy.lib.recfunctions.**apply_along_fields**(*func*, *arr*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L1084-L1123)

  将函数“func”简单的应用于结构化数组的各个字段的。

  这类似于 *apply_along_axis*，但将结构化数组的字段视为额外轴。这些字段首先被转换为类型提升规则后 [``numpy.result_type``](https://numpy.org/devdocs/reference/generated/numpy.result_type.html#numpy.result_type) 应用于字段的dtypes 的公共类型。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  func | function | 要应用于“field”维度的函数。此函数必须支持轴参数，如np.mean、np.sum 等。
  arr | ndarray | 要应用func的结构化数组。

  **返回值**：

  参数名 | 数据类型 | 描述
  ---|---|---
  out | ndarray | 恢复操作的结果

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
  ...              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
  >>> rfn.apply_along_fields(np.mean, b)
  array([ 2.66666667,  5.33333333,  8.66666667, 11.        ])
  >>> rfn.apply_along_fields(np.mean, b[['x', 'z']])
  array([ 3. ,  5.5,  9. , 11. ])
  ```

- numpy.lib.recfunctions.**assign_fields_by_name**(*dst*, *src*, *zero_unassigned=True*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L1130-L1166)

  通过字段名称将值从一个结构化数组分配到另一个结构化数组。

  通常在numpy>=1.14中，将一个结构化数组分配给另一个结构化数组会 “按位置” 复制字段，这意味着来自src的第一个字段被复制到DST的第一个字段，依此类推，与字段名称无关。

  此函数改为复制 “按字段名”，以便从src中的同名字段分配DST中的字段。这对嵌套结构递归适用。这就是在 numpy>=1.6 到 <=1.13 中结构赋值的工作方式。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  dst | ndarray | 略
  src | ndarray | 分配期间的源数组和目标数组。
  zero_unassigned | bool，可选 | 如果为 True，则用值0(零)填充dst中src中没有匹配字段的字段。这是numpy<=1.13的行为。如果为false，则不修改这些字段。

- numpy.lib.recfunctions.**drop_fields**(*base*, *drop_names*, *usemask=True*, *asrecarray=False*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L523-L583)

  返回一个新数组，其中 *drop_names* 中的字段已删除。

  支持嵌套字段。

  参数名 | 数据类型 | 描述
  ---|---|---
  base | array | 输入的数组
  drop_names | string or sequence | 与要删除的字段名称对应的字符串或字符串序列。
  usemask | {False, True}, optional | 是否返回掩码数组。
  asrecarray | string or sequence, optional |  是返回recarray还是mrecarray(asrecarray=True)，还是返回具有灵活dtype的普通ndarray或掩码数组。默认值为false。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
  ...   dtype=[('a', np.int64), ('b', [('ba', np.double), ('bb', np.int64)])])
  >>> rfn.drop_fields(a, 'a')
  array([((2., 3),), ((5., 6),)],
        dtype=[('b', [('ba', '<f8'), ('bb', '<i8')])])
  >>> rfn.drop_fields(a, 'ba')
  array([(1, (3,)), (4, (6,))], dtype=[('a', '<i8'), ('b', [('bb', '<i8')])])
  >>> rfn.drop_fields(a, ['ba', 'bb'])
  array([(1,), (4,)], dtype=[('a', '<i8')])
  ```

- numpy.lib.recfunctions.**find_duplicates**(*a*, *key=None*, *ignoremask=True*, *return_index=False*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L1313-L1368)

  沿给定键查找结构化数组中的重复项。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  a | array-like | 输入的数组
  key | {string, None}, optional | 要检查重复项的字段的名称。如果没有，则按记录执行搜索
  ignoremask | {True, False}, optional | 是否应丢弃淹码数据或将其视为重复数据。
  return_index | {False, True}, optional | 是否返回重复值的索引。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> ndtype = [('a', int)]
  >>> a = np.ma.array([1, 1, 1, 2, 2, 3, 3],
  ...         mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
  >>> rfn.find_duplicates(a, ignoremask=True, return_index=True)
  (masked_array(data=[(1,), (1,), (2,), (2,)],
              mask=[(False,), (False,), (False,), (False,)],
        fill_value=(999999,),
              dtype=[('a', '<i8')]), array([0, 1, 3, 4]))
  ```

- numpy.lib.recfunctions.**flatten_descr**(*ndtype*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L184-L207)

  展平结构化数据类型描述。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> ndtype = np.dtype([('a', '<i4'), ('b', [('ba', '<f8'), ('bb', '<i4')])])
  >>> rfn.flatten_descr(ndtype)
  (('a', dtype('int32')), ('ba', dtype('float64')), ('bb', dtype('int32')))
  ```

- numpy.lib.recfunctions.**get_fieldstructure**(*adtype*, *lastname=None*, *parents=None*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L240-L284)

  返回一个字典，其中的字段索引其父字段的列表。

  此函数用于简化对嵌套在其他字段中的字段的访问。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  adtype | np.dtype | 传入数据类型
  lastname | optional | 上次处理的字段名称(在递归过程中内部使用)。
  parents | dictionary | 父字段的字典(在递归期间间隔使用)。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> ndtype =  np.dtype([('A', int),
  ...                     ('B', [('BA', int),
  ...                            ('BB', [('BBA', int), ('BBB', int)])])])
  >>> rfn.get_fieldstructure(ndtype)
  ... # XXX: possible regression, order of BBA and BBB is swapped
  {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'], 'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}
  ```

- numpy.lib.recfunctions.**get_names**(*adtype*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L113-L146)

  以元组的形式返回输入数据类型的字段名称。

  **参数表**：
  
  参数名 | 数据类型 | 描述
  ---|---|---
  adtype | dtype | 输入数据类型

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> rfn.get_names(np.empty((1,), dtype=int))
  Traceback (most recent call last):
      ...
  AttributeError: 'numpy.ndarray' object has no attribute 'names'
  ```

  ``` python
  >>> rfn.get_names(np.empty((1,), dtype=[('A',int), ('B', float)]))
  Traceback (most recent call last):
      ...
  AttributeError: 'numpy.ndarray' object has no attribute 'names'
  >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])
  >>> rfn.get_names(adtype)
  ('a', ('b', ('ba', 'bb')))
  ```

- numpy.lib.recfunctions.**get_names_flat**(*adtype*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L149-L181)

  以元组的形式返回输入数据类型的字段名称。嵌套结构预先展平。

  **参数表**：
  
  参数名 | 数据类型 | 描述
  ---|---|---
  adtype | dtype | 输入数据类型

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> rfn.get_names_flat(np.empty((1,), dtype=int)) is None
  Traceback (most recent call last):
      ...
  AttributeError: 'numpy.ndarray' object has no attribute 'names'
  >>> rfn.get_names_flat(np.empty((1,), dtype=[('A',int), ('B', float)]))
  Traceback (most recent call last):
      ...
  AttributeError: 'numpy.ndarray' object has no attribute 'names'
  >>> adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])
  >>> rfn.get_names_flat(adtype)
  ('a', 'b', 'ba', 'bb')
  ```

- numpy.lib.recfunctions.**join_by**(*key*, *r1*, *r2*, *jointype='inner'*, *r1postfix='1'*, *r2postfix='2'*, *defaults=None*, *usemask=True*, *asrecarray=False*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L1377-L1554)

  在键（*key*）上加入数组 *r1* 和 *r2*。

  键应该是字符串或与用于连接数组的字段相对应的字符串序列。如果在两个输入数组中找不到*键*字段，则会引发异常。*r1* 和 *r2* 都不应该有任何沿着 *键* 的重复项：重复项的存在将使输出相当不可靠。请注意，算法不会查找重复项。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  key | {string, sequence} | 与用于比较的字段相对应的字符串或字符串序列。
  r1, r2 | arrays | 结构化数组。
  jointype | {‘inner’, ‘outer’, ‘leftouter’}, optional | 如果是'inner'，则返回r1和r2共有的元素。 如果是'outer'，则返回公共元素以及不在r2中的r1元素和不在r2中的元素。 如果是'leftouter'，则返回r1中的公共元素和r1的元素。
  r1postfix | string, optional | 附加到r1的字段名称的字符串，这些字段存在于r2中但没有键。
  r2postfix | string, optional | 附加到r1字段名称的字符串，这些字段存在于r1中但没有键。
  defaults | {dictionary}, optional | 字典将字段名称映射到相应的默认值。
  usemask | {True, False}, optional | 是否返回MaskedArray（或MaskedRecords是asrecarray == True）或ndarray。
  asrecarray | {False, True}, optional | 是否返回重新排列（如果usemask == True则返回MaskedRecords）或仅返回灵活类型的ndarray。

  ::: tip 提示

  - 输出按 key 排序。
  - 通过删除不在两个数组的键中的字段并连接结果来形成临时数组。然后对该数组进行排序，并选择公共条目。
  通过用所选条目填充字段来构造输出。如果存在一些重复的…，则不保留匹配

  :::

- numpy.lib.recfunctions.**merge_arrays**(*seqarrays*, *fill_value=-1*, *flatten=False*, *usemask=False*, *asrecarray=False*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L383-L516)

  按字段合并数组。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  seqarrays | sequence of ndarrays | 数组序列
  fill_value | {float}, optional | 填充值用于填充较短的数组上的缺失数据。
  flatten | {False, True}, optional | 是否折叠嵌套字段。
  usemask | {False, True}, optional | 是否返回掩码数组。
  asrecarray | {False, True}, optional |是否返回重新排列（MaskedRecords）。

  ::: tip 提示

  - 如果没有掩码，将使用某些内容填充缺少的值，具体取决于其对应的类型：
    - ``-1``      对于整数
    - ``-1.0``    对于浮点数
    - ``'-'``     对于字符
    - ``'-1'``    对于字符串
    - ``True``    对于布尔值
    - ``-1``      对于整数
    - ``-1.0``    对于浮点数
    - ``'-'``     对于字符
    - ``'-1'``    对于字符串
    - ``True``    对于布尔值
  - XXX: 我只是凭经验获得这些值

  :::

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> rfn.merge_arrays((np.array([1, 2]), np.array([10., 20., 30.])))
  array([( 1, 10.), ( 2, 20.), (-1, 30.)],
        dtype=[('f0', '<i8'), ('f1', '<f8')])
  ```

  ``` python
  >>> rfn.merge_arrays((np.array([1, 2], dtype=np.int64),
  ...         np.array([10., 20., 30.])), usemask=False)
  array([(1, 10.0), (2, 20.0), (-1, 30.0)],
          dtype=[('f0', '<i8'), ('f1', '<f8')])
  >>> rfn.merge_arrays((np.array([1, 2]).view([('a', np.int64)]),
  ...               np.array([10., 20., 30.])),
  ...              usemask=False, asrecarray=True)
  rec.array([( 1, 10.), ( 2, 20.), (-1, 30.)],
            dtype=[('a', '<i8'), ('f1', '<f8')])
  ```

- numpy.lib.recfunctions.**rec_append_fields**(*base*, *names*, *data*, *dtypes=None*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L751-L783)

  向现有数组添加新字段。

  字段的名称使用 *names* 参数给出，相应的值使用 *data* 参数。如果追加单个字段，则 *names*、*data* 和 *dtypes* 不必是列表，值就行。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  base | array | 要扩展的输入数组。
  names | string, sequence | 与新字段名称对应的字符串或字符串序列。
  data | array or sequence of arrays | 存储要添加到基础的字段的数组或数组序列。
  dtypes | sequence of datatypes, optional | 数据类型或数据类型序列。 如果为None，则根据数据估计数据类型。


  **返回值**：

  参数名 | 数据类型 | 描述
  ---|---|---
  appended_array | np.recarray | 略

  ::: tip 另见

  [``append_fields``](#numpy.lib.recfunctions.append_fields)

  :::

- numpy.lib.recfunctions.**rec_drop_fields**(*base*, *drop_names*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L615-L620)

  返回一个新的 numpy.recarray，其中 *drop_names* 中的字段已删除。

- numpy.lib.recfunctions.**rec_join**(*key*, *r1*, *r2*, *jointype='inner'*, *r1postfix='1'*, *r2postfix='2'*, *defaults=None*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L1563-L1576)

  在键上加入数组 *r1* 和 *r2*。join_by的替代方法，它总是返回一个 np.recarray。

  ::: tip 另见
  
  [``join_by``](#numpy.lib.recfunctions.join_by)

  :::

- numpy.lib.recfunctions.**recursive_fill_fields**(*input*, *output*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L43-L79)

  使用输入中的字段填充输出中的字段，并支持嵌套结构。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  input | ndarray | 输入的数组
  output | ndarray | 输出的数组

  ::: tip 提示

  - *输出*应至少与*输入*的大小相同


  :::

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> a = np.array([(1, 10.), (2, 20.)], dtype=[('A', np.int64), ('B', np.float64)])
  >>> b = np.zeros((3,), dtype=a.dtype)
  >>> rfn.recursive_fill_fields(a, b)
  array([(1, 10.), (2, 20.), (0,  0.)], dtype=[('A', '<i8'), ('B', '<f8')])
  ```

- numpy.lib.recfunctions.**rename_fields**(*base*, *namemapper*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L627-L664)

  重命名来自灵活数据类型 ndarray 或 recarray 的字段。

  支持嵌套字段。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  base | ndarray | 必须修改其字段的输入数组。
  namemapper | dictionary | 将旧字段名映射到新版本的字典对象。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],
  ...   dtype=[('a', int),('b', [('ba', float), ('bb', (float, 2))])])
  >>> rfn.rename_fields(a, {'a':'A', 'bb':'BB'})
  array([(1, (2., [ 3., 30.])), (4, (5., [ 6., 60.]))],
        dtype=[('A', '<i8'), ('b', [('ba', '<f8'), ('BB', '<f8', (2,))])])
  ```

- numpy.lib.recfunctions.**repack_fields**(*a*, *align=False*, *recurse=False*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L790-L869)

  在内存中重新打包结构化数组或dtype的字段。

  结构化数据类型的内存布局允许任意字节偏移的字段。这意味着字段可以通过填充字节来分隔，它们的偏移量可以是非单调增加的，并且它们可以重叠。

  此方法删除所有重叠并重新排序内存中的字段，使它们具有增加的字节偏移量，并根据 *align* 选项添加或删除填充字节，该选项的行为类似于 ``np.dtype`` 的 *align* 选项。

  如果 *align=False*，则此方法生成“压缩”内存布局，其中每个字段从前一个字段结束的字节开始，并删除所有填充字节。

  如果 *align=True*，则此方法通过根据需要添加填充字节来生成 “对齐” 内存布局，其中每个字段的偏移量是其对齐方式的倍数，而总项目大小是最大对齐方式的倍数。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  a | ndarray or dtype | 要重新打包字段的数组或数据类型。
  align | boolean | 如果为真，则使用“对齐”内存布局，否则使用“打包”布局。
  recurse | boolean | 如果为True，还会重新打包嵌套结构。

  **返回值**：

  参数名 | 数据类型 | 描述
  ---|---|---
  repacked | ndarray or dtype | 带字段重新打包的副本，如果不需要重新打包，则副本本身。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> def print_offsets(d):
  ...     print("offsets:", [d.fields[name][1] for name in d.names])
  ...     print("itemsize:", d.itemsize)
  ...
  >>> dt = np.dtype('u1, <i8, <f8', align=True)
  >>> dt
  dtype({'names':['f0','f1','f2'], 'formats':['u1','<i8','<f8'], 'offsets':[0,8,16], 'itemsize':24}, align=True)
  >>> print_offsets(dt)
  offsets: [0, 8, 16]
  itemsize: 24
  >>> packed_dt = rfn.repack_fields(dt)
  >>> packed_dt
  dtype([('f0', 'u1'), ('f1', '<i8'), ('f2', '<f8')])
  >>> print_offsets(packed_dt)
  offsets: [0, 1, 9]
  itemsize: 17
  ```

- numpy.lib.recfunctions.**require_fields**(*array*, *required_dtype*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L1171-L1212)

  使用字段名赋值将结构化数组强制转换为新的dtype。

  此函数按名称从旧数组分配到新数组，因此输出数组中字段的值是源数组中具有相同名称的字段的值。这具有创建新的ndarray的效果，该ndarray仅包含Required_dtype“必需”的字段。

  如果在输入数组中不存在Required_dtype中的字段名称，则会在输出数组中创建该字段并将其设置为0。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  a | ndarray | 要强制转换的数组
  required_dtype | dtype | 输出数组的数据类型

  **Returns**：

  参数名 | 数据类型 | 描述
  ---|---|---
  out | ndarray | 具有新dtype的数组，具有从具有相同名称的输入数组中的字段复制的字段值

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> a = np.ones(4, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
  >>> rfn.require_fields(a, [('b', 'f4'), ('c', 'u1')])
  array([(1., 1), (1., 1), (1., 1), (1., 1)],
    dtype=[('b', '<f4'), ('c', 'u1')])
  >>> rfn.require_fields(a, [('b', 'f4'), ('newf', 'u1')])
  array([(1., 0), (1., 0), (1., 0), (1., 0)],
    dtype=[('b', '<f4'), ('newf', 'u1')])
  ```

- numpy.lib.recfunctions.**stack_arrays**(*arrays*, *defaults=None*, *usemask=True*, *asrecarray=False*, *autoconvert=False*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L1220-L1305)

  按字段叠加数组字段

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  arrays | array or sequence | 输入数组序列。
  defaults | dictionary, optional | 字典将字段名称映射到相应的默认值。
  usemask | {True, False}, optional | 是否返回MaskedArray(或MaskedRecords为asrecarray=True)或ndarray。
  asrecarray | {False, True}, optional | 是返回一个recarray(如果usemask=True则返回MaskedRecords)，还是只返回一个灵活类型的ndarray。
  autoconvert | {False, True}, optional | 是否自动将字段类型强制转换为最大值。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> x = np.array([1, 2,])
  >>> rfn.stack_arrays(x) is x
  True
  >>> z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])
  >>> zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
  ...   dtype=[('A', '|S3'), ('B', np.double), ('C', np.double)])
  >>> test = rfn.stack_arrays((z,zz))
  >>> test
  masked_array(data=[(b'A', 1.0, --), (b'B', 2.0, --), (b'a', 10.0, 100.0),
                    (b'b', 20.0, 200.0), (b'c', 30.0, 300.0)],
              mask=[(False, False,  True), (False, False,  True),
                    (False, False, False), (False, False, False),
                    (False, False, False)],
        fill_value=(b'N/A', 1.e+20, 1.e+20),
              dtype=[('A', 'S3'), ('B', '<f8'), ('C', '<f8')])
  ```

- numpy.lib.recfunctions.**structured_to_unstructured**(*arr*, *dtype=None*, *copy=False*, *casting='unsafe'*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L894-L977)

  将和n-D结构化数组转换为(n+1)-D非结构化数组。

  新的数组将具有新的最后一个维度，其大小等于输入数组的字段元素的数量。如果未提供，则根据应用于所有字段数据类型的numpy类型提升规则确定输出数据类型。

  嵌套字段以及任何子数组字段的每个元素都算作单个字段元素。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  arr | ndarray | 要转换的结构化数组或数据类型。不能包含对象数据类型。
  dtype | dtype, optional | 输出非结构化数组的数据类型。
  copy | bool, optional | 请参见将参数复制到ndarray.astype。如果为true，则始终返回副本。如果为false，并且满足dtype要求，则返回视图。
  casting | {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional | 请参见转换ndarray.astype的参数。控制可能发生的数据转换类型。

  **返回值**：

  参数名 | 数据类型 | 描述
  ---|---|---
  unstructured | ndarray | 多一维的非结构化数组。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
  >>> a
  array([(0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.]),
        (0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.])],
        dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])
  >>> rfn.structured_to_unstructured(a)
  array([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
  ```

  ``` python
  >>> b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],
  ...              dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
  >>> np.mean(rfn.structured_to_unstructured(b[['x', 'z']]), axis=-1)
  array([ 3. ,  5.5,  9. , 11. ])
  ```

- numpy.lib.recfunctions.**unstructured_to_structured**(*arr*, *dtype=None*, *names=None*, *align=False*, *copy=False*, *casting='unsafe'*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/recfunctions.py#L984-L1079)

  将n-D非结构化数组转换为(n-1)-D结构化数组。

  输入数组的最后一维被转换为结构，字段元素的数量等于输入数组的最后一维的大小。默认情况下，所有输出字段都具有输入数组的dtype，但是可以提供具有相等数量的field-element的输出结构化dtype。

  嵌套字段以及任何子数组字段的每个元素都计入字段元素的数量。

  **参数表**：

  参数名 | 数据类型 | 描述
  ---|---|---
  arr | ndarray | 要转换的非结构化数组或数据类型。
  dtype | dtype, optional | 输出数组的结构化数据类型。
  names | list of strings, optional | 如果未提供dtype，则按顺序指定输出dtype的字段名称。字段dtype将与输入数组相同。
  align | boolean, optional | 是否创建对齐的内存布局。
  copy | bool, optional | 请参见将参数复制到ndarray.astype。如果为true，则始终返回副本。如果为false，并且满足dtype要求，则返回视图。
  casting | {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional | 请参见转换ndarray.astype的参数。控制可能发生的数据转换类型。

  **返回值**：

  参数名 | 数据类型 | 描述
  ---|---|---
  structured | ndarray | 维数较少的结构化数组。

  **示例**：

  ``` python
  >>> from numpy.lib import recfunctions as rfn
  >>> dt = np.dtype([('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
  >>> a = np.arange(20).reshape((4,5))
  >>> a
  array([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]])
  >>> rfn.unstructured_to_structured(a, dt)
  array([( 0, ( 1.,  2), [ 3.,  4.]), ( 5, ( 6.,  7), [ 8.,  9.]),
        (10, (11., 12), [13., 14.]), (15, (16., 17), [18., 19.])],
        dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])
  ```
