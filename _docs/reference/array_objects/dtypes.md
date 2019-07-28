# 数据类型对象

数据类型对象（numpy.dtype类的实例）描述了如何解释与数组项对应的固定大小的内存块中的字节。 它描述了数据的以下方面：

1. 数据类型（整数，浮点数，Python对象等）
1. 数据的大小（例如整数中有多少字节）
1. 数据的字节顺序（little-endian或big-endian）
1. 如果数据类型是结构化的，则是其他数据类型的集合（例如，描述由整数和浮点数组成的数组项），
     1. 结构“字段”的名称是什么，通过它们可以访问它们，
     1. 每个字段的数据类型是什么，以及
     1. 每个字段占用的内存块的哪个部分。
1. 如果数据类型是子数组，那么它的形状和数据类型是什么。

为了描述标量数据的类型，NumPy中有几种内置标量类型，用于各种整数精度，浮点数等。从数组中提取的项目，例如通过索引，将是一个类型为Python的对象 是与数组的数据类型关联的标量类型。

请注意，标量类型不是dtype对象，即使在NumPy中需要数据类型规范时它们也可以代替它们。

通过创建其字段包含其他数据类型的数据类型来形成结构化数据类型。 每个字段都有一个可以访问它的名称。 父数据类型应足够大，以包含其所有字段; 父类几乎总是基于允许任意项大小的void类型。 结构化数据类型还可以在其字段中包含嵌套的结构化子数组数据类型。

最后，数据类型可以描述本身是另一种数据类型的项目数组的项目。 但是，这些子数组必须具有固定的大小。

如果使用描述子数组的数据类型创建数组，则在创建数组时，子数组的维度将附加到数组的形状。 结构化类型字段中的子数组的行为有所不同，请参阅字段访问。

子数组始终具有C连续的内存布局。

**例子**

包含32位大端整数的简单数据类型:(有关构造的详细信息，请参阅[指定和构造数据类型](#指定和构造数据类型)）

```python
>>> dt = np.dtype('>i4')
>>> dt.byteorder
'>'
>>> dt.itemsize
4
>>> dt.name
'int32'
>>> dt.type is np.int32
True
```

相应的数组标量类型是``int32``。

**例子**

结构化数据类型，包含16个字符的字符串（在字段'name'中）和包含两个64位浮点数的子数组（在字段'grade'中）：

```python
>>> dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
>>> dt['name']
dtype('|U16')
>>> dt['grades']
dtype(('float64',(2,)))
```

此数据类型的数组项包装在一个数组标量类型中，该类型也有两个字段：

```python
>>> x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
>>> x[1]
('John', [6.0, 7.0])
>>> x[1]['grades']
array([ 6.,  7.])
>>> type(x[1])
<type 'numpy.void'>
>>> type(x[1]['grades'])
<type 'numpy.ndarray'>
```

## 指定和构造数据类型

每当NumPy函数或方法中需要数据类型时，都可以提供dtype对象或可以转换为一个对象的对象。 这种转换由dtype构造函数完成：

> dtype(obj[, align, copy]) 创建数据类型对象。

可以转换为数据类型对象的内容如下所述：

dtype对象

> 按原样使用。

**None（无）**

> 默认数据类型: float_.

数组标量类型

> 24个内置数组标量类型对象都转换为关联的数据类型对象。 对于他们的子类也是如此。
> 请注意，并非所有数据类型信息都可以提供类型对象：例如，灵活数据类型的默认项大小为0，并且要求显式给定的大小有用。
 
**例子**

```python
>>>
>>> dt = np.dtype(np.int32)      # 32-bit integer
>>> dt = np.dtype(np.complex128) # 128-bit complex floating-point number
```

**通用类型**

> 通用分层类型对象根据关联转换为相应的类型对象：

数据类型 | 关联对象
---|---
number, inexact, floating | float
complexfloating | cfloat
integer, signedinteger | int_
unsignedinteger | uint
character | string
generic, flexible | void
Built-in Python | types

**内置Python类型**

当用于生成dtype对象时，几个python类型等效于相应的数组标量：

标量 | dtype
---|---
int | int_
bool | bool_
float  float_
complex | cfloat
bytes | bytes_
str | bytes_ (Python2) 或 unicode_ (Python3)
unicode | unicode_
buffer | void
(all others) | object_

请注意，str指的是空终止字节或unicode字符串，具体取决于Python版本。 在代码目标中，Python 2和3都应该使用np.unicode_作为字符串的dtype。 请参阅字符串类型的注释。

**例子**

```python
>>>
>>> dt = np.dtype(float)   # Python-compatible floating-point number
>>> dt = np.dtype(int)     # Python-compatible integer
>>> dt = np.dtype(object)  # Python object
```

**带.dtype的类型**

> 具有dtype属性的任何类型对象：将直接访问和使用该属性。 该属性必须返回可转换为dtype对象的内容。
> 可以转换几种字符串。 可以使用'>'（big-endian），'<'（little-endian）或'='（hardware-native, the default）来预先识别字符串，以指定字节顺序。

**单字符串**

> 每个内置数据类型都有一个唯一标识它的字符代码（更新的数字类型代码）。

**例子**

```python
>>>
>>> dt = np.dtype('b')  # byte, native byte order
>>> dt = np.dtype('>H') # big-endian unsigned short
>>> dt = np.dtype('<f') # little-endian single-precision float
>>> dt = np.dtype('d')  # double-precision floating-point number
```

数组协议类型字符串（请参阅数组接口）

第一个字符指定数据类型，其余字符指定每个项目的字节数，Unicode除外，其中它被解释为字符数。项目大小必须与现有类型相对应，否则将引发错误。支持的种类是：

- | -
---|---
'?' | boolean
'b' | (signed) byte
'B' | unsigned byte
'i' | (signed) integer
'u' | unsigned integer
'f' | floating-point
'c' | complex-floating point
'm' | timedelta
'M' | datetime
'O' | (Python) objects
'S', 'a' | zero-terminated bytes (not recommended)
'U' | Unicode string
'V' | raw data (void)

**例子**

```python
>>>
>>> dt = np.dtype('i4')   # 32-bit signed integer
>>> dt = np.dtype('f8')   # 64-bit floating-point number
>>> dt = np.dtype('c16')  # 128-bit complex floating-point number
>>> dt = np.dtype('a25')  # 25-length zero-terminated bytes
>>> dt = np.dtype('U25')  # 25-character string
```

**关于字符串类型的注意**

为了向后兼容Python 2，``S``和类型字符串保持零终止字节，np.string_继续映射到``np.bytes_``。 要在Python 3中使用实际字符串，请使用``U``或``np.unicode_``。 对于不需要零终止的带符号字节，可以使用b或i1。

带逗号分隔字段的字符串

> 用于指定结构化数据类型格式的简写符号是以逗号分隔的基本格式字符串。
> 此上下文中的基本格式是可选的形状说明符，后跟数组协议类型字符串。 如果形状具有多个维度，则需要在该形状上使用括号。 NumPy允许对格式进行修改，因为任何可以唯一标识类型的字符串都可用于指定字段中的数据类型。 生成的数据类型字段命名为'f0'，'f1'，...，'f <N-1>'，其中N（> 1）是字符串中逗号分隔的基本格式的数量。 如果提供了可选的形状说明符，则相应字段的数据类型描述子数组。

**例子**

- 名为f0的字段，包含32位整数
- 名为f1的字段，包含一个2 x 3的64位浮点数子数组
- 名为f2的字段，包含32位浮点数

```python
>>> dt = np.dtype("i4, (2,3)f8, f4")
```

- 名为f0的字段，包含3个字符的字符串
- 名为f1的字段，包含一个包含64位无符号整数的shape(3,)子数组
- field named f2 containing a 3 x 4 sub-array containing 10-character strings

```python
>>> dt = np.dtype("a3, 3u8, (3,4)a10")
```

输入字符串

**numpy.sctypeDict**.keys() 中的任意字符串。

**例子**

```python
>>> dt = np.dtype('uint32')   # 32-bit unsigned integer
>>> dt = np.dtype('Float64')  # 64-bit floating-point number
```

(flexible_dtype, itemsize)

第一个参数必须是转换为零大小的灵活数据类型对象的对象，第二个参数是提供所需itemsize的整数。

**例子**

```python
>>> dt = np.dtype((np.void, 10))  # 10-byte wide data block
>>> dt = np.dtype(('U', 10))   # 10-character unicode string
```

(fixed_dtype, shape)

第一个参数是可以转换为固定大小的数据类型对象的任何对象。 第二个参数是此类型的所需形状。 如果shape参数为1，则data-type对象等效于fixed dtype。 如果shape是元组，则新的dtype定义给定形状的子数组。

**例子**

```python
>>> dt = np.dtype((np.int32, (2,2)))          # 2 x 2 integer sub-array
>>> dt = np.dtype(('U10', 1))                 # 10-character string
>>> dt = np.dtype(('i4, (2,3)f8, f4', (2,3))) # 2 x 3 structured sub-array
```

[(field_name, field_dtype, field_shape), ...]

> obj应该是一个字段列表，其中每个字段由长度为2或3的元组描述。（相当于__array_interface__属性中的descr项。）
> 第一个元素field_name是字段名称（如果这是''，则分配标准字段名称'f＃'）。 字段名称也可以是2元组的字符串，其中第一个字符串是“标题”（可以是任何字符串或unicode字符串）或字段的元数据，可以是任何对象，第二个字符串是 “name”必须是有效的Python标识符。
> 第二个元素field_dtype可以是任何可以解释为数据类型的元素。
> 如果此字段表示第二个元素中数据类型的数组，则可选的第三个元素field_shape包含形状。 请注意，第三个参数等于1的3元组相当于2元组。
> 此样式不接受dtype构造函数中的align，因为假定所有内存都由数组接口的描述来计算。

**例子**

字段大（big-endian 32位整数）和little（little-endian 32位整数）的数据类型：

```python
>>> dt = np.dtype([('big', '>i4'), ('little', '<i4')])
```

数据类型包含字段R，G，B，A，每个都是无符号的8位整数：

```python
>>> dt = np.dtype([('R','u1'), ('G','u1'), ('B','u1'), ('A','u1')])
```

{'names': ..., 'formats': ..., 'offsets': ..., 'titles': ..., 'itemsize': ...}

> 此样式有两个必需键和三个可选键。 名称和格式键是必需的。 它们各自的值是具有字段名称和字段格式的等长列表。 字段名称必须是字符串，字段格式可以是dtype构造函数接受的任何对象。
> 当提供可选的键偏移和标题时，它们的值必须是与名称和格式列表长度相同的列表。 偏移值是每个字段的字节偏移（整数）列表，而标题值是每个字段的标题列表（如果该字段不需要标题，则可以使用无）。 标题可以是任何字符串或unicode对象，并将向标题键入的字段字典中添加另一个条目，并引用相同的字段元组，该元组将包含标题作为附加元组成员。
> itemsize键允许设置dtype的总大小，并且必须是足够大的整数，以便所有字段都在dtype内。 如果正在构造的dtype是对齐的，则itemsize也必须可以被struct alignment对齐。

**例子**

包含字段r，g，b，a的数据类型，每个都是一个8位无符号整数：

```python
>>> dt = np.dtype({'names': ['r','g','b','a'],
...                'formats': [uint8, uint8, uint8, uint8]})
```

带有字段r和b（带有给定标题）的数据类型，两者都是8位无符号整数，第一个位于字段开头的字节位置0，第二个位于位置2：

```python
>>> dt = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
...                'offsets': [0, 2],
...                'titles': ['Red pixel', 'Blue pixel']})
```

{'field1': ..., 'field2': ..., ...}

> 不鼓励使用这种用法，因为它与其他基于dict的构造方法不一致。 如果你有一个名为“names”的字段和一个名为“formats”的字段，则会发生冲突。
> 此样式允许传入数据类型对象的fields属性。
> obj应该包含引用（数据类型，偏移量）或（数据类型，偏移量，标题）元组的字符串或unicode键。

**例子**

数据类型包含字段col1（字节位置0处的10个字符的字符串），col2（字节位置10处的32位浮点数）和col3（字节位置14处的整数）：

```python
>>> dt = np.dtype({'col1': ('U10', 0), 'col2': (float32, 10),
    'col3': (int, 14)})
```

(base_dtype, new_dtype)

> 在NumPy 1.7及更高版本中，此表单允许将base_dtype解释为结构化dtype。 使用此dtype创建的数组将具有基础dtype base_dtype，但将具有取自new_dtype的字段和标志。 这对于创建自定义结构化dtypes很有用，就像在记录数组中一样。
> 此表单还可以指定具有重叠字段的结构类型，其功能类似于C中的“联合”类型。但是，不鼓励使用此类用法，并且首选联合机制。
> 两个参数必须可转换为具有相同总大小的数据类型对象。 .. admonition:: Example
> 32位整数，其前两个字节通过字段real解释为整数，后两个字节通过字段imag解释。

```python
>>> dt = np.dtype((np.int32,{'real':(np.int16, 0),'imag':(np.int16, 2)})
```

32位整数，被解释为由包含8位整数的形状 (4, ) 的子数组组成：

```python
>>> dt = np.dtype((np.int32, (np.int8, 4)))
```

32位整数，包含字段r，g，b，a，将整数中的4个字节解释为四个无符号整数：

```python
>>> dt = np.dtype(('i4', [('r','u1'),('g','u1'),('b','u1'),('a','u1')]))
```

**dtype**

NumPy数据类型描述是dtype类的实例。

**属性**

数据类型由以下dtype属性描述：

- dtype.type	用于实例化此数据类型的标量的类型对象。
- dtype.kind	标识一般数据类型的字符代码（'biufcmMOSUV'之一）。
- dtype.char	21种不同内置类型中每种类型的唯一字符代码。
- dtype.num	21种不同内置类型中每种类型的唯一编号。
- dtype.str	此数据类型对象的array-protocol typestring。

数据大小依次描述如下：

- dtype.name	此数据类型的位宽名称。
- dtype.itemsize	此数据类型对象的元素大小。

此数据的字节顺序：

- dtype.byteorder	一个字符，指示此数据类型对象的字节顺序。

有关结构化数据类型中子数据类型的信息：

- dtype.fields	为此数据类型定义的命名字段字典，或 None。
- dtype.names	有序的字段名称列表，如果没有字段，则为None。

对于描述子数组的数据类型：

- dtype.subdtype	如果此dtype描述子数组，则为元组（item_dtype，shape），否则为None。
- dtype.shape	如果此数据类型描述子数组，则子数组的形状元组，否则为 ()。

提供附加信息的属性：

- dtype.hasobject	布尔值，指示此dtype是否包含任何字段或子数据类型中的任何引用计数对象。
- dtype.flags	描述如何解释此数据类型的位标志。
- dtype.isbuiltin	整数表示此dtype与内置dtypes的关系。
- dtype.isnative	布尔值，指示此dtype的字节顺序是否为平台本机。
- dtype.descr	PEP3118接口描述了数据类型。
- dtype.alignment	根据编译器，此数据类型所需的对齐（字节）。

## 方法

数据类型具有以下更改字节顺序的方法：

- dtype.newbyteorder([new_order])	返回具有不同字节顺序的新dtype。

以下方法实现了pickle（腌制）协议：

- dtype.__reduce__
- dtype.__setstate__