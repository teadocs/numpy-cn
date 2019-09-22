# 数据类型对象（``dtype``）

数据类型对象（[``numpy.dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype)类的实例）描述了如何解释与数组项对应的固定大小的内存块中的字节。
它描述了数据的以下几个方面：

1. 数据类型（整型、浮点型、Python对象等）。
1. 数据的大小（例如整数中有多少字节）。
1. 数据的顺序（[little-endian](https://numpy.org/devdocs/glossary.html#term-little-endian) 或 [big-endian](https://numpy.org/devdocs/glossary.html#term-big-endian)）。
1. 如果数据类型是[结构化数据类型](https://numpy.org/devdocs/glossary.html#term-structured-data-type)，则是其他数据类型的集合(例如，描述由整数和浮点数组成的数组项)。
    1. 结构的 “[字段](https://numpy.org/devdocs/glossary.html#term-field)” 的名称是什么，通过这些名称可以[访问](indexing.html#arrays-indexing-fields)它们。
    1. 每个 [字段](https://numpy.org/devdocs/glossary.html#term-field) 的数据类型是什么，以及
    1. 每个字段占用内存块的哪一部分。
1. 如果数据类型是子数组，那么它的形状和数据类型是什么。

为了描述标量数据的类型，在NumPy中存在用于整数、浮点数等的各种精度的几个[内置标量类型](scalars.html)。
*例如*，*通过* 索引从数组中提取的项将是其类型是与数组的数据类型相关联的标量类型的 Python 对象。

请注意，标量类型不是 [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) 对象，即使在NumPy中需要数据类型规范时可以使用它们来代替dtype对象。

结构化数据类型是通过创建其[字段](https://numpy.org/devdocs/glossary.html#term-field)包含其他数据类型的数据类型来形成的。
每个字段都有一个可以用来[访问](indexing.html)它的名称。
父数据类型应该有足够的大小来包含它的所有字段；父数据类型几乎总是基于void类型，该类型允许任意的项大小。
结构化数据类型还可以在其字段中包含嵌套的结构子数组数据类型。

最后，数据类型可以描述本身就是另一种数据类型的项的数组的项。然而，这些子数组必须具有固定的大小。

如果使用描述子数组的数据类型创建数组，则在创建数组时，子数组的维数将附加到数组的形状。
结构化类型字段中的子数组的行为有所不同，请参阅[字段访问](indexing.html)。

子数组总是具有 C-contiguous 的内存布局。

**示例：**

包含32位 big-endia 整数的简单数据类型：（有关构造的详细信息，请参阅[指定和构造数据类型](#指定和构造数据类型)）

``` python
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

对应的数组标量类型为 ``int32``。

**示例：**

结构化数据类型，包含16个字符的字符串（在字段“name”中）和两个64位浮点数的子数组（在字段“grades”中）：

``` python
>>> dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
>>> dt['name']
dtype('|U16')
>>> dt['grades']
dtype(('float64',(2,)))
```

此数据类型的数组的项包装在[数组标量](scalars.html)类型中，该数组标量类型也有两个字段：

``` python
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

每当在NumPy函数或方法中需要数据类型时，都可以提供 [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) 
对象或可以转换为 [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) 的对象。此类转换由dtype构造函数完成：

方法 | 描述
---|---
[dtype](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype)(obj[, align, copy]) | 创建数据类型对象。

可以转换为数据类型对象的内容如下：

- [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) object

  按原样使用。

- ``None``

  默认数据类型：`float_``。

- Array-scalar 类型

  24个内置[数组标量类型对象](scalars.html)都转换为关联的数据类型对象。对于他们的子类也是如此。

  请注意，并非所有数据类型信息都可以与type-object一起提供：例如，灵活数据类型的默认 *itemsize* 为 0，并且需要显式给定的大小才有用。

  ``` python
  >>> dt = np.dtype(np.int32)      # 32-bit integer
  >>> dt = np.dtype(np.complex128) # 128-bit complex floating-point number
  ```

- 泛型类型

  泛型分层类型对象根据关联转换为相应的类型对象：

  类型 | 类型
  ---|---
  number, inexact, [float](https://docs.python.org/dev/library/functions.html#float)ing | float
  complexfloating | cfloat
  integer, signedinteger | int_
  unsignedinteger | uint
  character | string
  [generic](https://numpy.org/devdocs/reference/generated/numpy.generic.html#numpy.generic), flexible | void

- 内置Python类型

  几个python类型在用于生成 [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) 对象时等效于相应的数组标量：

  类型 | 类型
  ---|---
  [int](https://docs.python.org/dev/library/functions.html#int) | int_
  [bool](https://docs.python.org/dev/library/functions.html#bool) | bool_
  [float](https://docs.python.org/dev/library/functions.html#float) | float_
  [complex](https://docs.python.org/dev/library/functions.html#complex) | cfloat
  [bytes](https://docs.python.org/dev/library/stdtypes.html#bytes) | bytes_
  [str](https://docs.python.org/dev/library/stdtypes.html#str) | bytes_ (Python2) 或者 unicode_ (Python3)
  unicode | unicode_
  buffer | void
  (all others) | object_

  请注意，``str`` 指的是以 null 结尾的字节或Unicode字符串，具体取决于Python版本。
  在同时面向Python 2和3的代码中，``np.unicode_`` 应该用作字符串的dtype。参见本页面的字符串类型说明。

  **示例：**

  ``` python
  >>> dt = np.dtype(float)   # Python-compatible floating-point number
  >>> dt = np.dtype(int)     # Python-compatible integer
  >>> dt = np.dtype(object)  # Python object
  ```

- 带有 ``.dtype`` 的类型

  具有 ``dtype`` 属性的任何类型对象：将直接访问和使用该属性。该属性必须返回可转换为dtype对象的内容。

可以转换几种类型的字符串。
可以在识别的字符串前面加上 ``'>'`` （[big-endian](https://numpy.org/devdocs/glossary.html#term-big-endian)）、
``'<'`` （[little-endian](https://numpy.org/devdocs/glossary.html#term-little-endian)） 
或 ``'='`` (hardware-native、默认值)，以指定字节顺序。

- 单字符串（One-character strings）

  每个内置数据类型都有一个字符代码（更新后的数字类型代码），用于唯一地标识它。

  **示例：**

  ``` python
  >>> dt = np.dtype('b')  # byte, native byte order
  >>> dt = np.dtype('>H') # big-endian unsigned short
  >>> dt = np.dtype('<f') # little-endian single-precision float
  >>> dt = np.dtype('d')  # double-precision floating-point number
  ```

- Array-protocol 类型字符串（请参阅 [数组接口](interface.html#数组接口)）

  第一个字符指定数据的类型，其余的字符指定每个项目的字节数，Unicode除外，在Unicode中，它被解释为字符数。
  项大小必须对应于现有类型，否则将引发错误。支持的类型包括：

  代码 | 类型
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

  **示例：**

  ``` python
  >>> dt = np.dtype('i4')   # 32-bit signed integer
  >>> dt = np.dtype('f8')   # 64-bit floating-point number
  >>> dt = np.dtype('c16')  # 128-bit complex floating-point number
  >>> dt = np.dtype('a25')  # 25-length zero-terminated bytes
  >>> dt = np.dtype('U25')  # 25-character string
  ```

  **关于字符串类型的注解：**

  为了与 Python 2 向后兼容，``S`` 和 ``a`` 类型字符串仍然是以零结尾的字节，``np.string_`` 继续映射到 ``np.bytes_``。
  要在Python3中使用实际字符串，请使用 ``U`` 或 ``np.unicode_``。
  对于不需要零终止的有符号字节，可以使用 ``b``` 或 ``i1``。

- 带有逗号分隔字段的字符串

  用于指定结构化数据类型的格式的简写表示法是基本格式的逗号分隔字符串。

  此上下文中的基本格式是可选的形状说明符，后跟数组协议类型字符串。
  如果形状有多个维度，则需要在形状上加上括号。
  NumPy允许对格式进行修改，因为可以唯一标识类型的任何字符串都可以用于指定字段中的数据类型。
  生成的数据类型字段命名为 ``'f0'``、``'f1'``、...、``'f'`` 其中 N (>1) 是字符串中以逗号分隔的基本格式的数量。
  如果提供了可选的形状说明符，则相应字段的数据类型描述一个子数组。

  **示例：**

  - 名为 ``f0`` 的字段包含32位整数。
  - 名为 ``f1`` 的字段，包含一个由64位浮点数组成的 2 x 3 子数组。
  - 名为 ``f2`` 的字段包含32位浮点数。

  ``` python
  >>> dt = np.dtype("i4, (2,3)f8, f4")
  ```

  - 名为 ``f0`` 的字段包含3个字符的字符串。
  - 名为 ``f1`` 的字段包含 shape(3,) 的子数组，其中包含64位无符号整数。
  - 名为 ``f2`` 的字段包含10个字符串的 3 x 4 子数组。

  ``` python
  >>> dt = np.dtype("a3, 3u8, (3,4)a10")
  ```

- 类型字符串

  ``numpy.sctypeDict``.keys() 中的任何字符串：

  **示例：**

  ``` python
  >>> dt = np.dtype('uint32')   # 32-bit unsigned integer
  >>> dt = np.dtype('Float64')  # 64-bit floating-point number
  ```

- ``(flexible_dtype, itemsize)``

  第一个参数必须是转换为零大小的灵活数据类型对象的对象，第二个参数是提供所需项大小的整数。

  **示例：**

  ``` python
  >>> dt = np.dtype((np.void, 10))  # 10-byte wide data block
  >>> dt = np.dtype(('U', 10))   # 10-character unicode string
  ```

- ``(fixed_dtype, shape)``

  第一个参数是可以转换为固定大小的数据类型对象的任何对象。
  第二个参数是此类型所需的形状。如果Shape参数为1，则数据类型对象等同于固定dtype。
  如果 *shape* 是一个元组，那么新的dtype定义了给定形状的子数组。

  **示例：**

  ``` python
  >>> dt = np.dtype((np.int32, (2,2)))          # 2 x 2 integer sub-array
  >>> dt = np.dtype(('U10', 1))                 # 10-character string
  >>> dt = np.dtype(('i4, (2,3)f8, f4', (2,3))) # 2 x 3 structured sub-array
  ```

- ``[(field_name, field_dtype, field_shape), ...]``

  *obj* 应该是字段列表，其中每个字段由长度为2或3的元组描述。(相当于 [``__array_interface__``](interface.html#__array_interface__) 属性中的 ``descr`` 项。)

  第一个元素 *field_name* 是字段名称(如果这是 ``''`` ，则分配标准字段名称 ``'f#'``)。字段名称也可以是字符串的2元组，其中第一个字符串是“title”（可以是任何字符串或Unicode字符串）或字段的元数据，该字段可以是任何对象，而第二个字符串是“name”，它必须是有效的Python标识符。

  第二个元素 *field_dtype* 可以是任何可以解释为数据类型的元素。

  如果该字段表示第二个元素中数据类型的数组，则可选的第三个元素 *field_shape* 包含形状。请注意，第三个参数等于1的3元组等同于2元组。

  这种样式不接受 [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) 构造函数中的 *align* ，因为假设所有内存都由数组接口描述说明。
  
  **示例：**

  具有字段 ``big`` (Big-Endian 32位整数）和 ``little``（Little-Endian 32位整数）的数据类型：

  ``` python
  >>> dt = np.dtype([('big', '>i4'), ('little', '<i4')])
  ```

  具有字段 ``R``、``G``、``B``、``A``的数据类型，每个字段都是无符号8位整数：

  ``` python
  >>> dt = np.dtype([('R','u1'), ('G','u1'), ('B','u1'), ('A','u1')])
  ```

- ``{'names': ..., 'formats': ..., 'offsets': ..., 'titles': ..., 'itemsize': ...}``

  这种样式有两个必需键和三个可选键。*名称（names）*和 *格式（formats）* 密钥是必需的。
  它们各自的值是具有字段名称和字段格式的等长列表。
  字段名称必须是字符串，字段格式可以是 [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) 构造函数接受的任何对象。

  当提供可选键 *偏移量（offsets）* 和 *标题（titles）* 时，它们的值都必须是与名称和 *格式* 列表长度相同的列表。
  偏移量值是每个字段的字节 *偏移量*（限于 [``ctypes.c_int``](https://docs.python.org/dev/library/ctypes.html#ctypes.c_int) ）的列表，而标题值是每个字段的 *标题* 列表（如果该字段不需要标题，则不能使用任何标题）。
  *标题* 可以是任何字符串或 ``unicode`` 对象，并且将向字段字典中添加由 *标题* 键入的另一个条目，
  并引用相同的字段元组，该字段元组将包含 *标题* 作为额外的元组成员。

  *itemsize* 键允许设置dtype的总大小，并且必须是足够大的整数，以便所有字段都在dtype内。如果正在构造的dtype是对齐的，
  那么 *itemsize* 也必须可以被struct对齐整除。总dtype项目大小限制为 [``ctypes.c_int``](https://docs.python.org/dev/library/ctypes.html#ctypes.c_int)。

  **示例：**

  具有字段 ``r``、``g``、``b``、``a`` 的数据类型，每个字段都是8位无符号整数：

  ``` python
  >>> dt = np.dtype({'names': ['r','g','b','a'],
  ...                'formats': [uint8, uint8, uint8, uint8]})
  ```

  具有字段 ``r`` 和 ``b``（具有给定标题）的数据类型，两者都是8位无符号整数，第一个位于从字段开始的字节位置0，第二个位于位置2：

  ``` python
  >>> dt = np.dtype({'names': ['r','b'], 'formats': ['u1', 'u1'],
  ...                'offsets': [0, 2],
  ...                'titles': ['Red pixel', 'Blue pixel']})
  ```

- ``{'field1': ..., 'field2': ..., ...}``

  不鼓励使用这种用法，因为它与其他基于dict的构造方法有歧义。
  如果您有一个名为 “Names” 的字段和一个名为 “Formats” 的字段，则会发生冲突。

  此样式允许传入数据类型对象的 [``fields``](https://numpy.org/devdocs/reference/generated/numpy.dtype.fields.html#numpy.dtype.fields) 属性。

  *obj* 应包含引用 ``(data-type，offset)`` 或 ``(data-type，offset，title)`` 元组的字符串或Unicode键。

  **示例：**

  包含字段 ``col1``（字节位置0处的10个字符串）、``col2``（字节位置10处的32位浮点）和 ``col3``（字节位置14处的整数）的数据类型：

  ``` python
  >>> dt = np.dtype({'col1': ('U10', 0), 'col2': (float32, 10),
      'col3': (int, 14)})
  ```

- ``(base_dtype, new_dtype)``

  在NumPy 1.7和更高版本中，这种形式允许 *base_dtype* 被解释为结构化dtype。
  使用此dtype创建的数组将具有基础dtype *base_dtype*，但将具有取自 *new_dtype* 的字段和标志。
  这对于创建自定义结构化数据类型非常有用，就像在 [记录数组](classes.html#记录数组（numpy-rec）) 中所做的那样。

  这种形式还使指定具有重叠字段的struct dtype成为可能，其功能类似于C中的“Union”类型。然而，不鼓励使用这种用法，而首选联合机制。

  这两个参数必须可以转换为具有相同总大小的数据类型对象。

  **示例：**

  32位整数，其前两个字节通过field ``real``解释为整数，后面两个字节通过field ``imag`` 解释为整数。

  ``` python
  >>> dt = np.dtype((np.int32,{'real':(np.int16, 0),'imag':(np.int16, 2)})
  ```

  32位整数，解释为由包含8位整数的 shape ``(4，)`` 的子数组成：

  ``` python
  >>> dt = np.dtype((np.int32, (np.int8, 4)))
  ```

  32位整数，包含字段 ``r``、``g``、``b``、``a``，将整数中的4个字节解释为四个无符号整数：

  ``` python
  >>> dt = np.dtype(('i4', [('r','u1'),('g','u1'),('b','u1'),('a','u1')]))
  ```

## ``dtype``

NumPy数据类型描述是 [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) 类的实例。

### 属性

数据的类型由以下 [``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) 属性描述：

方法 | 描述
---|---
[dtype.type](https://numpy.org/devdocs/reference/generated/numpy.dtype.type.html#numpy.dtype.type) | 用于实例化此数据类型的标量的类型对象。
[dtype.kind](https://numpy.org/devdocs/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind) | 识别一般数据类型的字符码( “biufcmMOSUV” 之一)。
[dtype.char](https://numpy.org/devdocs/reference/generated/numpy.dtype.char.html#numpy.dtype.char) | 21种不同内置类型中的每一种都有唯一的字符码。
[dtype.num](https://numpy.org/devdocs/reference/generated/numpy.dtype.num.html#numpy.dtype.num) | 21种不同内置类型中每种类型的唯一编号。
[dtype.str](https://numpy.org/devdocs/reference/generated/numpy.dtype.str.html#numpy.dtype.str) | 此数据类型对象的数组协议类型字符串。

数据的大小依次由以下内容描述：

方法 | 描述
---|---
[dtype.name](https://numpy.org/devdocs/reference/generated/numpy.dtype.name.html#numpy.dtype.name) | 此数据类型的位宽名称。
[dtype.itemsize](https://numpy.org/devdocs/reference/generated/numpy.dtype.itemsize.html#numpy.dtype.itemsize) | 此数据类型对象的元素大小。

此数据的字符顺序：

方法 | 描述
---|---
[dtype.byteorder](https://numpy.org/devdocs/reference/generated/numpy.dtype.byteorder.html#numpy.dtype.byteorder) | 指示此数据类型对象的字节顺序的字符。

有关结构化数据类型中的[子数据类型](https://numpy.org/devdocs/glossary.html#term-structured-data-type)的信息：

方法 | 描述
---|---
[dtype.fields](https://numpy.org/devdocs/reference/generated/numpy.dtype.fields.html#numpy.dtype.fields) | 为此数据类型定义的命名字段的字典，或 ``None``。
[dtype.names](https://numpy.org/devdocs/reference/generated/numpy.dtype.names.html#numpy.dtype.names) | 字段名称的有序列表，如果没有字段，则为 ``None``。

对于描述子数组的数据类型：

方法 | 描述
---|---
[dtype.subdtype](https://numpy.org/devdocs/reference/generated/numpy.dtype.subdtype.html#numpy.dtype.subdtype) | 元组 ``(item_dtype，shape)``。如果此dtype描述子数组，则无其他。
[dtype.shape](https://numpy.org/devdocs/reference/generated/numpy.dtype.shape.html#numpy.dtype.shape) | 如果此数据类型描述子数组，则为子数组的Shape元组，否则为 ``()``。

提供附加信息的属性：

方法 | 描述
---|---
[dtype.hasobject](https://numpy.org/devdocs/reference/generated/numpy.dtype.hasobject.html#numpy.dtype.hasobject) | 指示此数据类型在任何字段或子数据类型中是否包含任何引用计数对象的布尔值。
[dtype.flags](https://numpy.org/devdocs/reference/generated/numpy.dtype.flags.html#numpy.dtype.flags) | 描述如何解释此数据类型的位标志。
[dtype.isbuiltin](https://numpy.org/devdocs/reference/generated/numpy.dtype.isbuiltin.html#numpy.dtype.isbuiltin) | 指示此数据类型如何与内置数据类型相关的整数。
[dtype.isnative](https://numpy.org/devdocs/reference/generated/numpy.dtype.isnative.html#numpy.dtype.isnative) | 指示此dtype的字节顺序是否为平台固有的布尔值。
[dtype.descr](https://numpy.org/devdocs/reference/generated/numpy.dtype.descr.html#numpy.dtype.descr) | *\_\_array_interface__* 数据类型说明。
[dtype.alignment](https://numpy.org/devdocs/reference/generated/numpy.dtype.alignment.html#numpy.dtype.alignment) | 根据编译器，此数据类型所需的对齐(字节)。
[dtype.base](https://numpy.org/devdocs/reference/generated/numpy.dtype.base.html#numpy.dtype.base) | 返回子数组的基本元素的dtype，而不考虑其尺寸或形状。

### 方法

数据类型具有以下更改字节顺序的方法：

方法 | 描述
---|---
[dtype.newbyteorder](https://numpy.org/devdocs/reference/generated/numpy.dtype.newbyteorder.html#numpy.dtype.newbyteorder)([new_order]) | 返回具有不同字节顺序的新dtype。

以下方法实现腌制（pickle）协议：

方法 | 描述
---|---
[dtype.__reduce__](https://numpy.org/devdocs/reference/generated/numpy.dtype.__reduce__.html#numpy.dtype.__reduce__)() | 帮助腌制（pickle）
[dtype.__setstate__](https://numpy.org/devdocs/reference/generated/numpy.dtype.__setstate__.html#numpy.dtype.__setstate__)() | 
