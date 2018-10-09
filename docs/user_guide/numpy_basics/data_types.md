<title>numpy数据类型 - <%-__DOC_NAME__ %></title>
<meta name="keywords" content="numpy数据类型" />

# 数据类型

另见：

> Data type objects

## 数组类型和类型之间的转换

Numpy支持比Python更多的数字类型。本部分显示哪些是可用的，以及如何修改数组的数据类型。

数据类型 | 描述
---|---
``bool_`` |布尔（True或False），存储为一个字节
``int_`` | 默认整数类型（与C``long``相同；通常是``int64``或``int32``）
INTC | 与C``int``（通常为``int32``或``int64``）相同
INTP | 用于索引的整数（与C``ssize_t``相同；通常是``int32``或``int64``）
INT8 | 字节（-128至127）
INT16 | 整数（-32768至32767）
INT32 | 整数（-2147483648至2147483647）
Int64的 | 整数（-9223372036854775808至9223372036854775807）
UINT8 | 无符号整数（0到255）
UINT16 | 无符号整数（0到65535）
UINT32 | 无符号整数（0到4294967295）
UINT64 | 无符号整数（0到18446744073709551615）
float_ | ``float64``的简写。
float16 | 半精度浮点：符号位，5位指数，10位尾数
FLOAT32 | 单精度浮点数：符号位，8位指数，23位尾数
float64 | 双精度浮点：符号位，11位指数，52位尾数
complex_ | ``complex128``的简写。
complex64 | 复数，由两个32位浮点数（实部和虚部）
complex128 | 复数，由两个64位浮点数（实部和虚部）

除了``intc``之外，还定义了平台相关的C整数类型``short``，``long``，``longlong``。

Numpy数值类型是``dtype``（data-type）对象的实例，每个类型具有唯一的特征。在你使用下面的语句导入NumPy后

```python
>>> import numpy as np
```

这些类型可以用``np.bool_``、``np.float32``等方式访问。

未在上表中列出的高级类型，请参见[结构化数组](/user_guide/numpy_basics/structured_arrays.html)部分。

有5个基本数字类型表示布尔（bool）、整数（int）、无符号整数（uint）、浮点数（float）和复数。那些在其名称中具有数字的类型表示类型的位的大小（即，需要多少位来表示存储器中的单个值）。某些类型，例如``int``和``intp``，根据平台（例如32位与64位机器）具有不同的位大小。当与存储器直接寻址的低级代码（例如C或Fortran）接口时，应该考虑这一点。

数据类型可以用作函数将python数字转换为数组标量（有关说明，请参阅数组标量部分）、将python数字序列转换为该类型的数组、或作为许多numpy函数或方法接受的dtype关键字参数。一些例子：

```python
>>> import numpy as np
>>> x = np.float32(1.0)
>>> x
1.0
>>> y = np.int_([1,2,4])
>>> y
array([1, 2, 4])
>>> z = np.arange(3, dtype=np.uint8)
>>> z
array([0, 1, 2], dtype=uint8)
```

数组类型也可以由字符代码引用，主要是为了保持与旧数据包（如Numeric）的向后兼容性。有些文档可能仍然提及这些文档，例如：

```python
>>> np.array([1, 2, 3], dtype='f')
array([ 1.,  2.,  3.], dtype=float32)
```

我们建议使用dtype对象。

要转换数组的类型，请使用.astype()方法（首选）或类型本身作为函数。例如：

```python
>>> z.astype(float)                 
array([  0.,  1.,  2.])
>>> np.int8(z)
array([0, 1, 2], dtype=int8)
```

请注意，上面我们使用Python float对象作为dtype。NumPy 知道int指代np.int_、bool表示np.bool_、 float为np.float_以及complex为np.complex_。其他数据类型没有Python等效的类型。

要确定数组的类型，请查看dtype属性：

```python
>>> z.dtype
dtype('uint8')
```

dtype对象还包含有关该类型的信息，例如其位宽和字节顺序。数据类型也可以间接用于查询类型的属性，例如是否为整数：

```python
>>> d = np.dtype(int)
>>> d
dtype('int32')

>>> np.issubdtype(d, int)
True

>>> np.issubdtype(d, float)
False
```

## 数组标量
Numpy通常返回数组的元素作为数组标量（与相关dtype的标量）。数组标量与Python标量不同，但大多数情况下它们可以互换使用（主要例外是Python版本比v2.x更早的版本，其中整数数组标量不能充当列表和元组的索引）。有一些例外情况，比如代码需要非常特定的标量属性，或者当它特别检查某个值是否为Python标量时。通常，使用相应的Python类型函数（例如``int``、``float``、``complex``、``str``，``unicode``）将数组标量显式转换为Python标量就很容易解决问题。

使用数组标量的主要优点是它们保留数组类型（Python可能没有可用的匹配标量类型，例如``int16``）。因此，使用数组标量可以确保数组和标量之间的相同行为，而不管该值是否在数组中。NumPy标量也有很多和数组相同的方法。

## 扩展精度
Python的浮点数通常是64位浮点数，几乎相当于``np.float64``。在某些不常见的情况下，使用Python的浮点数更精确。这在numpy中是否可行，取决于硬件和开发的环境：具体来说，x86机器提供80位精度的硬件浮点数，大多数C编译器提供它为``long double``类型，MSVC（Windows版本的标准）让``long double``和``double``（64位）完全一样。Numpy使编译器的``long double``为``np.longdouble``（复数为``np.clongdouble``）。你可以用``np.finfo(np.longdouble)``找出你的numpy提供的是什么。

Numpy 不提供比 C 语言里 ``long double`` 更高精度的数据类型，特别是 128 位的IEEE 四倍精度的数据类型（FORTRAN的 ``REAL*16``） 不可用。

为了有效地进行内存的校准，``np.longdouble``通常以零位进行填充，即96或者128位， 哪个更有效率取决于硬件和开发环境；通常在32位系统上它们被填充到96位，而在64位系统上它们通常被填充到128位。``np.longdouble``被填充到系统默认值；为需要特定填充的用户提供了``np.float96``和``np.float128``。尽管它们的名称是这样叫的, 但是``np.float96``和``np.float128``只提供与``np.longdouble``一样的精度, 即大多数x86机器上的80位和标准Windows版本中的64位。

请注意，即使``np.longdouble``提供比python ``float``更多的精度，也很容易失去额外的精度，因为python通常强制值通过``float``传递值。例如，``%``格式操作符要求将其参数转换为标准python类型，因此即使请求了许多小数位，也不可能保留扩展精度。使用值``1 + np.finfo(np.longdouble).eps``测试你的代码非常有用。