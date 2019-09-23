# 数据类型

::: tip 另见

[数据类型对象](/reference/arrays/dtypes.html)

:::

## 数组类型之间的转换

NumPy支持比Python更多种类的数字类型。本节显示了哪些可用，以及如何修改数组的数据类型。

支持的原始类型与 C 中的原始类型紧密相关：

Numpy 的类型 | C 的类型 | 描述
---|---|---
np.bool | bool | 存储为字节的布尔值（True或False）
np.byte | signed char | 平台定义
np.ubyte | unsigned char | 平台定义
np.short | short | 平台定义
np.ushort | unsigned short | 平台定义
np.intc | int | 平台定义
np.uintc | unsigned int | 平台定义
np.int_ | long | 平台定义
np.uint | unsigned long | 平台定义
np.longlong | long long | 平台定义
np.ulonglong | unsigned long long | 平台定义
np.half / np.float16 |   | 半精度浮点数：符号位，5位指数，10位尾数
np.single | float | 平台定义的单精度浮点数：通常为符号位，8位指数，23位尾数
np.double | double | 平台定义的双精度浮点数：通常为符号位，11位指数，52位尾数。
np.longdouble | long double | 平台定义的扩展精度浮点数
np.csingle | float complex | 复数，由两个单精度浮点数（实部和虚部）表示
np.cdouble | double complex | 复数，由两个双精度浮点数（实部和虚部）表示。
np.clongdouble | long double complex | 复数，由两个扩展精度浮点数（实部和虚部）表示。

由于其中许多都具有依赖于平台的定义，因此提供了一组固定大小的别名：

Numpy 的类型 | C 的类型 | 描述
---|---|---
np.int8 | int8_t | 字节（-128到127）
np.int16 | int16_t | 整数（-32768至32767）
np.int32 | int32_t | 整数（-2147483648至2147483647）
np.int64 | int64_t | 整数（-9223372036854775808至9223372036854775807）
np.uint8 | uint8_t | 无符号整数（0到255）
np.uint16 | uint16_t | 无符号整数（0到65535）
np.uint32 | uint32_t | 无符号整数（0到4294967295）
np.uint64 | uint64_t | 无符号整数（0到18446744073709551615）
np.intp | intptr_t | 用于索引的整数，通常与索引相同 ssize_t
np.uintp | uintptr_t | 整数大到足以容纳指针
np.float32 | float |  
np.float64 / np.float_ | double | 请注意，这与内置python float的精度相匹配。
np.complex64 | float complex | 复数，由两个32位浮点数（实数和虚数组件）表示
np.complex128 / np.complex_ | double complex | 请注意，这与内置python 复合体的精度相匹配。

NumPy数值类型是``dtype``（数据类型）对象的实例，每个对象都具有独特的特征。使用后导入NumPy

``` python
>>> import numpy as np
```

在dtypes可作为``np.bool_``，``np.float32``等等。

上表中未列出的高级类型将在[结构化数组](rec.html#structured-arrays)中进行探讨。

有5种基本数字类型表示布尔值（bool），整数（int），无符号整数（uint）浮点（浮点数）和复数。名称中带有数字的那些表示该类型的位大小（即，在内存中表示单个值需要多少位）。某些类型（例如 ``int`` 和 ``intp``）具有不同的位，取决于平台（例如，32位与64位计算机）。在与寻址原始内存的低级代码（例如C或Fortran）连接时，应考虑这一点。

数据类型可以用作将python数转换为数组标量的函数（请参阅数组标量部分以获得解释），将python数字序列转换为该类型的数组，或作为许多numpy函数或方法接受的dtype关键字的参数。一些例子：

``` python
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

数组类型也可以通过字符代码引用，主要是为了保持与较旧的包（如Numeric）的向后兼容性。有些文档可能仍然引用这些，例如：

``` python
>>> np.array([1, 2, 3], dtype='f')
array([ 1.,  2.,  3.], dtype=float32)
```

我们建议使用dtype对象。

要转换数组的类型，请使用 .astype() 方法（首选）或类型本身作为函数。例如：

``` python
>>> z.astype(float)                 
array([  0.,  1.,  2.])
>>> np.int8(z)
array([0, 1, 2], dtype=int8)
```

注意，在上面，我们使用 *Python* 的 float对象作为dtype。NumPy的人都知道``int``是指``np.int_``，``bool``意味着``np.bool_``，这``float``是``np.float_``和``complex``是``np.complex_``。其他数据类型没有Python等价物。

要确定数组的类型，请查看dtype属性：

``` python
>>> z.dtype
dtype('uint8')
```

dtype对象还包含有关类型的信息，例如其位宽和字节顺序。数据类型也可以间接用于查询类型的属性，例如它是否为整数：

``` python
>>> d = np.dtype(int)
>>> d
dtype('int32')

>>> np.issubdtype(d, np.integer)
True

>>> np.issubdtype(d, np.floating)
False
```

## 数组标量

NumPy通常将数组元素作为数组标量返回（带有关联dtype的标量）。数组标量与Python标量不同，但在大多数情况下它们可以互换使用（主要的例外是早于v2.x的Python版本，其中整数数组标量不能作为列表和元组的索引）。有一些例外，例如当代码需要标量的非常特定的属性或者它特定地检查值是否是Python标量时。通常，存在的问题很容易被显式转换数组标量到Python标量，采用相应的Python类型的功能（例如，固定的``int``，``float``，``complex``，``str``，``unicode``）。

使用数组标量的主要优点是它们保留了数组类型（Python可能没有匹配的标量类型，例如``int16``）。因此，使用数组标量可确保数组和标量之间的相同行为，无论值是否在数组内。NumPy标量也有许多与数组相同的方法。

## 溢出错误

当值需要比数据类型中的可用内存更多的内存时，NumPy数值类型的固定大小可能会导致溢出错误。例如，numpy.power对于64位整数正确计算 ``100 * 10 * 8``，但对于32位整数给出1874919424（不正确）。

``` python
>>> np.power(100, 8, dtype=np.int64)
10000000000000000
>>> np.power(100, 8, dtype=np.int32)
1874919424
```

NumPy和Python整数类型的行为在整数溢出方面存在显着差异，并且可能会使用户期望NumPy整数的行为类似于Python ``int``。与 NumPy 不同，Python 的大小``int`` 是灵活的。这意味着Python整数可以扩展以容纳任何整数并且不会溢出。

NumPy分别提供[``numpy.iinfo``](https://numpy.org/devdocs/reference/generated/numpy.iinfo.html#numpy.iinfo)并[``numpy.finfo``](https://numpy.org/devdocs/reference/generated/numpy.finfo.html#numpy.finfo)验证NumPy整数和浮点值的最小值或最大值：

``` python
>>> np.iinfo(np.int) # Bounds of the default integer on this system.
iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)
>>> np.iinfo(np.int32) # Bounds of a 32-bit integer
iinfo(min=-2147483648, max=2147483647, dtype=int32)
>>> np.iinfo(np.int64) # Bounds of a 64-bit integer
iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)
```

如果64位整数仍然太小，则结果可能会转换为浮点数。浮点数提供了更大但不精确的可能值范围。

``` python
>>> np.power(100, 100, dtype=np.int64) # Incorrect even with 64-bit int
0
>>> np.power(100, 100, dtype=np.float64)
1e+200
```

## 扩展精度

Python 的浮点数通常是64位浮点数，几乎等同于 ``np.float64`` 。在某些不寻常的情况下，使用更精确的浮点数可能会很有用。这在numpy中是否可行取决于硬件和开发环境：具体地说，x86机器提供80位精度的硬件浮点，虽然大多数C编译器提供这一点作为它们的 ``long double`` 类型，MSVC(Windows构建的标准)使 ``long double`` 等同于 ``double`` (64位)。NumPy使编译器的 ``long double`` 作为 ``np.longdouble`` 可用(而 ``np.clongdouble`` 用于复数)。您可以使用 ``np.finfo(np.longdouble)`` 找出 numpy提供了什么。

NumPy不提供比C的 ``long double`` 更高精度的dtype；特别是128位IEEE四精度数据类型(FORTRAN的 ``REAL*16`` )不可用。

为了有效地进行内存的校准，``np.longdouble``通常以零位进行填充，即96或者128位， 哪个更有效率取决于硬件和开发环境；通常在32位系统上它们被填充到96位，而在64位系统上它们通常被填充到128位。``np.longdouble``被填充到系统默认值；为需要特定填充的用户提供了``np.float96``和``np.float128``。尽管它们的名称是这样叫的, 但是``np.float96``和``np.float128``只提供与``np.longdouble``一样的精度, 即大多数x86机器上的80位和标准Windows版本中的64位。

请注意，即使``np.longdouble``提供比python ``float``更多的精度，也很容易失去额外的精度，因为python通常强制值通过``float``传递值。例如，``%``格式操作符要求将其参数转换为标准python类型，因此即使请求了许多小数位，也不可能保留扩展精度。使用值``1 + np.finfo(np.longdouble).eps``测试你的代码非常有用。
