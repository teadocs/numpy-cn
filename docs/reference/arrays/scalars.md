# 标量

Python只定义了一种特定数据类（只有一种整数类型，一种浮点类型等）。这在不需要关心数据在计算机中表示的所有方式的应用中是方便的。然而，对于科学计算，通常需要更多的控制。

在NumPy中，有24种新的基本Python类型来描述不同类型的标量。这些类型描述符主要基于CPython编写的C语言中可用的类型，其他几种类型与Python的类型兼容。

数组标量具有与之相同的属性和方法[``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)。[[1]](#id2)这允许人们将数组中的项目部分地放在与数组相同的基础上，从而平滑混合标量和数组操作时产生的粗糙边缘。

数组标量存在于数据类型的层次结构中（请参见下图）。
可以使用层次结构检测它们：例如，如果Val是数组标量对象，则 ``isinstance(val，np.generic)`` 将返回 ``True``。
或者，可以使用数据类型层次结构的其他成员来确定存在哪种数组标量。
因此，例如，如果val是复数值类型，则 ``isinstance(val，np.complexfloat)`` 将返回 ``True``，
而如果 *val* 是灵活的itemsize数组类型之一（``string``、``unicode``、``void``），
则 ``isinstance(val，np.Flexible)`` 将返回 ``True``。

![dtype-hierarchy](/static/images/dtype-hierarchy.png)

**图**：表示数组数据类型的类型对象的层次结构。
未示出两种整数类型``intp``、``uintp``它们仅指向保存平台指针的整数类型。
所有数字类型也可以使用位宽名称获得。

[[1]](#id1)但是，数组标量是不可变的，因此没有数组标量属性可以设置。

## 内置标量类型

内置标量类型如下所示。连同它们的（主要是）C衍生的名称时，整数，浮点数，和复杂的数据类型也可使用位宽度约定，以便正确的大小的数组可以总是确保（例如``int8``，``float64``，
 ``complex128``）。还提供了两个别名（``intp``和``uintp``）指向足以容纳C指针的整数类型。类似C的名称与字符代码相关联，如表中所示。但是，不鼓励使用字符代码。

一些标量类型基本上等同于基本的Python类型，因此从它们以及通用数组标量类型继承：

数组标量类型 | 相关的Python类型
---|---
int_ | IntType （仅限Python 2）
float_ | FloatType
complex_ | ComplexType
bytes_ | BytesType
unicode_ | UnicodeType

该``bool_``数据类型是非常类似的Python
 ``BooleanType``，但不继承它，因为Python的
 ``BooleanType``不允许自己被继承，并在C级的实际布尔数据的大小是不一样的一个Python布尔标量。

::: danger 警告

该``bool_``类型不是该类型的子类``int_``（``bool_``甚至不是数字类型）。这与Python [``bool``](https://docs.python.org/dev/library/functions.html#bool)作为int的子类的默认实现不同。

:::

::: danger 警告

的``int_``类型并**没有**从继承
 [``int``](https://docs.python.org/dev/library/functions.html#int)内置Python 3下，因为类型[``int``](https://docs.python.org/dev/library/functions.html#int)不再是固定宽度的整数类型。

:::

NumPy中的默认数据类型是``float_``。

在下表中，``platform?``表示该类型可能并非在所有平台上都可用。指出了与不同C或Python类型的兼容性：如果两种类型的数据具有相同的大小并以相同的方式解释，则它们是兼容的。

布尔（Booleans）：

类型 | 备注 | 字符代码
---|---|---
bool_ | 兼容：Python bool | '?'
bool8 | 8位 |  

整数（Integers）：

类型 | 备注 | 字符代码
---|---|---
byte | 兼容：C char | 'b'
short | 兼容：C短 | 'h'
intc | 兼容：C int | 'i'
int_ | 兼容：Python int | 'l'
longlong | 兼容：C长 | 'q'
intp | 大到足以适合指针 | 'p'
int8 | 8位 |  
int16 | 16位 |  
int32 | 32位 |  
int64 | 64位 |  

无符号整数（Unsigned integers）：

类型 | 备注 | 字符代码
---|---|---
ubyte | compatible：C unsigned char | 'B'
ushort | 兼容：C unsigned short | 'H'
uintc | compatible：C unsigned int | 'I'
uint | 兼容：Python int | 'L'
ulonglong | 兼容：C长 | 'Q'
uintp | 大到足以适合指针 | 'P'
uint8 | 8位 |  
uint16 | 16位 |  
uint32 | 32位 |  
uint64 | 64位 |  

浮点数字（Floating-point numbers）：

类型 | 备注 | 字符代码
---|---|---
half |   | 'e'
single | 兼容：C浮动 | 'f'
double | 兼容：C双 |  
float_ | 兼容：Python float | 'd'
longfloat | 兼容：C长浮 | 'g'
float16 | 16位 |  
float32 | 32位 |  
float64 | 64位 |  
float96 | 96位，平台？ |  
float128 | 128位，平台？ |  

复杂的浮点数（Complex floating-point numbers）：

类型 | 备注 | 字符代码
---|---|---
csingle |   | 'F'
complex_ | 兼容：Python复杂 | 'D'
clongfloat |   | 'G'
complex64 | 两个32位浮点数 |  
complex128 | 两个64位浮点数 |  
complex192 | 两个96位浮动平台？ |  
complex256 | 两个128位浮点数，平台？ |  

任何Python对象（Any Python object）：

类型 | 备注 | 字符代码
---|---|---
object_ | 任何Python对象 | 'O'

::: tip 注意

实际存储在对象数组中的数据（ *即* 具有dtype的数组``object_``）是对Python对象的引用，而不是对象本身。因此，对象数组的行为更像普通的Python [``lists``](https://docs.python.org/dev/library/stdtypes.html#list)，因为它们的内容不必是相同的Python类型。

对象类型也是特殊的，因为包含``object_``项的数组
 不会``object_``在项访问时返回对象，而是返回数组项引用的实际对象。

:::

以下数据类型是**灵活的**：它们没有预定义的大小，并且它们描述的数据在不同的数组中可以具有不同的长度。（在字符代码中``#``是一个整数，表示数据类型包含多少个元素。）

类型 | 备注 | 字符代码
---|---|---
bytes_ | 兼容：Python字节 | 'S#'
unicode_ | 兼容：Python unicode / str | 'U#'
void |   | 'V#'

::: danger 警告

请参阅[字符串类型的注解](arrays.dtypes.html#string-dtype-note)。

数字兼容性：如果您在数字代码中使用了旧的类型代码字符（从未推荐过），
则需要将其中一些更改为新字符。
特别是，所需的更改是 ``c -> S1``, ``b -> B``, ``1 -> b``, ``s -> h``, ``w -> H`` 和 ``u -> I``。
这些更改使类型字符约定与其他Python更加一致 诸如 [``struct``](https://docs.python.org/dev/library/struct.html#module-struct) 模块之类的模块。

:::

## 属性

数组标量对象的 ``数组优先级`` 为``NPY_SCALAR_PRIORITY`` (-1，000，000.0)。它们也(还)没有 [``ctypes``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ctypes.html#numpy.ndarray.ctypes) 属性。否则，它们与数组共享相同的属性：

方法 | 描述
---|---
[generic.flags](https://numpy.org/devdocs/reference/generated/numpy.generic.flags.html#numpy.generic.flags) | 标志的整数值
[generic.shape](https://numpy.org/devdocs/reference/generated/numpy.generic.shape.html#numpy.generic.shape) | 数组维度的元组
[generic.strides](https://numpy.org/devdocs/reference/generated/numpy.generic.strides.html#numpy.generic.strides) | 每个维度中的字节元组步骤
[generic.ndim](https://numpy.org/devdocs/reference/generated/numpy.generic.ndim.html#numpy.generic.ndim) | 数组维数
[generic.data](https://numpy.org/devdocs/reference/generated/numpy.generic.data.html#numpy.generic.data) | 指向数据开始的指针
[generic.size](https://numpy.org/devdocs/reference/generated/numpy.generic.size.html#numpy.generic.size) | gentype中的元素数量
[generic.itemsize](https://numpy.org/devdocs/reference/generated/numpy.generic.itemsize.html#numpy.generic.itemsize) | 一个元素的长度，以字节为单位
[generic.base](https://numpy.org/devdocs/reference/generated/numpy.generic.base.html#numpy.generic.base) | 基础对象
[generic.dtype](https://numpy.org/devdocs/reference/generated/numpy.generic.dtype.html#numpy.generic.dtype) | 获取数组数据描述符
[generic.real](https://numpy.org/devdocs/reference/generated/numpy.generic.real.html#numpy.generic.real) | 标量的真实部分
[generic.imag](https://numpy.org/devdocs/reference/generated/numpy.generic.imag.html#numpy.generic.imag) | 标量的虚部
[generic.flat](https://numpy.org/devdocs/reference/generated/numpy.generic.flat.html#numpy.generic.flat) | 标量的一维视图
[generic.T](https://numpy.org/devdocs/reference/generated/numpy.generic.T.html#numpy.generic.T) | 颠倒
[generic.__array_interface__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_interface__.html#numpy.generic.__array_interface__) | 数组协议：Python端
[generic.__array_struct__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_struct__.html#numpy.generic.__array_struct__) | 数组协议：struct
[generic.__array_priority__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_priority__.html#numpy.generic.__array_priority__) | 数组优先级。
[generic.__array_wrap__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_wrap__.html#numpy.generic.__array_wrap__)() | sc .__ array_wrap __（obj）从数组返回标量

## 索引

::: tip 另见

[索引](indexing.html)、
[数据类型对象（dtype）](dtypes.html)

:::

数组标量可以像0维数组一样索引：如果 *x* 是数组标量，

- ``x[()]`` 返回数组标量的副本
- ``x[...]`` 返回0维 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)
- ``x['field-name']``返回字段 *field-name中* 的数组标量。（例如， *x* 可以包含字段，当它对应于结构化数据类型时。）

## 方法

数组标量具有与数组完全相同的方法。这些方法的默认行为是在内部将标量转换为等效的0维数组并调用相应的数组方法。
此外，数组标量的数学运算被定义，使得在相同的硬件标志被设置，
并用于解释结果作为通函数（[ufunc](/reference/ufuncs.html)），
使得用于ufuncs错误状态也延续到上数组标量的数学。

以上规则的例外情况如下：

方法 | 描述
---|---
[generic](https://numpy.org/devdocs/reference/generated/numpy.generic.html#numpy.generic) | numpy标量类型的基类。
[generic.__array__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array__.html#numpy.generic.__array__)() | sc .__ array __（dtype）从带有指定dtype的标量返回0-dim数组
[generic.__array_wrap__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_wrap__.html#numpy.generic.__array_wrap__)() | sc .__ array_wrap __（obj）从数组返回标量
[generic.squeeze](https://numpy.org/devdocs/reference/generated/numpy.generic.squeeze.html#numpy.generic.squeeze)() | 未实现（虚拟属性）
[generic.byteswap](https://numpy.org/devdocs/reference/generated/numpy.generic.byteswap.html#numpy.generic.byteswap)() | 未实现（虚拟属性）
[generic.__reduce__](https://numpy.org/devdocs/reference/generated/numpy.generic.__reduce__.html#numpy.generic.__reduce__)() | 泡菜的助手
[generic.__setstate__](https://numpy.org/devdocs/reference/generated/numpy.generic.__setstate__.html#numpy.generic.__setstate__)() | 
[generic.setflags](https://numpy.org/devdocs/reference/generated/numpy.generic.setflags.html#numpy.generic.setflags)() | 未实现（虚拟属性）

## 定义新类型

有两种方法可以有效地定义新的数组标量类型（除了从内置标量类型组合结构化类型[dtypes](dtypes.html)）：一种方法是简单地子类化
 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)并覆盖感兴趣的方法。
 这将在一定程度上起作用，但内部某些行为由数组的数据类型修复。要完全自定义数组的数据类型，
 您需要定义新的数据类型，并使用NumPy进行注册。
 这些新类型只能使用[NumPy C-API](/reference/c-api/index.html)在C中定义。