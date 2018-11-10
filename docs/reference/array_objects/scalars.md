# 标量

Python只定义了一种特定数据类（只有一种整数类型，一种浮点类型等）。 这在不需要关注数据在计算机中表示的所有方式的应用中是方便的。 然而，对于科学计算，通常需要控制更多。

在NumPy中，有24种新的基本Python类型来描述不同类型的标量。 这些类型描述符主要基于CPython编写的C语言中可用的类型，其他几种类型与Python的类型兼容。

数组标量与ndarray具有相同的属性和方法。[1]这使得人们可以在与数组相同的基础上处理数组中的部分项，从而平滑混合标量和数组操作时产生的粗糙边缘。

数组标量存在于数据类型的层次结构中（请参见下图）。 可以使用层次结构检测它们：例如，如果val是数组标量对象，则isinstance(val, np.generic)将返回True。 或者，可以使用数据类型层次结构的其他成员来确定存在何种类型的数组标量。 因此，例如，如果val是复值类型，则isinstance(val, np.complexfloating)将返回True，而如果val是灵活的itemsize数组类型之一，则isinstance(val, np.flexible)将返回true（string，unicode，void）。

![标量](/static/images/dtype-hierarchy.png)

**数字**: 表示数组数据类型的类型对象的层次结构。未显示的是两个整数类型intp和uintp，它们只是指向保存平台指针的整数类型。所有的数字类型也可以使用位宽名称来获得。

*但是，数组标量是不可变的，因此没有一个数组标量属性是可设置的。*

## 内建标量类型

内置标量类型如下所示。 除了它们（大多数）C派生的名称，整数，浮点和复数数据类型也可以使用位宽约定，以便始终可以确保正确大小的数组（例如int8，float64，complex128）。 还提供了指向足够大以容纳C指针的整数类型的两个别名（intp和uintp）。 类似C的名称与字符代码相关联，如表中所示。 但是，不鼓励使用字符代码。

一些标量类型基本上等同于基本的Python类型，因此从它们以及通用数组标量类型继承：

数组标量类型 | 相关的Python类型
---|---
``int_`` | ``IntType`` (只有Python 2可用)
``float_`` | ``FloatType``
``complex_`` | ``ComplexType``
``bytes_`` | ``BytesType``
``unicode_`` | ``UnicodeType``

``bool_`` 数据类型与Python``BooleanType``非常相似，但不会从它继承，因为Python的``BooleanType``不允许自己继承，而在C级别的大小 实际的bool数据与Python布尔标量不同。

<div class="warning-warp">
<b>警告</b>
<p>bool_类型不是int_类型的子类（bool_甚至不是数字类型）。 这与Python作为int的子类的bool的默认实现不同。
</p>
</div>

<div class="warning-warp">
<b>警告</b>
<p>int_type不从Python 3中的int内置继承，因为int类型不再是固定宽度的整数类型。.</p>
</div>

**小贴士**
NumPy中的默认数据类型是float_。

在下表中，``platform？``表示该类型可能并非在所有平台上都可用。 指出了与不同C或Python类型的兼容性：如果两种类型的数据具有相同的大小并以相同的方式解释，则它们是兼容的。

布尔值：

类型 | 备注 | 字符代码
---|---|---
``bool_`` | compatible: Python bool | ``'?'``
``bool8`` | 8 bits |

整形：

类型 | 备注 | 字符代码
---|---|---
byte | compatible: C char | 'b'
short | compatible: C short | 'h'
intc | compatible: C int | 'i'
int_ | compatible: Python int | 'l'
longlong | compatible: C long long | 'q'
intp | large enough to fit a pointer | 'p'
int8 | 8 bits |  
int16 | 16 bits |  
int32 | 32 bits |  
int64 | 64 bits |  

无符号整形：

类型 | 备注 | 字符代码
---|---|---
ubyte | 兼容: C unsigned char | 'B'
ushort | 兼容: C unsigned short | 'H'
uintc | 兼容: C unsigned int | 'I'
uint | 兼容: Python int | 'L'
ulonglong | 兼容: C long long | 'Q'
uintp |大到足以适合指针 | 'P'
uint8 | 8 bits | -
uint16 | 16 bits | -
uint32 | 32 bits | -
uint64 | 64 bits | -

浮点类型：

类型 | 备注 | 字符代码
---|---|---
half |   | 'e'
single | 兼容: C float | 'f'
double | 兼容: C double |  
float_ | 兼容: Python float | 'd'
longfloat | 兼容: C long float | 'g'
float16 | 16 bits |  
float32 | 32 bits |  
float64 | 64 bits |  
float96 | 96 bits, platform? |  
float128 | 128 bits, platform? |  

复杂的浮点数：

类型 | 备注 | 字符代码
---|---|---
csingle |   | 'F'
complex_ | 兼容: Python complex 类型 | 'D'
clongfloat |   | 'G'
complex64 | 两个 32-bit 浮点数 |  
complex128 | 两个 64-bit 浮点数 |  
complex192 | 两个 96-bit 浮点数, platform? |  
complex256 | 两个 128-bit 浮点数, platform? |  

任意Python对象：

类型 | 备注 | 字符代码
---|---|---
object_ | 一个 Python 对象 | 'O'

> **注意**
> 实际存储在对象数组中的数据（即具有dtype object_的数组）是对Python对象的引用，而不是对象本身。 因此，对象数组的行为更像通常的Python列表，因为它们的内容不必是相同的Python类型。

对象类型也是特殊的，因为包含object_ items的数组不会在项访问时返回object_对象，而是返回数组项引用的实际对象。

以下数据类型是灵活的。 它们没有预定义的大小：它们描述的数据在不同的数组中可以具有不同的长度。 （在字符代码中＃是一个整数，表示数据类型包含多少个元素。）

类型 | 备注 | 字符代码
---|---|---
bytes_ | 兼容: Python bytes | 'S#'
unicode_ | 兼容: Python unicode/str | 'U#'
void | - | 'V#'


<div class="warning-warp">
<b>警告</b>
<p>请参阅字符串类型的解释。</p>
<p>
379/5000
数字兼容性：如果你在数字代码中使用了旧的类型代码字符（从未推荐过），则需要将其中一些更改为新字符。 特别是，所需的更改是c -> S1，b -> B，1 -> b，s -> h，w -> H和 u -> I.这些更改使类型字符约定与其他Python更加一致 诸如struct模块之类的模块。</p>
</div>

## 属性

数组标量对象具有 ``NPY_SCALAR_PRIORITY``（-1,000,000.0）的 ``数组优先级``。 他们还没有（还）有一个``ctypes``属性。 否则，它们与数组共享相同的属性：

属性 | 描述
---|---
generic.flags | 标志的整数值。
generic.shape | 数组维度的元组。
generic.strides | 每个维度中的字节元组步骤。
generic.ndim | 数组维数。
generic.data | 指向数据的开始
generic.size | gentype中的元素数量
generic.itemsize | 一个元素的长度，以字节为单位
generic.base | 基本的对象
generic.dtype | 获取数组数据描述符
generic.real | 标量的真实部分
generic.imag | 标量的虚部
generic.flat | 标量的一维视图
generic.T | 转置
generic.__array_interface__ | 数组协议：Python方面
generic.__array_struct__ | 数组协议：struct
generic.__array_priority__ | Array priority.
generic.__array_wrap__ | sc.__array_wrap__(obj) 从数组返回标量

## 索引

另见：

> 索引，数据类型对象（dtype）

数组标量可以像0维数组一样索引：如果x是数组标量，

- x[()] 返回数组标量的副本
- x[...] 返回一个0维的ndarray
- x['field-name'] 返回字段field-name中的数组标量。 （例如，x可以包含字段，当它对应于结构化数据类型时。）

## 方法

数组标量与数组具有完全相同的方法。 这些方法的默认行为是在内部将标量转换为等效的0维数组并调用相应的数组方法。 此外，定义了数组标量的数学运算，以便设置相同的硬件标志并用于解释ufunc的结果，以便用于ufuncs的错误状态也会延续到数组标量的数学运算。

以下规则的例外情况如下：

属性 | 描述
---|---
generic | numpy标量类型的基类。
generic.__array__ | sc.__array__(|type) 返回 0 维的数组
generic.__array_wrap__ | sc.__array_wrap__(obj) 从数组返回标量
generic.squeeze | 未实现（虚拟属性）
generic.byteswap | 未实现（虚拟属性）
generic.__reduce__ | 关于“腌制”的帮助
generic.__setstate__ | - 
generic.setflags | 未实现（虚拟属性）

## 定义新类型

有两种方法可以有效地定义新的数组标量类型（除了从内置标量类型组合结构化类型dtypes）：一种方法是简单地子类化ndarray并覆盖感兴趣的方法。 这将在一定程度上起作用，但内部某些行为由数组的数据类型修复。 要完全自定义数组的数据类型，你需要定义新的数据类型，并使用NumPy进行注册。 这些新类型只能使用NumPy C-API在C中定义。