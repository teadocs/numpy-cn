---
meta:
  - name: keywords
    content: NumPy 通函数ufunc
  - name: description
    content: 通函数（或简称为ufunc） 是一种ndarrays以逐元素方式操作的函数，支持数组广播，类型转换和其他一些标准功能。
---

# 通函数（``ufunc``）

通函数（或简称为[ufunc](https://numpy.org/devdocs/glossary.html#term-ufunc)）
是一种[``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)以逐元素方式操作的函数，支持[数组广播](#ufuncs-broadcasting)，[类型转换](#ufuncs-casting)和其他一些标准功能。也就是说，ufunc是一个函数的 “ [矢量化](https://numpy.org/devdocs/glossary.html#term-vectorization) ” 包装器，它接受固定数量的特定输入并产生固定数量的特定输出。

在NumPy中，通函数是``numpy.ufunc``类的实例 。
许多内置函数都是在编译的C代码中实现的。
基本的ufuncs对标量进行操作，但也有一种通用类型，基本元素是子数组（向量，矩阵等），
广播是在其他维度上完成的。也可以``ufunc``使用[``frompyfunc``](https://numpy.org/devdocs/reference/generated/numpy.frompyfunc.html#numpy.frompyfunc)工厂函数生成自定义实例。

## 广播

每个通函数接受数组输入并通过在输入上逐元素地执行核心功能来生成数组输出（其中元素通常是标量，但可以是用于广义ufunc的向量或更高阶子数组）。
应用标准广播规则，以便仍然可以有效地操作不共享完全相同形状的输入。
广播可以通过四个规则来理解：

1. 所有输入数组都[``ndim``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ndim.html#numpy.ndarray.ndim)小于最大的输入数组，[``ndim``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ndim.html#numpy.ndarray.ndim)其形状前面有1个。
1. 输出形状的每个维度的大小是该维度中所有输入大小的最大值。
1. 如果输入在特定维度中的大小与该维度中的输出大小匹配，或者其值正好为1，则可以在计算中使用该输入。
1. 如果输入的形状尺寸为1，则该维度中的第一个数据条目将用于沿该维度的所有计算。换句话说，[ufunc](https://numpy.org/devdocs/glossary.html#term-ufunc)的步进机械
 将不会沿着该维度步进（对于该维度，[步幅](/reference/arrays/ndarray.html#内存布局)将为0）。

整个NumPy使用广播来决定如何处理不同形状的数组; 例如，所有算术运算（``+``，
 ``-``，``*``之间，...）[``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)的数组操作之前广播。

如果上述规则产生有效结果，则将一组数组称为“可广播”到相同的形状， *即* 满足下列条件之一：

1. 数组都具有完全相同的形状。
1. 数组都具有相同的维数，每个维度的长度是公共长度或1。
1. 尺寸太小的数组可以使其形状前置为长度为1的尺寸以满足属性2。

如果``a.shape``是 (5,1)，``b.shape``是 (1,6)，``c.shape``是 (6，)并且``d.shape``是 () 使得 *d* 是标量，则 *a* ， *b* ， *c* 和 *d* 都可以广播到维度 (5, 6); 和：

- *a* 的作用类似于（5,6）数组，其中 [:, 0] 广播到其他列，
- *b* 的作用类似于（5,6）数组，其中 b[0, :] 广播到其他行，
- *c* 就像一个（1,6）数组，因此像一个（5,6）数组，其中 c[:] 广播到每一行，最后，
- *d* 的作用类似于（5,6）数组，其中重复单个值。

## 输出类型确定

[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)如果所有输入参数都不是，则ufunc（及其方法）的输出不一定
 是[``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)。实际上，如果任何输入定义了一个[``__array_ufunc__``](arrays.classes.html#numpy.class.__array_ufunc__)方法，控件将完全传递给该函数，即重写ufunc
 。

如果没有任何输入覆盖ufunc，则所有输出数组将被传递给输入的 [``__array_prepare__``](arrays.classes.html#numpy.class.__array_prepare__) 和[``__array_wrap__``](arrays.classes.html#numpy.class.__array_wrap__) 方法（除了 [``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 和scalars），
这些方法定义它并且具有通用函数的任何其他输入的最高 [``__array_priority__``](arrays.classes.html#numpy.class.__array_priority__) 。 
ndarray的默认 [``__array_priority__``](arrays.classes.html#numpy.class.__array_priority__) 为0.0，子类型的默认 [``__array_priority__``](arrays.classes.html#numpy.class.__array_priority__) 为0.0。
矩阵的 [``__array_priority__``](arrays.classes.html#numpy.class.__array_priority__) 等于10.0。

所有ufunc也可以获取输出参数。如有必要，输出将转换为提供的输出数组的数据类型。如果将带有[``__array__``](arrays.classes.html#numpy.class.__array__)方法的类用于输出，则结果将写入返回的对象[``__array__``](arrays.classes.html#numpy.class.__array__)。然后，如果类也有一个[``__array_prepare__``](arrays.classes.html#numpy.class.__array_prepare__)方法，则调用它，因此可以根据ufunc的上下文确定元数据（由ufunc本身组成的上下文，传递给ufunc的参数和ufunc域。）数组对象返回者
 [``__array_prepare__``](arrays.classes.html#numpy.class.__array_prepare__)传递给ufunc进行计算。最后，如果类也有一个[``__array_wrap__``](arrays.classes.html#numpy.class.__array_wrap__)方法，返回的
 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)结果将在将控制权传递给调用者之前传递给该方法。

## 使用内部缓冲区

在内部，缓冲区用于未对齐的数据，交换的数据以及必须从一种数据类型转换为另一种数据类型的数据。内部缓冲区的大小可以基于每个线程设置。最多可以
创建指定大小的缓冲区来处理来自ufunc的所有输入和输出的数据。缓冲区的默认大小为10,000个元素。每当需要基于缓冲区的计算，但所有输入数组都小于缓冲区大小时，将在计算进行之前复制那些行为不当或类型不正确的数组。因此，调整缓冲区的大小可能会改变各种类型的ufunc计算完成的速度。可以使用该函数访问用于设置此变量的简单界面

方法 | 描述
---|---
[setbufsize](https://numpy.org/devdocs/reference/generated/numpy.setbufsize.html#numpy.setbufsize)(size) | 设置ufuncs中使用的缓冲区的大小。

## 错误处理

通用功能可以使硬件中的特殊浮点状态寄存器跳闸（例如除零）。如果在您的平台上可用，则在计算期间将定期检查这些寄存器。错误处理基于每个线程进行控制，并且可以使用这些函数进行配置

方法 | 描述
---|---
[seterr](https://numpy.org/devdocs/reference/generated/numpy.seterr.html#numpy.seterr)([all, divide, over, under, invalid]) | 设置如何处理浮点错误。
[seterrcall](https://numpy.org/devdocs/reference/generated/numpy.seterrcall.html#numpy.seterrcall)(func) | 设置浮点错误回调函数或日志对象。

## 映射规则

::: tip 注意

在NumPy 1.6.0中，创建了一个类型提升API来封装用于确定输出类型的机制。有关详细信息，
 请参阅函数
 [``result_type``](https://numpy.org/devdocs/reference/generated/numpy.result_type.html#numpy.result_type)，。[``promote_types``](https://numpy.org/devdocs/reference/generated/numpy.promote_types.html#numpy.promote_types)[``min_scalar_type``](https://numpy.org/devdocs/reference/generated/numpy.min_scalar_type.html#numpy.min_scalar_type)

:::

每个ufunc的核心是一维跨步循环，它实现特定类型组合的实际功能。创建ufunc时，会给出一个内部循环的静态列表以及ufunc操作的相应类型签名列表。ufunc机器使用此列表来确定用于特定情况的内部循环。
您可以检查[``.types``](https://numpy.org/devdocs/reference/generated/numpy.ufunc.types.html#numpy.ufunc.types)特定ufunc 的属性，
以查看哪些类型组合具有已定义的内部循环以及它们生成的输出类型（为简洁起见，在所述输出中使用[字符代码](/reference/arrays/scalars.htmll#内置标量类型)）。

每当ufunc没有提供的输入类型的核心循环实现时，必须在一个或多个输入上进行强制转换。如果无法找到输入类型的实现，则算法搜索具有类型签名的实现，所有输入都可以“安全地”强制转换到该实现。在所有必要的类型转换之后，选择并执行在其内部循环列表中找到的第一个循环。回想一下，ufuncs期间的内部副本（甚至对于转换）限制为内部缓冲区的大小（可由用户设置）。

::: tip 注意

NumPy中的通用功能足够灵活，可以具有混合类型签名。因此，例如，可以定义使用浮点和整数值的通函数。
请参阅[``ldexp``](https://numpy.org/devdocs/reference/generated/numpy.ldexp.html#numpy.ldexp)
示例。

:::

通过以上描述，强制转换规则基本上是通过何时可以将数据类型 “安全地” 强制转换为另一数据类型的问题来实现的。这个问题的答案可以通过函数调用在Python中确定：[``can_cast(fromtype, totype)``](https://numpy.org/devdocs/reference/generated/numpy.can_cast.html#numpy.can_cast)。下图显示了对作者的64位系统上的24个内部支持的类型进行调用的结果。您可以使用图中给出的代码为您的系统生成此表。

显示32位系统的“可以安全转换”表的代码段。

``` python
>>> def print_table(ntypes):
...     print 'X',
...     for char in ntypes: print char,
...     print
...     for row in ntypes:
...         print row,
...         for col in ntypes:
...             print int(np.can_cast(row, col)),
...         print
>>> print_table(np.typecodes['All'])
X ? b h i l q p B H I L Q P e f d g F D G S U V O M m
? 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
b 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0
h 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0
i 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1 0 0
l 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1 0 0
q 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1 0 0
p 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1 0 0
B 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
H 0 0 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0
I 0 0 0 0 1 1 1 0 0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 0 0
L 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 0 1 1 1 1 1 1 0 0
Q 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 0 1 1 1 1 1 1 0 0
P 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 0 1 1 1 1 1 1 0 0
e 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0
f 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0
d 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1 0 0
g 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 1 0 0
F 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0
D 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0
G 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0
U 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0
V 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
O 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0
M 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
m 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
```

您应该注意，虽然表中包含了完整性，但是ufuncs无法对“S”，“U”和“V”类型进行操作。另请注意，在32位系统上，整数类型可能具有不同的大小，从而导致表稍微改变。

混合标量数组操作使用一组不同的强制转换规则，以确保标量不能“向上”数组，除非标量是一种根本不同类型的数据（ *即* ，在数据类型层次结构中的不同层次结构下），而不是数组。此规则使您可以在代码中使用标量常量（在Python类型中，相应地在ufunc中进行解释），而不必担心标量常量的精度是否会导致大型（小精度）数组的上转。

## 覆盖Ufunc行为

类（包括ndarray子类）可以通过定义某些特殊方法来覆盖ufunc对它们的作用。有关详细信息，请参阅
 [标准数组子类](/reference/arrays/classes.html)。

## ``ufunc``

### 可选的关键字参数

所有ufunc都采用可选的关键字参数。其中大多数代表高级用法，通常不会被使用。

- *out* 

    *版本1.6中的新功能。* 

    第一个输出可以作为位置参数或关键字参数提供。关键字'out'参数与位置参数不兼容。

    *版本1.10中的新功能。* 

    'out'关键字参数应该是一个元组，每个输出有一个条目（对于由ufunc分配的数组，它可以是 *None* ）。对于具有单个输出的ufunc，传递单个数组（而不是包含单个数组的元组）也是有效的。

    不推荐将'out'关键字参数中的单个数组传递给具有多个输出的ufunc，并且将在numpy 1.10中引发警告，并在将来的版本中引发错误。

    如果'out'为None（默认值），则创建一个未初始化的返回数组。然后在广播“where”为True的位置填充输出数组的ufunc结果。如果'where'是标量True（默认值），那么这对应于填充的整个输出。请注意，未明确填充的输出将保留未初始化的值。

- *where* 

    *版本1.7中的新功能。* 

    接受与操作数一起广播的布尔数组。值True表示计算该位置的ufunc，值False表示仅将值保留在输出中。此参数不能用于广义ufunc，因为它们采用非标量输入。

    请注意，如果创建了未初始化的返回数组，则值False将使这些值保持**未初始化状态**。

- *axes* 

    *版本1.15中的新功能。* 

    具有广义ufunc应对其操作的轴的索引的元组的列表。
    例如，对于适合于矩阵乘法的 ``(i，j)，(j，k)->(i，k)`` 的签名，基本元素是二维矩阵，
    并且这些基本元素被认为存储在每个自变量的最后两个轴中。相应的axes关键字将是 ``[(-2，-1)，(-2，-1)，(-2，-1)]``。
    为了简单起见，对于在一维数组（向量）上操作的广义ufuncs，
    接受单个整数而不是单元素元组，并且对于其所有输出都是标量的广义ufuncs，
    可以省略输出元组。

- *axis* 

    *版本1.15中的新功能。* 

    广义ufunc应在其上运行的单个轴。这是在单个共享核心维度上运行的ufunc的快捷方式，
    相当于传入每个单核维度参数的``axes``条目``(axis,)``以及``()``所有其他参数。
    例如，对于签名``(i),(i)->()``，它相当于传入 ``axes=[(axis,), (axis,), ()]``。

- *keepdims* 

    *版本1.15中的新功能。* 

    如果将此值设置为 *True* ，则减小的轴将作为尺寸为1的尺寸保留在结果中，以便结果将针对输入正确广播。此选项只能用于对输入进行操作的通用ufunc，这些输入都具有相同数量的核心维度，并且输出没有核心维度，即具有类似``(i),(i)->()``或的签名``(m,m)->()``。如果使用，可以使用``axes``和控制输出中尺寸的位置``axis``。

- *casting* 

    *版本1.6中的新功能。* 

    可能是'不'，'等于'，'安全'，'same_kind'或'不安全'。有关[``can_cast``](https://numpy.org/devdocs/reference/generated/numpy.can_cast.html#numpy.can_cast)参数值的说明，请参阅。

    提供允许何种类型转换的策略。为了与以前版本的NumPy兼容，对于numpy <1.7，默认为“unsafe”。在numpy 1.7中，开始过渡到'same_kind'，其中ufunc为“不安全”规则允许的呼叫产生DeprecationWarning，但不在'same_kind'规则下。从numpy 1.10开始，默认为'same_kind'。

- *order* 

    *版本1.6中的新功能。* 

    指定输出数组的计算迭代顺序/内存布局。默认为“K”。'C'表示输出应该是C连续的，'F'表示F-连续，'A'表示F-连续，如果输入是F-连续的而且也不是C-连续的，否则是C-连续的，'K' '意味着尽可能地匹配输入的元素排序。

- *dtype* 

    *版本1.6中的新功能。* 

    覆盖计算和输出数组的dtype。与 *signature* 类似。

- *subok* 

    *版本1.6中的新功能。* 

    默认为true。如果设置为false，则输出将始终为严格数组，而不是子类型。

- *signature* 

    数据类型、数据类型的元组或指示ufunc的输入和输出类型的特殊签名字符串。
    此参数允许您为1-d循环提供在基础计算中使用的特定签名。
    如果为ufunc指定的循环不存在，则引发TypeError。
    通常，通过将输入类型与可用的输入类型进行比较并搜索具有所有输入都可以安全强制转换到的数据类型的循环，可以自动找到合适的循环。
    此关键字参数允许您绕过该搜索并选择特定循环。
    可用签名的列表由ufunc对象的types属性提供。为了向后兼容，该参数也可以作为sig提供，
    但最好使用长形式。请注意，这不应与存储在ufunc对象的签名属性中的通用ufunc签名混淆。

- *extobj* 

    长度为1、2或3的列表，指定ufunc缓冲区大小、错误模式整数和错误回调函数。
    通常，这些值在特定于线程的字典中查找。在这里传递它们可以绕过查找并使用为错误模式提供的低级规范。
    这可能是有用的，例如，作为对循环中的小数组上需要许多ufunc调用的计算的优化。

### 属性

通用功能具有一些信息属性。没有属性可以设置。

属性 | 描述
---|---
\_\_doc__ | 每个ufunc的文档字符串。docstring的第一部分是根据输出数量，名称和输入数量动态生成的。docstring的第二部分在创建时提供，并与ufunc一起存储。
\_\_name__ | ufunc的名称。

方法 | 描述
---|---
[ufunc.nin](https://numpy.org/devdocs/reference/generated/numpy.ufunc.nin.html#numpy.ufunc.nin) | 输入数量。
[ufunc.nout](https://numpy.org/devdocs/reference/generated/numpy.ufunc.nout.html#numpy.ufunc.nout) | 输出数量。
[ufunc.nargs](https://numpy.org/devdocs/reference/generated/numpy.ufunc.nargs.html#numpy.ufunc.nargs) | 参数的数量。
[ufunc.ntypes](https://numpy.org/devdocs/reference/generated/numpy.ufunc.ntypes.html#numpy.ufunc.ntypes) | 类型数量。
[ufunc.types](https://numpy.org/devdocs/reference/generated/numpy.ufunc.types.html#numpy.ufunc.types) | 返回包含input-> output类型的列表。
[ufunc.identity](https://numpy.org/devdocs/reference/generated/numpy.ufunc.identity.html#numpy.ufunc.identity) | 身份价值。
[ufunc.signature](https://numpy.org/devdocs/reference/generated/numpy.ufunc.signature.html#numpy.ufunc.signature) | 广义ufunc操作的核心元素的定义。

### 方法

所有ufunc都有四种方法。但是，这些方法仅对采用两个输入参数并返回一个输出参数的标量ufunc有意义。试图在其他ufunc上调用这些方法会导致a
 [``ValueError``](https://docs.python.org/dev/library/exceptions.html#ValueError)。类似reduce的方法都采用 *axis* 关键字，
 *dtype* 关键字和 *out* 关键字，并且数组必须都具有维> = 1。*轴* 关键字指定将在其上进行缩减的数组的轴（具有负数）值倒计时）。通常，它是一个整数，但是[``ufunc.reduce``](https://numpy.org/devdocs/reference/generated/numpy.ufunc.reduce.html#numpy.ufunc.reduce)，它也可以是一次[``int``](https://docs.python.org/dev/library/functions.html#int)减少多个轴或 *无* ，以减少所有轴的元组。在 *D型* 关键字允许您管理天真使用时出现的非常常见的问题[``ufunc.reduce``](https://numpy.org/devdocs/reference/generated/numpy.ufunc.reduce.html#numpy.ufunc.reduce)。有时您可能拥有某种数据类型的数组，并希望将其所有元素相加，但结果不适合数组的数据类型。如果您有一个单字节整数数组，通常会发生这种情况。的 *D型细胞* 关键字允许以改变在其上发生还原（以及因此输出的类型）的数据类型。因此，您可以确保输出是一种精度足以处理输出的数据类型。改变减少类型的责任主要取决于你。有一个例外：如果没有 *dtype* 给出减少“add”或“multiply”操作，然后如果输入类型是整数（或布尔）数据类型且小于``int_``数据类型的大小 ，它将在内部向上转换为``int_`` 或``uint``） 数据类型。最后， *out* 关键字允许您提供一个输出数组（对于单输出ufunc，这是当前唯一支持的;对于将来的扩展，但是，可以传入一个带有单个参数的元组）。如果给出 *out* ，则忽略 *dtype* 参数。

Ufuncs还有第五种方法，允许使用花哨的索引来执行就地操作。在使用花式索引的维度上不使用缓冲，因此花式索引可以多次列出项目，并且将对该项目的上一个操作的结果执行操作。

方法 | 描述
---|---
[ufunc.reduce](https://numpy.org/devdocs/reference/generated/numpy.ufunc.reduce.html#numpy.ufunc.reduce)(a[, axis, dtype, out, …]) | 减少一个接一个的尺寸，由沿一个轴施加ufunc。
[ufunc.accumulate](https://numpy.org/devdocs/reference/generated/numpy.ufunc.accumulate.html#numpy.ufunc.accumulate)(array[, axis, dtype, out]) | 累积将运算符应用于所有元素的结果。
[ufunc.reduceat](https://numpy.org/devdocs/reference/generated/numpy.ufunc.reduceat.html#numpy.ufunc.reduceat)(a, indices[, axis, dtype, out]) | 在单个轴上使用指定切片执行（局部）缩减。
[ufunc.outer](https://numpy.org/devdocs/reference/generated/numpy.ufunc.outer.html#numpy.ufunc.outer)(A, B, **kwargs) | 将ufunc op应用于所有对（a，b），其中a中的a和b中的b。
[ufunc.at](https://numpy.org/devdocs/reference/generated/numpy.ufunc.at.html#numpy.ufunc.at)(a, indices[, b]) | 对'index'指定的元素在操作数'a'上执行无缓冲的就地操作。

::: danger 警告

对数组类型进行类似reduce的操作，其数据类型的范围“太小”，无法处理结果，将以静默方式进行换行。应该[``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype)用来增加减少发生的数据类型的大小。

:::

## 可用ufuncs 

目前在 [``numpy``](index.html#module-numpy) 中定义了一种或多种类型的60多种通用功能，涵盖了各种各样的操作。
当使用相关的中缀符号时，在数组上自动调用这些ufunc中的一些（例如，当写入 ``a + b`` 并且 *a* 或 *b* 是 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 时，在内部调用 [``add(a, b)``](https://numpy.org/devdocs/reference/generated/numpy.add.html#numpy.add)）。
尽管如此，您可能仍希望使用ufunc调用以使用可选的输出参数将输出放置在您选择的对象（或多个对象）中。

回想一下，每个ufunc都是逐个元素运行的。因此，每个标量ufunc将被描述为如果作用于一组标量输入以返回一组标量输出。

::: tip 注意

即使您使用可选的输出参数，ufunc仍会返回其输出。

:::

### 数学运算

方法 | 描述
---|---
[add](https://numpy.org/devdocs/reference/generated/numpy.add.html#numpy.add)(x1, x2, /[, out, where, cast, order, ...]) | 按元素添加参数。
[subtract](https://numpy.org/devdocs/reference/generated/numpy.subtract.html#numpy.subtract)(x1, x2, /[, out, where, cast, ...]) | 从元素方面减去参数。
[multiply](https://numpy.org/devdocs/reference/generated/numpy.multiply.html#numpy.multiply)(x1, x2, /[, out, where, cast, ...]) | 在元素方面乘以论证。
[divide](https://numpy.org/devdocs/reference/generated/numpy.divide.html#numpy.divide)(x1, x2, /[, out, where, cast, ...]) | 以元素方式返回输入的真正除法。
[logaddexp](https://numpy.org/devdocs/reference/generated/numpy.logaddexp.html#numpy.logaddexp)(x1, x2, /[, out, where, cast, ...]) | 输入的取幂之和的对数。
[logaddexp2](https://numpy.org/devdocs/reference/generated/numpy.logaddexp2.html#numpy.logaddexp2)(x1, x2, /[, out, where, cast, ...]) | base-2中输入的取幂之和的对数。
[true_divide](https://numpy.org/devdocs/reference/generated/numpy.true_divide.html#numpy.true_divide)(x1, x2, /[, out, where, ...]) | 以元素方式返回输入的真正除法。
[floor_divide](https://numpy.org/devdocs/reference/generated/numpy.floor_divide.html#numpy.floor_divide)(x1, x2, /[, out, where, ...]) | 返回小于或等于输入除法的最大整数。
[negative](https://numpy.org/devdocs/reference/generated/numpy.negative.html#numpy.negative)(x, /[, out, where, cast, order, ...]) | 数字否定, 元素方面。
[positive](https://numpy.org/devdocs/reference/generated/numpy.positive.html#numpy.positive)(x, /[, out, where, cast, order, ...]) | 数字正面, 元素方面。
[power](https://numpy.org/devdocs/reference/generated/numpy.power.html#numpy.power)(x1, x2, /[, out, where, cast, ...]) | 第一个数组元素从第二个数组提升到幂, 逐个元素。
[remainder](https://numpy.org/devdocs/reference/generated/numpy.remainder.html#numpy.remainder)(x1, x2, /[, out, where, cast, ...]) | 返回除法元素的余数。
[mod](https://numpy.org/devdocs/reference/generated/numpy.mod.html#numpy.mod)(x1, x2, /[, out, where, cast, order, ...]) | 返回除法元素的余数。
[fmod](https://numpy.org/devdocs/reference/generated/numpy.fmod.html#numpy.fmod)(x1, x2, /[, out, where, cast, ...]) | 返回除法的元素余数。
[divmod](https://numpy.org/devdocs/reference/generated/numpy.divmod.html#numpy.divmod)(x1, x2 [, out1, out2], /[[, out, ...]) | 同时返回逐元素的商和余数。
[absolute](https://numpy.org/devdocs/reference/generated/numpy.absolute.html#numpy.absolute)(x, /[, out, where, cast, order, ...]) | 逐个元素地计算绝对值。
[fabs](https://numpy.org/devdocs/reference/generated/numpy.fabs.html#numpy.fabs)(x, /[, out, where, cast, order, ...]) | 以元素方式计算绝对值。
[rint](https://numpy.org/devdocs/reference/generated/numpy.rint.html#numpy.rint)(x, /[, out, where, cast, order, ...]) | 将数组的元素舍入为最接近的整数。
[sign](https://numpy.org/devdocs/reference/generated/numpy.sign.html#numpy.sign)(x, /[, out, where, cast, order, ...]) | 返回数字符号的元素指示。
[heaviside](https://numpy.org/devdocs/reference/generated/numpy.heaviside.html#numpy.heaviside)(x1, x2, /[, out, where, cast, ...]) | 计算Heaviside阶跃函数。
[conj](https://numpy.org/devdocs/reference/generated/numpy.conj.html#numpy.conj)(x, /[, out, where, cast, order, ...]) | 以元素方式返回复共轭。
[conjugate](https://numpy.org/devdocs/reference/generated/numpy.conjugate.html#numpy.conjugate)(x, /[, out, where, cast, ...]) | 以元素方式返回复共轭。
[exp](https://numpy.org/devdocs/reference/generated/numpy.exp.html#numpy.exp)(x, /[, out, where, cast, order, ...]) | 计算输入数组中所有元素的指数。
[exp2](https://numpy.org/devdocs/reference/generated/numpy.exp2.html#numpy.exp2)(x, /[, out, where, cast, order, ...]) | 计算输入数组中所有 p 的 2\*\*p。
[log](https://numpy.org/devdocs/reference/generated/numpy.log.html#numpy.log)(x, /[, out, where, cast, order, ...]) | 自然对数, 元素方面。
[log2](https://numpy.org/devdocs/reference/generated/numpy.log2.html#numpy.log2)(x, /[, out, where, cast, order, ...]) | x的基数为2的对数。
[log10](https://numpy.org/devdocs/reference/generated/numpy.log10.html#numpy.log10)(x, /[, out, where, cast, order, ...]) | 以元素方式返回输入数组的基数10对数。
[expm1](https://numpy.org/devdocs/reference/generated/numpy.expm1.html#numpy.expm1)(x, /[, out, where, cast, order, ...]) | 计算数组中的所有元素。exp(x) - 1
[log1p](https://numpy.org/devdocs/reference/generated/numpy.log1p.html#numpy.log1p)(x, /[, out, where, cast, order, ...]) | 返回一个加上输入数组的自然对数, 逐个元素。
[sqrt](https://numpy.org/devdocs/reference/generated/numpy.sqrt.html#numpy.sqrt)(x, /[, out, where, cast, order, ...]) | 以元素方式返回数组的非负平方根。
[square](https://numpy.org/devdocs/reference/generated/numpy.square.html#numpy.square)(x, /[, out, where, cast, order, ...]) | 返回输入的元素方块。
[cbrt](https://numpy.org/devdocs/reference/generated/numpy.cbrt.html#numpy.cbrt)(x, /[, out, where, cast, order, ...]) | 以元素方式返回数组的立方根。
[reciprocal](https://numpy.org/devdocs/reference/generated/numpy.reciprocal.html#numpy.reciprocal)(x, /[, out, where, cast, ...]) | 以元素方式返回参数的倒数。
[gcd](https://numpy.org/devdocs/reference/generated/numpy.gcd.html#numpy.gcd)(x1, x2, /[, out, where, cast, order, ...]) | 返回 \| x1 \| 和的最大公约数  \| x2 \| 。
[lcm](https://numpy.org/devdocs/reference/generated/numpy.lcm.html#numpy.lcm)(x1, x2, /[, out, where, cast, order, ...]) | 返回  \| x1 \|  和的最小公倍数  \| x2 \| 。

::: tip 提示 

可选的输出参数可用于帮助您节省大型计算的内存。
如果您的数组很大，由于临时计算空间的创建和（稍后）破坏，复杂的表达式可能需要比绝对必要的时间更长的时间。
例如，表达式 ``G = a * b + c`` 等于 ``t1 = A * B; G = T1 + C; del t1``。
它将更快地执行为 ``G = A * B; add(G, C, G)`` 与 ``G = A * B; G + = C`` 相同.

:::

### 三角函数

当需要角度时, 所有三角函数都使用弧度。度与弧度的比率是

方法 | 描述
---|---
[sin](https://numpy.org/devdocs/reference/generated/numpy.sin.html#numpy.sin)(x, /[, out, where, cast, order, ...]) | 三角正弦, 元素方式。
[cos](https://numpy.org/devdocs/reference/generated/numpy.cos.html#numpy.cos)(x, /[, out, where, cast, order, ...]) | 余弦元素。
[tan](https://numpy.org/devdocs/reference/generated/numpy.tan.html#numpy.tan)(x, /[, out, where, cast, order, ...]) | 计算切线元素。
[arcsin](https://numpy.org/devdocs/reference/generated/numpy.arcsin.html#numpy.arcsin)(x, /[, out, where, cast, order, ...]) | 反向正弦, 元素方式。
[arccos](https://numpy.org/devdocs/reference/generated/numpy.arccos.html#numpy.arccos)(x, /[, out, where, cast, order, ...]) | 三角反余弦, 元素方式。
[arctan](https://numpy.org/devdocs/reference/generated/numpy.arctan.html#numpy.arctan)(x, /[, out, where, cast, order, ...]) | 三角反正切, 逐元素。
[arctan2](https://numpy.org/devdocs/reference/generated/numpy.arctan2.html#numpy.arctan2)(x1, x2, /[, out, where, cast, ...]) | x1/x2正确选择象限的逐元素反正切。
[hypot](https://numpy.org/devdocs/reference/generated/numpy.hypot.html#numpy.hypot)(x1, x2, /[, out, where, cast, ...]) | 给定直角三角形的“腿”, 返回其斜边。
[sinh](https://numpy.org/devdocs/reference/generated/numpy.sinh.html#numpy.sinh)(x, /[, out, where, cast, order, ...]) | 双曲正弦, 元素。
[cosh](https://numpy.org/devdocs/reference/generated/numpy.cosh.html#numpy.cosh)(x, /[, out, where, cast, order, ...]) | 双曲余弦, 元素。
[tanh](https://numpy.org/devdocs/reference/generated/numpy.tanh.html#numpy.tanh)(x, /[, out, where, cast, order, ...]) | 计算双曲正切元素。
[arcsinh](https://numpy.org/devdocs/reference/generated/numpy.arcsinh.html#numpy.arcsinh)(x, /[, out, where, cast, order, ...]) | 逆双曲正弦元素。
[arccosh](https://numpy.org/devdocs/reference/generated/numpy.arccosh.html#numpy.arccosh)(x, /[, out, where, cast, order, ...]) | 反双曲余弦, 元素。
[arctanh](https://numpy.org/devdocs/reference/generated/numpy.arctanh.html#numpy.arctanh)(x, /[, out, where, cast, order, ...]) | 逆双曲正切元素。
[deg2rad](https://numpy.org/devdocs/reference/generated/numpy.deg2rad.html#numpy.deg2rad)(x, /[, out, where, cast, order, ...]) | 将角度从度数转换为弧度。
[rad2deg](https://numpy.org/devdocs/reference/generated/numpy.rad2deg.html#numpy.rad2deg)(x, /[, out, where, cast, order, ...]) | 将角度从弧度转换为度数。

### 位运算函数

这些函数都需要整数参数, 并且它们操纵这些参数的位模式。

方法 | 描述
---|---
[bitwise_and](https://numpy.org/devdocs/reference/generated/numpy.bitwise_and.html#numpy.bitwise_and)(x1, x2, /[, out, where, ...]) | 逐个元素地计算两个数组的逐位AND。
[bitwise_or](https://numpy.org/devdocs/reference/generated/numpy.bitwise_or.html#numpy.bitwise_or)(x1, x2, /[, out, where, cast, ...]) | 逐个元素地计算两个数组的逐位OR。
[bitwise_xor](https://numpy.org/devdocs/reference/generated/numpy.bitwise_xor.html#numpy.bitwise_xor)(x1, x2, /[, out, where, ...]) | 逐个元素地计算两个数组的逐位XOR。
[invert](https://numpy.org/devdocs/reference/generated/numpy.invert.html#numpy.invert)(x, /[, out, where, cast, order, ...]) | 计算逐位反转, 或逐位NOT, 逐元素计算。
[left_shift](https://numpy.org/devdocs/reference/generated/numpy.left_shift.html#numpy.left_shift)(x1, x2, /[, out, where, cast, ...]) | 将整数位移到左侧。
[right_shift](https://numpy.org/devdocs/reference/generated/numpy.right_shift.html#numpy.right_shift)(x1, x2, /[, out, where, ...]) | 将整数位移到右侧。

### 比较函数

方法 | 描述
---|---
[greater](https://numpy.org/devdocs/reference/generated/numpy.greater.html#numpy.greater)(x1, x2, /[, out, where, cast, ...]) | 以元素方式返回(x1 > x2)的真值。
[greater_equal](https://numpy.org/devdocs/reference/generated/numpy.greater_equal.html#numpy.greater_equal)(x1, x2, /[, out, where, ...]) | 以元素方式返回(x1 >= x2)的真值。
[less](https://numpy.org/devdocs/reference/generated/numpy.less.html#numpy.less)(x1, x2, /[, out, where, cast, ...]) | 返回(x1 < x2)元素的真值。
[less_equal](https://numpy.org/devdocs/reference/generated/numpy.less_equal.html#numpy.less_equal)(x1, x2, /[, out, where, cast, ...]) | 以元素方式返回(x1 =< x2)的真值。
[not_equal](https://numpy.org/devdocs/reference/generated/numpy.not_equal.html#numpy.not_equal)(x1, x2, /[, out, where, cast, ...]) | 以元素方式返回(x1 != x2)。
[equal](https://numpy.org/devdocs/reference/generated/numpy.equal.html#numpy.equal)(x1, x2, /[, out, where, cast, ...]) | 以元素方式返回(x1 == x2)。

::: danger 警告

不要使用Python关键字``and``并``or``组合逻辑数组表达式。这些关键字将测试整个数组的真值(不是你想象的逐个元素)。使用按位运算符＆和| 代替。

:::

方法 | 描述
---|---
[logical_and](https://numpy.org/devdocs/reference/generated/numpy.logical_and.html#numpy.logical_and)(x1, x2, /[, out, where, ...]) | 计算x1和x2元素的真值。
[logical_or](https://numpy.org/devdocs/reference/generated/numpy.logical_or.html#numpy.logical_or)(x1, x2, /[, out, where, cast, ...]) | 计算x1 OR x2元素的真值。
[logical_xor](https://numpy.org/devdocs/reference/generated/numpy.logical_xor.html#numpy.logical_xor)(x1, x2, /[, out, where, ...]) | 以元素方式计算x1 XOR x2的真值。
[logical_not](https://numpy.org/devdocs/reference/generated/numpy.logical_not.html#numpy.logical_not)(x, /[, out, where, cast, ...]) | 计算NOT x元素的真值。

::: danger 警告

逐位运算符＆和| 是执行逐个元素数组比较的正确方法。
确保您理解运算符优先级：``(a > 2) ＆ (a < 5)`` 是正确的语法，
因为 ``a > 2 & a < 5`` 将导致错误，因为首先计算 ``2 ＆ a``。

:::

方法 | 描述
---|---
[maximum](https://numpy.org/devdocs/reference/generated/numpy.maximum.html#numpy.maximum)(x1, x2, /[, out, where, cast, ...]) | 元素最大的数组元素。

::: tip 提示

Python函数``max()``将在一维数组中找到最大值, 但它将使用较慢的序列接口来实现。最大ufunc的reduce方法要快得多。此外, 该``max()``方法不会给出具有多
个维度的数组所期望的答案。reduce的minimal方法还允许您计算数组的总最小值。

:::

方法 | 描述
---|---
[minimum](https://numpy.org/devdocs/reference/generated/numpy.minimum.html#numpy.minimum)(x1, x2, /[, out, where, cast, ...]) | 元素最小的数组元素。

::: danger 警告

``maximum(a，b)`` 的行为与 ``max(a，b)`` 的行为不同。作为 ufunc，``maximum(a，b)`` 执行 *a* 和 *b* 的逐个元素比较，并根据两个数组中哪个元素较大来选择结果的每个元素。相反，``max(a，b)`` 将对象a和b视为一个整体，查看 ``a > b`` 的(总)真值，并使用它返回a或b(作为一个整体)。在 ``minimum(a，b)`` 和 ``min(a，b)`` 之间存在类似的差异。

:::

方法 | 描述
---|---
[fmax](https://numpy.org/devdocs/reference/generated/numpy.fmax.html#numpy.fmax)(x1, x2, /[, out, where, cast, ...]) | 元素最大的数组元素。
[fmin](https://numpy.org/devdocs/reference/generated/numpy.fmin.html#numpy.fmin)(x1, x2, /[, out, where, cast, ...]) | 元素最小的数组元素。

### 浮动函数

回想一下, 所有这些函数都在一个数组上逐个元素地工作, 返回一个数组输出。该描述仅详细说明了一个操作。

方法 | 描述
---|---
[isfinite](https://numpy.org/devdocs/reference/generated/numpy.isfinite.html#numpy.isfinite)(x, /[, out, where, cast, order, ...]) | 测试元素的有限性(不是无穷大或不是数字)。
[isinf](https://numpy.org/devdocs/reference/generated/numpy.isinf.html#numpy.isinf)(x, /[, out, where, cast, order, ...]) | 正面或负面无穷大的元素测试。
[isnan](https://numpy.org/devdocs/reference/generated/numpy.isnan.html#numpy.isnan)(x, /[, out, where, cast, order, ...]) | 测试NaN的元素, 并将结果作为布尔数组返回。
[isnat](https://numpy.org/devdocs/reference/generated/numpy.isnat.html#numpy.isnat)(x, /[, out, where, cast, order, ...]) | 为NaT(不是时间)测试元素, 并将结果作为布尔数组返回。
[fabs](https://numpy.org/devdocs/reference/generated/numpy.fabs.html#numpy.fabs)(x, /[, out, where, cast, order, ...]) | 以元素方式计算绝对值。
[signbit](https://numpy.org/devdocs/reference/generated/numpy.signbit.html#numpy.signbit)(x, /[, out, where, cast, order, ...]) | 返回元素为True设置signbit(小于零)。
[copysign](https://numpy.org/devdocs/reference/generated/numpy.copysign.html#numpy.copysign)(x1, x2, /[, out, where, cast, ...]) | 将元素x1的符号更改为x2的符号。
[nextafter](https://numpy.org/devdocs/reference/generated/numpy.nextafter.html#numpy.nextafter)(x1, x2, /[, out, where, cast, ...]) | 将x1之后的下一个浮点值返回x2(元素方向)。
[spacing](https://numpy.org/devdocs/reference/generated/numpy.spacing.html#numpy.spacing)(x, /[, out, where, cast, order, ...]) | 返回x与最近的相邻数字之间的距离。
[modf](https://numpy.org/devdocs/reference/generated/numpy.modf.html#numpy.modf)(x [, out1, out2], /[[, out, where, ...]) | 以元素方式返回数组的小数和整数部分。
[ldexp](https://numpy.org/devdocs/reference/generated/numpy.ldexp.html#numpy.ldexp)(x1, x2, /[, out, where, cast, ...]) | 以元素方式返回x1 * 2 ** x2。
[frexp](https://numpy.org/devdocs/reference/generated/numpy.frexp.html#numpy.frexp)(x [, out1, out2], /[[, out, where, ...]) | 将x的元素分解为尾数和二进制指数。
[fmod](https://numpy.org/devdocs/reference/generated/numpy.fmod.html#numpy.fmod)(x1, x2, /[, out, where, cast, ...]) | 返回除法的元素余数。
[floor](https://numpy.org/devdocs/reference/generated/numpy.floor.html#numpy.floor)(x, /[, out, where, cast, order, ...]) | 以元素方式返回输入的底限。
[ceil](https://numpy.org/devdocs/reference/generated/numpy.ceil.html#numpy.ceil)(x, /[, out, where, cast, order, ...]) | 以元素方式返回输入的上限。
[trunc](https://numpy.org/devdocs/reference/generated/numpy.trunc.html#numpy.trunc)(x, /[, out, where, cast, order, ...]) | 以元素方式返回输入的截断值。
