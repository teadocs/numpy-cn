# 标准数组子类

NumPy中的``ndarray``是一种“新式”Python内置类型。 因此，如果需要，它可以从（在Python或C中）继承。 因此，它可以为许多有用的类奠定基础。 通常，是否对数组对象进行子类化或简单地将核心数组组件用作新类的内部部分是一个困难的决定，并且可以简单地作为选择问题。 NumPy有几个工具可以简化新对象与其他数组对象的交互方式，因此最终选择可能并不重要。 简化问题的一种方法是问自己，你感兴趣的对象是否可以替换为单个数组，还是确实需要两个或更多数组。

请注意，``asarray``始终返回基类ndarray。 如果你确信使用数组对象可以处理ndarray的任何子类，则可以使用``asanyarray``来允许子类在子例程中更加干净地传播。 原则上，子类可以重新定义数组的任何方面，因此，在严格的指导下，“asanyarray”将很少有用。 但是，数组对象的大多数子类都不会重新定义数组对象的某些方面，例如缓冲区接口或数组的属性。 然而，一个重要的例子是你的子程序可能无法处理数组的任意子类的原因是矩阵将“*”运算符重新定义为矩阵乘法，而不是逐个元素乘法。

## 特殊属性和方法

另见：

> Subclassing ndarray

NumPy提供了几个类可以自定义的钩子：

### ``class.__array_ufunc__`` *(ufunc, method, \*inputs, \*\*kwargs)*
*版本1.13中的新功能。*

> **注意**
> API是临时的，即我们尚不保证向后兼容性。

任何类，ndarray子类或非类，都可以定义此方法或将其设置为“无”以覆盖NumPy的ufuncs的行为。 这与Python的`__mul__``和其他二进制操作例程非常相似。

- ufunc是被调用的ufunc对象。
- method是一个字符串，表示调用了哪个Ufunc方法（``__call__``，``reduce``，``reduceat``，``accumulate``，``outer``，``内在``）。
- inputs是``ufunc``的输入参数的元组。
- kwargs是一个包含ufunc的可选输入参数的字典。 如果给出，任何``out``参数，无论是位置还是关键字，都会在kwargs中作为``tuple``传递。 有关详细信息，请参阅通用函数（ufunc）中的讨论。

如果未执行请求的操作，该方法应该返回操作的结果，或者返回“NotImplemented”。

如果输入或输出参数之一具有`__array_ufunc__``方法，则执行它而不是ufunc。 如果多个参数实现``__array_ufunc__``，则按顺序尝试它们：超类之前的子类，输出之前的输入，否则从左到右。 返回“NotImplemented”之外的第一个例程确定结果。 如果所有``__array_ufunc__``操作都返回NotImplemented，则会引发``TypeError``。

> **注意**
> 我们打算将numpy函数重新实现为（generalized）Ufunc，在这种情况下，它们可以被``__array_ufunc__``方法覆盖。 一个主要的候选者是``matmul``，它当前不是Ufunc，但可以相对容易地被重写为（一组）广义Ufuncs。 使用诸如``median``，``min``和``argsort``等功能也可能发生同样的情况。

与python中的一些其他特殊方法一样，例如``__hash__``和``__iter__``，可以通过设置`__array_ufunc__`` = None来指示你的类不支持ufunc。 在设置``__array_ufunc__`` = None的对象上调用时，Ufuncs总是引发TypeError。

`__array_ufunc__``的存在也会影响``ndarray``如何处理二进制操作，如``arr + obj``和arr <obj，当arr是一个ndarray而obj是一个自定义类的实例。 有两种可能性。 如果obj .__ array_ufunc__存在而不是None，则ndarray .__ add__和friends将委托给ufunc机制，这意味着arr + obj变为np.add（arr，obj），然后添加调用obj .__ array_ufunc__。 如果要定义一个类似于数组的对象，这将非常有用。

或者，如果obj .__ array_ufunc__设置为None，那么作为一种特殊情况，像ndarray .__ add__这样的特殊方法会注意到这一点并无条件地引发TypeError。 如果要创建通过二进制操作与数组交互的对象，但这些对象本身不是数组，则此选项非常有用。 例如，单位处理系统可能有一个对象m代表“米”单位，并希望支持语法arr * m来表示该数组具有“米”单位，但不希望以其他方式通过ufuncs与数组交互 或者其他。 这可以通过设置__array_ufunc__ = None并定义__mul__和__rmul__方法来完成。 （注意，这意味着编写始终返回NotImplemented的__array_ufunc__与设置__array_ufunc__ = None不完全相同：在前一种情况下，arr + obj将引发TypeError，而在后一种情况下，可以定义__radd__方法 防止这种情况。）

以上内容不适用于就地运算符，ndarray永远不会返回NotImplemented。 因此，arr + = obj总是会导致TypeError。 这是因为对于阵列就地操作一般不能用简单的反向操作代替。 （例如，默认情况下，arr + = obj将转换为arr = arr + obj，即arr将被替换，与内部数组操作的预期相反。）

> **注意**
> 如果你定义__array_ufunc__：

> - 如果你不是ndarray的子类，我们建议你的类定义像_dadray那样委托给ufuncs的特殊方法，比如__add__和__lt__。一个简单的方法是从``NDArrayOperatorsMixin``继承。
> - 如果你继承``ndarray``，我们建议你将所有覆盖逻辑放在__array_ufunc__中，而不是覆盖特殊方法。 这确保了类层次结构只在一个地方确定，而不是由ufunc机制和二进制操作规则（它优先考虑子类的特殊方法;强制执行单一地方层次结构的另一种方法，将__array_ufunc__设置为 没有，看起来非常意外，因而令人困惑，因为那时子类根本无法使用ufuncs）。
> - ``ndarray``定义了自己的__array_ufunc__，如果没有参数覆盖，则计算ufunc，否则返回NotImplemented。 这对于__array_ufunc__将其自己的类的任何实例转换为ndarray的子类可能很有用：然后它可以使用super（）.__ array_ufunc __（* inputs，** kwargs）将它们传递给它的超类，最后在可能之后返回结果反变换。 这种做法的优点是它确保可以有一个扩展行为的子类层次结构。 有关详细信息，请参阅子类化ndarray。

> **注意**
> 如果一个类定义了``__array_ufunc__``方法，则会禁用下面针对ufuncs描述的`__array_wrap__``，`__array_prepare__``，``__array_priority__``机制（最终可能会弃用）。

### ``class.__array_finalize__``(obj)

只要系统在内部从obj分配一个新数组，就会调用此方法，其中obj是ndarray的子类（子类型）。 它可以用于在构造之后更改self的属性（例如，以确保2-d矩阵），或者从“父”更新元信息。子类继承此方法的默认实现，该方法不执行任何操作。

### ``class.__array_prepare__``(array, context=None)

在每个ufunc的开头，在具有最高数组优先级的输入对象上调用此方法，或者在指定了一个输出对象的情况下调用此方法。 传入输出数组，返回的任何内容都传递给ufunc。 子类继承此方法的默认实现，它只返回未修改的输出数组。 子类可以选择使用此方法将输出数组转换为子类的实例，并在将数组返回到ufunc进行计算之前更新元数据。

> **注意**
> 对于ufuncs，希望最终弃用这个方法，而不是__array_ufunc__。

### ``class.__array_wrap__``(array, context=None)

在每个ufunc的末尾，在具有最高数组优先级的输入对象上调用此方法，如果指定了一个输出对象，则调用此方法。 传入ufunc-computed数组，并将返回的任何内容传递给用户。 子类继承此方法的默认实现，该实现将数组转换为对象类的新实例。 子类可以选择使用此方法将输出数组转换为子类的实例，并在将数组返回给用户之前更新元数据。

> **注意**
> 对于ufuncs，希望最终弃用这个方法而不是`__array_ufunc__``。

### ``class.__array_priority__``
此属性的值用于确定在返回对象的Python类型有多种可能性的情况下要返回的对象类型。 子类为此属性继承默认值0.0。

> **注意**
> 对于ufuncs，希望最终弃用这个方法，而不是__array_ufunc__。

### ``class.__array__``([dtype])

如果使用具有`__array__``方法的类（ndarray子类）作为ufunc的输出对象，则结果将被写入由`__array__``返回的对象。 在输入数组上进行类似的转换。

## 矩阵对象

``matrix``对象继承自ndarray，因此，它们具有相同的ndarrays属性和方法。 但是，矩阵对象有六个重要区别，当你使用矩阵但希望它们像数组一样时，可能会导致意外结果：

1. 可以使用字符串表示法创建Matrix对象，以允许使用Matlab样式的语法，其中空格分隔列和分号（';'）分隔行。

1. Matrix对象始终是二维的。 这具有深远意义，因为m.ravel()仍然是二维的（第一维中为1），项选择返回二维对象，因此序列行为与数组根本不同。

1. 矩阵对象覆盖乘法是矩阵乘法。 确保你对可能希望接收矩阵的函数有所了解。 特别是考虑到当m是矩阵时asanyarray（m）返回矩阵的事实。

1. 矩阵物体覆盖功率以使矩阵升高为功率。 关于在使用asanyarray(...)获取数组对象的函数内部使用电源的相同警告适用于此事实。

1. 矩阵对象的默认__array_priority__是10.0，因此与ndarrays的混合操作总是产生矩阵。

1. 矩阵具有特殊属性，使计算更容易。这些是

    ``matrix.T``	返回矩阵的转置。
    ``matrix.H``	返回self的（复数）共轭转置。
    ``matrix.I``	返回可逆self的（乘法）逆。
    ``matrix.A``	将自己作为ndarray对象返回。


<div class="warning-warp">
<b>警告</b>

<p>矩阵对象覆盖乘法，'*'和幂，'**'分别是矩阵乘法和矩阵幂。 如果你的子例程可以接受子类并且你没有转换为基类数组，则必须使用ufuncs multiply和power来确保你对所有输入执行正确的操作。</p>
</div>

矩阵类是ndarray的Python子类，可以用作如何构造自己的ndarray子类的参考。 可以从其他矩阵，字符串以及可以转换为ndarray的任何其他内容创建矩阵。 名称“mat”是NumPy中“matrix”的别名。

``matrix``(data[, dtype, copy])	从类数组对象或数据字符串返回矩阵。
``asmatrix``(data[, dtype])	将输入解释为矩阵。
``bmat``(obj[, ldict, gdict]) 从字符串，嵌套序列或数组构建矩阵对象。

示例1：从字符串创建矩阵

```python
>>> a=mat('1 2 3; 4 5 3')
>>> print (a*a.T).I
[[ 0.2924 -0.1345]
 [-0.1345  0.0819]]
```

示例2：从嵌套序列创建矩阵

```python
>>> mat([[1,5,10],[1.0,3,4j]])
matrix([[  1.+0.j,   5.+0.j,  10.+0.j],
        [  1.+0.j,   3.+0.j,   0.+4.j]])
```

示例3：从数组创建矩阵

```python
>>> mat(random.rand(3,3)).T
matrix([[ 0.7699,  0.7922,  0.3294],
        [ 0.2792,  0.0101,  0.9219],
        [ 0.3398,  0.7571,  0.8197]])
```

## 内存映射文件数组

内存映射文件对于使用常规布局读取和/或修改大文件的小段非常有用，而无需将整个文件读入内存。 ndarray的一个简单子类使用内存映射文件作为数组的数据缓冲区。 对于小文件，将整个文件读入内存的开销通常并不重要，但是对于使用内存映射的大型文件可以节省大量资源。

内存映射文件数组有一个额外的方法（除了它们从ndarray继承的那些）：。flush()必须由用户手动调用，以确保对数组的任何更改实际上都写入磁盘。

``memmap``	为存储在磁盘上的二进制文件中的数组创建内存映射。
``memmap.flush``()	将数组中的任何更改写入磁盘上的文件。

例子：

```python
>>> a = memmap('newfile.dat', dtype=float, mode='w+', shape=1000)
>>> a[10] = 10.0
>>> a[30] = 30.0
>>> del a
>>> b = fromfile('newfile.dat', dtype=float)
>>> print b[10], b[30]
10.0 30.0
>>> a = memmap('newfile.dat', dtype=float)
>>> print a[10], a[30]
10.0 30.0
```

## Character arrays (``numpy.char``)

另见：

> Creating character arrays (numpy.char)

> **注意**
> chararray类的存在是为了向后兼容Numarray，不建议用于新开发。 从numpy 1.4开始，如果需要字符串数组，建议使用dtype object_，string_或unicode_的数组，并使用numpy.char模块中的free函数进行快速矢量化字符串操作。

这些是string_type或unicode_类型的增强数组。 这些数组继承自ndarray，但是在（逐个）元素的基础上特别定义了操作+，*和％。 这些操作在字符类型的标准ndarray上不可用。 此外，chararray具有所有标准字符串（和unicode）方法，在逐个元素的基础上执行它们。 也许创建chararray的最简单方法是使用self.view（chararray），其中self是str或unicode数据类型的ndarray。 但是，也可以使用numpy.chararray构造函数或通过numpy.char.array函数创建chararray：

- ``chararray``(shape[, itemsize，unicode, ...])提供字符串和unicode值数组的方便视图。
- ``core.defchararray.array``(obj[, itemsize, ...]) 创建一个 ``chararray``.

与str数据类型的标准ndarray的另一个不同之处在于chararray继承了Numarray引入的特性，即在项检索和比较操作中将忽略数组中任何元素末尾的空白空间。

## 记录数组(``numpy.rec``)

另见:

> 创建记录数组（numpy.rec），数据类型例程，数据类型对象（dtype）。

NumPy提供了``recarray``类，它允许访问结构化数组的字段作为属性，以及相应的标量数据类型对象记录。

- ``recarray``	构造一个允许使用属性进行字段访问的ndarray。
- ``record``	一种数据类型标量，允许字段访问作为属性查找。

## 掩码数组 (``numpy.ma``)

另见：

> Masked arrays

## 标准容器类

为了向后兼容并作为标准的“容器”类，Numeric的UserArray已经被带到NumPy并命名为``numpy.lib.user_array.container``容器类是一个Python类，其self.array属性是一个ndarray。 使用numpy.lib.user_array.container比使用ndarray本身更容易进行多重继承，因此默认包含它。 除了提及它的存在之外，这里没有记录，因为如果可以的话，我们鼓励你直接使用ndarray类。

``numpy.lib.user_array.container``(data[, …]) 标准容器类，便于多重继承。

## 数组迭代器

迭代器是数组处理的强大概念。 本质上，迭代器实现了一个通用的for循环。 如果myiter是一个迭代器对象，那么Python代码：

```python
for val in myiter:
    ...
    some code involving val
    ...
```

重复调用``val = myiter.next()``直到迭代器引发``StopIteration``。有几种方法可以迭代可能有用的数组：默认迭代，平面迭代和N维枚举。

### 默认迭代

ndarray对象的默认迭代器是序列类型的默认Python迭代器。 因此，当数组对象本身用作迭代器时。 默认行为等效于：

```python
for i in range(arr.shape[0]):
    val = arr[i]
```

此默认迭代器从数组中选择维度为N-1的子数组。 这可以是用于定义递归算法的有用构造。 要遍历整个数组，需要N个for循环。

```python
>>> a = arange(24).reshape(3,2,4)+10
>>> for val in a:
...     print 'item:', val
item: [[10 11 12 13]
 [14 15 16 17]]
item: [[18 19 20 21]
 [22 23 24 25]]
item: [[26 27 28 29]
 [30 31 32 33]]
```

### 平面迭代

- ``ndarray.flat`` 数组上的一维迭代器。

如前所述，ndarray对象的flat属性返回一个迭代器，它将以C风格的连续顺序循环遍历整个数组。

```python
>>> for i, val in enumerate(a.flat):
...     if i%5 == 0: print i, val
0 10
5 15
10 20
15 25
20 30
```

在这里，我使用了内置的枚举迭代器来返回迭代器索引以及值。

### N维枚举

- ``ndenumerate``(arr) 多维索引迭代器。

有时在迭代时获得N维索引可能很有用。 ndenumerate迭代器可以实现这一点。

```python
>>> for i, val in ndenumerate(a):
...     if sum(i)%5 == 0: print i, val
(0, 0, 0) 10
(1, 1, 3) 25
(2, 0, 3) 29
(2, 1, 2) 32
```

### 广播迭代器

- ``broadcast``	制作一个模仿广播的对象。

广播的一般概念也可以使用``broadcast``迭代器从Python获得。 此对象将N个对象作为输入，并返回一个迭代器，该迭代器返回元组，提供广播结果中的每个输入序列元素。

```python
>>> for val in broadcast([[1,0],[2,3]],[0,1]):
...     print val
(1, 0)
(0, 1)
(2, 0)
(3, 1)
```
