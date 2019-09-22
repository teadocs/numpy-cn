# 迭代数组

[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)NumPy 1.6中引入的迭代器对象提供了许多灵活的方法来以系统的方式访问一个或多个数组的所有元素。
本页介绍了在Python中使用该对象进行数组计算的一些基本方法，然后总结了如何在Cython中加速内部循环。
由于Python暴露
[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer) 是C数组迭代器API的相对简单的映射，
因此这些想法​​还将提供有关使用C或C ++的数组迭代的帮助。

## 单数组迭代

可以使用的最基本任务[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)是访问数组的每个元素。使用标准Python迭代器接口逐个提供每个元素。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a):
...     print(x, end=' ')
...
0 1 2 3 4 5
```

对于此迭代，需要注意的重要一点是，选择顺序以匹配数组的内存布局，而不是使用标准C或Fortran排序。
这样做是为了提高访问效率，反映了默认情况下只需要访问每个元素而不关心特定排序的想法。
我们可以通过迭代前一个数组的转置来看到这一点，而不是以C顺序获取该转置的副本。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a.T):
...     print(x, end=' ')
...
0 1 2 3 4 5
```

``` python
>>> for x in np.nditer(a.T.copy(order='C')):
...     print(x, end=' ')
...
0 3 1 4 2 5
```

*a* 和 *aT* 的元素以相同的顺序遍历，即它们存储在内存中的顺序，
而 *a.T.copy(order='C')* 的元素以不同的顺序访问，因为它们被放入不同的内存中布局。

### 控制迭代顺序

有时，无论内存中元素的布局如何，以特定顺序访问数组的元素都很重要。
该[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)对象提供了一个 *命令* 参数来控制迭代的这个方面。
具有上述行为的默认值是order ='K'以保持现有订单。
对于C顺序，可以使用order ='C'覆盖它，对于Fortran顺序，可以使用order ='F'覆盖它。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a, order='F'):
...     print(x, end=' ')
...
0 3 1 4 2 5
>>> for x in np.nditer(a.T, order='C'):
...     print(x, end=' ')
...
0 3 1 4 2 5
```

### 修改数组值

默认情况下，[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)将输入操作数视为只读对象。
为了能够修改数组元素，必须使用 *'readwrite'* 或 *'writeonly'*  每操作数标志指定读写或只写模式。

然后，nditer将生成可写的缓冲区数组，您可以修改它们。
但是，因为一旦迭代完成，nditer必须将此缓冲区数据复制回原始数组，
所以必须通过两种方法之一发出迭代结束时的信号。你可以：

- 使用 *with* 语句将nditer用作上下文管理器，并在退出上下文时写回临时数据。
- 完成迭代后调用迭代器的 *close* 方法，这将触发回写。

一旦调用 *close* 或退出其上下文，就不能再迭代nditer 。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> with np.nditer(a, op_flags=['readwrite']) as it:
...    for x in it:
...        x[...] = 2 * x
...
>>> a
array([[ 0,  2,  4],
       [ 6,  8, 10]])
```

### 使用外部循环

在所有实施例中，到目前为止，的元素 *一个* 由迭代器一次一个地提供，
因为所有的循环逻辑是内部的迭代器。虽然这很简单方便，但效率不高。
更好的方法是将一维最内层循环移动到迭代器外部的代码中。
这样，NumPy的矢量化操作可以用在被访问元素的较大块上。

该[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)会尽量提供尽可能大的内循环块。
通过强制'C'和'F'顺序，我们得到不同的外部循环大小。通过指定迭代器标志来启用此模式。

观察到默认情况下保持本机内存顺序，迭代器能够提供单个一维块，而在强制Fortran命令时，它必须提供三个块，每个块包含两个元素。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a, flags=['external_loop']):
...     print(x, end=' ')
...
[0 1 2 3 4 5]
```

``` python
>>> for x in np.nditer(a, flags=['external_loop'], order='F'):
...     print(x, end=' ')
...
[0 3] [1 4] [2 5]
```

### 跟踪索引或多索引

在迭代期间，您可能希望在计算中使用当前元素的索引。
例如，您可能希望以内存顺序访问数组的元素，但使用C顺序，Fortran顺序或多维索引来查找不同数组中的值。

Python迭代器协议没有从迭代器查询这些附加值的自然方法，
因此我们引入了一个替代语法来迭代[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)。
此语法显式使用迭代器对象本身，因此在迭代期间可以轻松访问其属性。使用此循环结构，可以通过索引到迭代器来访问当前值，并且正在跟踪的索引是属性 *索引* 或 *multi_index，* 具体取决于请求的内容。

遗憾的是，Python交互式解释器在循环的每次迭代期间打印出while循环内的表达式的值。我们使用这个循环结构修改了示例中的输出，以便更具可读性。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3)
>>> it = np.nditer(a, flags=['f_index'])
>>> while not it.finished:
...     print("%d <%d>" % (it[0], it.index), end=' ')
...     it.iternext()
...
0 <0> 1 <2> 2 <4> 3 <1> 4 <3> 5 <5>
```

``` python
>>> it = np.nditer(a, flags=['multi_index'])
>>> while not it.finished:
...     print("%d <%s>" % (it[0], it.multi_index), end=' ')
...     it.iternext()
...
0 <(0, 0)> 1 <(0, 1)> 2 <(0, 2)> 3 <(1, 0)> 4 <(1, 1)> 5 <(1, 2)>
```

``` python
>>> it = np.nditer(a, flags=['multi_index'], op_flags=['writeonly'])
>>> with it:
....    while not it.finished:
...         it[0] = it.multi_index[1] - it.multi_index[0]
...         it.iternext()
...
>>> a
array([[ 0,  1,  2],
       [-1,  0,  1]])
```

跟踪索引或多索引与使用外部循环不兼容，因为它需要每个元素具有不同的索引值。如果您尝试组合这些标志，该[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)对象将引发异常

**示例：**

``` python
>>> a = np.zeros((2,3))
>>> it = np.nditer(a, flags=['c_index', 'external_loop'])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked
```

### 缓冲数组元素

在强制迭代顺序时，我们观察到外部循环选项可以以较小的块提供元素，因为不能以恒定的步幅以适当的顺序访问元素。
编写C代码时，这通常很好，但在纯Python代码中，这可能会导致性能显着降低。

通过启用缓冲模式，迭代器提供给内部循环的块可以变得更大，从而显着减少Python解释器的开销。
在强制Fortran迭代顺序的示例中，当启用缓冲时，内部循环可以一次性查看所有元素。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a, flags=['external_loop'], order='F'):
...     print(x, end=' ')
...
[0 3] [1 4] [2 5]
```

``` python
>>> for x in np.nditer(a, flags=['external_loop','buffered'], order='F'):
...     print(x, end=' ')
...
[0 3 1 4 2 5]
```

### 作为特定数据类型迭代

有时需要将数组视为与存储数据类型不同的数据类型。例如，即使被操作的数组是32位浮点数，也可能希望对64位浮点数进行所有计算。除了编写低级C代码外，通常最好让迭代器处理复制或缓冲，而不是自己在内部循环中转换数据类型。

有两种机制允许这样做，临时副本和缓冲模式。对于临时副本，使用新数据类型创建整个数组的副本，然后在副本中完成迭代。通过在所有迭代完成后更新原始数组的模式允许写访问。临时副本的主要缺点是临时副本可能消耗大量内存，特别是如果迭代数据类型具有比原始类型更大的项目大小。

缓冲模式可以缓解内存使用问题，并且比制作临时副本更加缓存友好。除特殊情况外，在迭代器外部需要立即使用整个数组，建议使用缓冲而不是临时复制。在NumPy中，ufuncs和其他函数使用缓冲来支持灵活的输入，并且内存开销最小。

在我们的示例中，我们将使用复杂数据类型处理输入数组，以便我们可以取负数的平方根。如果数据类型不精确匹配，则在不启用复制或缓冲模式的情况下，迭代器将引发异常。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3) - 3
>>> for x in np.nditer(a, op_dtypes=['complex128']):
...     print(np.sqrt(x), end=' ')
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Iterator operand required copying or buffering, but neither copying nor buffering was enabled
```

在复制模式下，'copy'被指定为每操作数标志。这样做是为了以操作时的方式提供控制。缓冲模式被指定为迭代器标志。

**示例：**

``` python
>>> a = np.arange(6).reshape(2,3) - 3
>>> for x in np.nditer(a, op_flags=['readonly','copy'],
...                 op_dtypes=['complex128']):
...     print(np.sqrt(x), end=' ')
...
1.73205080757j 1.41421356237j 1j 0j (1+0j) (1.41421356237+0j)
```

``` python
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['complex128']):
...     print(np.sqrt(x), end=' ')
...
1.73205080757j 1.41421356237j 1j 0j (1+0j) (1.41421356237+0j)
```

迭代器使用NumPy的转换规则来确定是否允许特定转换。默认情况下，它会强制执行“安全”投射。这意味着，例如，如果您尝试将64位浮点数组视为32位浮点数组，则会引发异常。在许多情况下，规则'same_kind'是最合理的规则，因为它允许从64位转换为32位浮点数，但不允许从float转换为int或从complex转换为float。

**示例：**

``` python
>>> a = np.arange(6.)
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['float32']):
...     print(x, end=' ')
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Iterator operand 0 dtype could not be cast from dtype('float64') to dtype('float32') according to the rule 'safe'
```

``` python
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['float32'],
...                 casting='same_kind'):
...     print(x, end=' ')
...
0.0 1.0 2.0 3.0 4.0 5.0
```

``` python
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['int32'], casting='same_kind'):
...     print(x, end=' ')
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Iterator operand 0 dtype could not be cast from dtype('float64') to dtype('int32') according to the rule 'same_kind'
```

需要注意的一点是，在使用读写或只写操作数时，将转换回原始数据类型。一个常见的情况是根据64位浮点数实现内部循环，并使用“same_kind”转换来允许处理其他浮点类型。在只读模式下，可以提供整数数组，读写模式将引发异常，因为转换回数组会违反转换规则。

**示例：**

``` python
>>> a = np.arange(6)
>>> for x in np.nditer(a, flags=['buffered'], op_flags=['readwrite'],
...                 op_dtypes=['float64'], casting='same_kind'):
...     x[...] = x / 2.0
...
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
TypeError: Iterator requested dtype could not be cast from dtype('float64') to dtype('int64'), the operand 0 dtype, according to the rule 'same_kind'
```

## 广播数组迭代

NumPy有一套规则来处理具有不同形状的数组，只要函数采用多个组合元素的操作数，就会应用这些规则。
这称为[广播](/reference/ufuncs.html#广播)。
[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)当您需要编写此类函数时，该对象可以为您应用这些规则。

作为示例，我们一起打印出一维和二维数组的结果。

**示例：**

``` python
>>> a = np.arange(3)
>>> b = np.arange(6).reshape(2,3)
>>> for x, y in np.nditer([a,b]):
...     print("%d:%d" % (x,y), end=' ')
...
0:0 1:1 2:2 0:3 1:4 2:5
```

当发生广播错误时，迭代器引发一个异常，其中包括输入形状以帮助诊断问题。

**示例：**

``` python
>>> a = np.arange(2)
>>> b = np.arange(6).reshape(2,3)
>>> for x, y in np.nditer([a,b]):
...     print("%d:%d" % (x,y), end=' ')
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (2) (2,3)
```

### 迭代器分配的输出数组

NumPy函数的一个常见情况是根据输入的广播分配输出，另外还有一个名为'out'的可选参数，其中结果将在提供时放置。该[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)对象提供了一种方便的习惯用法，使得支持这种机制非常容易。

我们将通过创建一个[``square``](https://numpy.org/devdocs/reference/generated/numpy.square.html#numpy.square)对其输入进行平方的函数来展示它是如何工作的。让我们从最小的函数定义开始，不包括'out'参数支持。

**示例：**

``` python
>>> def square(a):
...     with np.nditer([a, None]) as it:
...         for x, y in it:
...             y[...] = x*x
...         return it.operands[1]
...
>>> square([1,2,3])
array([1, 4, 9])
```

默认情况下，对于[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)作为None传入的操作数，使用标志'allocate'和'writeonly'。这意味着我们只能为迭代器提供两个操作数，并处理其余的操作数。

当添加'out'参数时，我们必须显式提供这些标志，因为如果有人将数组作为'out'传入，迭代器将默认为'readonly'，并且我们的内部循环将失败。'readonly'是输入数组的默认值，是为了防止无意中触发减少操作的混淆。如果默认为'readwrite'，则任何广播操作也会触发减少，这个主题将在本文档的后面部分介绍。

虽然我们正在使用它，但我们还会引入'no_broadcast'标志，这将阻止输出被广播。这很重要，因为我们只需要为每个输出输入一个输入值。聚合多个输入值是还原操作，需要特殊处理。它已经引发错误，因为必须在迭代器标志中显式启用减少，但是对于最终用户来说，禁用广播导致的错误消息更容易理解。要了解如何将平方函数推广到缩减，请查看关于Cython的部分中的平方和函数。

为了完整起见，我们还将添加'external_loop'和'buffered'标志，因为出于性能原因，这些标志通常是您需要的。

**示例：**

``` python
>>> def square(a, out=None):
...     it = np.nditer([a, out],
...             flags = ['external_loop', 'buffered'],
...             op_flags = [['readonly'],
...                         ['writeonly', 'allocate', 'no_broadcast']])
...     with it:
...         for x, y in it:
...             y[...] = x*x
...         return it.operands[1]
...
```

``` python
>>> square([1,2,3])
array([1, 4, 9])
```

``` python
>>> b = np.zeros((3,))
>>> square([1,2,3], out=b)
array([ 1.,  4.,  9.])
>>> b
array([ 1.,  4.,  9.])
```

``` python
>>> square(np.arange(6).reshape(2,3), out=b)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in square
ValueError: non-broadcastable output operand with shape (3) doesn't match the broadcast shape (2,3)
```

### 外部产品迭代

任何二进制操作都可以像外部产品一样扩展到数组操作[``outer``](https://numpy.org/devdocs/reference/generated/numpy.outer.html#numpy.outer)，
并且该[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)对象提供了一种通过显式映射操作数的轴来实现此目的的方法。
也可以使用[``newaxis``](/reference/constants.html#numpy.newaxis)
索引来完成此操作，但我们将向您展示如何直接使用nditer *op_axes* 参数来完成此操作而不使用中间视图。

我们将做一个简单的外部产品，将第一个操作数的尺寸放在第二个操作数的尺寸之前。
所述 *op_axes* 参数需要轴中的一个列表中的每个操作数，并从迭代器的轴来的操作数的轴提供了一个映射。

假设第一个操作数是一维的，第二个操作数是二维的。迭代器将具有三个维度，
因此 *op_axes*  将具有两个3元素列表。第一个列表选取第一个操作数的一个轴，
其余迭代器轴为-1，最终结果为[0，-1，-1]。
第二个列表选取第二个操作数的两个轴，但不应与第一个操作数中选取的轴重叠。
它的列表是[-1,0,1]。输出操作数以标准方式映射到迭代器轴，因此我们可以提供None而不是构造另一个列表。

内循环中的操作是简单的乘法。与外部产品有关的一切都由迭代器设置处理。

**示例：**

``` python
>>> a = np.arange(3)
>>> b = np.arange(8).reshape(2,4)
>>> it = np.nditer([a, b, None], flags=['external_loop'],
...             op_axes=[[0, -1, -1], [-1, 0, 1], None])
>>> with it:
...     for x, y, z in it:
...         z[...] = x*y
...     result = it.operands[2]  # same as z
...
>>> result
array([[[ 0,  0,  0,  0],
        [ 0,  0,  0,  0]],
       [[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
       [[ 0,  2,  4,  6],
        [ 8, 10, 12, 14]]])
```

请注意，一旦迭代器关闭，我们就无法访问[``operands``](https://numpy.org/devdocs/reference/generated/numpy.nditer.operands.html#numpy.nditer.operands)
，必须使用在上下文管理器中创建的引用。

### 减少迭代

每当可写操作数的元素少于完整迭代空间时，该操作数就会减少。该[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)对象要求将任何减少操作数标记为读写，并且仅当'reduce_ok'作为迭代器标志提供时才允许减少。

举一个简单的例子，考虑获取数组中所有元素的总和。

**示例：**

``` python
>>> a = np.arange(24).reshape(2,3,4)
>>> b = np.array(0)
>>> with np.nditer([a, b], flags=['reduce_ok', 'external_loop'],
...                     op_flags=[['readonly'], ['readwrite']]) as it:
...     for x,y in it:
...         y[...] += x
...
>>> b
array(276)
>>> np.sum(a)
276
```

在组合缩减和分配的操作数时，事情会有点棘手。在迭代开始之前，必须将任何减少操作数初始化为其起始值。下面是我们如何能做到这一点，采取沿款项的最后一个轴 *一个* 。

**示例：**

``` python
>>> a = np.arange(24).reshape(2,3,4)
>>> it = np.nditer([a, None], flags=['reduce_ok', 'external_loop'],
...             op_flags=[['readonly'], ['readwrite', 'allocate']],
...             op_axes=[None, [0,1,-1]])
>>> with it:
...     it.operands[1][...] = 0
...     for x, y in it:
...         y[...] += x
...     result = it.operands[1]
...
>>> result
array([[ 6, 22, 38],
       [54, 70, 86]])
>>> np.sum(a, axis=2)
array([[ 6, 22, 38],
       [54, 70, 86]])
```

要进行缓冲还原，需要在设置过程中进行另一次调整。通常，迭代器构造涉及将第一个数据缓冲区从可读数组复制到缓冲区中。任何减少操作数都是可读的，因此可以将其读入缓冲区。不幸的是，在完成此缓冲操作之后操作数的初始化将不会反映在迭代开始的缓冲区中，并且将产生垃圾结果。

迭代器标志“delay_bufalloc”允许迭代器分配的缩减操作数与缓冲一起存在。设置此标志后，迭代器将使其缓冲区保持未初始化状态，直到它收到重置为止，之后它将为常规迭代做好准备。如果我们也启用缓冲，上面的示例如何查看。

**示例：**

``` python
>>> a = np.arange(24).reshape(2,3,4)
>>> it = np.nditer([a, None], flags=['reduce_ok', 'external_loop',
...                                  'buffered', 'delay_bufalloc'],
...             op_flags=[['readonly'], ['readwrite', 'allocate']],
...             op_axes=[None, [0,1,-1]])
>>> with it:
...     it.operands[1][...] = 0
...     it.reset()
...     for x, y in it:
...         y[...] += x
...     result = it.operands[1]
...
>>> result
array([[ 6, 22, 38],
       [54, 70, 86]])
```

## 将内循环放在Cython中

那些希望从低级别操作中获得真正良好性能的人应该强烈考虑直接使用C中提供的迭代API，但对于那些不熟悉C或C ++的人来说，Cython是一个良好的中间地带，具有合理的性能权衡。对于[``nditer``](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer)对象，这意味着让迭代器处理广播，dtype转换和缓冲，同时给Cython提供内部循环。

对于我们的例子，我们将创建一个平方和函数。首先，让我们在简单的Python中实现这个功能。我们想要支持类似于numpy [``sum``](https://numpy.org/devdocs/reference/generated/numpy.sum.html#numpy.sum)函数的'axis'参数，因此我们需要为 *op_axes* 参数构造一个列表。这是这个看起来如何。

**示例：**

``` python
>>> def axis_to_axeslist(axis, ndim):
...     if axis is None:
...         return [-1] * ndim
...     else:
...         if type(axis) is not tuple:
...             axis = (axis,)
...         axeslist = [1] * ndim
...         for i in axis:
...             axeslist[i] = -1
...         ax = 0
...         for i in range(ndim):
...             if axeslist[i] != -1:
...                 axeslist[i] = ax
...                 ax += 1
...         return axeslist
...
>>> def sum_squares_py(arr, axis=None, out=None):
...     axeslist = axis_to_axeslist(axis, arr.ndim)
...     it = np.nditer([arr, out], flags=['reduce_ok', 'external_loop',
...                                       'buffered', 'delay_bufalloc'],
...                 op_flags=[['readonly'], ['readwrite', 'allocate']],
...                 op_axes=[None, axeslist],
...                 op_dtypes=['float64', 'float64'])
...     with it:
...         it.operands[1][...] = 0
...         it.reset()
...         for x, y in it:
...             y[...] += x*x
...         return it.operands[1]
...
>>> a = np.arange(6).reshape(2,3)
>>> sum_squares_py(a)
array(55.0)
>>> sum_squares_py(a, axis=-1)
array([  5.,  50.])
```

为了Cython-ize这个函数，我们用Cython代码替换内部循环（y [...] + = x * x），这些代码专门用于float64 dtype。启用“external_loop”标志后，提供给内部循环的数组将始终为一维，因此需要进行的检查非常少。

这是sum_squares.pyx的列表：

``` python
import numpy as np
cimport numpy as np
cimport cython

def axis_to_axeslist(axis, ndim):
    if axis is None:
        return [-1] * ndim
    else:
        if type(axis) is not tuple:
            axis = (axis,)
        axeslist = [1] * ndim
        for i in axis:
            axeslist[i] = -1
        ax = 0
        for i in range(ndim):
            if axeslist[i] != -1:
                axeslist[i] = ax
                ax += 1
        return axeslist

@cython.boundscheck(False)
def sum_squares_cy(arr, axis=None, out=None):
    cdef np.ndarray[double] x
    cdef np.ndarray[double] y
    cdef int size
    cdef double value

    axeslist = axis_to_axeslist(axis, arr.ndim)
    it = np.nditer([arr, out], flags=['reduce_ok', 'external_loop',
                                      'buffered', 'delay_bufalloc'],
                op_flags=[['readonly'], ['readwrite', 'allocate']],
                op_axes=[None, axeslist],
                op_dtypes=['float64', 'float64'])
    with it:
        it.operands[1][...] = 0
        it.reset()
        for xarr, yarr in it:
            x = xarr
            y = yarr
            size = x.shape[0]
            for i in range(size):
               value = x[i]
               y[i] = y[i] + value * value
        return it.operands[1]
```

在这台机器上，将.pyx文件构建到模块中如下所示，但您可能需要找到一些Cython教程来告诉您系统配置的具体信息：

``` python
$ cython sum_squares.pyx
$ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -I/usr/include/python2.7 -fno-strict-aliasing -o sum_squares.so sum_squares.c
```

从Python解释器运行它会产生与我们的本机Python / NumPy代码相同的答案。

**示例：**

``` python
>>> from sum_squares import sum_squares_cy
>>> a = np.arange(6).reshape(2,3)
>>> sum_squares_cy(a)
array(55.0)
>>> sum_squares_cy(a, axis=-1)
array([  5.,  50.])
```

在IPython中做一点时间表明，Cython内部循环的减少的开销和内存分配提供了比简单的Python代码和使用NumPy的内置和函数的表达式更好的加速：

``` python
>>> a = np.random.rand(1000,1000)

>>> timeit sum_squares_py(a, axis=-1)
10 loops, best of 3: 37.1 ms per loop

>>> timeit np.sum(a*a, axis=-1)
10 loops, best of 3: 20.9 ms per loop

>>> timeit sum_squares_cy(a, axis=-1)
100 loops, best of 3: 11.8 ms per loop

>>> np.all(sum_squares_cy(a, axis=-1) == np.sum(a*a, axis=-1))
True

>>> np.all(sum_squares_py(a, axis=-1) == np.sum(a*a, axis=-1))
True
```
