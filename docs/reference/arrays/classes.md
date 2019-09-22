# 标准数组子类

::: tip 注意

可以对 ``numpy.ndarray`` 进行子类化，
但如果您的目标是创建具有 *修改* 的行为的数组，
就像用于分布式计算的Dask数组和用于基于GPU的计算的cupy数组一样，则不鼓励子类化。
相反，建议使用 numpy 的[调度机制](/user/basics/dispatch.html#basics-dispatch)。

:::

如果需要，可以从（ Python 或 C ）继承 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)。
因此，它可以形成许多有用的类的基础。
通常是对数组对象进行子类，还是简单地将核心数组件用作新类的内部部分，这是一个困难的决定，可能只是一个选择的问题。
NumPy有几个工具可以简化新对象与其他数组对象的交互方式，因此最终选择可能并不重要。
简化问题的一种方法是问问自己，您感兴趣的对象是否可以替换为单个数组，或者它的核心是否真的需要两个或更多个数组。

注意，[``asarray``](https://numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray) 总是返回基类ndarray。
如果您确信使用数组对象可以处理ndarray的任何子类，
那么可以使用 [``asanyarray``](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray) 来允许子类通过您的子例程更干净地传播。
原则上，子类可以重新定义数组的任何方面，因此，在严格的指导原则下，[``asanyarray``](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray) 很少有用。
然而，数组对象的大多数子类不会重新定义数组对象的某些方面，例如Buffer接口或数组的属性。
但是，子例程可能无法处理数组的任意子类的一个重要示例是，矩阵将 “*” 运算符重新定义为矩阵乘法，而不是逐个元素的乘法。

## 特殊属性和方法

::: tip 另见

[子类化ndarray](/user/basics/subclassing.html)

:::

NumPy提供了几个类可以自定义的钩子：

- ``class.__array_ufunc__``(*ufunc*, *method*, **inputs*, ***kwargs*)

  *版本1.13中的新功能。* 

  任何类，ndarray子类是否可以定义此方法或将其设置为 [``None``](https://docs.python.org/dev/library/constants.html#None)  以覆盖NumPy的ufuncs的行为。
  这与Python的__mul__和其他二进制操作例程非常相似。

  - *ufunc* 是被调用的ufunc对象。
  - *method* 是一个字符串，指示调用了哪个Ufunc方法(``"__call__"``，``"reduce"``，``"acculate"``，``"outer"``，``"internal"`` 之一)。
  - *inputs* 是 ``ufunc`` 的输入参数的元组。
  - *kwargs* 是包含ufunc的可选输入参数的字典。
  如果给定，任何 ``out`` 参数（包括位置参数和关键字）都将作为kwargs中的 [``元组``](https://docs.python.org/dev/library/stdtypes.html#tuple) 传递。有关详细信息，请参阅 [通函数（ufunc）](/reference/ufuncs.html) 中的讨论。

  该方法应返回操作的结果，如果未实现请求的操作，则返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented)。

  如果输入或输出参数之一具有 [``__array_ufunc__``](#numpy.class.__array_ufunc__) 方法，则执行该方法而不是ufunc。
  如果多个参数实现 [``__array_ufunc__``](#numpy.class.__array_ufunc__) ，则按顺序尝试它们：子类在超类之前，输入在输出之前，否则从左到右。
  返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) 以外的内容的第一个例程确定结果。
  如果所有 [``__array_ufunc__``](#numpy.class.__array_ufunc__) 操作都返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented)，则会引发 [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError)。

  ::: tip 注意

  我们打算将numpy函数重新实现为(（通用的）ufunc，在这种情况下，
  它们将可能被 ``__array_ufunc__`` 方法覆盖。
  一个主要的候选是 [``matmul``](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul)，它目前不是Ufunc，
  但可以相对容易地重写为（一组）通用Ufuncs。
  对于[``median``](https://numpy.org/devdocs/reference/generated/numpy.median.html#numpy.median)、
  [``amin``](https://numpy.org/devdocs/reference/generated/numpy.amin.html#numpy.amin)和 [``argsort``](https://numpy.org/devdocs/reference/generated/numpy.argsort.html#numpy.argsort) 等函数，可能会发生相同的情况。
  :::

  与python中的其他一些特殊方法一样，例如 ``__hash__`` 和 ``__iter__``，
  可以通过设置 ``__array_ufunc__ = None`` 来指示您的类不支持ufuncs。
  当对设置 ``__array_ufunc__ = None`` 的对象调用时，ufuncs 总是引发 [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError)。

  当 arr 是 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 且 ``obj`` 是自定义类的实例时，
  [``__array_ufunc__``](#numpy.class.__array_ufunc__)的存在也会影响 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 处理 ``arr + obj`` 和 ``arr < obj`` 等二进制操作的方式。
  有两种可能性。如果 ``obj.__array_ufunc__`` 存在而不是 [None](https://docs.python.org/dev/library/constants.html#None)，
  则 ``ndarray.__add__`` 和 friends 将委托给 ufunc 机器，
  这意味着 ``arr + obj`` 变为 ``np.add(arr，obj)``，然后 [add](https://numpy.org/devdocs/reference/generated/numpy.add.html#numpy.add) 调用 ``obj.__array_ufunc__``。
  如果您想定义一个像数组一样作用的对象，这是很有用的。

  或者，如果 ``obj.__array_ufunc__`` 设置为 [``None``](https://docs.python.org/dev/library/constants.html#None)，
  那么作为一种特殊情况，像 ``ndarray.__add__`` 这样的特殊方法会注意到这一点，并 *无条件* 地引发 [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError)。
  如果要创建通过二进制操作与数组交互的对象，但本身不是数组，则这很有用。
  例如，单位处理系统可能具有表示 “meters” 单位的对象 ``m``，并且希望支持语法 ``arr * m`` 来表示数组具有 “meters” 单位，
  但不想通过 ufuncs 或其他方式与数组交互。
  这可以通过设置 ``__array_ufunc_ = none`` 并定义 ``__mul_`` 和 ``__rmul__`` 方法来完成。
  （请注意，这意味着编写始终返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) 的``__array_ufunc__`` 与设置 ``__array_ufunc__ = none`` 并不完全相同：
  在前一种情况下，``arr + obj`` 将引发 [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError)，而在后一种情况下，可以定义 ``__radd__`` 方法来防止这种情况。）

  以上不适用于 in-place 操作符，[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 从不返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented)。因此，``arr += obj`` 总是会导致 [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError)。这是因为对于数组来说，in-place 操作一般不能被简单的反向操作所取代。(例如，默认情况下，``arr += obj`` 将被转换为 ``arr= arr + obj``，即 ``arr`` 将被替换，这与就地数组操作的预期相反。)

  ::: tip 注意

  如果定义 ``__array_ufunc__``：

  - 如果您不是 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 的子类，我们建议您的类定义特殊的方法，如 ``__add__`` 和 ``__lt__``，它们像 ndarray 一样委托给 ufuncs。一种简单的方法是从 [``NDArrayOperatorsMixin``](https://numpy.org/devdocs/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin) 子类。
  - 如果您是 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 的子类，
  我们建议您将所有覆盖逻辑放在 [``__array_ufunc__``](#numpy.class.__array_ufunc__) 中，并且不要覆盖特殊方法。
  这确保了类层次结构只在一个地方确定，而不是由ufunc机制和二元运算规则单独确定（优先考虑子类的特殊方法；强制实施只有一个位置的层次结构的替代方法，即将[``__array_ufunc__``](#numpy.class.__array_ufunc__) 设置为 [``None``](https://docs.python.org/dev/library/constants.html#None)，似乎非常出乎意料，因此令人困惑，因为这样子类将根本不能与ufuncs一起工作）。
  - [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 定义自己的 [``__array_ufunc__``](#numpy.class.__array_ufunc__)，如果没有参数有覆盖，则计算ufunc，否则返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented)。这对于 [``__array_ufunc__``](#numpy.class.__array_ufunc__) 将其自身类的任何实例转换为 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 的子类可能很有用：然后它可以使用 ``super().__array_ufunc__(*inputs, **kwargs)`` 将这些实例传递给其超类，并最终在可能的反向转换后返回结果。这种做法的优点是，它确保了有可能具有扩展行为的子类的层次结构。
  有关详细信息，请参见 [子类化ndarray](/user/basics/subclassing.html)。

  :::

  ::: tip 注意

  如果类定义了 [``__array_ufunc__``](#numpy.class.__array_ufunc__) 方法，这将禁用下面为 ufuncs 描述的 [``__array_wrap__``](#numpy.class.__array_wrap__)、[``__array_prepare__``](#numpy.class.__array_prepare__)、[``__array_priority__``](#numpy.class.__array_priority__) 机制（最终可能会被弃用）。

  :::

- ``class.__array_function__``(*func*, *types*, *args*, *kwargs*)

  *版本1.16中的新增功能。* 

  ::: tip 注意

  - 在NumPy 1.17中，默认情况下启用该协议，但可以禁用该协议，设置 ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION = 0``。
  - 在NumPy 1.16中，您需要设置环境变量
  - 在NumPy 1.16中，您需要在导入NumPy之前设置环境变量 ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION = 1`` 以使用NumPy函数覆盖。
  - 最终，期望始终启用``__array_function__``。

  :::

  - ``func`` 是由NumPy的公共API公开的任意可调用的，它以 ``func(*args, **kwargs)`` 的形式调用。
  - ``types`` 是来自实现 __array_function__ 的原始NumPy函数调用的唯一参数类型的[集合](collections.abc.Collection)。
  - 元组 ``args`` 和dict ``kwargs`` 直接从原始调用传递。

  为了方便 ``__array_function__`` 实现者，``types`` 为所有参数类型提供了 ``'__array_function__'`` 属性。
  这允许实现者快速识别他们应该遵循其他参数的 ``__array_function__`` 实现的情况。
  实现不应该依赖于``类型``的迭代顺序。
  
  ``__array_function__`` 的大多数实现都将从两个检查开始：

  1. 给定的函数是否知道如何重载？
  1. 我们知道如何处理的类型的所有参数都是？

  如果这些条件成立，``__array_function__`` 应该返回调用 ``func(*args, **kwargs)`` 实现的结果。
  否则，它应该返回 sentinel 值 ``NotImplemented``，表示该函数未由这些类型实现。

  对 ``__array_function__`` 的返回值没有一般要求，尽管大多数合理的实现应该返回与函数参数之一具有相同类型的数组。

  还可以方便地定义用于注册 ``__array_function__`` 实现的自定义装饰器（比如下面的``实现（implements）``)。

  ``` python
  HANDLED_FUNCTIONS = {}

  class MyArray:
      def __array_function__(self, func, types, args, kwargs):
          if func not in HANDLED_FUNCTIONS:
              return NotImplemented
          # Note: this allows subclasses that don't override
          # __array_function__ to handle MyArray objects
          if not all(issubclass(t, MyArray) for t in types):
              return NotImplemented
          return HANDLED_FUNCTIONS[func](*args, **kwargs)

  def implements(numpy_function):
      """Register an __array_function__ implementation for MyArray objects."""
      def decorator(func):
          HANDLED_FUNCTIONS[numpy_function] = func
          return func
      return decorator

  @implements(np.concatenate)
  def concatenate(arrays, axis=0, out=None):
      ...  # implementation of concatenate for MyArray objects

  @implements(np.broadcast_to)
  def broadcast_to(array, shape):
      ...  # implementation of broadcast_to for MyArray objects
  ```

  请注意，``__array_function__`` 实现不需要包含 *所有* 相应的NumPy函数的可选参数
  （例如，上面的 ``broadcast_to`` 省略了无关的 ``subok`` 参数）。
  如果在NumPy函数调用中明确使用了可选参数，则它们仅传递给 ``__array_function__``。

  就像内置特殊方法（如 ``__add__`` ）的情况一样，
  正确编写的 ``__array_function__`` 方法应始终在遇到未知类型时返回 ``NotImplemented``。
  否则，如果操作还包含一个对象，则无法从另一个对象正确覆盖NumPy函数。

  在大多数情况下，``__array_function__`` 的调度规则与 ``__array_ufunc__`` 的调度规则相匹配。尤其是：

  - NumPy将从所有指定的输入中收集 ``__array_function__`` 的实现，并按顺序调用它们：超类之前的子类，否则从左到右。
  请注意，在涉及子类的某些边缘情况下，这与Python的[当前行为](https://bugs.python.org/issue30140)略有不同。
  - ``__array_function__`` 的实现表明它们可以通过返回除 ``NotImplemented`` 之外的任何值来处理操作。
  - 如果所有 ``__array_function__`` 方法都返回 ``NotImplemented``，NumPy 将引发 ``TypeError``。

  如果不存在 ``__array_function__`` 方法，NumPy将默认调用自己的实现，用于NumPy数组。
  例如，当所有类似数组的参数都是Python数字或列表时，会出现这种情况。
  （NumPy数组确实有一个 ``__array_function__`` 方法，
  如下所示，但如果除NumPy数组子类之外的任何参数都实现了 ``__array_function__``，它总是返回 ``NotImplemented``。）

  与 ``__array_ufunc__`` 的当前行为的一个偏差是，NumPy将仅对每个唯一类型的第一个参数调用 ``__array_function__``。
  这与Python[调用反射方法的规则相匹配](https://docs.python.org/3/reference/datamodel.html#object.__ror__)，这确保了检查重载具有可接受的性能，即使存在大量重载参数。

- ``class.__array_finalize__``(*obj*)

  只要系统在内部从 *obj* 分配一个新数组，就会调用此方法，其中 *obj* 是 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 的子类（子类型）。 
  它可以用于在构造之后更改 *self* 的属性（例如，以确保二维矩阵），
  或者从“父”更新元信息。子类继承此方法的默认实现，该方法不执行任何操作。

- ``class.__array_prepare__``(*array*, *context=None*)

  在每个ufunc的开头，在具有最高数组优先级的输入对象上调用此方法，如果指定了一个输出对象，则调用此方法。
  传入输出数组，返回的任何内容都传递给 [ufunc](/reference/ufuncs.html#输出类型确定)。
  子类继承此方法的默认实现，它只是简单地返回未修改的输出数组。
  子类可以选择使用此方法将输出数组转换为子类的实例，并在将数组返回到ufunc进行计算之前更新元数据。

  ::: tip 注意

  对于 ufuncs，希望最终弃用这个方法，而不是 [``__array_ufunc__``](#numpy.class.__array_ufunc__)。

  :::

- ``class.__array_wrap__``(*array*, *context=None*)

  在每个 [ufunc](/reference/ufuncs.html#输出类型确定) 的末尾，
  对具有最高数组优先级的输入对象或输出对象（如果指定了一个）调用此方法。
  传入ufunc计算的数组，并将返回的任何内容传递给用户。
  子类继承此方法的默认实现，该方法将数组转换为对象类的新实例。
  子类可以选择使用此方法将输出数组转换为子类的实例，并在将数组返回给用户之前更新元数据。

  ::: tip 注意

  对于ufuncs，希望最终弃用此方法，取而代之的是 [``__array_ufunc__``](#numpy.class.__array_ufunc__)。

  :::

- ``class.__array_priority__``

  该属性的值用于确定在返回对象的Python类型有多种可能性的情况下返回哪种类型的对象。子类继承此属性的默认值0.0。

  ::: tip 注意

  对于ufuncs，希望最终弃用此方法，取而代之的是 [``__array_ufunc__``](#numpy.class.__array_ufunc__)。

  :::

- ``class.__array__``([*dtype*])

  如果将具有 [``__array__``](#numpy.class.__array__) 方法的类（是否ndarray子类）用作 [ufunc](/reference/ufuncs.html#输出类型确定) 的输出对象，
  则结果将写入 [``__array__``](#numpy.class.__array__) 返回的对象。在输入数组上进行类似的转换。

## 矩阵对象

::: tip 注意

强烈建议*不要*使用矩阵子类。如下所述，它使得编写一致处理矩阵和规则数组的函数变得非常困难。
目前，它们主要用于与 ``scipy.sparse`` 进行交互。
但是，我们希望为此用途提供一种替代方法，并最终删除 ``matrix`` 子类。

:::

[``matrix``](https://numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix) 对象继承自ndarray，因此它们具有ndarray的相同属性和方法。
但是，当您使用矩阵但希望它们像数组一样工作时，可能会导致意外结果的矩阵对象有六个重要差异：

1. 可以使用字符串表示法创建 Matrix 对象，以允许 Matlab 样式的语法，其中空格分隔列，分号 (';') 分隔行。
1. Matrix对象始终是二维的。这具有深远的含义，因为 m.ravel() 仍然是二维的(第一维为1)，并且 item selection 返回二维对象，因此序列行为与数组根本不同。
1. 矩阵对象覆盖乘法成为矩阵乘法。确保您了解您可能希望接收矩阵的函数的这一点。特别是考虑到当m是矩阵时 asanyarray(m) 返回矩阵的事实。
1. 矩阵对象超越幂被矩阵提升为幂。关于在使用 asanyarray(…) 的函数中使用电源的相同警告。获取此事实的数组对象保持。
1. 矩阵对象的默认 \_\_array_priority__ 为10.0，因此与ndarray的混合运算始终生成矩阵。
1. 矩阵具有使计算更容易的特殊属性。这些是

    方法 | 描述
    ---|---
    matrix.T | 返回矩阵的转置。
    matrix.H | 返回self的（复数）共轭转置。
    matrix.I | 返回可逆self的（乘法）逆。
    matrix.A | 将self作为ndarray对象返回。

::: danger 警告

Matrix对象覆盖乘法 ‘\*’ 和 ‘\*\*’（幂），分别为矩阵乘法和矩阵幂。
如果您的子例程可以接受子类，并且您没有转换为基类数组，
则必须使用ufuncs multiply和power来确保对所有输入执行正确的操作。

:::

Matrix类是ndarray的Python子类，可以用作如何构造自己的ndarray子类的参考。
矩阵可以从其他矩阵、字符串和任何可以转换为 ``ndarray`` 的东西中创建。名称 “mat” 是NumPy中 “matrix” 的别名。

方法 | 描述
---|---
[matrix](https://numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix)(data[, dtype, copy]) | **注意：** 不建议再使用这个类，即使是线性的
[asmatrix](https://numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | 将输入解析为矩阵。
[bmat](https://numpy.org/devdocs/reference/generated/numpy.bmat.html#numpy.bmat)(obj[, ldict, gdict]) | 从字符串、嵌套序列或数组构建矩阵对象。

示例1：从字符串创建矩阵

``` python
>>> a=mat('1 2 3; 4 5 3')
>>> print (a*a.T).I
[[ 0.2924 -0.1345]
 [-0.1345  0.0819]]
```

示例2：从嵌套序列创建矩阵

``` python
>>> mat([[1,5,10],[1.0,3,4j]])
matrix([[  1.+0.j,   5.+0.j,  10.+0.j],
        [  1.+0.j,   3.+0.j,   0.+4.j]])
```

示例 3: 从数组创建矩阵

``` python
>>> mat(random.rand(3,3)).T
matrix([[ 0.7699,  0.7922,  0.3294],
        [ 0.2792,  0.0101,  0.9219],
        [ 0.3398,  0.7571,  0.8197]])
```

## 内存映射文件数组

内存映射文件对于读取和/或修改具有规则布局的大文件的小段非常有用，
而无需将整个文件读入内存。
ndarray的一个简单子类使用内存映射文件作为数组的数据缓冲区。
对于小文件，将整个文件读入内存的开销通常不大，但是对于大文件，使用内存映射可以节省大量资源。

内存映射文件数组还有一个额外的方法（除了它们从ndarray继承的方法之外）：[``.flush()``](https://numpy.org/devdocs/reference/generated/numpy.memmap.flush.html#numpy.memmap.flush)，
用户必须手动调用该方法，以确保对数组的任何更改都实际写入磁盘。

方法 | 描述
---|---
[memmap](https://numpy.org/devdocs/reference/generated/numpy.memmap.html#numpy.memmap) | 创建存储在磁盘上二进制文件中的数组的内存映射。
[memmap.flush](https://numpy.org/devdocs/reference/generated/numpy.memmap.flush.html#numpy.memmap.flush)(self) | 将数组中的任何更改写入磁盘上的文件。

示例：

``` python
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

## 字符数组（``numpy.char``）

::: tip 另见

[创建字符数组（numpy.char）](/reference/routines/array-creation.html#routines-array-creation-char)

:::

::: tip 注意

[``chararray``](https://numpy.org/devdocs/reference/generated/numpy.chararray.html#numpy.chararray) 类的存在是为了与Numarray向后兼容，
不建议在新开发中使用它。从 numpy 1.4 开始，
如果需要字符串数组，建议使用[``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) ``object_``、 ``string_`` 或 ``unicode_`` 的数组，
并使用 [``numpy.char``](/reference/routines/char.html#module-numpy.char) 模块中的自由函数进行快速矢量化字符串操作。

:::

这些是 ``string_`` 类型或 ``unicode_`` 类型的增强型数组。
这些数组继承自 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) ，但在（逐个）元素的基础上特别定义了操作 ``+``, ``*``, 和 ``%`` 。
这些操作在字符类型的标准 [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 上不可用。
此外，[``chararray``](https://numpy.org/devdocs/reference/generated/numpy.chararray.html#numpy.chararray) 具有所有标准 [``string``](https://docs.python.org/dev/library/stdtypes.html#str)（和``unicode`` ）方法，在逐个元素的基础上执行它们。
也许创建chararray的最简单方法是使用 [``self.view(chararray)``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.view.html#numpy.ndarray.view)，
其中 *self* 是str或unicode数据类型的ndarray。
但是，也可以使用 [``numpy.chararray``](https://numpy.org/devdocs/reference/generated/numpy.chararray.html#numpy.chararray) 构造函数或通过 [``numpy.char.array``](https://numpy.org/devdocs/reference/generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array) 函数创建chararray：

方法 | 描述
---|---
[chararray](https://numpy.org/devdocs/reference/generated/numpy.chararray.html#numpy.chararray)(shape[, itemsize, unicode, …]) | 提供有关字符串和unicode值数组的便捷视图。
[core.defchararray.array](https://numpy.org/devdocs/reference/generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array)(obj[, itemsize, …]) | 创建一个chararray。

与 str 数据类型的标准 ndarray 的另一个不同之处是 chararray 继承了由 Numarray 引入的特性，
即在项检索和比较操作中，数组中任何元素末尾的空格都将被忽略。

## 记录数组（``numpy.rec``）

::: tip 另见

[Creating record arrays (numpy.rec)](/reference/routines/array-creation.html#routines-array-creation-rec)、
[Data type routines](/reference/routines/dtype.html#routines-dtype)、
[Data type objects (dtype)](arrays.dtypes.html#arrays-dtypes)。

:::

NumPy提供了 [``recarray``](https://numpy.org/devdocs/reference/generated/numpy.recarray.html#numpy.recarray) 类，允许将结构化数组的字段作为属性进行访问，
以及相应的标量数据类型对象 [``记录``](https://numpy.org/devdocs/reference/generated/numpy.record.html#numpy.record)。

方法 | 描述
---|---
[recarray](https://numpy.org/devdocs/reference/generated/numpy.recarray.html#numpy.recarray) | 构造一个允许使用属性进行字段访问的ndarray。
[record](https://numpy.org/devdocs/reference/generated/numpy.record.html#numpy.record) | 一种数据类型标量，允许字段访问作为属性查找。

## 掩码数组（``numpy.ma``）

::: tip 另见

[Masked arrays](maskedarray.html)

:::

## 标准容器类

为了向后兼容并作为标准的“容器”类，
Numeric的UserArray已被引入NumPy并命名为 [``numpy.lib.user_array.container``](https://numpy.org/devdocs/reference/generated/numpy.lib.user_array.container.html#numpy.lib.user_array.container) 容器类是一个Python类，
其self.array属性是一个ndarray。
使用numpy.lib.user_array.container比使用ndarray本身更容易进行多重继承，因此默认包含它。
除了提及它的存在之外，这里没有记录，因为如果可以的话，我们鼓励你直接使用ndarray类。

方法 | 描述
---|---
[numpy.lib.user_array.container](https://numpy.org/devdocs/reference/generated/numpy.lib.user_array.container.html#numpy.lib.user_array.container)(data[, …]) | 标准容器类，便于多重继承。

## 数组迭代器

迭代器是数组处理的强大概念。本质上，迭代器实现了一个通用的for循环。
如果 *myiter* 是一个迭代器对象，那么Python代码：

``` python
for val in myiter:
    ...
    some code involving val
    ...
```

重复调用 ``val = next(myiter)``，直到迭代器引发 [``StopIteration``](https://docs.python.org/dev/library/exceptions.html#StopIteration)。
有几种方法可以迭代可能有用的数组：默认迭代，平面迭代和-dimensional枚举。

### 默认迭代

ndarray对象的默认迭代器是序列类型的默认Python迭代器。
因此，当数组对象本身用作迭代器时。默认行为相当于：

``` python
for i in range(arr.shape[0]):
    val = arr[i]
```

此默认迭代器从数组中选择维度的子数组。
这可以是用于定义递归算法的有用构造。
要遍历整个数组，需要for循环。

``` python
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

### Flat 迭代

方法 | 描述
---|---
[ndarray.flat](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | 数组上的一维迭代器。

如前所述，ndarray 对象的 flat 属性返回一个迭代器，它将以C风格的连续顺序循环遍历整个数组。

``` python
>>> for i, val in enumerate(a.flat):
...     if i%5 == 0: print i, val
0 10
5 15
10 20
15 25
20 30
```

在这里，我使用了内置的枚举迭代器来返回迭代器索引和值。

### N维枚举

方法 | 描述
---|---
[ndenumerate](https://numpy.org/devdocs/reference/generated/numpy.ndenumerate.html#numpy.ndenumerate)(arr) | 多维索引迭代器。

有时在迭代时获取N维索引可能是有用的。ndenumerate迭代器可以实现这一点。

``` python
>>> for i, val in ndenumerate(a):
...     if sum(i)%5 == 0: print i, val
(0, 0, 0) 10
(1, 1, 3) 25
(2, 0, 3) 29
(2, 1, 2) 32
```

### 广播迭代器

方法 | 描述
---|---
[broadcast](https://numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast) | 创建一个模仿广播的对象。

广播的一般概念也可以使用 [``broadcast``](https://numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast) 迭代器从Python获得。
此对象将对象作为输入，并返回一个迭代器，该迭代器返回元组，提供广播结果中的每个输入序列元素。

``` python
>>> for val in broadcast([[1,0],[2,3]],[0,1]):
...     print val
(1, 0)
(0, 1)
(2, 0)
(3, 1)
```
