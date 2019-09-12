# Standard array subclasses

::: tip 注意

可以对 ``numpy.ndarray`` 进行子类化，
但如果您的目标是创建具有 *修改* 的行为的数组，
就像用于分布式计算的Dask数组和用于基于GPU的计算的cupy数组一样，则不鼓励子类化。
相反，建议使用 numpy 的[调度机制](https://numpy.org/devdocs/user/basics.dispatch.html#basics-dispatch)。

:::

如果需要，可以从（ Python 或 C ）继承 [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray)。
因此，它可以形成许多有用的类的基础。
通常是对数组对象进行子类，还是简单地将核心数组件用作新类的内部部分，这是一个困难的决定，可能只是一个选择的问题。
NumPy有几个工具可以简化新对象与其他数组对象的交互方式，因此最终选择可能并不重要。
简化问题的一种方法是问问自己，您感兴趣的对象是否可以替换为单个数组，或者它的核心是否真的需要两个或更多个数组。

注意，[``asarray``](generated/numpy.asarray.html#numpy.asarray) 总是返回基类ndarray。
如果您确信使用数组对象可以处理ndarray的任何子类，
那么可以使用 [``asanyarray``](generated/numpy.asanyarray.html#numpy.asanyarray) 来允许子类通过您的子例程更干净地传播。
原则上，子类可以重新定义数组的任何方面，因此，在严格的指导原则下，[``asanyarray``](generated/numpy.asanyarray.html#numpy.asanyarray) 很少有用。
然而，数组对象的大多数子类不会重新定义数组对象的某些方面，例如Buffer接口或数组的属性。
但是，子例程可能无法处理数组的任意子类的一个重要示例是，矩阵将 “*” 运算符重新定义为矩阵乘法，而不是逐个元素的乘法。

## 特殊属性和方法

::: tip 另见

[Subclassing ndarray](https://numpy.org/devdocs/user/basics.subclassing.html#basics-subclassing)

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
  如果给定，任何 ``out`` 参数（包括位置参数和关键字）都将作为kwargs中的 [``元组``](https://docs.python.org/dev/library/stdtypes.html#tuple) 传递。有关详细信息，请参阅 [泛函数(ufunc)](ufuncs.html#ufuncs) 中的讨论。

  该方法应返回操作的结果，如果未实现请求的操作，则返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented)。

  如果输入或输出参数之一具有 [``__array_ufunc__``](#numpy.class.__array_ufunc__) 方法，则执行该方法而不是ufunc。
  如果多个参数实现 [``__array_ufunc__``](#numpy.class.__array_ufunc__) ，则按顺序尝试它们：子类在超类之前，输入在输出之前，否则从左到右。
  返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) 以外的内容的第一个例程确定结果。
  如果所有 [``__array_ufunc__``](#numpy.class.__array_ufunc__) 操作都返回 [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented)，则会引发 [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError)。

  ::: tip 注意

  我们打算将numpy函数重新实现为(（通用的）ufunc，在这种情况下，
  它们将可能被 ``__array_ufunc__`` 方法覆盖。
  一个主要的候选是 [``matmul``](generated/numpy.matmul.html#numpy.matmul)，它目前不是Ufunc，
  但可以相对容易地重写为（一组）通用Ufuncs。
  对于[``median``](generated/numpy.median.html#numpy.median)、
  [``amin``](generated/numpy.amin.html#numpy.amin)和 [``argsort``](generated/numpy.argsort.html#numpy.argsort) 等函数，可能会发生相同的情况。
  :::

  Like with some other special methods in python, such as ``__hash__`` and
  ``__iter__``, it is possible to indicate that your class does *not*
  support ufuncs by setting ``__array_ufunc__ = None``. Ufuncs always raise
  [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError) when called on an object that sets
  ``__array_ufunc__ = None``.

  The presence of [``__array_ufunc__``](#numpy.class.__array_ufunc__) also influences how
  [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray) handles binary operations like ``arr + obj`` and ``arr
  < obj`` when ``arr`` is an [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray) and ``obj`` is an instance
  of a custom class. There are two possibilities. If
  ``obj.__array_ufunc__`` is present and not [``None``](https://docs.python.org/dev/library/constants.html#None), then
  ``ndarray.__add__`` and friends will delegate to the ufunc machinery,
  meaning that ``arr + obj`` becomes ``np.add(arr, obj)``, and then
  [``add``](generated/numpy.add.html#numpy.add) invokes ``obj.__array_ufunc__``. This is useful if you
  want to define an object that acts like an array.

  Alternatively, if ``obj.__array_ufunc__`` is set to [``None``](https://docs.python.org/dev/library/constants.html#None), then as a
  special case, special methods like ``ndarray.__add__`` will notice this
  and *unconditionally* raise [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError). This is useful if you want to
  create objects that interact with arrays via binary operations, but
  are not themselves arrays. For example, a units handling system might have
  an object ``m`` representing the “meters” unit, and want to support the
  syntax ``arr * m`` to represent that the array has units of “meters”, but
  not want to otherwise interact with arrays via ufuncs or otherwise. This
  can be done by setting ``__array_ufunc__ = None`` and defining ``__mul__``
  and ``__rmul__`` methods. (Note that this means that writing an
  ``__array_ufunc__`` that always returns [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) is not
  quite the same as setting ``__array_ufunc__ = None``: in the former
  case, ``arr + obj`` will raise [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError), while in the latter
  case it is possible to define a ``__radd__`` method to prevent this.)

  The above does not hold for in-place operators, for which [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray)
  never returns [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented).  Hence, ``arr += obj`` would always
  lead to a [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError).  This is because for arrays in-place operations
  cannot generically be replaced by a simple reverse operation.  (For
  instance, by default, ``arr += obj`` would be translated to ``arr =
  arr + obj``, i.e., ``arr`` would be replaced, contrary to what is expected
  for in-place array operations.)

  ::: tip 注意

  If you define ``__array_ufunc__``:

  - If you are not a subclass of [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray), we recommend your
  class define special methods like ``__add__`` and ``__lt__`` that
  delegate to ufuncs just like ndarray does.  An easy way to do this
  is to subclass from [``NDArrayOperatorsMixin``](generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin).
  - If you subclass [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray), we recommend that you put all your
  override logic in ``__array_ufunc__`` and not also override special
  methods. This ensures the class hierarchy is determined in only one
  place rather than separately by the ufunc machinery and by the binary
  operation rules (which gives preference to special methods of
  subclasses; the alternative way to enforce a one-place only hierarchy,
  of setting [``__array_ufunc__``](#numpy.class.__array_ufunc__) to [``None``](https://docs.python.org/dev/library/constants.html#None), would seem very
  unexpected and thus confusing, as then the subclass would not work at
  all with ufuncs).
  - [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray) defines its own [``__array_ufunc__``](#numpy.class.__array_ufunc__), which,
  evaluates the ufunc if no arguments have overrides, and returns
  [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) otherwise. This may be useful for subclasses
  for which [``__array_ufunc__``](#numpy.class.__array_ufunc__) converts any instances of its own
  class to [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray): it can then pass these on to its
  superclass using ``super().__array_ufunc__(*inputs, **kwargs)``,
  and finally return the results after possible back-conversion. The
  advantage of this practice is that it ensures that it is possible
  to have a hierarchy of subclasses that extend the behaviour. See
  [Subclassing ndarray](https://numpy.org/devdocs/user/basics.subclassing.html#basics-subclassing) for details.

  :::

  ::: tip 注意

  If a class defines the [``__array_ufunc__``](#numpy.class.__array_ufunc__) method,
  this disables the [``__array_wrap__``](#numpy.class.__array_wrap__),
  [``__array_prepare__``](#numpy.class.__array_prepare__), [``__array_priority__``](#numpy.class.__array_priority__) mechanism
  described below for ufuncs (which may eventually be deprecated).

  :::


- ``class.__array_function__``(*func*, *types*, *args*, *kwargs*)

  *New in version 1.16.* 

  ::: tip 注意

  - In NumPy 1.17, the protocol is enabled by default, but can be disabled
  with ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0``.
  - In NumPy 1.16, you need to set the environment variable
  ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1`` before importing NumPy to use
  NumPy function overrides.
  - Eventually, expect to ``__array_function__`` to always be enabled.

  :::

  - ``func`` is an arbitrary callable exposed by NumPy’s public API,
  which was called in the form ``func(*args, **kwargs)``.
  - ``types`` is a [collection](collections.abc.Collection)
  of unique argument types from the original NumPy function call that
  implement ``__array_function__``.
  - The tuple ``args`` and dict ``kwargs`` are directly passed on from the
  original call.

  As a convenience for ``__array_function__`` implementors, ``types``
  provides all argument types with an ``'__array_function__'`` attribute.
  This allows implementors to quickly identify cases where they should defer
  to ``__array_function__`` implementations on other arguments.
  Implementations should not rely on the iteration order of ``types``.

  Most implementations of ``__array_function__`` will start with two
  checks:

  1. Is the given function something that we know how to overload?
  1. Are all arguments of a type that we know how to handle?

  If these conditions hold, ``__array_function__`` should return the result
  from calling its implementation for ``func(*args, **kwargs)``.  Otherwise,
  it should return the sentinel value ``NotImplemented``, indicating that the
  function is not implemented by these types.

  There are no general requirements on the return value from
  ``__array_function__``, although most sensible implementations should
  probably return array(s) with the same type as one of the function’s
  arguments.

  It may also be convenient to define a custom decorators (``implements``
  below) for registering ``__array_function__`` implementations.

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

  Note that it is not required for ``__array_function__`` implementations to
  include *all* of the corresponding NumPy function’s optional arguments
  (e.g., ``broadcast_to`` above omits the irrelevant ``subok`` argument).
  Optional arguments are only passed in to ``__array_function__`` if they
  were explicitly used in the NumPy function call.

  Just like the case for builtin special methods like ``__add__``, properly
  written ``__array_function__`` methods should always return
  ``NotImplemented`` when an unknown type is encountered. Otherwise, it will
  be impossible to correctly override NumPy functions from another object
  if the operation also includes one of your objects.

  For the most part, the rules for dispatch with ``__array_function__``
  match those for ``__array_ufunc__``. In particular:

  - NumPy will gather implementations of ``__array_function__`` from all
  specified inputs and call them in order: subclasses before
  superclasses, and otherwise left to right. Note that in some edge cases
  involving subclasses, this differs slightly from the
  [current behavior](https://bugs.python.org/issue30140) of Python.
  - Implementations of ``__array_function__`` indicate that they can
  handle the operation by returning any value other than
  ``NotImplemented``.
  - If all ``__array_function__`` methods return ``NotImplemented``,
  NumPy will raise ``TypeError``.

  If no ``__array_function__`` methods exists, NumPy will default to calling
  its own implementation, intended for use on NumPy arrays. This case arises,
  for example, when all array-like arguments are Python numbers or lists.
  (NumPy arrays do have a ``__array_function__`` method, given below, but it
  always returns ``NotImplemented`` if any argument other than a NumPy array
  subclass implements ``__array_function__``.)

  One deviation from the current behavior of ``__array_ufunc__`` is that
  NumPy will only call ``__array_function__`` on the *first* argument of each
  unique type. This matches Python’s [rule for calling reflected methods](https://docs.python.org/3/reference/datamodel.html#object.__ror__), and
  this ensures that checking overloads has acceptable performance even when
  there are a large number of overloaded arguments.

- ``class.__array_finalize__``(*obj*)

  This method is called whenever the system internally allocates a
  new array from *obj*, where *obj* is a subclass (subtype) of the
  [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray). It can be used to change attributes of *self*
  after construction (so as to ensure a 2-d matrix for example), or
  to update meta-information from the “parent.” Subclasses inherit
  a default implementation of this method that does nothing.


- ``class.__array_prepare__``(*array*, *context=None*)

  At the beginning of every [ufunc](ufuncs.html#ufuncs-output-type), this
  method is called on the input object with the highest array
  priority, or the output object if one was specified. The output
  array is passed in and whatever is returned is passed to the ufunc.
  Subclasses inherit a default implementation of this method which
  simply returns the output array unmodified. Subclasses may opt to
  use this method to transform the output array into an instance of
  the subclass and update metadata before returning the array to the
  ufunc for computation.

  ::: tip 注意

  For ufuncs, it is hoped to eventually deprecate this method in
  favour of [``__array_ufunc__``](#numpy.class.__array_ufunc__).

  :::


- ``class.__array_wrap__``(*array*, *context=None*)

  At the end of every [ufunc](ufuncs.html#ufuncs-output-type), this method
  is called on the input object with the highest array priority, or
  the output object if one was specified. The ufunc-computed array
  is passed in and whatever is returned is passed to the user.
  Subclasses inherit a default implementation of this method, which
  transforms the array into a new instance of the object’s class.
  Subclasses may opt to use this method to transform the output array
  into an instance of the subclass and update metadata before
  returning the array to the user.

  ::: tip 注意

  For ufuncs, it is hoped to eventually deprecate this method in
  favour of [``__array_ufunc__``](#numpy.class.__array_ufunc__).

  :::

- ``class.__array_priority__``

  The value of this attribute is used to determine what type of
  object to return in situations where there is more than one
  possibility for the Python type of the returned object. Subclasses
  inherit a default value of 0.0 for this attribute.

  ::: tip 注意

  For ufuncs, it is hoped to eventually deprecate this method in
  favour of [``__array_ufunc__``](#numpy.class.__array_ufunc__).

  :::

- ``class.__array__``([*dtype*])

  If a class (ndarray subclass or not) having the [``__array__``](#numpy.class.__array__)
  method is used as the output object of an [ufunc](ufuncs.html#ufuncs-output-type), results will be written to the object
  returned by [``__array__``](#numpy.class.__array__). Similar conversion is done on
  input arrays.

## 矩阵对象

::: tip 注意

It is strongly advised *not* to use the matrix subclass.  As described
below, it makes writing functions that deal consistently with matrices
and regular arrays very difficult. Currently, they are mainly used for
interacting with ``scipy.sparse``. We hope to provide an alternative
for this use, however, and eventually remove the ``matrix`` subclass.

:::

[``matrix``](generated/numpy.matrix.html#numpy.matrix) objects inherit from the ndarray and therefore, they
have the same attributes and methods of ndarrays. There are six
important differences of matrix objects, however, that may lead to
unexpected results when you use matrices but expect them to act like
arrays:

1. Matrix objects can be created using a string notation to allow Matlab-style syntax where spaces separate columns and semicolons (‘;’) separate rows.
1. Matrix objects are always two-dimensional. This has far-reaching implications, in that m.ravel() is still two-dimensional (with a 1 in the first dimension) and item selection returns two-dimensional objects so that sequence behavior is fundamentally different than arrays.
1. Matrix objects over-ride multiplication to be matrix-multiplication. Make sure you understand this for functions that you may want to receive matrices. Especially in light of the fact that asanyarray(m) returns a matrix when m is a matrix.
1. Matrix objects over-ride power to be matrix raised to a power. The same warning about using power inside a function that uses asanyarray(…) to get an array object holds for this fact.
1. The default \_\_array_priority__ of matrix objects is 10.0, and therefore mixed operations with ndarrays always produce matrices.
1. Matrices have special attributes which make calculations easier. These are

    方法 | 描述
    ---|---
    matrix.T | Returns the transpose of the matrix.
    matrix.H | Returns the (complex) conjugate transpose of self.
    matrix.I | Returns the (multiplicative) inverse of invertible self.
    matrix.A | Return self as an ndarray object.

::: danger 警告

Matrix objects over-ride multiplication, ‘*’, and power, ‘**’, to
be matrix-multiplication and matrix power, respectively. If your
subroutine can accept sub-classes and you do not convert to base-
class arrays, then you must use the ufuncs multiply and power to
be sure that you are performing the correct operation for all
inputs.

:::

The matrix class is a Python subclass of the ndarray and can be used
as a reference for how to construct your own subclass of the ndarray.
Matrices can be created from other matrices, strings, and anything
else that can be converted to an ``ndarray`` . The name “mat “is an
alias for “matrix “in NumPy.

方法 | 描述
---|---
[matrix](generated/numpy.matrix.html#numpy.matrix)(data[, dtype, copy]) | **Note:** It is no longer recommended to use this class, even for linear
[asmatrix](generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | Interpret the input as a matrix.
[bmat](generated/numpy.bmat.html#numpy.bmat)(obj[, ldict, gdict]) | Build a matrix object from a string, nested sequence, or array.

Example 1: Matrix creation from a string

``` python
>>>>>> a=mat('1 2 3; 4 5 3')
>>> print (a*a.T).I
[[ 0.2924 -0.1345]
 [-0.1345  0.0819]]
```

Example 2: Matrix creation from nested sequence

``` python
>>>>>> mat([[1,5,10],[1.0,3,4j]])
matrix([[  1.+0.j,   5.+0.j,  10.+0.j],
        [  1.+0.j,   3.+0.j,   0.+4.j]])
```

示例 3: 从数组创建矩阵

``` python
>>>>>> mat(random.rand(3,3)).T
matrix([[ 0.7699,  0.7922,  0.3294],
        [ 0.2792,  0.0101,  0.9219],
        [ 0.3398,  0.7571,  0.8197]])
```

## 内存映射文件数组

内存映射文件对于读取和/或修改具有规则布局的大文件的小段非常有用，
而无需将整个文件读入内存。
ndarray的一个简单子类使用内存映射文件作为数组的数据缓冲区。
对于小文件，将整个文件读入内存的开销通常不大，但是对于大文件，使用内存映射可以节省大量资源。

内存映射文件数组还有一个额外的方法（除了它们从ndarray继承的方法之外）：[``.flush()``](generated/numpy.memmap.flush.html#numpy.memmap.flush)，
用户必须手动调用该方法，以确保对阵列的任何更改都实际写入磁盘。

方法 | 描述
---|---
[memmap](generated/numpy.memmap.html#numpy.memmap) | 创建存储在磁盘上二进制文件中的数组的内存映射。
[memmap.flush](generated/numpy.memmap.flush.html#numpy.memmap.flush)(self) | 将数组中的任何更改写入磁盘上的文件。

示例：

``` python
>>>>>> a = memmap('newfile.dat', dtype=float, mode='w+', shape=1000)
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

[创建字符数组(numpy.char)](routines.array-creation.html#routines-array-creation-char)

:::

::: tip 注意

[``chararray``](generated/numpy.chararray.html#numpy.chararray) 类的存在是为了与Numarray向后兼容，不建议在新开发中使用它。从 numpy 1.4 开始，如果需要字符串数组，建议使用[``dtype``](generated/numpy.dtype.html#numpy.dtype) ``object_``、 ``string_`` 或 ``unicode_`` 的数组，并使用 [``numpy.char``](routines.char.html#module-numpy.char) 模块中的自由函数进行快速矢量化字符串操作。

:::

这些是 ``string_`` 类型或 ``unicode_`` 类型的增强型数组。
这些数组继承自 [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray) ，但在（逐个）元素的基础上特别定义了操作 ``+``, ``*``, 和 ``%`` 。
这些操作在字符类型的标准 [``ndarray``](generated/numpy.ndarray.html#numpy.ndarray) 上不可用。
此外，[``chararray``](generated/numpy.chararray.html#numpy.chararray) 具有所有标准 [``string``](https://docs.python.org/dev/library/stdtypes.html#str)（和``unicode`` ）方法，在逐个元素的基础上执行它们。
也许创建chararray的最简单方法是使用 [``self.view(chararray)``](generated/numpy.ndarray.view.html#numpy.ndarray.view)，
其中 *self* 是str或unicode数据类型的ndarray。
但是，也可以使用 [``numpy.chararray``](generated/numpy.chararray.html#numpy.chararray) 构造函数或通过 [``numpy.char.array``](generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array) 函数创建chararray：

方法 | 描述
---|---
[chararray](generated/numpy.chararray.html#numpy.chararray)(shape[, itemsize, unicode, …]) | 提供有关字符串和unicode值数组的便捷视图。
[core.defchararray.array](generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array)(obj[, itemsize, …]) | 创建一个chararray。

与 str 数据类型的标准 ndarray 的另一个不同之处是 chararray 继承了由 Numarray 引入的特性，
即在项检索和比较操作中，数组中任何元素末尾的空格都将被忽略。

## 记录数组（``numpy.rec``）

::: tip 另见

[Creating record arrays (numpy.rec)](routines.array-creation.html#routines-array-creation-rec), [Data type routines](routines.dtype.html#routines-dtype),
[Data type objects (dtype)](arrays.dtypes.html#arrays-dtypes).

:::

NumPy提供了 [``recarray``](generated/numpy.recarray.html#numpy.recarray) 类，允许将结构化数组的字段作为属性进行访问，
以及相应的标量数据类型对象 [``记录``](generated/numpy.record.html#numpy.record)。

方法 | 描述
---|---
[recarray](generated/numpy.recarray.html#numpy.recarray) | 构造一个允许使用属性进行字段访问的ndarray。
[record](generated/numpy.record.html#numpy.record) | 一种数据类型标量，允许字段访问作为属性查找。

## 掩码数组（``numpy.ma``）

::: tip 另见

[Masked arrays](maskedarray.html#maskedarray)

:::

## 标准容器类

为了向后兼容并作为标准的“容器”类，
Numeric的UserArray已被引入NumPy并命名为 [``numpy.lib.user_array.container``](generated/numpy.lib.user_array.container.html#numpy.lib.user_array.container) 容器类是一个Python类，
其self.array属性是一个ndarray。
使用numpy.lib.user_array.container比使用ndarray本身更容易进行多重继承，因此默认包含它。
除了提及它的存在之外，这里没有记录，因为如果可以的话，我们鼓励你直接使用ndarray类。

方法 | 描述
---|---
[numpy.lib.user_array.container](generated/numpy.lib.user_array.container.html#numpy.lib.user_array.container)(data[, …]) | 标准容器类，便于多重继承。

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
>>>>>> a = arange(24).reshape(3,2,4)+10
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
[ndarray.flat](generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | 数组上的一维迭代器。

如前所述，ndarray 对象的 flat 属性返回一个迭代器，它将以C风格的连续顺序循环遍历整个数组。

``` python
>>>>>> for i, val in enumerate(a.flat):
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
[ndenumerate](generated/numpy.ndenumerate.html#numpy.ndenumerate)(arr) | 多维索引迭代器。

有时在迭代时获取N维索引可能是有用的。ndenumerate迭代器可以实现这一点。

``` python
>>>>>> for i, val in ndenumerate(a):
...     if sum(i)%5 == 0: print i, val
(0, 0, 0) 10
(1, 1, 3) 25
(2, 0, 3) 29
(2, 1, 2) 32
```

### 广播迭代器

方法 | 描述
---|---
[broadcast](generated/numpy.broadcast.html#numpy.broadcast) | 创建一个模仿广播的对象。

广播的一般概念也可以使用 [``broadcast``](generated/numpy.broadcast.html#numpy.broadcast) 迭代器从Python获得。
此对象将对象作为输入，并返回一个迭代器，该迭代器返回元组，提供广播结果中的每个输入序列元素。

``` python
>>>>>> for val in broadcast([[1,0],[2,3]],[0,1]):
...     print val
(1, 0)
(0, 1)
(2, 0)
(3, 1)
```
