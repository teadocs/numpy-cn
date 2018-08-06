# 标准数组子类

NumPy中的``ndarray``是一种“新式”Python内置类型。 因此，如果需要，它可以从（在Python或C中）继承。 因此，它可以为许多有用的类奠定基础。 通常，是否对数组对象进行子类化或简单地将核心数组组件用作新类的内部部分是一个困难的决定，并且可以简单地作为选择问题。 NumPy有几个工具可以简化新对象与其他数组对象的交互方式，因此最终选择可能并不重要。 简化问题的一种方法是问自己，您感兴趣的对象是否可以替换为单个数组，还是确实需要两个或更多数组。

请注意，``asarray``始终返回基类ndarray。 如果您确信使用数组对象可以处理ndarray的任何子类，则可以使用``asanyarray``来允许子类在子例程中更加干净地传播。 原则上，子类可以重新定义数组的任何方面，因此，在严格的指导下，“asanyarray”将很少有用。 但是，数组对象的大多数子类都不会重新定义数组对象的某些方面，例如缓冲区接口或数组的属性。 然而，一个重要的例子是你的子程序可能无法处理数组的任意子类的原因是矩阵将“*”运算符重新定义为矩阵乘法，而不是逐个元素乘法。

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
> - 如果您继承``ndarray``，我们建议您将所有覆盖逻辑放在__array_ufunc__中，而不是覆盖特殊方法。 这确保了类层次结构只在一个地方确定，而不是由ufunc机制和二进制操作规则（它优先考虑子类的特殊方法;强制执行单一地方层次结构的另一种方法，将__array_ufunc__设置为 没有，看起来非常意外，因而令人困惑，因为那时子类根本无法使用ufuncs）。
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

At the end of every ufunc, this method is called on the input object with the highest array priority, or the output object if one was specified. The ufunc-computed array is passed in and whatever is returned is passed to the user. Subclasses inherit a default implementation of this method, which transforms the array into a new instance of the object’s class. Subclasses may opt to use this method to transform the output array into an instance of the subclass and update metadata before returning the array to the user.

> **Note**
> For ufuncs, it is hoped to eventually deprecate this method in favour of ``__array_ufunc__``.

### ``class.__array_priority__``
The value of this attribute is used to determine what type of object to return in situations where there is more than one possibility for the Python type of the returned object. Subclasses inherit a default value of 0.0 for this attribute.

> **Note**
> For ufuncs, it is hoped to eventually deprecate this method in favour of __array_ufunc__.

### ``class.__array__``([dtype])

If a class (ndarray subclass or not) having the ``__array__`` method is used as the output object of an ufunc, results will be written to the object returned by ``__array__``. Similar conversion is done on input arrays.

## Matrix objects

``matrix`` objects inherit from the ndarray and therefore, they have the same attributes and methods of ndarrays. There are six important differences of matrix objects, however, that may lead to unexpected results when you use matrices but expect them to act like arrays:

1. Matrix objects can be created using a string notation to allow Matlab-style syntax where spaces separate columns and semicolons (‘;’) separate rows.

1. Matrix objects are always two-dimensional. This has far-reaching implications, in that m.ravel() is still two-dimensional (with a 1 in the first dimension) and item selection returns two-dimensional objects so that sequence behavior is fundamentally different than arrays.

1. Matrix objects over-ride multiplication to be matrix-multiplication. Make sure you understand this for functions that you may want to receive matrices. Especially in light of the fact that asanyarray(m) returns a matrix when m is a matrix.

1. Matrix objects over-ride power to be matrix raised to a power. The same warning about using power inside a function that uses asanyarray(…) to get an array object holds for this fact.

1. The default __array_priority__ of matrix objects is 10.0, and therefore mixed operations with ndarrays always produce matrices.

1. Matrices have special attributes which make calculations easier. These are

    ``matrix.T``	Returns the transpose of the matrix.
    ``matrix.H``	Returns the (complex) conjugate transpose of self.
    ``matrix.I``	Returns the (multiplicative) inverse of invertible self.
    ``matrix.A``	Return self as an ndarray object.


<div class="warning-warp">
<b>Warning</b>

<p>Matrix objects over-ride multiplication, ‘*’, and power, ‘**’, to be matrix-multiplication and matrix power, respectively. If your subroutine can accept sub-classes and you do not convert to base- class arrays, then you must use the ufuncs multiply and power to be sure that you are performing the correct operation for all inputs.</p>
</div>

The matrix class is a Python subclass of the ndarray and can be used as a reference for how to construct your own subclass of the ndarray. Matrices can be created from other matrices, strings, and anything else that can be converted to an ndarray . The name “mat “is an alias for “matrix “in NumPy.

``matrix``(data[, dtype, copy])	Returns a matrix from an array-like object, or from a string of data.
``asmatrix``(data[, dtype])	Interpret the input as a matrix.
``bmat``(obj[, ldict, gdict])	Build a matrix object from a string, nested sequence, or array.

Example 1: Matrix creation from a string

```python
>>> a=mat('1 2 3; 4 5 3')
>>> print (a*a.T).I
[[ 0.2924 -0.1345]
 [-0.1345  0.0819]]
```

Example 2: Matrix creation from nested sequence

```python
>>> mat([[1,5,10],[1.0,3,4j]])
matrix([[  1.+0.j,   5.+0.j,  10.+0.j],
        [  1.+0.j,   3.+0.j,   0.+4.j]])
```

Example 3: Matrix creation from an array

```python
>>> mat(random.rand(3,3)).T
matrix([[ 0.7699,  0.7922,  0.3294],
        [ 0.2792,  0.0101,  0.9219],
        [ 0.3398,  0.7571,  0.8197]])
```

## Memory-mapped file arrays

Memory-mapped files are useful for reading and/or modifying small segments of a large file with regular layout, without reading the entire file into memory. A simple subclass of the ndarray uses a memory-mapped file for the data buffer of the array. For small files, the over-head of reading the entire file into memory is typically not significant, however for large files using memory mapping can save considerable resources.

Memory-mapped-file arrays have one additional method (besides those they inherit from the ndarray): .flush() which must be called manually by the user to ensure that any changes to the array actually get written to disk.

``memmap``	Create a memory-map to an array stored in a binary file on disk.
``memmap.flush``()	Write any changes in the array to the file on disk.

Example:

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

> **Note**
> The chararray class exists for backwards compatibility with Numarray, it is not recommended for new development. Starting from numpy 1.4, if one needs arrays of strings, it is recommended to use arrays of dtype object_, string_ or unicode_, and use the free functions in the numpy.char module for fast vectorized string operations.

These are enhanced arrays of either string_ type or unicode_ type. These arrays inherit from the ndarray, but specially-define the operations +, *, and % on a (broadcasting) element-by-element basis. These operations are not available on the standard ndarray of character type. In addition, the chararray has all of the standard string (and unicode) methods, executing them on an element-by-element basis. Perhaps the easiest way to create a chararray is to use self.view(chararray) where self is an ndarray of str or unicode data-type. However, a chararray can also be created using the numpy.chararray constructor, or via the numpy.char.array function:

- ``chararray``(shape[, itemsize, unicode, …])	Provides a convenient view on arrays of string and unicode values.
- ``core.defchararray.array``(obj[, itemsize, …])	Create a ``chararray``.

Another difference with the standard ndarray of str data-type is that the chararray inherits the feature introduced by Numarray that white-space at the end of any element in the array will be ignored on item retrieval and comparison operations.

## Record arrays (``numpy.rec``)

另见:

> Creating record arrays (numpy.rec), Data type routines, Data type objects (dtype).

NumPy provides the ``recarray`` class which allows accessing the fields of a structured array as attributes, and a corresponding scalar data type object record.

- ``recarray``	Construct an ndarray that allows field access using attributes.
- ``record``	A data-type scalar that allows field access as attribute lookup.

## Masked arrays (``numpy.ma``)

另见：

> Masked arrays

## Standard container class

For backward compatibility and as a standard “container “class, the UserArray from Numeric has been brought over to NumPy and named ``numpy.lib.user_array.container`` The container class is a Python class whose self.array attribute is an ndarray. Multiple inheritance is probably easier with numpy.lib.user_array.container than with the ndarray itself and so it is included by default. It is not documented here beyond mentioning its existence because you are encouraged to use the ndarray class directly if you can.

``numpy.lib.user_array.container``(data[, …])	Standard container-class for easy multiple-inheritance.

## Array Iterators

Iterators are a powerful concept for array processing. Essentially, iterators implement a generalized for-loop. If myiter is an iterator object, then the Python code:

```python
for val in myiter:
    ...
    some code involving val
    ...
```

calls ``val = myiter.next()`` repeatedly until ``StopIteration`` is raised by the iterator. There are several ways to iterate over an array that may be useful: default iteration, flat iteration, and N-dimensional enumeration.

### Default iteration

The default iterator of an ndarray object is the default Python iterator of a sequence type. Thus, when the array object itself is used as an iterator. The default behavior is equivalent to:

```python
for i in range(arr.shape[0]):
    val = arr[i]
```

This default iterator selects a sub-array of dimension N-1 from the array. This can be a useful construct for defining recursive algorithms. To loop over the entire array requires N for-loops.

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

### Flat iteration

- ``ndarray.flat``	A 1-D iterator over the array.

As mentioned previously, the flat attribute of ndarray objects returns an iterator that will cycle over the entire array in C-style contiguous order.

```python
>>> for i, val in enumerate(a.flat):
...     if i%5 == 0: print i, val
0 10
5 15
10 20
15 25
20 30
```

Here, I’ve used the built-in enumerate iterator to return the iterator index as well as the value.

### N-dimensional enumeration

- ``ndenumerate``(arr)	Multidimensional index iterator.

Sometimes it may be useful to get the N-dimensional index while iterating. The ndenumerate iterator can achieve this.

```python
>>> for i, val in ndenumerate(a):
...     if sum(i)%5 == 0: print i, val
(0, 0, 0) 10
(1, 1, 3) 25
(2, 0, 3) 29
(2, 1, 2) 32
```

### Iterator for broadcasting

- ``broadcast``	Produce an object that mimics broadcasting.

The general concept of broadcasting is also available from Python using the ``broadcast`` iterator. This object takes N objects as inputs and returns an iterator that returns tuples providing each of the input sequence elements in the broadcasted result.

```python
>>> for val in broadcast([[1,0],[2,3]],[0,1]):
...     print val
(1, 0)
(0, 1)
(2, 0)
(3, 1)
```
