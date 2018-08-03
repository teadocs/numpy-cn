# 迭代数组

NumPy 1.6中引入的迭代器对象``nditer``提供了许多灵活的方式来以系统的方式访问一个或多个数组的所有元素。 本页介绍了在Python中使用该对象进行数组计算的一些基本方法，然后总结了如何在Cython中加速内部循环。 由于``nditer``的Python曝光是C数组迭代器API的相对简单的映射，因此这些想法还将提供有关使用C或C ++的数组迭代的帮助。

## 单数组迭代

使用``nditer``可以完成的最基本的任务是访问数组的每个元素。 使用标准Python迭代器接口逐个提供每个元素。

**例子**

```python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a):
...     print x,
...
0 1 2 3 4 5
```

要注意这个迭代，重要的是选择顺序来匹配数组的内存布局，而不是使用标准的C或Fortran排序。 这样做是为了提高访问效率，反映了默认情况下只需要访问每个元素而不关心特定排序的想法。 我们可以通过迭代前一个数组的转置来看到这一点，而不是以C顺序获取该转置的副本。

**例子**

```python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a.T):
...     print x,
...
0 1 2 3 4 5
>>> for x in np.nditer(a.T.copy(order='C')):
...     print x,
...
0 3 1 4 2 5
```

a和aT的元素以相同的顺序遍历，即它们存储在内存中的顺序，而* aTcopy（order ='C'）*的元素以不同的顺序访问，因为它们已被放入 不同的内存布局。

### 控制迭代顺序

有时，以特定顺序访问数组的元素很重要，而不管内存中元素的布局如何。 ``nditer``对象提供了一个order参数来控制迭代的这个方面。 具有上述行为的默认值是order ='K'以保持现有订单。 对于C顺序，可以使用order ='C'覆盖它，对于Fortran顺序，可以使用order ='F'覆盖它。

**例子**

```python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a, order='F'):
...     print x,
...
0 3 1 4 2 5
>>> for x in np.nditer(a.T, order='C'):
...     print x,
...
0 3 1 4 2 5
```

### 修改数组值

默认情况下，``nditer``将输入数组视为只读对象。 要修改数组元素，必须指定读写或只写模式。 这是用每操作数标志控制的。

Python中的常规赋值只是更改本地或全局变量字典中的引用，而不是修改现有变量。 这意味着简单地分配给x不会将值放入数组的元素中，而是将x作为数组元素引用切换为对指定值的引用。 要实际修改数组的元素，x应该用省略号索引。

**例子**

```python
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> for x in np.nditer(a, op_flags=['readwrite']):
...     x[...] = 2 * x
...
>>> a
array([[ 0,  2,  4],
       [ 6,  8, 10]])
```

### 使用外部循环

在目前为止的所有示例中，a的元素由迭代器一次提供一个，因为所有循环逻辑都是迭代器的内部逻辑。 虽然这很简单方便，但效率不高。 更好的方法是将一维最内层循环移动到迭代器外部的代码中。 这样，NumPy的矢量化操作可以用在被访问元素的较大块上。

``nditer``将尝试提供尽可能大的内部循环块。 通过强制'C'和'F'顺序，我们得到不同的外部循环大小。 通过指定迭代器标志来启用此模式。

观察到默认情况下保持本机内存顺序，迭代器能够提供单个一维块，而在强制Fortran命令时，它必须提供三个块，每个块有两个元素。

**例子**

```python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a, flags=['external_loop']):
...     print x,
...
[0 1 2 3 4 5]
```

```python
>>> for x in np.nditer(a, flags=['external_loop'], order='F'):
...     print x,
...
[0 3] [1 4] [2 5]
```

### 跟踪索引或多索引

在迭代期间，您可能希望在计算中使用当前元素的索引。 例如，您可能希望按内存顺序访问数组的元素，但使用C顺序，Fortran顺序或多维索引来查找不同数组中的值。

Python迭代器协议没有一种从迭代器查询这些附加值的自然方法，因此我们引入了一种用``nditer``迭代的替代语法。 此语法显式使用迭代器对象本身，因此在迭代期间可以轻松访问其属性。 使用此循环结构，可以通过索引到迭代器来访问当前值，并且正在跟踪的索引是属性索引或multi_index，具体取决于请求的内容。

遗憾的是，Python交互式解释器在循环的每次迭代期间打印出while循环内的表达式的值。 我们使用这个循环结构修改了示例中的输出，以便更具可读性。

**例子**

```python
>>> a = np.arange(6).reshape(2,3)
>>> it = np.nditer(a, flags=['f_index'])
>>> while not it.finished:
...     print "%d < %d>" % (it[0], it.index),
...     it.iternext()
...
0 <0> 1 <2> 2 <4> 3 <1> 4 <3> 5 <5>
```

```python
>>> it = np.nditer(a, flags=['multi_index'])
>>> while not it.finished:
...     print "%d < %s>" % (it[0], it.multi_index),
...     it.iternext()
...
0 <(0, 0)> 1 <(0, 1)> 2 <(0, 2)> 3 <(1, 0)> 4 <(1, 1)> 5 <(1, 2)>
```

```python
>>> it = np.nditer(a, flags=['multi_index'], op_flags=['writeonly'])
>>> while not it.finished:
...     it[0] = it.multi_index[1] - it.multi_index[0]
...     it.iternext()
...
>>> a
array([[ 0,  1,  2],
       [-1,  0,  1]])
```

跟踪索引或多索引与使用外部循环不兼容，因为它需要每个元素具有不同的索引值。 如果您尝试组合这些标志，``nditer``对象将引发异常

**例子**

```python
>>> a = np.zeros((2,3))
>>> it = np.nditer(a, flags=['c_index', 'external_loop'])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked
```

### 缓冲数组元素

当强制迭代次序时，我们观察到外部循环选项可以以较小的块提供元素，因为不能以恒定的步幅以适当的顺序访问元素。 编写C代码时，这通常很好，但在纯Python代码中，这可能会导致性能显着降低。

通过启用缓冲模式，迭代器提供给内部循环的块可以变得更大，从而显着减少Python解释器的开销。 在强制Fortran迭代顺序的示例中，当启用缓冲时，内部循环可以一次性查看所有元素。

**例子**

```python
>>> a = np.arange(6).reshape(2,3)
>>> for x in np.nditer(a, flags=['external_loop'], order='F'):
...     print x,
...
[0 3] [1 4] [2 5]
```

```python
>>> for x in np.nditer(a, flags=['external_loop','buffered'], order='F'):
...     print x,
...
[0 3 1 4 2 5]
```

### 迭代为特定数据类型

有时需要将数组视为与存储数据类型不同的数据类型。 例如，即使被操作的数组是32位浮点数，也可能希望对64位浮点数进行所有计算。 除了在编写低级C代码时，通常最好让迭代器处理复制或缓冲，而不是在内部循环中自己转换数据类型。

有两种机制允许这样做，临时副本和缓冲模式。 对于临时副本，使用新数据类型创建整个数组的副本，然后在副本中完成迭代。 通过在所有迭代完成后更新原始数组的模式允许写访问。 临时副本的主要缺点是临时副本可能消耗大量内存，特别是如果迭代数据类型具有比原始类型更大的项目大小。

缓冲模式减轻了内存使用问题，并且比制作临时副本更加缓存友好。 除特殊情况外，在迭代器外部需要立即使用整个数组，建议使用缓冲而不是临时复制。 在NumPy中，ufuncs和其他函数使用缓冲来支持灵活的输入，并且内存开销最小。

在我们的示例中，我们将使用复杂数据类型处理输入数组，以便我们可以取负数的平方根。 如果没有启用副本或缓冲模式，如果数据类型不精确匹配，迭代器将引发异常。

**例子**

```python
>>> a = np.arange(6).reshape(2,3) - 3
>>> for x in np.nditer(a, op_dtypes=['complex128']):
...     print np.sqrt(x),
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Iterator operand required copying or buffering, but neither copying nor buffering was enabled
```

In copying mode, ‘copy’ is specified as a per-operand flag. This is done to provide control in a per-operand fashion. Buffering mode is specified as an iterator flag.

**Example**

```python
>>> a = np.arange(6).reshape(2,3) - 3
>>> for x in np.nditer(a, op_flags=['readonly','copy'],
...                 op_dtypes=['complex128']):
...     print np.sqrt(x),
...
1.73205080757j 1.41421356237j 1j 0j (1+0j) (1.41421356237+0j)
```

```python
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['complex128']):
...     print np.sqrt(x),
...
1.73205080757j 1.41421356237j 1j 0j (1+0j) (1.41421356237+0j)
```

The iterator uses NumPy’s casting rules to determine whether a specific conversion is permitted. By default, it enforces ‘safe’ casting. This means, for example, that it will raise an exception if you try to treat a 64-bit float array as a 32-bit float array. In many cases, the rule ‘same_kind’ is the most reasonable rule to use, since it will allow conversion from 64 to 32-bit float, but not from float to int or from complex to float.

**Example**

```python
>>> a = np.arange(6.)
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['float32']):
...     print x,
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Iterator operand 0 dtype could not be cast from dtype('float64') to dtype('float32') according to the rule 'safe'
```

```python
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['float32'],
...                 casting='same_kind'):
...     print x,
...
0.0 1.0 2.0 3.0 4.0 5.0
>>> for x in np.nditer(a, flags=['buffered'], op_dtypes=['int32'], casting='same_kind'):
...     print x,
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Iterator operand 0 dtype could not be cast from dtype('float64') to dtype('int32') according to the rule 'same_kind'
```

One thing to watch out for is conversions back to the original data type when using a read-write or write-only operand. A common case is to implement the inner loop in terms of 64-bit floats, and use ‘same_kind’ casting to allow the other floating-point types to be processed as well. While in read-only mode, an integer array could be provided, read-write mode will raise an exception because conversion back to the array would violate the casting rule.

**Example**

```python
>>> a = np.arange(6)
>>> for x in np.nditer(a, flags=['buffered'], op_flags=['readwrite'],
...                 op_dtypes=['float64'], casting='same_kind'):
...     x[...] = x / 2.0
...
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
TypeError: Iterator requested dtype could not be cast from dtype('float64') to dtype('int64'), the operand 0 dtype, according to the rule 'same_kind'
```

## Broadcasting Array Iteration

NumPy has a set of rules for dealing with arrays that have differing shapes which are applied whenever functions take multiple operands which combine element-wise. This is called broadcasting. The ``nditer`` object can apply these rules for you when you need to write such a function.

As an example, we print out the result of broadcasting a one and a two dimensional array together.

**Example**

```python
>>> a = np.arange(3)
>>> b = np.arange(6).reshape(2,3)
>>> for x, y in np.nditer([a,b]):
...     print "%d:%d" % (x,y),
...
0:0 1:1 2:2 0:3 1:4 2:5
```

When a broadcasting error occurs, the iterator raises an exception which includes the input shapes to help diagnose the problem.

**Example**

```python
>>> a = np.arange(2)
>>> b = np.arange(6).reshape(2,3)
>>> for x, y in np.nditer([a,b]):
...     print "%d:%d" % (x,y),
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (2) (2,3)
```

### Iterator-Allocated Output Arrays

A common case in NumPy functions is to have outputs allocated based on the broadcasting of the input, and additionally have an optional parameter called ‘out’ where the result will be placed when it is provided. The ``nditer`` object provides a convenient idiom that makes it very easy to support this mechanism.

We’ll show how this works by creating a function ``square`` which squares its input. Let’s start with a minimal function definition excluding ‘out’ parameter support.

**Example**

```python
>>> def square(a):
...     it = np.nditer([a, None])
...     for x, y in it:
...          y[...] = x*x
...     return it.operands[1]
...
>>> square([1,2,3])
array([1, 4, 9])
```

By default, the ``nditer`` uses the flags ‘allocate’ and ‘writeonly’ for operands that are passed in as None. This means we were able to provide just the two operands to the iterator, and it handled the rest.

When adding the ‘out’ parameter, we have to explicitly provide those flags, because if someone passes in an array as ‘out’, the iterator will default to ‘readonly’, and our inner loop would fail. The reason ‘readonly’ is the default for input arrays is to prevent confusion about unintentionally triggering a reduction operation. If the default were ‘readwrite’, any broadcasting operation would also trigger a reduction, a topic which is covered later in this document.

While we’re at it, let’s also introduce the ‘no_broadcast’ flag, which will prevent the output from being broadcast. This is important, because we only want one input value for each output. Aggregating more than one input value is a reduction operation which requires special handling. It would already raise an error because reductions must be explicitly enabled in an iterator flag, but the error message that results from disabling broadcasting is much more understandable for end-users. To see how to generalize the square function to a reduction, look at the sum of squares function in the section about Cython.

For completeness, we’ll also add the ‘external_loop’ and ‘buffered’ flags, as these are what you will typically want for performance reasons.

**Example**

```python
>>> def square(a, out=None):
...     it = np.nditer([a, out],
...             flags = ['external_loop', 'buffered'],
...             op_flags = [['readonly'],
...                         ['writeonly', 'allocate', 'no_broadcast']])
...     for x, y in it:
...         y[...] = x*x
...     return it.operands[1]
...
```

```python
>>> square([1,2,3])
array([1, 4, 9])
```

```python
>>> b = np.zeros((3,))
>>> square([1,2,3], out=b)
array([ 1.,  4.,  9.])
>>> b
array([ 1.,  4.,  9.])
```

```python
>>> square(np.arange(6).reshape(2,3), out=b)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in square
ValueError: non-broadcastable output operand with shape (3) doesn't match the broadcast shape (2,3)
```

### Outer Product Iteration

Any binary operation can be extended to an array operation in an outer product fashion ``like`` in outer, and the ``nditer`` object provides a way to accomplish this by explicitly mapping the axes of the operands. It is also possible to do this with ``newaxis`` indexing, but we will show you how to directly use the nditer op_axes parameter to accomplish this with no intermediate views.

We’ll do a simple outer product, placing the dimensions of the first operand before the dimensions of the second operand. The op_axes parameter needs one list of axes for each operand, and provides a mapping from the iterator’s axes to the axes of the operand.

Suppose the first operand is one dimensional and the second operand is two dimensional. The iterator will have three dimensions, so op_axes will have two 3-element lists. The first list picks out the one axis of the first operand, and is -1 for the rest of the iterator axes, with a final result of [0, -1, -1]. The second list picks out the two axes of the second operand, but shouldn’t overlap with the axes picked out in the first operand. Its list is [-1, 0, 1]. The output operand maps onto the iterator axes in the standard manner, so we can provide None instead of constructing another list.

The operation in the inner loop is a straightforward multiplication. Everything to do with the outer product is handled by the iterator setup.

**Example**

```python
>>> a = np.arange(3)
>>> b = np.arange(8).reshape(2,4)
>>> it = np.nditer([a, b, None], flags=['external_loop'],
...             op_axes=[[0, -1, -1], [-1, 0, 1], None])
>>> for x, y, z in it:
...     z[...] = x*y
...
>>> it.operands[2]
array([[[ 0,  0,  0,  0],
        [ 0,  0,  0,  0]],
       [[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],
       [[ 0,  2,  4,  6],
        [ 8, 10, 12, 14]]])
```

### Reduction Iteration

Whenever a writeable operand has fewer elements than the full iteration space, that operand is undergoing a reduction. The ``nditer`` object requires that any reduction operand be flagged as read-write, and only allows reductions when ‘reduce_ok’ is provided as an iterator flag.

For a simple example, consider taking the sum of all elements in an array.

**Example**

```python
>>> a = np.arange(24).reshape(2,3,4)
>>> b = np.array(0)
>>> for x, y in np.nditer([a, b], flags=['reduce_ok', 'external_loop'],
...                     op_flags=[['readonly'], ['readwrite']]):
...     y[...] += x
...
>>> b
array(276)
>>> np.sum(a)
276
```

Things are a little bit more tricky when combining reduction and allocated operands. Before iteration is started, any reduction operand must be initialized to its starting values. Here’s how we can do this, taking sums along the last axis of a.

**Example**

```python
>>> a = np.arange(24).reshape(2,3,4)
>>> it = np.nditer([a, None], flags=['reduce_ok', 'external_loop'],
...             op_flags=[['readonly'], ['readwrite', 'allocate']],
...             op_axes=[None, [0,1,-1]])
>>> it.operands[1][...] = 0
>>> for x, y in it:
...     y[...] += x
...
>>> it.operands[1]
array([[ 6, 22, 38],
       [54, 70, 86]])
>>> np.sum(a, axis=2)
array([[ 6, 22, 38],
       [54, 70, 86]])
```

To do buffered reduction requires yet another adjustment during the setup. Normally the iterator construction involves copying the first buffer of data from the readable arrays into the buffer. Any reduction operand is readable, so it may be read into a buffer. Unfortunately, initialization of the operand after this buffering operation is complete will not be reflected in the buffer that the iteration starts with, and garbage results will be produced.

The iterator flag “delay_bufalloc” is there to allow iterator-allocated reduction operands to exist together with buffering. When this flag is set, the iterator will leave its buffers uninitialized until it receives a reset, after which it will be ready for regular iteration. Here’s how the previous example looks if we also enable buffering.

**Example**

```python
>>> a = np.arange(24).reshape(2,3,4)
>>> it = np.nditer([a, None], flags=['reduce_ok', 'external_loop',
...                                  'buffered', 'delay_bufalloc'],
...             op_flags=[['readonly'], ['readwrite', 'allocate']],
...             op_axes=[None, [0,1,-1]])
>>> it.operands[1][...] = 0
>>> it.reset()
>>> for x, y in it:
...     y[...] += x
...
>>> it.operands[1]
array([[ 6, 22, 38],
       [54, 70, 86]])
```

## Putting the Inner Loop in Cython

Those who want really good performance out of their low level operations should strongly consider directly using the iteration API provided in C, but for those who are not comfortable with C or C++, Cython is a good middle ground with reasonable performance tradeoffs. For the ``nditer`` object, this means letting the iterator take care of broadcasting, dtype conversion, and buffering, while giving the inner loop to Cython.

For our example, we’ll create a sum of squares function. To start, let’s implement this function in straightforward Python. We want to support an ‘axis’ parameter similar to the numpy ``sum`` function, so we will need to construct a list for the op_axes parameter. Here’s how this looks.

**Example**

```python
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
...     it.operands[1][...] = 0
...     it.reset()
...     for x, y in it:
...         y[...] += x*x
...     return it.operands[1]
...
>>> a = np.arange(6).reshape(2,3)
>>> sum_squares_py(a)
array(55.0)
>>> sum_squares_py(a, axis=-1)
array([  5.,  50.])
```

To Cython-ize this function, we replace the inner loop (y[…] += x*x) with Cython code that’s specialized for the float64 dtype. With the ‘external_loop’ flag enabled, the arrays provided to the inner loop will always be one-dimensional, so very little checking needs to be done.

Here’s the listing of sum_squares.pyx:

```python
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

On this machine, building the .pyx file into a module looked like the following, but you may have to find some Cython tutorials to tell you the specifics for your system configuration.:

```sh
$ cython sum_squares.pyx
$ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -I/usr/include/python2.7 -fno-strict-aliasing -o sum_squares.so sum_squares.c
```

Running this from the Python interpreter produces the same answers as our native Python/NumPy code did.

**Example**

```python
>>> from sum_squares import sum_squares_cy
>>> a = np.arange(6).reshape(2,3)
>>> sum_squares_cy(a)
array(55.0)
>>> sum_squares_cy(a, axis=-1)
array([  5.,  50.])
```

Doing a little timing in IPython shows that the reduced overhead and memory allocation of the Cython inner loop is providing a very nice speedup over both the straightforward Python code and an expression using NumPy’s built-in sum function.:

```python
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