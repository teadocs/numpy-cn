# Miscellaneous

## IEEE 754浮点特殊值

在numpy中特殊值定义为: nan, inf,

NaNs 可用作一种简陋的掩饰物 (如果你并不在乎初始的值是什么的话)

注意：不能使用等号运算符来测试NAN。例如：

```python
>>> myarr = np.array([1., 0., np.nan, 3.])
>>> np.nonzero(myarr == np.nan)
(array([], dtype=int64),)
>>> np.nan == np.nan  # is always False! Use special numpy functions instead.
False
>>> myarr[myarr == np.nan] = 0. # doesn't work
>>> myarr
array([  1.,   0.,  NaN,   3.])
>>> myarr[np.isnan(myarr)] = 0. # use this instead find
>>> myarr
array([ 1.,  0.,  0.,  3.])
```

其他的相关的特殊值判断函数

```python
isinf():    True if value is inf
isfinite(): True if not nan or inf
nan_to_num(): Map nan to 0, inf to max float, -inf to min float
```

除了从结果中排除nans之外，下是对应的常用函数：

```python
nansum()
nanmax()
nanmin()
nanargmax()
nanargmin()

>>> x = np.arange(10.)
>>> x[3] = np.nan
>>> x.sum()
nan
>>> np.nansum(x)
42.0
```

## numpy是如何处理数字异常的

默认``invalid`` 是 ``'warn'``, ``divide``, ``underflow`` 是 ``overflow`` 和 ``'ignore'``。 但是这可以改变，并且可以针对不同类型的异常单独设置。有以下不同的行为模式：

> - 'ignore' : 当异常发生时，不作任何动作。
> - 'warn' : 打印一个RuntimeWarning(通过Python ``warnings`` 模块)。
> - 'raise' : 抛出一个FloatingPointError。
> - 'call' : 调用使用seterrall函数指定的函数。
> - 'print' : 直接将警告打印到 ``stdout``。
> - 'log' : 在seterrall指定的日志对象中记录错误。

可以为所有类型的错误或特定错误设置以下行为：

> - all : 适用于所有数值异常。
> - invalid : 生成nans时。
> - divide : 除以零(整数也是如此！)
> - overflow : 浮点溢出。
> - underflow : 浮点下溢。

请注意，整数除零由相同的机器处理。 这些行为是基于每个线程设置的。

## Examples

```python
>>> oldsettings = np.seterr(all='warn')
>>> np.zeros(5,dtype=np.float32)/0.
invalid value encountered in divide
>>> j = np.seterr(under='ignore')
>>> np.array([1.e-100])**10
>>> j = np.seterr(invalid='raise')
>>> np.sqrt(np.array([-1.]))
FloatingPointError: invalid value encountered in sqrt
>>> def errorhandler(errstr, errflag):
...      print("saw stupid error!")
>>> np.seterrcall(errorhandler)
<function err_handler at 0x...>
>>> j = np.seterr(all='call')
>>> np.zeros(5, dtype=np.int32)/0
FloatingPointError: invalid value encountered in divide
saw stupid error!
>>> j = np.seterr(**oldsettings) # restore previous
...                              # error-handling settings
```

## Interfacing to C

Only a survey of the choices. Little detail on how each works.

1. Bare metal, wrap your own C-code manually.
    > - Plusses:
    >   - Efficient
    >   - No dependencies on other tools
    > - Minuses:
    >   - Lots of learning overhead:
    >   - need to learn basics of Python C API
    >   - need to learn basics of numpy C API
    >   - need to learn how to handle reference counting and love it.
    >   - Reference counting often difficult to get right.
    >   - getting it wrong leads to memory leaks, and worse, segfaults
    >   - API will change for Python 3.0!
1. Cython
    > - Plusses:
    >   - avoid learning C API's
    >   - no dealing with reference counting
    >   - can code in pseudo python and generate C code
    >   - can also interface to existing C code
    >   - should shield you from changes to Python C api
    >   - has become the de-facto standard within the scientific Python community
    >   - fast indexing support for arrays
    > - Minuses:
    >   - Can write code in non-standard form which may become obsolete
    >   - Not as flexible as manual wrapping
1. ctypes
    > - Plusses:
    >   - part of Python standard library
    >   - good for interfacing to existing sharable libraries, particularly Windows DLLs
    >   - avoids API/reference counting issues
    >   - good numpy support: arrays have all these in their ctypes attribute:
    >       ```python
    >       a.ctypes.data              a.ctypes.get_strides
    >       a.ctypes.data_as           a.ctypes.shape
    >       a.ctypes.get_as_parameter  a.ctypes.shape_as
    >       a.ctypes.get_data          a.ctypes.strides
    >       a.ctypes.get_shape         a.ctypes.strides_as
    >       ```
    > - Minuses:
    >   - can't use for writing code to be turned into C extensions, only a wrapper tool.
1. SWIG (automatic wrapper generator)
    > - Plusses:
    >   - around a long time
    >   - multiple scripting language support
    >   - C++ support
    >   - Good for wrapping large (many functions) existing C libraries
    > - Minuses:
    >   - generates lots of code between Python and the C code
    >   - can cause performance problems that are nearly impossible to optimize out
    >   - interface files can be hard to write
    >   - doesn't necessarily avoid reference counting issues or needing to know API's
1. scipy.weave
    > - Plusses:
    >   - can turn many numpy expressions into C code
    >   - dynamic compiling and loading of generated C code
    >   - can embed pure C code in Python module and have weave extract, generate interfaces and compile, etc.
    > - Minuses:
    >   - Future very uncertain: it's the only part of Scipy not ported to Python 3 and is effectively - deprecated in favor of Cython.
1. Psyco
    > - Plusses:
    >   - Turns pure python into efficient machine code through jit-like optimizations
    >   - very fast when it optimizes well
    > - Minuses:
    >   - Only on intel (windows?)
    >   - Doesn't do much for numpy?

## Interfacing to Fortran:

The clear choice to wrap Fortran code is f2py.

Pyfort is an older alternative, but not supported any longer. Fwrap is a newer project that looked promising but isn't being developed any longer.

## Interfacing to C++:

> 1. Cython
> 1. CXX
> 1. Boost.python
> 1. SWIG
> 1. SIP (used mainly in PyQT)