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

请注意，整数除零由相同的处理器处理，且这些行为是基于每个线程设置的。

## 例子

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

## 与C相关的接口

只针对下列选项进行阐述，阐述每一项工作原理的部分细节。

1. 不借助任何工具, 手动打包你的C语言代码。
    > - 加分项（优点）:
    >   - 高效
    >   - 不依赖其他的工具
    > - 减分项（缺点）:
    >   - 大量的学习开销。
    >   - 需要学习Python C API的基础知识。
    >   - 需要学习numpy C API的基础知识。
    >   - 需要学习如何处理引用计数并且熟练掌握。
    >   - 引用计数通常很难做到正确。
    >   - 错误导致内存泄漏，更糟糕的是段错误。
    >   - Python 3.0的API变化会很大。
1. Cython
    > - 加分项（优点）:
    >   - 避免学习C API。
    >   - 不需要处理引用计数。
    >   - 可以在伪python中编码并生成C代码。
    >   - 还可以接入现有的C代码的接口。
    >   - 即便是Python的api改变了也不会对你有任何影响。
    >   - 已经成为了科学Python社区中的权威标准。
    >   - 对数组的快速索引的支持。
    > - 减分项（缺点）:
    >   - 可以用非标准形式编写可能过时的代码。
    >   - 不如手动打包灵活。
1. ctypes
    > - 加分项（优点）:
    >   - Python标准库的一部分
    >   - 适用于连接现有的可共享库，尤其是Windows DLL
    >   - 避免 API/reference 的引用计数问题。
    >   - 良好的numpy支持：数组在ctypes属性中包含所有这些：
    >       ```python
    >       a.ctypes.data              a.ctypes.get_strides
    >       a.ctypes.data_as           a.ctypes.shape
    >       a.ctypes.get_as_parameter  a.ctypes.shape_as
    >       a.ctypes.get_data          a.ctypes.strides
    >       a.ctypes.get_shape         a.ctypes.strides_as
    >       ```
    > - 减分项（缺点）:
    >   - 不能把编写代码转换为C的扩展，只能用于打包工具。
1. SWIG (自动打包工具)
    > - 加分项（优点）:
    >   - 耗时长。
    >   - 多脚本语言支持
    >   - C++ 支持
    >   - 适用于打包大型的（包含许多函数）现有C库。
    > - 减分项（缺点）:
    >   - 在Python和C代码之间生成大量代码
    >   - 可能导致几乎无法优化的性能问题
    >   - 接口文件很难写
    >   - 不一定避免引用计数问题　或　必须要了解大部分的API。
1. scipy.weave
    > - 加分项（优点）:
    >   - 可以将许多numpy表达式转换为C代码
    >   - 动态编译和加载生成的C代码
    >   - 可在Python模块中嵌入纯C代码，并具有编织、提取、生成接口和编译等功能。
    > - 减分项（缺点）:
    >   - 未来非常不确定：它是Scipy中唯一没有移植到Python 3的部分，并且实际上已被弃用且不支持Cython。
1. Psyco
    > - 加分项（优点）:
    >   - 通过类似jit的优化将纯python转换为高效的机器代码
    >   - 当它优化得很好时非常快
    > - 减分项（缺点）:
    >   - 只在intel（也许是只能在windows上）上
    >   - 对numpy没有多大作用？

## 与Fortran的接口：

包装Fortran代码的明确选择是f2py。

Pyfort是一个较落后的选择，而且很长的时间已经没有人维护了。Fwrap是一个看起来很有希望但已经流产了的项目。

## 与C++的接口:

> 1. Cython
> 1. CXX
> 1. Boost.python
> 1. SWIG
> 1. SIP (主要用于PyQT)