---
meta:
  - name: keywords
    content: NumPy 其他杂项
  - name: description
    content: 在 NumPy 中定义的特殊值可以通过：nan，inf，NaNs 可以用作简陋的占位类型（如果你并不在乎初始的值是什么的话）...
---

# 其他杂项

## IEEE 754 浮点特殊值

在 NumPy 中定义的特殊值可以通过：nan，inf，

NaNs 可以用作简陋的占位类型（如果你并不在乎初始的值是什么的话）

注意：不能使用相等来测试 NaN。例如：

``` python
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


``` python
isinf():    True if value is inf
isfinite(): True if not nan or inf
nan_to_num(): Map nan to 0, inf to max float, -inf to min float
```

除了从结果中排除nans之外，以下内容对应于常用函数：

``` python
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

## NumPy 如何处理数字异常的

默认值为 ``Warn`` 表示无效、``Divide``和溢出，``Ignore``表示下溢。
但是这是可以更改的，并且可以针对不同种类的异常单独设置。不同的行为包括：

- 'ignore'：发生异常时不采取任何措施。
- 'warn'：打印 *RuntimeWarning* （通过Python [``warnings``](https://docs.python.org/dev/library/warnings.html#module-warnings)模块）。
- 'raise'：引发 *FloatingPointError* 。
- 'call'：调用使用 *seterrcall* 函数指定的函数。
- 'print'：直接打印警告``stdout``。
- 'log'：在 *seterrcall* 指定的Log对象中记录错误。

可以针对各种错误或特定错误设置这些行为：

- all：适用于所有数字异常
- 无效：生成NaN时
- 除以：除以零（对于整数！）
- 溢出：浮点溢出
- 下溢：浮点下溢

注意，整数除零由相同的机器处理。这些行为是基于每个线程设置的。

## 示例

``` python
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

## 连接到 C 的方式

只针对下列选项进行阐述，阐述每一项工作原理的部分细节。

1. 不借助任何工具, 手动打包你的C语言代码。
    - 加分项（优点）:
        - 高效
        - 不依赖于其他工具
    - 减分项（缺点）:
        - 大量的学习开销：
        - 需要学习Python C API的基础知识
        - 需要学习numpy C API的基础知识
        - 需要学习如何处理引用计数并喜欢它。
        - 引用计数通常很难正确。
        - 错误导致内存泄漏，更糟糕的是段错误。
        - Python可能会改变API！
1. Cython
    - 加分项（优点）:
      - 避免学习C API
      - 没有涉及引用计数
      - 可以在伪python中编码并生成C代码
      - 也可以与现有的C代码接口
      - 应该保护你免受Python C api的更改
      - 已经成为科学Python社区中事实上的标准
      - 对数组的快速索引支持
    - 减分项（缺点）:
      - 可以用非标准形式编写可能过时的代码
      - 不如手动包装灵活
1. ctypes
    - 加分项（优点）:
        - Python标准库的一部分
        - 适用于连接现有的可共享库，尤其是Windows DLL
        - 避免API /引用计数问题
        - 良好的numpy支持：数组在ctypes属性中包含所有这些：

        ``` python
        a.ctypes.data              a.ctypes.get_strides
        a.ctypes.data_as           a.ctypes.shape
        a.ctypes.get_as_parameter  a.ctypes.shape_as
        a.ctypes.get_data          a.ctypes.strides
        a.ctypes.get_shape         a.ctypes.strides_as
        ```
    - 减分项（缺点）:
        - 不能用于编写代码转换为C扩展，只能用于包装工具。
1. SWIG（自动包装发生器）
    - 加分项（优点）:
        - 很长一段时间
        - 多脚本语言支持
        - C ++支持
        - 适用于包装大型（许多功能）现有C库
    - 减分项（缺点）:
      - 在Python和C代码之间生成大量代码
      - 可能导致几乎无法优化的性能问题
      - 接口文件很难写
      - 不一定避免引用计数问题或需要知道API
1. scipy.weave
    - 加分项（优点）:
      - 可以将许多numpy表达式转换为C代码
      - 动态编译和加载生成的C代码
      - 可以在Python模块中嵌入纯C代码，并编织提取，生成接口和编译等。
    - 减分项（缺点）:
      - 未来非常不确定：它是Scipy中唯一没有移植到Python 3的部分，并且有效地弃用了Cython。
1. Psyco
    - 加分项（优点）:
      - 通过类似jit的优化将纯python转换为高效的机器代码
      - 当它优化得很好时非常快
    - 减分项（缺点）:
      - 只在intel（windows？）上
      - 对numpy没有多大作用？

## Fortran 的接口：

包装 Fortran 代码的明确选择是 [f2py](/f2py/)。

Pyfort是一个较旧的选择，但不再支持。Fwrap是一个看起来很有希望但不再开发的新项目。

## 连接到 C++ 有以下几个方式：

1. Cython
1. CXX
1. Boost.Python
1. SWIG
1. SIP（主要用于PyQT）
