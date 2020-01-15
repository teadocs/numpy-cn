---
meta:
  - name: keywords
    content: 自定义数组,容器
  - name: description
    content: NumPy 的分派机制(在numpy版本v1.16中引入)是编写与numpy API兼容并提供numpy功能的自定义实现的自定义N维数组容器的推荐方法。
---

# 编写自定义数组容器

NumPy 的分派机制(在numpy版本v1.16中引入)是编写与numpy API兼容并提供numpy功能的自定义实现的自定义N维数组容器的推荐方法。
应用包括 [dask](http://dask.pydata.org) 数组(分布在多个节点上的N维数组) 
和 [cupy](https://docs-cupy.chainer.org/en/stable/) 数组(GPU上的N维数组)。

为了获得编写自定义数组容器的感觉，我们将从一个简单的示例开始，该示例具有相当狭窄的实用程序，但说明了所涉及的概念。

``` python
>>> import numpy as np
>>> class DiagonalArray:
...     def __init__(self, N, value):
...         self._N = N
...         self._i = value
...     def __repr__(self):
...         return f"{self.__class__.__name__}(N={self._N}, value={self._i})"
...     def __array__(self):
...         return self._i * np.eye(self._N)
...
```

我们的自定义数组可以实例化，如下所示：

``` python
>>> arr = DiagonalArray(5, 1)
>>> arr
DiagonalArray(N=5, value=1)
```

我们可以使用 [``numpy.array``](https://numpy.org/devdocs/reference/generated/numpy.array.html#numpy.array) 或 [``numpy.asarray``](https://numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray), 转换为numpy数组，这将调用它的 ``__array__`` 方法来获得标准 ``numpy.ndarray``。

``` python
>>> np.asarray(arr)
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
```

如果我们使用 numpy 函数对 ``arr`` 进行操作，numpy 将再次使用 ``__array__``接口将其转换为数组，然后以通常的方式应用该函数。

``` python
>>> np.multiply(arr, 2)
array([[2., 0., 0., 0., 0.],
       [0., 2., 0., 0., 0.],
       [0., 0., 2., 0., 0.],
       [0., 0., 0., 2., 0.],
       [0., 0., 0., 0., 2.]])
```

注意，返回类型是标准 ``numpy.ndarray``。

``` python
>>> type(arr)
numpy.ndarray
```

我们如何通过此函数传递我们的自定义数组类型？Numpy允许类指示它希望通过交互 ``__array_ufunc__`` 和 ``__array_function__`` 以自定义方式处理计算。
让我们一次拿一个，从 ``__array_ufunc__`` 开始。
此方法涵盖 [Universal functions (ufunc)](/reference/ufuncs.html#ufuncs)，
这是一类函数，包括例如 [``numpy.multiply``](/reference/generated/numpy.multiply.html#numpy.multiply) 
和 [``numpy.sin``](/reference/generated/numpy.sin.html#numpy.sin)。

``_array_ufunc_`` 获得：

- ``ufunc``, 一个类似 ``numpy.multiply`` 的函数
- ``method``，一个字符串，区分 ``numpy.multiply(...)``。
以及``numpy.multiy.outer``、``numpy.multiy.accumate``等变体。对于常见情况，``numpy.multiply(...)``，``method='__call__'``。
- ``inputs``, 可能是不同类型的混合
- ``kwargs``, 传递给函数的关键字参数

对于这个例子，我们将只处理方法 ``'__call__``。

``` python
>>> from numbers import Number
>>> class DiagonalArray:
...     def __init__(self, N, value):
...         self._N = N
...         self._i = value
...     def __repr__(self):
...         return f"{self.__class__.__name__}(N={self._N}, value={self._i})"
...     def __array__(self):
...         return self._i * np.eye(self._N)
...     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
...         if method == '__call__':
...             N = None
...             scalars = []
...             for input in inputs:
...                 if isinstance(input, Number):
...                     scalars.append(input)
...                 elif isinstance(input, self.__class__):
...                     scalars.append(input._i)
...                     if N is not None:
...                         if N != self._N:
...                             raise TypeError("inconsistent sizes")
...                     else:
...                         N = self._N
...                 else:
...                     return NotImplemented
...             return self.__class__(N, ufunc(*scalars, **kwargs))
...         else:
...             return NotImplemented
...
```

现在让我们的自定义数组类型通过numpy的函数。

``` python
>>> arr = DiagonalArray(5, 1)
>>> np.multiply(arr, 3)
DiagonalArray(N=5, value=3)
>>> np.add(arr, 3)
DiagonalArray(N=5, value=4)
>>> np.sin(arr)
DiagonalArray(N=5, value=0.8414709848078965)
```

此时 ``arr + 3`` 不起作用。

``` python
>>> arr + 3
TypeError: unsupported operand type(s) for *: 'DiagonalArray' and 'int'
```

为了支持它，我们需要定义Python接口 ``__add__``， ``__lt__`` 等，以便调度到相应的ufunc。 我们可以通过继承mixin [``NDArrayOperatorsMixin``](https://numpy.org/devdocs/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin) 来方便地实现这一点。

``` python
>>> import numpy.lib.mixins
>>> class DiagonalArray(numpy.lib.mixins.NDArrayOperatorsMixin):
...     def __init__(self, N, value):
...         self._N = N
...         self._i = value
...     def __repr__(self):
...         return f"{self.__class__.__name__}(N={self._N}, value={self._i})"
...     def __array__(self):
...         return self._i * np.eye(self._N)
...     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
...         if method == '__call__':
...             N = None
...             scalars = []
...             for input in inputs:
...                 if isinstance(input, Number):
...                     scalars.append(input)
...                 elif isinstance(input, self.__class__):
...                     scalars.append(input._i)
...                     if N is not None:
...                         if N != self._N:
...                             raise TypeError("inconsistent sizes")
...                     else:
...                         N = self._N
...                 else:
...                     return NotImplemented
...             return self.__class__(N, ufunc(*scalars, **kwargs))
...         else:
...             return NotImplemented
...
```

``` python
>>> arr = DiagonalArray(5, 1)
>>> arr + 3
DiagonalArray(N=5, value=4)
>>> arr > 0
DiagonalArray(N=5, value=True)
```

现在让我们来解决 ``__array_function__``。 我们将创建将 numpy 函数映射到我们的自定义变体的 dict。

``` python
>>> HANDLED_FUNCTIONS = {}
>>> class DiagonalArray(numpy.lib.mixins.NDArrayOperatorsMixin):
...     def __init__(self, N, value):
...         self._N = N
...         self._i = value
...     def __repr__(self):
...         return f"{self.__class__.__name__}(N={self._N}, value={self._i})"
...     def __array__(self):
...         return self._i * np.eye(self._N)
...     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
...         if method == '__call__':
...             N = None
...             scalars = []
...             for input in inputs:
...                 # In this case we accept only scalar numbers or DiagonalArrays.
...                 if isinstance(input, Number):
...                     scalars.append(input)
...                 elif isinstance(input, self.__class__):
...                     scalars.append(input._i)
...                     if N is not None:
...                         if N != self._N:
...                             raise TypeError("inconsistent sizes")
...                     else:
...                         N = self._N
...                 else:
...                     return NotImplemented
...             return self.__class__(N, ufunc(*scalars, **kwargs))
...         else:
...             return NotImplemented
...    def __array_function__(self, func, types, args, kwargs):
...        if func not in HANDLED_FUNCTIONS:
...            return NotImplemented
...        # Note: this allows subclasses that don't override
...        # __array_function__ to handle DiagonalArray objects.
...        if not all(issubclass(t, self.__class__) for t in types):
...            return NotImplemented
...        return HANDLED_FUNCTIONS[func](*args, **kwargs)
...
```

一个便捷的模式是定义一个可用于向 ``HANDLED_FUNCTIONS`` 添加函数的装饰器 ``实现``。

``` python
>>> def implements(np_function):
...    "Register an __array_function__ implementation for DiagonalArray objects."
...    def decorator(func):
...        HANDLED_FUNCTIONS[np_function] = func
...        return func
...    return decorator
...
```

现在我们为 ``DiagonalArray`` 编写numpy函数的实现。
为了完整性，为了支持使用 ``arr.sum()``，
添加一个调用 ``numpy.sum(self)`` 的方法 ``sum``，对于 ``mean`` 来说也是一样的。

``` python
>>> @implements(np.sum)
... def sum(a):
...     "Implementation of np.sum for DiagonalArray objects"
...     return arr._i * arr._N
...
>>> @implements(np.mean)
... def sum(a):
...     "Implementation of np.mean for DiagonalArray objects"
...     return arr._i / arr._N
...
>>> arr = DiagonalArray(5, 1)
>>> np.sum(arr)
5
>>> np.mean(arr)
0.2
```

如果用户尝试使用 ``HANDLED_FUNCTIONS`` 中未包含的任何numpy函数，
则numpy将引发 ``TypeError``，表示不支持此操作。
例如，连接两个 ``DiagonalArrays`` 不会产生另一个对角线数组，因此不支持它。

``` python
>>> np.concatenate([arr, arr])
TypeError: no implementation found for 'numpy.concatenate' on types that implement __array_function__: [<class '__main__.DiagonalArray'>]
```

另外，我们的 ``sum`` 和 ``mean`` 实现不接受numpy实现的可选参数。

``` python
>>> np.sum(arr, axis=0)
TypeError: sum() got an unexpected keyword argument 'axis'
```

用户总是可以选择使用 ``numpy.asarray`` 转换为普通的 [``numpy.asarray``](/reference/generated/numpy.asarray.html#numpy.asarray)，并使用标准的numpy。

``` python
>>> np.concatenate([np.asarray(arr), np.asarray(arr)])
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.],
       [1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
```

有关自定义数组容器的更完整工作示例，请参阅[dask源代码](https://github.com/dask/dask)和[cupy源代码](https://github.com/cupy/cupy)。

另外可以看一下 [NEP 18](http://www.numpy.org/neps/nep-0018-array-function-protocol.html)。
