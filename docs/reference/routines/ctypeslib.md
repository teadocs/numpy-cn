---
meta:
  - name: keywords
    content: C-Types,cnumpy,外部接口
  - name: description
    content: 从ctypes数组或指针创建numpy数组。numpy数组与ctypes对象共享内存。如果从ctypes指针转换，则必须给定Shape参数。如果从ctypes数组转换，则忽略shape参数
---

# C-Types外部函数接口（``numpy.ctypeslib``）

  从ctypes数组或指针创建numpy数组。

  numpy数组与ctypes对象共享内存。

  如果从ctypes指针转换，则必须给定Shape参数。
  如果从ctypes数组转换，则忽略shape参数

- ``numpy.ctypeslib.``as_ctypes(*obj*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L526-L541)[¶](#numpy.ctypeslib.as_ctypes)

  从numpy数组创建并返回ctypes对象。
  实际上，任何公开 \_\_array_interface__的内容都是可以接受的。

- ``numpy.ctypeslib.``as_ctypes_type(*dtype*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L464-L502)

  将数据类型转换为ctype类型。

  **参数：**

  类型 | 描述
  ---|---
  dtype : dtype | 转换的dtype

  **返回：**

  类型 | 描述
  ---|---
  ctype | ctype标量，并集，数组或结构

  **异常：**

  类型 | 描述
  ---|---
  NotImplementedError | 如果无法进行转换

  ::: tip 注意

  此功能不会在两个方向上无损地往返。

  - ``np.dtype(as_ctypes_type(dt))`` 将会:
    - 插入填充字段
    - 按偏移量对要排序的字段进行重新排序
    - 放弃字段标题

  - ``as_ctypes_type(np.dtype(ctype))`` 将会:
    - 丢弃 [``ctypes.Structures``](https://docs.python.org/dev/library/ctypes.html#ctypes.Structure) 和 [``ctypes.Union``](https://docs.python.org/dev/library/ctypes.html#ctypes.Union) 的类名
    - 将单元素[``ctypes.Unions``](https://docs.python.org/dev/library/ctypes.html#ctypes.Union) 转换为单元素 [``ctypes.Structures``](https://docs.python.org/dev/library/ctypes.html#ctypes.Structure)。
    - 插入填充字段

  :::

- ``numpy.ctypeslib.``ctypes_load_library(*\*args*, *\*\*kwds*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/lib/utils.py#L98-L101)
  
  ``ctypes_load_library`` 已弃用，请改用 ``load_library``！
  
  可以使用 \>\>\> lib = ctypes.cdll[\<full_path_name\>] \# doctest: +SKIP 加载库

  但是有跨平台的考虑，例如库文件扩展名，
  此外，Windows将只加载它找到的具有该名称的第一个库。

  为方便起见，NumPy提供了load_library函数。

  **参数：**
  类型 | 描述
  ---|---
  libname : str | 库的名称，可以使用‘lib’作为前缀，但不带扩展名。
  loader_path : str | 可以找到库的路径。

  **返回：**
  类型 | 描述
  ---|---
  ctypes.cdll[libpath] : library object | 一个 ctypes 库对象

  **异常：**
  类型 | 描述
  ---|---
  OSError | 如果没有具有预期扩展名的库，或者库有缺陷且无法加载。

- ``numpy.ctypeslib.``load_library(*libname*, *loader_path*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L93-L157)

  可以使用 \>\>\> lib = ctypes.cdll[\<full_path_name\>] \# doctest: +SKIP 加载库

  但是有跨平台的考虑，例如库文件扩展名，
  此外，Windows将只加载它找到的具有该名称的第一个库。

  为方便起见，NumPy提供了load_library函数。

  **参数：**
  类型 | 描述
  ---|---
  libname : str | 库的名称，可以使用‘lib’作为前缀，但不带扩展名。
  loader_path : str | 可以找到库的路径。

  **返回：**
  类型 | 描述
  ---|---
  ctypes.cdll[libpath] : library object | 一个 ctypes 库对象

  **异常：**
  类型 | 描述
  ---|---
  OSError | 如果没有具有预期扩展名的库，或者库有缺陷且无法加载。

- ``numpy.ctypeslib.``ndpointer(*dtype=None*, *ndim=None*, *shape=None*, *flags=None*)[[点击查看源码]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L231-L346)

  数组检查restype/argtypes。

  在restypes和argtypes规范中，ndpoint实例用于描述ndarray。
  这种方法比使用 ``POINTER(c_double)`` 更灵活，因为可以指定几个限制，
  这些限制在调用ctypes函数时进行验证。这些包括数据类型、维度数量、形状和标志。
  如果给定的数组不满足指定的限制，则引发``TypeError``。

  **参数：**

  类型 | 描述
  ---|---
  dtype : data-type, optional | 数组数据类型。
  ndim : int, optional | 数组维数。
  shape : tuple of ints, optional | 数组形状。
  flags : str or tuple of str | A数组标志；可能是以下一项或多项：

  flags 的可能项：
    - C_CONTIGUOUS / C / CONTIGUOUS
    - F_CONTIGUOUS / F / FORTRAN
    - OWNDATA / O
    - WRITEABLE / W
    - ALIGNED / A
    - WRITEBACKIFCOPY / X
    - UPDATEIFCOPY / U

  **返回：**
  类型 | 描述
  ---|---
  klass : ndpointer type object | 类型对象，它是一个_ndtpr实例，包含dtype，ndim，shape和flags信息。

  **异常：**
  类型 | 描述
  ---|---
  TypeError | 如果给定数组不满足指定限制。

  **示例：**

  ``` python
  >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
  ...                                                  ndim=1,
  ...                                                  flags='C_CONTIGUOUS')]
  ... #doctest: +SKIP
  >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
  ... #doctest: +SKIP
  ```
