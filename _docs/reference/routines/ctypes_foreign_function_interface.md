# C-Types外部功能接口

- ``numpy.ctypeslib.as_array`` (obj, shape=None) [source](https://github.com/numpy/numpy/blob/v1.14.5/numpy/ctypeslib.py#L422-L436)
从ctypes数组或ctypes POINTER创建一个numpy数组。 numpy数组与ctypes对象共享内存。

如果从ctypes POINTER转换，则必须给出size参数。 如果从ctypes数组转换，则忽略size参数

- ``numpy.ctypeslib.as_ctypes``(obj) [source](http://github.com/numpy/numpy/blob/v1.14.5/numpy/ctypeslib.py#L438-L454)
从numpy数组创建并返回ctypes对象。 实际上接受了暴露__array_interface__的任何东西。

- ``numpy.ctypeslib.ctypes_load_library`` (*args, **kwds) [source](http://github.com/numpy/numpy/blob/v1.14.5/numpy/lib/utils.py#L98-L101)
    不推荐使用``ctypes_load_library``，而是使用``load_library``！

    可以使用 lib = ctypes.cdll [<full_path_name>]加载库

    但是存在跨平台的考虑因素，例如库文件扩展，以及Windows将只加载它找到的具有该名称的第一个库。 为方便起见，NumPy提供load_library函数。

    - **Parameters**:
        - **libname** : str
            库的名称，可以将“lib”作为前缀，但不带扩展名。
        - **loader_path** : str
            提供这个库的路径

    - **Returns**:	
        - **ctypes.cdll[libpath]** : library object
            一个ctypes库对象

    - **Raises**:
        - **OSError**
            如果没有具有预期扩展名的库，或者库有缺陷且无法加载。

- ``numpy.ctypeslib.load_library``(libname, loader_path) [source](https://github.com/numpy/numpy/blob/v1.14.5/numpy/ctypeslib.py#L91-L155)
    可以使用>>> lib = ctypes.cdll [<full_path_name>]加载库

    但是存在跨平台的考虑因素，例如库文件扩展，以及Windows将只加载它找到的具有该名称的第一个库。 为方便起见，NumPy提供load_library函数。

    - **Parameters**:	
        - **libname** : str
            库的名称，可以将“lib”作为前缀，但不带扩展名。
        - **loader_path** : str
            提供这个库的路径

    - **Returns**:
        - **ctypes.cdll[libpath]** : library object
            一个ctypes库对象

    - **Raises**:	
        - **OSError**
            如果没有具有预期扩展名的库，或者库有缺陷且无法加载。

- ``numpy.ctypeslib.ndpointer`` (dtype=None, ndim=None, shape=None, flags=None) [source](http://github.com/numpy/numpy/blob/v1.14.5/numpy/ctypeslib.py#L219-L320)
    数组检查restype / argtypes。

    ndpointer实例用于描述restypes和argtypes规范中的ndarray。 这种方法比使用“POINTER（c_double）”更灵活，因为可以指定几个限制，这些限制在调用ctypes函数时得到验证。 这些包括数据类型，维度数量，形状和标志。 如果给定的数组不满足指定的限制，则引发“TypeError”。

    - **Parameters**:	
        - **dtype** : 数据类型，可选
            数组数据类型
        - **ndim** : int, 可选
            数组维数。
        - **shape** : 整数元组，可选
            阵列形状。
        - **flags** : str的str或tuple
            - 数组标志; 可能是以下一种或多种：
                - C_CONTIGUOUS / C / CONTIGUOUS
                - F_CONTIGUOUS / F / FORTRAN
                - OWNDATA / O
                - WRITEABLE / W
                - ALIGNED / A
                - WRITEBACKIFCOPY / X
                - UPDATEIFCOPY / U

    - **Returns**:	
        - **klass** : ndpointer类型对象
            一个类型对象，它是一个包含dtype，ndim，shape和flags信息的_ndtpr实例。

    - **Raises**:	
        - **TypeError**
            如果给定的数组不满足指定的限制。

### 例子

```python
>>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
...                                                  ndim=1,
...                                                  flags='C_CONTIGUOUS')]
... 
>>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
... 
```