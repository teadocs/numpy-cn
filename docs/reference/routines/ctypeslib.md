# C-Types Foreign Function Interface (``numpy.ctypeslib``)


``numpy.ctypeslib.````as_array``(*obj*, *shape=None*)[[source]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L505-L523)[¶](#numpy.ctypeslib.as_array)

Create a numpy array from a ctypes array or POINTER.

The numpy array shares the memory with the ctypes object.

The shape parameter must be given if converting from a ctypes POINTER.
The shape parameter is ignored if converting from a ctypes array


``numpy.ctypeslib.````as_ctypes``(*obj*)[[source]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L526-L541)[¶](#numpy.ctypeslib.as_ctypes)

Create and return a ctypes object from a numpy array.  Actually
anything that exposes the __array_interface__ is accepted.


``numpy.ctypeslib.````as_ctypes_type``(*dtype*)[[source]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L464-L502)[¶](#numpy.ctypeslib.as_ctypes_type)

Convert a dtype into a ctypes type.

Parameters:
---
dtype : dtype

The dtype to convert


Returns:
ctype

A ctype scalar, union, array, or struct


Raises:
NotImplementedError

If the conversion is not possible

Notes

This function does not losslessly round-trip in either direction.

``np.dtype(as_ctypes_type(dt))`` will:

- insert padding fields
- reorder fields to be sorted by offset
- discard field titles

``as_ctypes_type(np.dtype(ctype))`` will:

- discard the class names of [``ctypes.Structure``](https://docs.python.org/dev/library/ctypes.html#ctypes.Structure)s and
[``ctypes.Union``](https://docs.python.org/dev/library/ctypes.html#ctypes.Union)s
- convert single-element [``ctypes.Union``](https://docs.python.org/dev/library/ctypes.html#ctypes.Union)s into single-element
[``ctypes.Structure``](https://docs.python.org/dev/library/ctypes.html#ctypes.Structure)s
- insert padding fields


``numpy.ctypeslib.````ctypes_load_library``(**args*, ***kwds*)[[source]](https://github.com/numpy/numpy/blob/master/numpy/lib/utils.py#L98-L101)[¶](#numpy.ctypeslib.ctypes_load_library)

[``ctypes_load_library``](#numpy.ctypeslib.ctypes_load_library) is deprecated, use [``load_library``](#numpy.ctypeslib.load_library) instead!

It is possible to load a library using 
>>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP

But there are cross-platform considerations, such as library file extensions,
plus the fact Windows will just load the first library it finds with that name.  
NumPy supplies the load_library function as a convenience.

Parameters:
---
libname : str

Name of the library, which can have ‘lib’ as a prefix, but without an extension.

loader_path : str

Where the library can be found.


Returns:
ctypes.cdll[libpath] : library object

A ctypes library object


Raises:
OSError

If there is no library with the expected extension, or the library is defective and cannot be loaded.


``numpy.ctypeslib.````load_library``(*libname*, *loader_path*)[[source]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L93-L157)[¶](#numpy.ctypeslib.load_library)

It is possible to load a library using 
>>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP

But there are cross-platform considerations, such as library file extensions,
plus the fact Windows will just load the first library it finds with that name.  
NumPy supplies the load_library function as a convenience.

Parameters:
---
libname : str

Name of the library, which can have ‘lib’ as a prefix, but without an extension.

loader_path : str

Where the library can be found.


Returns:
ctypes.cdll[libpath] : library object

A ctypes library object


Raises:
OSError

If there is no library with the expected extension, or the library is defective and cannot be loaded.


``numpy.ctypeslib.````ndpointer``(*dtype=None*, *ndim=None*, *shape=None*, *flags=None*)[[source]](https://github.com/numpy/numpy/blob/master/numpy/ctypeslib.py#L231-L346)[¶](#numpy.ctypeslib.ndpointer)

Array-checking restype/argtypes.

An ndpointer instance is used to describe an ndarray in restypes
and argtypes specifications.  This approach is more flexible than
using, for example, ``POINTER(c_double)``, since several restrictions
can be specified, which are verified upon calling the ctypes function.
These include data type, number of dimensions, shape and flags.  If a
given array does not satisfy the specified restrictions,
a ``TypeError`` is raised.

Parameters:
---
dtype : data-type, optional

Array data-type.

ndim : int, optional

Number of array dimensions.

shape : tuple of ints, optional

Array shape.

flags : str or tuple of str

Array flags; may be one or more of:

C_CONTIGUOUS / C / CONTIGUOUS
F_CONTIGUOUS / F / FORTRAN
OWNDATA / O
WRITEABLE / W
ALIGNED / A
WRITEBACKIFCOPY / X
UPDATEIFCOPY / U

Returns:
klass : ndpointer type object

A type object, which is an _ndtpr instance containing dtype, ndim, shape and flags information.


Raises:
TypeError

If a given array does not satisfy the specified restrictions.

Examples

``` python
>>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
...                                                  ndim=1,
...                                                  flags='C_CONTIGUOUS')]
... #doctest: +SKIP
>>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
... #doctest: +SKIP
```