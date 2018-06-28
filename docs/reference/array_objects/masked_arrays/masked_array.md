# MaskedArray类

## class ``numpy.ma.MaskedArray``[source]

> A subclass of ndarray designed to manipulate numerical arrays with missing data.

An instance of MaskedArray can be thought as the combination of several elements:

- The data, as a regular numpy.ndarray of any shape or datatype (the data).
- A boolean mask with the same shape as the data, where a True value indicates that the corresponding element of the data is invalid. The special value nomask is also acceptable for arrays without named fields, and indicates that no data is invalid.
- A fill_value, a value that may be used to replace the invalid entries in order to return a standard numpy.ndarray.

## Attributes and properties of masked arrays

另见：

> Array Attributes

### ``MaskedArray.data``

Returns the underlying data, as a view of the masked array. If the underlying data is a subclass of numpy.ndarray, it is returned as such.

```python
>>> x = ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
>>> x.data
matrix([[1, 2],
        [3, 4]])
```

The type of the data can be accessed through the ``baseclass`` attribute.

### ``MaskedArray.mask``

Returns the underlying mask, as an array with the same shape and structure as the data, but where all fields are atomically booleans. A value of ``True`` indicates an invalid entry.

### ``MaskedArray.recordmask``

Returns the mask of the array if it has no named fields. For structured arrays, returns a ndarray of booleans where entries are ``True`` if all the fields are masked, ``False`` otherwise:

```python
>>> x = ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
...        dtype=[('a', int), ('b', int)])
>>> x.recordmask
array([False, False,  True, False, False])
```

### ``MaskedArray.fill_value``

Returns the value used to fill the invalid entries of a masked array. The value is either a scalar (if the masked array has no named fields), or a 0-D ndarray with the same ``dtype`` as the masked array if it has named fields.

The default filling value depends on the datatype of the array:

datatype | default
---|---
bool | True
int | 999999
float | 1.e20
complex | 1.e20+0j
object | ‘?’
string | ‘N/A’

### ``MaskedArray.baseclass``

Returns the class of the underlying data.

```python
>>> x =  ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 0], [1, 0]])
>>> x.baseclass
<class 'numpy.matrixlib.defmatrix.matrix'>
```

### ``MaskedArray.sharedmask``

Returns whether the mask of the array is shared between several masked arrays. If this is the case, any modification to the mask of one array will be propagated to the others.

### ``MaskedArray.hardmask``

Returns whether the mask is hard (True) or soft (False). When the mask is hard, masked entries cannot be unmasked.

As ``MaskedArray`` is a subclass of ``ndarray``, a masked array also inherits all the attributes and properties of a ndarray instance.

method | desc
---|---
MaskedArray.base | Base object if memory is from some other object.
MaskedArray.ctypes | An object to simplify the interaction of the array with the ctypes module.
MaskedArray.dtype | Data-type of the array’s elements.
MaskedArray.flags | Information about the memory layout of the array.
MaskedArray.itemsize | Length of one array element in bytes.
MaskedArray.nbytes | Total bytes consumed by the elements of the array.
MaskedArray.ndim | Number of array dimensions.
MaskedArray.shape | Tuple of array dimensions.
MaskedArray.size | Number of elements in the array.
MaskedArray.strides | Tuple of bytes to step in each dimension when traversing an array.
MaskedArray.imag | Imaginary part.
MaskedArray.real | Real part
MaskedArray.flat | Flat version of the array.
MaskedArray.__array_priority__ | -