# The N-dimensional array (``ndarray``)

An [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) is a (usually fixed-size) multidimensional
container of items of the same type and size. The number of dimensions
and items in an array is defined by its [``shape``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape),
which is a [``tuple``](https://docs.python.org/dev/library/stdtypes.html#tuple) of *N* non-negative integers that specify the
sizes of each dimension. The type of items in the array is specified by
a separate [data-type object (dtype)](arrays.dtypes.html#arrays-dtypes), one of which
is associated with each ndarray.

As with other container objects in Python, the contents of an
[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) can be accessed and modified by [indexing or
slicing](arrays.indexing.html#arrays-indexing) the array (using, for example, *N* integers),
and via the methods and attributes of the [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray).

Different [``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) can share the same data, so that
changes made in one [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) may be visible in another. That
is, an ndarray can be a *“view”* to another ndarray, and the data it
is referring to is taken care of by the *“base”* ndarray. ndarrays can
also be views to memory owned by Python [``strings``](https://docs.python.org/dev/library/stdtypes.html#str) or
objects implementing the ``buffer`` or [array](arrays.interface.html#arrays-interface) interfaces.

**Example:**

A 2-dimensional array of size 2 x 3, composed of 4-byte integer
elements:

``` python
>>> x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
>>> type(x)
<type 'numpy.ndarray'>
>>> x.shape
(2, 3)
>>> x.dtype
dtype('int32')
```

The array can be indexed using Python container-like syntax:

``` python
>>> # The element of x in the *second* row, *third* column, namely, 6.
>>> x[1, 2]
```

For example [slicing](arrays.indexing.html#arrays-indexing) can produce views of the array:

``` python
>>> y = x[:,1]
>>> y
array([2, 5])
>>> y[0] = 9 # this also changes the corresponding element in x
>>> y
array([9, 5])
>>> x
array([[1, 9, 3],
       [4, 5, 6]])
```

## Constructing arrays

New arrays can be constructed using the routines detailed in
[Array creation routines](routines.array-creation.html#routines-array-creation), and also by using the low-level
[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) constructor:

method | description
---|---
[ndarray](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)(shape[, dtype, buffer, offset, …]) | An array object represents a multidimensional, homogeneous array of fixed-size items.


## Indexing arrays

Arrays can be indexed using an extended Python slicing syntax,
``array[selection]``.  Similar syntax is also used for accessing
fields in a [structured data type](https://numpy.org/devdocs/glossary.html#term-structured-data-type).

::: tip See also

[Array Indexing](arrays.indexing.html#arrays-indexing).

:::

## Internal memory layout of an ndarray

An instance of class [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) consists of a contiguous
one-dimensional segment of computer memory (owned by the array, or by
some other object), combined with an indexing scheme that maps *N*
integers into the location of an item in the block.  The ranges in
which the indices can vary is specified by the [``shape``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape) of the array. How many bytes each item takes and how
the bytes are interpreted is defined by the [data-type object](arrays.dtypes.html#arrays-dtypes) associated with the array.

A segment of memory is inherently 1-dimensional, and there are many
different schemes for arranging the items of an *N*-dimensional array
in a 1-dimensional block. NumPy is flexible, and [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)
objects can accommodate any *strided indexing scheme*. In a strided
scheme, the N-dimensional index <img class="math" src="/static/images/math/edb5f8b6064d0edc2bc57a1714249e0eae1a33e3.svg" alt="(n_0, n_1, ..., n_{N-1})"/> corresponds to the offset (in bytes):

<center>
<img src="/static/images/math/1388948b609ce9a1d9ae0380d361628d6b385812.svg" alt="n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k"/>
</center>

from the beginning of the memory block associated with the
array. Here,  are integers which specify the [``strides``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.strides.html#numpy.ndarray.strides) of the array. The [column-major](https://numpy.org/devdocs/glossary.html#term-column-major) order (used,
for example, in the Fortran language and in *Matlab*) and
[row-major](https://numpy.org/devdocs/glossary.html#term-row-major) order (used in C) schemes are just specific kinds of
strided scheme, and correspond to memory that can be *addressed* by the strides:

<center>
<img src="/static/images/math/af328186eedd2e4200b34e0e6a31acae4dbc9d20.svg" alt="n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k"/>
</center>

where <img class="math" src="/static/images/math/5e6cfb16a1d0565098e1a35072ef6fbfef092db3.svg" alt="d_j"/> *= self.shape[j]*.

Both the C and Fortran orders are [contiguous](https://docs.python.org/dev/glossary.html#term-contiguous), *i.e.,*
single-segment, memory layouts, in which every part of the
memory block can be accessed by some combination of the indices.

While a C-style and Fortran-style contiguous array, which has the corresponding
flags set, can be addressed with the above strides, the actual strides may be
different. This can happen in two cases:

1. If ``self.shape[k] == 1`` then for any legal index ``index[k] == 0``.
This means that in the formula for the offset  and thus
 and the value of  *= self.strides[k]* is
arbitrary.
1. If an array has no elements (``self.size == 0``) there is no legal
index and the strides are never used. Any array with no elements may be
considered C-style and Fortran-style contiguous.

Point 1. means that ``self`` and ``self.squeeze()`` always have the same
contiguity and ``aligned`` flags value. This also means
that even a high dimensional array could be C-style and Fortran-style
contiguous at the same time.

An array is considered aligned if the memory offsets for all elements and the
base offset itself is a multiple of *self.itemsize*. Understanding
*memory-alignment* leads to better performance on most hardware.

::: tip Note

Points (1) and (2) are not yet applied by default. Beginning with
NumPy 1.8.0, they are applied consistently only if the environment
variable ``NPY_RELAXED_STRIDES_CHECKING=1`` was defined when NumPy
was built. Eventually this will become the default.

You can check whether this option was enabled when your NumPy was
built by looking at the value of ``np.ones((10,1),
order='C').flags.f_contiguous``. If this is ``True``, then your
NumPy has relaxed strides checking enabled.

:::

::: danger Warning

It does *not* generally hold that ``self.strides[-1] == self.itemsize``
for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for
Fortran-style contiguous arrays is true.

:::

Data in new [``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) is in the [row-major](https://numpy.org/devdocs/glossary.html#term-row-major)
(C) order, unless otherwise specified, but, for example, [basic
array slicing](arrays.indexing.html#arrays-indexing) often produces [views](https://numpy.org/devdocs/glossary.html#term-view)
in a different scheme.

::: tip Note

Several algorithms in NumPy work on arbitrarily strided arrays.
However, some algorithms require single-segment arrays. When an
irregularly strided array is passed in to such algorithms, a copy
is automatically made.

:::

## Array attributes

Array attributes reflect information that is intrinsic to the array
itself. Generally, accessing an array through its attributes allows
you to get and sometimes set intrinsic properties of the array without
creating a new array. The exposed attributes are the core parts of an
array and only some of them can be reset meaningfully without creating
a new array. Information on each attribute is given below.

### Memory layout

The following attributes contain information about the memory layout
of the array:

method | description
---|---
[ndarray.flags](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flags.html#numpy.ndarray.flags) | Information about the memory layout of the array.
[ndarray.shape](https://numpy.org/devdocs/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape) | Tuple of array dimensions.
[ndarray.strides](https://numpy.org/devdocs/reference/generated/numpy.ndarray.strides.html#numpy.ndarray.strides) | Tuple of bytes to step in each dimension when traversing an array.
[ndarray.ndim](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ndim.html#numpy.ndarray.ndim) | Number of array dimensions.
[ndarray.data](https://numpy.org/devdocs/reference/generated/numpy.ndarray.data.html#numpy.ndarray.data) | Python buffer object pointing to the start of the array’s data.
[ndarray.size](https://numpy.org/devdocs/reference/generated/numpy.ndarray.size.html#numpy.ndarray.size) | Number of elements in the array.
[ndarray.itemsize](https://numpy.org/devdocs/reference/generated/numpy.ndarray.itemsize.html#numpy.ndarray.itemsize) | Length of one array element in bytes.
[ndarray.nbytes](https://numpy.org/devdocs/reference/generated/numpy.ndarray.nbytes.html#numpy.ndarray.nbytes) | Total bytes consumed by the elements of the array.
[ndarray.base](https://numpy.org/devdocs/reference/generated/numpy.ndarray.base.html#numpy.ndarray.base) | Base object if memory is from some other object.

### Data type

::: tip See also

[Data type objects](arrays.dtypes.html#arrays-dtypes)

:::

The data type object associated with the array can be found in the
[``dtype``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.dtype.html#numpy.ndarray.dtype) attribute:

method | description
---|---
[ndarray.dtype](https://numpy.org/devdocs/reference/generated/numpy.ndarray.dtype.html#numpy.ndarray.dtype) | Data-type of the array’s elements.

### Other attributes

method | description
---|---
[ndarray.T](https://numpy.org/devdocs/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T) | The transposed array.
[ndarray.real](https://numpy.org/devdocs/reference/generated/numpy.ndarray.real.html#numpy.ndarray.real) | The real part of the array.
[ndarray.imag](https://numpy.org/devdocs/reference/generated/numpy.ndarray.imag.html#numpy.ndarray.imag) | The imaginary part of the array.
[ndarray.flat](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | A 1-D iterator over the array.
[ndarray.ctypes](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ctypes.html#numpy.ndarray.ctypes) | An object to simplify the interaction of the array with the ctypes module.

### Array interface

::: tip See also

[The Array Interface](arrays.interface.html#arrays-interface).

:::

method | description
---|---
[\_\_array_interface__](arrays.interface.html#__array_interface__) | Python-side of the array interface
\_\_array_struct__ | C-side of the array interface

### ``ctypes`` foreign function interface

method | description
---|---
[ndarray.ctypes](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ctypes.html#numpy.ndarray.ctypes) | An object to simplify the interaction of the array with the ctypes module.

## Array methods

An [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) object has many methods which operate on or with
the array in some fashion, typically returning an array result. These
methods are briefly explained below. (Each method’s docstring has a
more complete description.)

For the following methods there are also corresponding functions in
[``numpy``](index.html#module-numpy): [``all``](https://numpy.org/devdocs/reference/generated/numpy.all.html#numpy.all), [``any``](https://numpy.org/devdocs/reference/generated/numpy.any.html#numpy.any), [``argmax``](https://numpy.org/devdocs/reference/generated/numpy.argmax.html#numpy.argmax),
[``argmin``](https://numpy.org/devdocs/reference/generated/numpy.argmin.html#numpy.argmin), [``argpartition``](https://numpy.org/devdocs/reference/generated/numpy.argpartition.html#numpy.argpartition), [``argsort``](https://numpy.org/devdocs/reference/generated/numpy.argsort.html#numpy.argsort), [``choose``](https://numpy.org/devdocs/reference/generated/numpy.choose.html#numpy.choose),
[``clip``](https://numpy.org/devdocs/reference/generated/numpy.clip.html#numpy.clip), [``compress``](https://numpy.org/devdocs/reference/generated/numpy.compress.html#numpy.compress), [``copy``](https://numpy.org/devdocs/reference/generated/numpy.copy.html#numpy.copy), [``cumprod``](https://numpy.org/devdocs/reference/generated/numpy.cumprod.html#numpy.cumprod),
[``cumsum``](https://numpy.org/devdocs/reference/generated/numpy.cumsum.html#numpy.cumsum), [``diagonal``](https://numpy.org/devdocs/reference/generated/numpy.diagonal.html#numpy.diagonal), [``imag``](https://numpy.org/devdocs/reference/generated/numpy.imag.html#numpy.imag), [``max``](https://numpy.org/devdocs/reference/generated/numpy.amax.html#numpy.amax),
[``mean``](https://numpy.org/devdocs/reference/generated/numpy.mean.html#numpy.mean), [``min``](https://numpy.org/devdocs/reference/generated/numpy.amin.html#numpy.amin), [``nonzero``](https://numpy.org/devdocs/reference/generated/numpy.nonzero.html#numpy.nonzero), [``partition``](https://numpy.org/devdocs/reference/generated/numpy.partition.html#numpy.partition),
[``prod``](https://numpy.org/devdocs/reference/generated/numpy.prod.html#numpy.prod), [``ptp``](https://numpy.org/devdocs/reference/generated/numpy.ptp.html#numpy.ptp), [``put``](https://numpy.org/devdocs/reference/generated/numpy.put.html#numpy.put), [``ravel``](https://numpy.org/devdocs/reference/generated/numpy.ravel.html#numpy.ravel), [``real``](https://numpy.org/devdocs/reference/generated/numpy.real.html#numpy.real),
[``repeat``](https://numpy.org/devdocs/reference/generated/numpy.repeat.html#numpy.repeat), [``reshape``](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape), [``round``](https://numpy.org/devdocs/reference/generated/numpy.around.html#numpy.around),
[``searchsorted``](https://numpy.org/devdocs/reference/generated/numpy.searchsorted.html#numpy.searchsorted), [``sort``](https://numpy.org/devdocs/reference/generated/numpy.sort.html#numpy.sort), [``squeeze``](https://numpy.org/devdocs/reference/generated/numpy.squeeze.html#numpy.squeeze), [``std``](https://numpy.org/devdocs/reference/generated/numpy.std.html#numpy.std),
[``sum``](https://numpy.org/devdocs/reference/generated/numpy.sum.html#numpy.sum), [``swapaxes``](https://numpy.org/devdocs/reference/generated/numpy.swapaxes.html#numpy.swapaxes), [``take``](https://numpy.org/devdocs/reference/generated/numpy.take.html#numpy.take), [``trace``](https://numpy.org/devdocs/reference/generated/numpy.trace.html#numpy.trace),
[``transpose``](https://numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose), [``var``](https://numpy.org/devdocs/reference/generated/numpy.var.html#numpy.var).

### Array conversion

method | description
---|---
[ndarray.item](https://numpy.org/devdocs/reference/generated/numpy.ndarray.item.html#numpy.ndarray.item)(*args) | Copy an element of an array to a standard Python scalar and return it.
[ndarray.tolist](https://numpy.org/devdocs/reference/generated/numpy.ndarray.tolist.html#numpy.ndarray.tolist)() | Return the array as an a.ndim-levels deep nested list of Python scalars.
[ndarray.itemset](https://numpy.org/devdocs/reference/generated/numpy.ndarray.itemset.html#numpy.ndarray.itemset)(*args) | Insert scalar into an array (scalar is cast to array’s dtype, if possible)
[ndarray.tostring](https://numpy.org/devdocs/reference/generated/numpy.ndarray.tostring.html#numpy.ndarray.tostring)([order]) | Construct Python bytes containing the raw data bytes in the array.
[ndarray.tobytes](https://numpy.org/devdocs/reference/generated/numpy.ndarray.tobytes.html#numpy.ndarray.tobytes)([order]) | Construct Python bytes containing the raw data bytes in the array.
[ndarray.tofile](https://numpy.org/devdocs/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile)(fid[, sep, format]) | Write array to a file as text or binary (default).
[ndarray.dump](https://numpy.org/devdocs/reference/generated/numpy.ndarray.dump.html#numpy.ndarray.dump)(file) | Dump a pickle of the array to the specified file.
[ndarray.dumps](https://numpy.org/devdocs/reference/generated/numpy.ndarray.dumps.html#numpy.ndarray.dumps)() | Returns the pickle of the array as a string.
[ndarray.astype](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype)(dtype[, order, casting, …]) | Copy of the array, cast to a specified type.
[ndarray.byteswap](https://numpy.org/devdocs/reference/generated/numpy.ndarray.byteswap.html#numpy.ndarray.byteswap)([inplace]) | Swap the bytes of the array elements
[ndarray.copy](https://numpy.org/devdocs/reference/generated/numpy.ndarray.copy.html#numpy.ndarray.copy)([order]) | Return a copy of the array.
[ndarray.view](https://numpy.org/devdocs/reference/generated/numpy.ndarray.view.html#numpy.ndarray.view)([dtype, type]) | New view of array with the same data.
[ndarray.getfield](https://numpy.org/devdocs/reference/generated/numpy.ndarray.getfield.html#numpy.ndarray.getfield)(dtype[, offset]) | Returns a field of the given array as a certain type.
[ndarray.setflags](https://numpy.org/devdocs/reference/generated/numpy.ndarray.setflags.html#numpy.ndarray.setflags)([write, align, uic]) | Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY), respectively.
[ndarray.fill](https://numpy.org/devdocs/reference/generated/numpy.ndarray.fill.html#numpy.ndarray.fill)(value) | Fill the array with a scalar value.

### Shape manipulation

For reshape, resize, and transpose, the single tuple argument may be
replaced with ``n`` integers which will be interpreted as an n-tuple.

method | description
---|---
[ndarray.reshape](https://numpy.org/devdocs/reference/generated/numpy.ndarray.reshape.html#numpy.ndarray.reshape)(shape[, order]) | Returns an array containing the same data with a new shape.
[ndarray.resize](https://numpy.org/devdocs/reference/generated/numpy.ndarray.resize.html#numpy.ndarray.resize)(new_shape[, refcheck]) | Change shape and size of array in-place.
[ndarray.transpose](https://numpy.org/devdocs/reference/generated/numpy.ndarray.transpose.html#numpy.ndarray.transpose)(*axes) | Returns a view of the array with axes transposed.
[ndarray.swapaxes](https://numpy.org/devdocs/reference/generated/numpy.ndarray.swapaxes.html#numpy.ndarray.swapaxes)(axis1, axis2) | Return a view of the array with axis1 and axis2 interchanged.
[ndarray.flatten](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten)([order]) | Return a copy of the array collapsed into one dimension.
[ndarray.ravel](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ravel.html#numpy.ndarray.ravel)([order]) | Return a flattened array.
[ndarray.squeeze](https://numpy.org/devdocs/reference/generated/numpy.ndarray.squeeze.html#numpy.ndarray.squeeze)([axis]) | Remove single-dimensional entries from the shape of a.

### Item selection and manipulation

For array methods that take an *axis* keyword, it defaults to
``None``. If axis is *None*, then the array is treated as a 1-D
array. Any other value for *axis* represents the dimension along which
the operation should proceed.

method | description
---|---
[ndarray.take](https://numpy.org/devdocs/reference/generated/numpy.ndarray.take.html#numpy.ndarray.take)(indices[, axis, out, mode]) | Return an array formed from the elements of a at the given indices.
[ndarray.put](https://numpy.org/devdocs/reference/generated/numpy.ndarray.put.html#numpy.ndarray.put)(indices, values[, mode]) | Set a.flat[n] = values[n] for all n in indices.
[ndarray.repeat](https://numpy.org/devdocs/reference/generated/numpy.ndarray.repeat.html#numpy.ndarray.repeat)(repeats[, axis]) | Repeat elements of an array.
[ndarray.choose](https://numpy.org/devdocs/reference/generated/numpy.ndarray.choose.html#numpy.ndarray.choose)(choices[, out, mode]) | Use an index array to construct a new array from a set of choices.
[ndarray.sort](https://numpy.org/devdocs/reference/generated/numpy.ndarray.sort.html#numpy.ndarray.sort)([axis, kind, order]) | Sort an array in-place.
[ndarray.argsort](https://numpy.org/devdocs/reference/generated/numpy.ndarray.argsort.html#numpy.ndarray.argsort)([axis, kind, order]) | Returns the indices that would sort this array.
[ndarray.partition](https://numpy.org/devdocs/reference/generated/numpy.ndarray.partition.html#numpy.ndarray.partition)(kth[, axis, kind, order]) | Rearranges the elements in the array in such a way that the value of the element in kth position is in the position it would be in a sorted array.
[ndarray.argpartition](https://numpy.org/devdocs/reference/generated/numpy.ndarray.argpartition.html#numpy.ndarray.argpartition)(kth[, axis, kind, order]) | Returns the indices that would partition this array.
[ndarray.searchsorted](https://numpy.org/devdocs/reference/generated/numpy.ndarray.searchsorted.html#numpy.ndarray.searchsorted)(v[, side, sorter]) | Find indices where elements of v should be inserted in a to maintain order.
[ndarray.nonzero](https://numpy.org/devdocs/reference/generated/numpy.ndarray.nonzero.html#numpy.ndarray.nonzero)() | Return the indices of the elements that are non-zero.
[ndarray.compress](https://numpy.org/devdocs/reference/generated/numpy.ndarray.compress.html#numpy.ndarray.compress)(condition[, axis, out]) | Return selected slices of this array along given axis.
[ndarray.diagonal](https://numpy.org/devdocs/reference/generated/numpy.ndarray.diagonal.html#numpy.ndarray.diagonal)([offset, axis1, axis2]) | Return specified diagonals.

### Calculation

Many of these methods take an argument named *axis*. In such cases,

- If *axis* is *None* (the default), the array is treated as a 1-D
array and the operation is performed over the entire array. This
behavior is also the default if self is a 0-dimensional array or
array scalar. (An array scalar is an instance of the types/classes
float32, float64, etc., whereas a 0-dimensional array is an ndarray
instance containing precisely one array scalar.)
- If *axis* is an integer, then the operation is done over the given
axis (for each 1-D subarray that can be created along the given axis).

A 3-dimensional array of size 3 x 3 x 3, summed over each of its
three axes

``` python
>>> x
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],
       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],
       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
>>> x.sum(axis=0)
array([[27, 30, 33],
       [36, 39, 42],
       [45, 48, 51]])
>>> # for sum, axis is the first keyword, so we may omit it,
>>> # specifying only its value
>>> x.sum(0), x.sum(1), x.sum(2)
(array([[27, 30, 33],
        [36, 39, 42],
        [45, 48, 51]]),
 array([[ 9, 12, 15],
        [36, 39, 42],
        [63, 66, 69]]),
 array([[ 3, 12, 21],
        [30, 39, 48],
        [57, 66, 75]]))
```

The parameter *dtype* specifies the data type over which a reduction
operation (like summing) should take place. The default reduce data
type is the same as the data type of *self*. To avoid overflow, it can
be useful to perform the reduction using a larger data type.

For several methods, an optional *out* argument can also be provided
and the result will be placed into the output array given. The *out*
argument must be an [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) and have the same number of
elements. It can have a different data type in which case casting will
be performed.

method | description
---|---
[ndarray.max](https://numpy.org/devdocs/reference/generated/numpy.ndarray.max.html#numpy.ndarray.max)([axis, out, keepdims, initial, …]) | Return the maximum along a given axis.
[ndarray.argmax](https://numpy.org/devdocs/reference/generated/numpy.ndarray.argmax.html#numpy.ndarray.argmax)([axis, out]) | Return indices of the maximum values along the given axis.
[ndarray.min](https://numpy.org/devdocs/reference/generated/numpy.ndarray.min.html#numpy.ndarray.min)([axis, out, keepdims, initial, …]) | Return the minimum along a given axis.
[ndarray.argmin](https://numpy.org/devdocs/reference/generated/numpy.ndarray.argmin.html#numpy.ndarray.argmin)([axis, out]) | Return indices of the minimum values along the given axis of a.
[ndarray.ptp](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ptp.html#numpy.ndarray.ptp)([axis, out, keepdims]) | Peak to peak (maximum - minimum) value along a given axis.
[ndarray.clip](https://numpy.org/devdocs/reference/generated/numpy.ndarray.clip.html#numpy.ndarray.clip)([min, max, out]) | Return an array whose values are limited to [min, max].
[ndarray.conj](https://numpy.org/devdocs/reference/generated/numpy.ndarray.conj.html#numpy.ndarray.conj)() | Complex-conjugate all elements.
[ndarray.round](https://numpy.org/devdocs/reference/generated/numpy.ndarray.round.html#numpy.ndarray.round)([decimals, out]) | Return a with each element rounded to the given number of decimals.
[ndarray.trace](https://numpy.org/devdocs/reference/generated/numpy.ndarray.trace.html#numpy.ndarray.trace)([offset, axis1, axis2, dtype, out]) | Return the sum along diagonals of the array.
[ndarray.sum](https://numpy.org/devdocs/reference/generated/numpy.ndarray.sum.html#numpy.ndarray.sum)([axis, dtype, out, keepdims, …]) | Return the sum of the array elements over the given axis.
[ndarray.cumsum](https://numpy.org/devdocs/reference/generated/numpy.ndarray.cumsum.html#numpy.ndarray.cumsum)([axis, dtype, out]) | Return the cumulative sum of the elements along the given axis.
[ndarray.mean](https://numpy.org/devdocs/reference/generated/numpy.ndarray.mean.html#numpy.ndarray.mean)([axis, dtype, out, keepdims]) | Returns the average of the array elements along given axis.
[ndarray.var](https://numpy.org/devdocs/reference/generated/numpy.ndarray.var.html#numpy.ndarray.var)([axis, dtype, out, ddof, keepdims]) | Returns the variance of the array elements, along given axis.
[ndarray.std](https://numpy.org/devdocs/reference/generated/numpy.ndarray.std.html#numpy.ndarray.std)([axis, dtype, out, ddof, keepdims]) | Returns the standard deviation of the array elements along given axis.
[ndarray.prod](https://numpy.org/devdocs/reference/generated/numpy.ndarray.prod.html#numpy.ndarray.prod)([axis, dtype, out, keepdims, …]) | Return the product of the array elements over the given axis
[ndarray.cumprod](https://numpy.org/devdocs/reference/generated/numpy.ndarray.cumprod.html#numpy.ndarray.cumprod)([axis, dtype, out]) | Return the cumulative product of the elements along the given axis.
[ndarray.all](https://numpy.org/devdocs/reference/generated/numpy.ndarray.all.html#numpy.ndarray.all)([axis, out, keepdims]) | Returns True if all elements evaluate to True.
[ndarray.any](https://numpy.org/devdocs/reference/generated/numpy.ndarray.any.html#numpy.ndarray.any)([axis, out, keepdims]) | Returns True if any of the elements of a evaluate to True.

## Arithmetic, matrix multiplication, and comparison operations

Arithmetic and comparison operations on [``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)
are defined as element-wise operations, and generally yield
[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) objects as results.

Each of the arithmetic operations (``+``, ``-``, ``*``, ``/``, ``//``,
``%``, ``divmod()``, ``**`` or ``pow()``, ``<<``, ``>>``, ``&``,
``^``, ``|``, ``~``) and the comparisons (``==``, ``<``, ``>``,
``<=``, ``>=``, ``!=``) is equivalent to the corresponding
universal function (or [ufunc](https://numpy.org/devdocs/glossary.html#term-ufunc) for short) in NumPy.  For
more information, see the section on [Universal Functions](ufuncs.html#ufuncs).

Comparison operators:

method | description
---|---
[ndarray.\__lt__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__lt__.html#numpy.ndarray.__lt__)(self, value, /) | Return self<value.
[ndarray.\__le__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__le__.html#numpy.ndarray.__le__)(self, value, /) | Return self<=value.
[ndarray.\__gt__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__gt__.html#numpy.ndarray.__gt__)(self, value, /) | Return self>value.
[ndarray.\__ge__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__ge__.html#numpy.ndarray.__ge__)(self, value, /) | Return self>=value.
[ndarray.\__eq__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__eq__.html#numpy.ndarray.__eq__)(self, value, /) | Return self==value.
[ndarray.\__ne__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__ne__.html#numpy.ndarray.__ne__)(self, value, /) | Return self!=value.

Truth value of an array (``bool``):

method | description
---|---
[ndarray.\_\_bool__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__bool__.html#numpy.ndarray.__bool__)(self, /) | self != 0

::: tip Note

Truth-value testing of an array invokes
[``ndarray.__bool__``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__bool__.html#numpy.ndarray.__bool__), which raises an error if the number of
elements in the array is larger than 1, because the truth value
of such arrays is ambiguous. Use [``.any()``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.any.html#numpy.ndarray.any) and
[``.all()``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.all.html#numpy.ndarray.all) instead to be clear about what is meant
in such cases. (If the number of elements is 0, the array evaluates
to ``False``.)

:::

Unary operations:

method | description
---|---
[ndarray.\_\_neg__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__neg__.html#numpy.ndarray.__neg__)(self, /) | -self
[ndarray.\_\_pos__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__pos__.html#numpy.ndarray.__pos__)(self, /) | +self
[ndarray.\_\_abs__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__abs__.html#numpy.ndarray.__abs__)(self) | 
[ndarray.\_\_invert__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__invert__.html#numpy.ndarray.__invert__)(self, /) | ~self

Arithmetic:

method | description
---|---
[ndarray.\_\_add__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__add__.html#numpy.ndarray.__add__)(self, value, /) | Return self+value.
[ndarray.\_\_sub__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__sub__.html#numpy.ndarray.__sub__)(self, value, /) | Return self-value.
[ndarray.\_\_mul__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__mul__.html#numpy.ndarray.__mul__)(self, value, /) | Return self*value.
[ndarray.\_\_truediv__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__truediv__.html#numpy.ndarray.__truediv__)(self, value, /) | Return self/value.
[ndarray.\_\_floordiv__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__floordiv__.html#numpy.ndarray.__floordiv__)(self, value, /) | Return self//value.
[ndarray.\_\_mod__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__mod__.html#numpy.ndarray.__mod__)(self, value, /) | Return self%value.
[ndarray.\_\_divmod__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__divmod__.html#numpy.ndarray.__divmod__)(self, value, /) | Return divmod(self, value).
[ndarray.\_\_pow__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__pow__.html#numpy.ndarray.__pow__)(self, value[, mod]) | Return pow(self, value, mod).
[ndarray.\_\_lshift__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__lshift__.html#numpy.ndarray.__lshift__)(self, value, /) | Return self<<value.
[ndarray.\_\_rshift__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__rshift__.html#numpy.ndarray.__rshift__)(self, value, /) | Return self>>value.
[ndarray.\_\_and__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__and__.html#numpy.ndarray.__and__)(self, value, /) | Return self&value.
[ndarray.\_\_or__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__or__.html#numpy.ndarray.__or__)(self, value, /) | Return self|value.
[ndarray.\_\_xor__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__xor__.html#numpy.ndarray.__xor__)(self, value, /) | Return self^value.

::: tip Note

- Any third argument to [``pow``](https://docs.python.org/dev/library/functions.html#pow) is silently ignored,
as the underlying [``ufunc``](https://numpy.org/devdocs/reference/generated/numpy.power.html#numpy.power) takes only two arguments.
- The three division operators are all defined; ``div`` is active
by default, ``truediv`` is active when
[``__future__``](https://docs.python.org/dev/library/__future__.html#module-__future__) division is in effect.
- Because [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) is a built-in type (written in C), the
``__r{op}__`` special methods are not directly defined.
- The functions called to implement many arithmetic special methods
for arrays can be modified using [``__array_ufunc__``](arrays.classes.html#numpy.class.__array_ufunc__).

:::

Arithmetic, in-place:

method | description
---|---
[ndarray.\_\_iadd__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__iadd__.html#numpy.ndarray.__iadd__)(self, value, /) | Return self+=value.
[ndarray.\_\_isub__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__isub__.html#numpy.ndarray.__isub__)(self, value, /) | Return self-=value.
[ndarray.\_\_imul__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__imul__.html#numpy.ndarray.__imul__)(self, value, /) | Return self*=value.
[ndarray.\_\_itruediv__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__itruediv__.html#numpy.ndarray.__itruediv__)(self, value, /) | Return self/=value.
[ndarray.\_\_ifloordiv__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__ifloordiv__.html#numpy.ndarray.__ifloordiv__)(self, value, /) | Return self//=value.
[ndarray.\_\_imod__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__imod__.html#numpy.ndarray.__imod__)(self, value, /) | Return self%=value.
[ndarray.\_\_ipow__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__ipow__.html#numpy.ndarray.__ipow__)(self, value, /) | Return self**=value.
[ndarray.\_\_ilshift__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__ilshift__.html#numpy.ndarray.__ilshift__)(self, value, /) | Return self<<=value.
[ndarray.\_\_irshift__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__irshift__.html#numpy.ndarray.__irshift__)(self, value, /) | Return self>>=value.
[ndarray.\_\_iand__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__iand__.html#numpy.ndarray.__iand__)(self, value, /) | Return self&=value.
[ndarray.\_\_ior__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__ior__.html#numpy.ndarray.__ior__)(self, value, /) | Return self|=value.
[ndarray.\_\_ixor__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__ixor__.html#numpy.ndarray.__ixor__)(self, value, /) | Return self^=value.

::: danger Warning

In place operations will perform the calculation using the
precision decided by the data type of the two operands, but will
silently downcast the result (if necessary) so it can fit back into
the array.  Therefore, for mixed precision calculations, ``A {op}=
B`` can be different than ``A = A {op} B``. For example, suppose
``a = ones((3,3))``. Then, ``a += 3j`` is different than ``a = a +
3j``: while they both perform the same computation, ``a += 3``
casts the result to fit back in ``a``, whereas ``a = a + 3j``
re-binds the name ``a`` to the result.

:::

Matrix Multiplication:

method | description
---|---
[ndarray.\_\_matmul__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__matmul__.html#numpy.ndarray.__matmul__)(self, value, /) | Return [self@value](mailto:self%40value).

::: tip Note

Matrix operators ``@`` and ``@=`` were introduced in Python 3.5
following PEP465. NumPy 1.10.0 has a preliminary implementation of ``@``
for testing purposes. Further documentation can be found in the
[``matmul``](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul) documentation.

:::

## Special methods

For standard library functions:

method | description
---|---
[ndarray.\_\_copy__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__copy__.html#numpy.ndarray.__copy__)() | Used if [copy.copy](https://docs.python.org/dev/library/copy.html#copy.copy) is called on an array.
[ndarray.\_\_deepcopy__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__deepcopy__.html#numpy.ndarray.__deepcopy__)() | Used if [copy.deepcopy](https://docs.python.org/dev/library/copy.html#copy.deepcopy) is called on an array.
[ndarray.\_\_reduce__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__reduce__.html#numpy.ndarray.__reduce__)() | For pickling.
[ndarray.\_\_setstate__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__setstate__.html#numpy.ndarray.__setstate__)(state, /) | For unpickling.

Basic customization:

method | description
---|---
[ndarray.\_\_new__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__new__.html#numpy.ndarray.__new__)(\*args, \*\*kwargs) | Create and return a new object.
[ndarray.\_\_array__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__array__.html#numpy.ndarray.__array__)() | Returns either a new reference to self if dtype is not given or a new array of provided data type if dtype is different from the current dtype of the array.
[ndarray.\_\_array_wrap__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__array_wrap__.html#numpy.ndarray.__array_wrap__)() | 

Container customization: (see [Indexing](arrays.indexing.html#arrays-indexing))

method | description
---|---
[ndarray.\_\_len__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__len__.html#numpy.ndarray.__len__)(self, /) | Return len(self).
[ndarray.\_\_getitem__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__getitem__.html#numpy.ndarray.__getitem__)(self, key, /) | Return self[key].
[ndarray.\_\_setitem__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__setitem__.html#numpy.ndarray.__setitem__)(self, key, value, /) | Set self[key] to value.
[ndarray.\_\_contains__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__contains__.html#numpy.ndarray.__contains__)(self, key, /) | Return key in self.

Conversion; the operations ``int``, ``float`` and
``complex``.
. They work only on arrays that have one element in them
and return the appropriate scalar.

method | description
---|---
[ndarray.\_\_int__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__int__.html#numpy.ndarray.__int__)(self) | none
[ndarray.\_\_float__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__float__.html#numpy.ndarray.__float__)(self) | none
[ndarray.\_\_complex__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__complex__.html#numpy.ndarray.__complex__)() | none

String representations:

method | description
---|---
[ndarray.\_\_str__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__str__.html#numpy.ndarray.__str__)(self, /) | Return str(self).
[ndarray.\_\_repr__](https://numpy.org/devdocs/reference/generated/numpy.ndarray.__repr__.html#numpy.ndarray.__repr__)(self, /) | Return repr(self).
