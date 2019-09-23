# Scalars

Python defines only one type of a particular data class (there is only
one integer type, one floating-point type, etc.). This can be
convenient in applications that don’t need to be concerned with all
the ways data can be represented in a computer.  For scientific
computing, however, more control is often needed.

In NumPy, there are 24 new fundamental Python types to describe
different types of scalars. These type descriptors are mostly based on
the types available in the C language that CPython is written in, with
several additional types compatible with Python’s types.

Array scalars have the same attributes and methods as [``ndarrays``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray). [[1]](#id2) This allows one to treat items of an array partly on
the same footing as arrays, smoothing out rough edges that result when
mixing scalar and array operations.

Array scalars live in a hierarchy (see the Figure below) of data
types. They can be detected using the hierarchy: For example,
``isinstance(val, np.generic)`` will return ``True`` if *val* is
an array scalar object. Alternatively, what kind of array scalar is
present can be determined using other members of the data type
hierarchy. Thus, for example ``isinstance(val, np.complexfloating)``
will return ``True`` if *val* is a complex valued type, while
``isinstance(val, np.flexible)`` will return true if *val* is one
of the flexible itemsize array types (``string``,
``unicode``, ``void``).

![dtype-hierarchy](/static/images/dtype-hierarchy.png)

**Figure:** Hierarchy of type objects representing the array data
types. Not shown are the two integer types ``intp`` and
``uintp`` which just point to the integer type that holds a
pointer for the platform. All the number types can be obtained
using bit-width names as well.

[[1]](#id1)However, array scalars are immutable, so none of the array scalar attributes are settable.

## Built-in scalar types

The built-in scalar types are shown below. Along with their (mostly)
C-derived names, the integer, float, and complex data-types are also
available using a bit-width convention so that an array of the right
size can always be ensured (e.g. ``int8``, ``float64``,
``complex128``). Two aliases (``intp`` and ``uintp``)
pointing to the integer type that is sufficiently large to hold a C pointer
are also provided. The C-like names are associated with character codes,
which are shown in the table. Use of the character codes, however,
is discouraged.

Some of the scalar types are essentially equivalent to fundamental
Python types and therefore inherit from them as well as from the
generic array scalar type:

Array scalar type | Related Python type
---|---
int_ | IntType (Python 2 only)
float_ | FloatType
complex_ | ComplexType
bytes_ | BytesType
unicode_ | UnicodeType

The ``bool_`` data type is very similar to the Python
``BooleanType`` but does not inherit from it because Python’s
``BooleanType`` does not allow itself to be inherited from, and
on the C-level the size of the actual bool data is not the same as a
Python Boolean scalar.

::: danger Warning

The ``bool_`` type is not a subclass of the ``int_`` type
(the ``bool_`` is not even a number type). This is different
than Python’s default implementation of [``bool``](https://docs.python.org/dev/library/functions.html#bool) as a
sub-class of int.

:::

::: danger Warning

The ``int_`` type does **not** inherit from the
[``int``](https://docs.python.org/dev/library/functions.html#int) built-in under Python 3, because type [``int``](https://docs.python.org/dev/library/functions.html#int) is no
longer a fixed-width integer type.

:::

The default data type in NumPy is ``float_``.

In the tables below, ``platform?`` means that the type may not be
available on all platforms. Compatibility with different C or Python
types is indicated: two types are compatible if their data is of the
same size and interpreted in the same way.

Booleans:

Type | Remarks | Character code
---|---|---
bool_ | compatible: Python bool | '?'
bool8 | 8 bits |  

Integers:

Type | Remarks | Character code
---|---|---
byte | compatible: C char | 'b'
short | compatible: C short | 'h'
intc | compatible: C int | 'i'
int_ | compatible: Python int | 'l'
longlong | compatible: C long long | 'q'
intp | large enough to fit a pointer | 'p'
int8 | 8 bits |  
int16 | 16 bits |  
int32 | 32 bits |  
int64 | 64 bits |  

Unsigned integers:

Type | Remarks | Character code
---|---|---
ubyte | compatible: C unsigned char | 'B'
ushort | compatible: C unsigned short | 'H'
uintc | compatible: C unsigned int | 'I'
uint | compatible: Python int | 'L'
ulonglong | compatible: C long long | 'Q'
uintp | large enough to fit a pointer | 'P'
uint8 | 8 bits |  
uint16 | 16 bits |  
uint32 | 32 bits |  
uint64 | 64 bits |  

Floating-point numbers:

Type | Remarks | Character code
---|---|---
half |   | 'e'
single | compatible: C float | 'f'
double | compatible: C double |  
float_ | compatible: Python float | 'd'
longfloat | compatible: C long float | 'g'
float16 | 16 bits |  
float32 | 32 bits |  
float64 | 64 bits |  
float96 | 96 bits, platform? |  
float128 | 128 bits, platform? |  

Complex floating-point numbers:

Type | Remarks | Character code
---|---|---
csingle |   | 'F'
complex_ | compatible: Python complex | 'D'
clongfloat |   | 'G'
complex64 | two 32-bit floats |  
complex128 | two 64-bit floats |  
complex192 | two 96-bit floats, platform? |  
complex256 | two 128-bit floats, platform? |  

Any Python object:

Type | Remarks | Character code
---|---|---
object_ | any Python object | 'O'

::: tip Note

The data actually stored in object arrays
(*i.e.*, arrays having dtype ``object_``) are references to
Python objects, not the objects themselves. Hence, object arrays
behave more like usual Python [``lists``](https://docs.python.org/dev/library/stdtypes.html#list), in the sense
that their contents need not be of the same Python type.

The object type is also special because an array containing
``object_`` items does not return an ``object_`` object
on item access, but instead returns the actual object that
the array item refers to.

:::

The following data types are **flexible**: they have no predefined
size and the data they describe can be of different length in different
arrays. (In the character codes ``#`` is an integer denoting how many
elements the data type consists of.)

Type | Remarks | Character code
---|---|---
bytes_ | compatible: Python bytes | 'S#'
unicode_ | compatible: Python unicode/str | 'U#'
void |   | 'V#'

::: danger Warning

See [Note on string types](arrays.dtypes.html#string-dtype-note).

Numeric Compatibility: If you used old typecode characters in your
Numeric code (which was never recommended), you will need to change
some of them to the new characters. In particular, the needed
changes are ``c -> S1``, ``b -> B``, ``1 -> b``, ``s -> h``, ``w ->
H``, and ``u -> I``. These changes make the type character
convention more consistent with other Python modules such as the
[``struct``](https://docs.python.org/dev/library/struct.html#module-struct) module.

:::

## Attributes

The array scalar objects have an ``array priority`` of [``NPY_SCALAR_PRIORITY``](c-api/array.html#c.NPY_SCALAR_PRIORITY)
(-1,000,000.0). They also do not (yet) have a [``ctypes``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.ctypes.html#numpy.ndarray.ctypes)
attribute. Otherwise, they share the same attributes as arrays:

method | description
---|---
[generic.flags](https://numpy.org/devdocs/reference/generated/numpy.generic.flags.html#numpy.generic.flags) | integer value of flags
[generic.shape](https://numpy.org/devdocs/reference/generated/numpy.generic.shape.html#numpy.generic.shape) | tuple of array dimensions
[generic.strides](https://numpy.org/devdocs/reference/generated/numpy.generic.strides.html#numpy.generic.strides) | tuple of bytes steps in each dimension
[generic.ndim](https://numpy.org/devdocs/reference/generated/numpy.generic.ndim.html#numpy.generic.ndim) | number of array dimensions
[generic.data](https://numpy.org/devdocs/reference/generated/numpy.generic.data.html#numpy.generic.data) | pointer to start of data
[generic.size](https://numpy.org/devdocs/reference/generated/numpy.generic.size.html#numpy.generic.size) | number of elements in the gentype
[generic.itemsize](https://numpy.org/devdocs/reference/generated/numpy.generic.itemsize.html#numpy.generic.itemsize) | length of one element in bytes
[generic.base](https://numpy.org/devdocs/reference/generated/numpy.generic.base.html#numpy.generic.base) | base object
[generic.dtype](https://numpy.org/devdocs/reference/generated/numpy.generic.dtype.html#numpy.generic.dtype) | get array data-descriptor
[generic.real](https://numpy.org/devdocs/reference/generated/numpy.generic.real.html#numpy.generic.real) | real part of scalar
[generic.imag](https://numpy.org/devdocs/reference/generated/numpy.generic.imag.html#numpy.generic.imag) | imaginary part of scalar
[generic.flat](https://numpy.org/devdocs/reference/generated/numpy.generic.flat.html#numpy.generic.flat) | a 1-d view of scalar
[generic.T](https://numpy.org/devdocs/reference/generated/numpy.generic.T.html#numpy.generic.T) | transpose
[generic.__array_interface__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_interface__.html#numpy.generic.__array_interface__) | Array protocol: Python side
[generic.__array_struct__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_struct__.html#numpy.generic.__array_struct__) | Array protocol: struct
[generic.__array_priority__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_priority__.html#numpy.generic.__array_priority__) | Array priority.
[generic.__array_wrap__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_wrap__.html#numpy.generic.__array_wrap__)() | sc.__array_wrap__(obj) return scalar from array

## Indexing

::: tip See also

[Indexing](arrays.indexing.html#arrays-indexing), [Data type objects (dtype)](arrays.dtypes.html#arrays-dtypes)

:::

Array scalars can be indexed like 0-dimensional arrays: if *x* is an
array scalar,

- ``x[()]`` returns a copy of array scalar
- ``x[...]`` returns a 0-dimensional [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)
- ``x['field-name']`` returns the array scalar in the field *field-name*.
(*x* can have fields, for example, when it corresponds to a structured data type.)

## Methods

Array scalars have exactly the same methods as arrays. The default
behavior of these methods is to internally convert the scalar to an
equivalent 0-dimensional array and to call the corresponding array
method. In addition, math operations on array scalars are defined so
that the same hardware flags are set and used to interpret the results
as for [ufunc](ufuncs.html#ufuncs), so that the error state used for ufuncs
also carries over to the math on array scalars.

The exceptions to the above rules are given below:

method | description
---|---
[generic](https://numpy.org/devdocs/reference/generated/numpy.generic.html#numpy.generic) | Base class for numpy scalar types.
[generic.__array__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array__.html#numpy.generic.__array__)() | sc.__array__(dtype) return 0-dim array from scalar with specified dtype
[generic.__array_wrap__](https://numpy.org/devdocs/reference/generated/numpy.generic.__array_wrap__.html#numpy.generic.__array_wrap__)() | sc.__array_wrap__(obj) return scalar from array
[generic.squeeze](https://numpy.org/devdocs/reference/generated/numpy.generic.squeeze.html#numpy.generic.squeeze)() | Not implemented (virtual attribute)
[generic.byteswap](https://numpy.org/devdocs/reference/generated/numpy.generic.byteswap.html#numpy.generic.byteswap)() | Not implemented (virtual attribute)
[generic.__reduce__](https://numpy.org/devdocs/reference/generated/numpy.generic.__reduce__.html#numpy.generic.__reduce__)() | helper for pickle
[generic.__setstate__](https://numpy.org/devdocs/reference/generated/numpy.generic.__setstate__.html#numpy.generic.__setstate__)() | 
[generic.setflags](https://numpy.org/devdocs/reference/generated/numpy.generic.setflags.html#numpy.generic.setflags)() | Not implemented (virtual attribute)

## Defining new types

There are two ways to effectively define a new array scalar type
(apart from composing structured types [dtypes](arrays.dtypes.html#arrays-dtypes) from
the built-in scalar types): One way is to simply subclass the
[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) and overwrite the methods of interest. This will work to
a degree, but internally certain behaviors are fixed by the data type of
the array.  To fully customize the data type of an array you need to
define a new data-type, and register it with NumPy. Such new types can only
be defined in C, using the [NumPy C-API](c-api/index.html#c-api).
