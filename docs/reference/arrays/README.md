# Array objects

NumPy provides an N-dimensional array type, the [ndarray](ndarray.html#arrays-ndarray), which describes a collection of “items” of the same
type. The items can be [indexed](indexing.html#arrays-indexing) using for
example N integers.

All ndarrays are [homogenous](https://numpy.org/devdocs/glossary.html#term-homogenous): every item takes up the same size
block of memory, and all blocks are interpreted in exactly the same
way. How each item in the array is to be interpreted is specified by a
separate [data-type object](dtypes.html#arrays-dtypes), one of which is associated
with every array. In addition to basic types (integers, floats,
*etc.*), the data type objects can also represent data structures.

An item extracted from an array, *e.g.*, by indexing, is represented
by a Python object whose type is one of the [array scalar types](scalars.html#arrays-scalars) built in NumPy. The array scalars allow easy manipulation
of also more complicated arrangements of data.

![threefundamental](/static/images/threefundamental.png)

**Figure** Conceptual diagram showing the relationship between the three
fundamental objects used to describe the data in an array: 1) the
ndarray itself, 2) the data-type object that describes the layout
of a single fixed-size element of the array, 3) the array-scalar
Python object that is returned when a single element of the array
is accessed.

- [The N-dimensional array (ndarray)](ndarray.html)
  - [Constructing arrays](ndarray.html#constructing-arrays)
  - [Indexing arrays](ndarray.html#indexing-arrays)
  - [Internal memory layout of an ndarray](ndarray.html#internal-memory-layout-of-an-ndarray)
  - [Array attributes](ndarray.html#array-attributes)
  - [Array methods](ndarray.html#array-methods)
  - [Arithmetic, matrix multiplication, and comparison operations](ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations)
  - [Special methods](ndarray.html#special-methods)
- [Scalars](scalars.html)
  - [Built-in scalar types](scalars.html#built-in-scalar-types)
  - [Attributes](scalars.html#attributes)
  - [Indexing](scalars.html#indexing)
  - [Methods](scalars.html#methods)
  - [Defining new types](scalars.html#defining-new-types)
- [Data type objects (dtype)](dtypes.html)
  - [Specifying and constructing data types](dtypes.html#specifying-and-constructing-data-types)
  - [dtype](dtypes.html#dtype)
- [Indexing](indexing.html)
  - [Basic Slicing and Indexing](indexing.html#basic-slicing-and-indexing)
  - [Advanced Indexing](indexing.html#advanced-indexing)
  - [Detailed notes](indexing.html#detailed-notes)
  - [Field Access](indexing.html#field-access)
  - [Flat Iterator indexing](indexing.html#flat-iterator-indexing)
- [Iterating Over Arrays](nditer.html)
  - [Single Array Iteration](nditer.html#single-array-iteration)
  - [Broadcasting Array Iteration](nditer.html#broadcasting-array-iteration)
  - [Putting the Inner Loop in Cython](nditer.html#putting-the-inner-loop-in-cython)
- [Standard array subclasses](classes.html)
  - [Special attributes and methods](classes.html#special-attributes-and-methods)
  - [Matrix objects](classes.html#matrix-objects)
  - [Memory-mapped file arrays](classes.html#memory-mapped-file-arrays)
  - [Character arrays (numpy.char)](classes.html#character-arrays-numpy-char)
  - [Record arrays (numpy.rec)](classes.html#record-arrays-numpy-rec)
  - [Masked arrays (numpy.ma)](classes.html#masked-arrays-numpy-ma)
  - [Standard container class](classes.html#standard-container-class)
  - [Array Iterators](classes.html#array-iterators)
- [Masked arrays](maskedarray.html)
  - [The numpy.ma module](maskedarray.generic.html)
  - [Using numpy.ma](maskedarray.generic.html#using-numpy-ma)
  - [Examples](maskedarray.generic.html#examples)
  - [Constants of the numpy.ma module](maskedarray.baseclass.html)
  - [The MaskedArray class](maskedarray.baseclass.html#the-maskedarray-class)
  - [MaskedArray methods](maskedarray.baseclass.html#maskedarray-methods)
  - [Masked array operations](routines.ma.html)
- [The Array Interface](interface.html)
  - [Python side](interface.html#python-side)
  - [C-struct access](interface.html#c-struct-access)
  - [Type description examples](interface.html#type-description-examples)
  - [Differences with Array interface (Version 2)](interface.html#differences-with-array-interface-version-2)
- [Datetimes and Timedeltas](datetime.html)
  - [Basic Datetimes](datetime.html#basic-datetimes)
  - [Datetime and Timedelta Arithmetic](datetime.html#datetime-and-timedelta-arithmetic)
  - [Datetime Units](datetime.html#datetime-units)
  - [Business Day Functionality](datetime.html#business-day-functionality)
  - [Changes with NumPy 1.11](datetime.html#changes-with-numpy-1-11)
  - [Differences Between 1.6 and 1.7 Datetimes](datetime.html#differences-between-1-6-and-1-7-datetimes)