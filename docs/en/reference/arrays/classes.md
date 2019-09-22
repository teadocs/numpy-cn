# Standard array subclasses

::: tip Note

Subclassing a ``numpy.ndarray`` is possible but if your goal is to create
an array with *modified* behavior, as do dask arrays for distributed
computation and cupy arrays for GPU-based computation, subclassing is
discouraged. Instead, using numpy’s
[dispatch mechanism](https://numpy.org/devdocs/user/basics.dispatch.html#basics-dispatch) is recommended.

:::

The [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) can be inherited from (in Python or in C)
if desired. Therefore, it can form a foundation for many useful
classes. Often whether to sub-class the array object or to simply use
the core array component as an internal part of a new class is a
difficult decision, and can be simply a matter of choice. NumPy has
several tools for simplifying how your new object interacts with other
array objects, and so the choice may not be significant in the
end. One way to simplify the question is by asking yourself if the
object you are interested in can be replaced as a single array or does
it really require two or more arrays at its core.

Note that [``asarray``](https://numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray) always returns the base-class ndarray. If
you are confident that your use of the array object can handle any
subclass of an ndarray, then [``asanyarray``](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray) can be used to allow
subclasses to propagate more cleanly through your subroutine. In
principal a subclass could redefine any aspect of the array and
therefore, under strict guidelines, [``asanyarray``](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray) would rarely be
useful. However, most subclasses of the array object will not
redefine certain aspects of the array object such as the buffer
interface, or the attributes of the array. One important example,
however, of why your subroutine may not be able to handle an arbitrary
subclass of an array is that matrices redefine the “*” operator to be
matrix-multiplication, rather than element-by-element multiplication.

## Special attributes and methods

::: tip See also

[Subclassing ndarray](https://numpy.org/devdocs/user/basics.subclassing.html#basics-subclassing)

:::

NumPy provides several hooks that classes can customize:


- ``class.__array_ufunc__``(*ufunc*, *method*, **inputs*, ***kwargs*)

  *New in version 1.13.* 

  Any class, ndarray subclass or not, can define this method or set it to
  [``None``](https://docs.python.org/dev/library/constants.html#None) in order to override the behavior of NumPy’s ufuncs. This works
  quite similarly to Python’s ``__mul__`` and other binary operation routines.

  - *ufunc* is the ufunc object that was called.
  - *method* is a string indicating which Ufunc method was called
  (one of ``"__call__"``, ``"reduce"``, ``"reduceat"``,
  ``"accumulate"``, ``"outer"``, ``"inner"``).
  - *inputs* is a tuple of the input arguments to the ``ufunc``.
  - *kwargs* is a dictionary containing the optional input arguments
  of the ufunc. If given, any ``out`` arguments, both positional
  and keyword, are passed as a [``tuple``](https://docs.python.org/dev/library/stdtypes.html#tuple) in *kwargs*. See the
  discussion in [Universal functions (ufunc)](ufuncs.html#ufuncs) for details.

  The method should return either the result of the operation, or
  [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) if the operation requested is not implemented.

  If one of the input or output arguments has a [``__array_ufunc__``](#numpy.class.__array_ufunc__)
  method, it is executed *instead* of the ufunc.  If more than one of the
  arguments implements [``__array_ufunc__``](#numpy.class.__array_ufunc__), they are tried in the
  order: subclasses before superclasses, inputs before outputs, otherwise
  left to right. The first routine returning something other than
  [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) determines the result. If all of the
  [``__array_ufunc__``](#numpy.class.__array_ufunc__) operations return [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented), a
  [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError) is raised.

  ::: tip Note

  We intend to re-implement numpy functions as (generalized)
  Ufunc, in which case it will become possible for them to be
  overridden by the ``__array_ufunc__`` method.  A prime candidate is
  [``matmul``](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul), which currently is not a Ufunc, but could be
  relatively easily be rewritten as a (set of) generalized Ufuncs. The
  same may happen with functions such as [``median``](https://numpy.org/devdocs/reference/generated/numpy.median.html#numpy.median),
  [``amin``](https://numpy.org/devdocs/reference/generated/numpy.amin.html#numpy.amin), and [``argsort``](https://numpy.org/devdocs/reference/generated/numpy.argsort.html#numpy.argsort).

  :::

  Like with some other special methods in python, such as ``__hash__`` and
  ``__iter__``, it is possible to indicate that your class does *not*
  support ufuncs by setting ``__array_ufunc__ = None``. Ufuncs always raise
  [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError) when called on an object that sets
  ``__array_ufunc__ = None``.

  The presence of [``__array_ufunc__``](#numpy.class.__array_ufunc__) also influences how
  [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) handles binary operations like ``arr + obj`` and ``arr
  < obj`` when ``arr`` is an [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) and ``obj`` is an instance
  of a custom class. There are two possibilities. If
  ``obj.__array_ufunc__`` is present and not [``None``](https://docs.python.org/dev/library/constants.html#None), then
  ``ndarray.__add__`` and friends will delegate to the ufunc machinery,
  meaning that ``arr + obj`` becomes ``np.add(arr, obj)``, and then
  [``add``](https://numpy.org/devdocs/reference/generated/numpy.add.html#numpy.add) invokes ``obj.__array_ufunc__``. This is useful if you
  want to define an object that acts like an array.

  Alternatively, if ``obj.__array_ufunc__`` is set to [``None``](https://docs.python.org/dev/library/constants.html#None), then as a
  special case, special methods like ``ndarray.__add__`` will notice this
  and *unconditionally* raise [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError). This is useful if you want to
  create objects that interact with arrays via binary operations, but
  are not themselves arrays. For example, a units handling system might have
  an object ``m`` representing the “meters” unit, and want to support the
  syntax ``arr * m`` to represent that the array has units of “meters”, but
  not want to otherwise interact with arrays via ufuncs or otherwise. This
  can be done by setting ``__array_ufunc__ = None`` and defining ``__mul__``
  and ``__rmul__`` methods. (Note that this means that writing an
  ``__array_ufunc__`` that always returns [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) is not
  quite the same as setting ``__array_ufunc__ = None``: in the former
  case, ``arr + obj`` will raise [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError), while in the latter
  case it is possible to define a ``__radd__`` method to prevent this.)

  The above does not hold for in-place operators, for which [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)
  never returns [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented).  Hence, ``arr += obj`` would always
  lead to a [``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError).  This is because for arrays in-place operations
  cannot generically be replaced by a simple reverse operation.  (For
  instance, by default, ``arr += obj`` would be translated to ``arr =
  arr + obj``, i.e., ``arr`` would be replaced, contrary to what is expected
  for in-place array operations.)

  ::: tip Note

  If you define ``__array_ufunc__``:

  - If you are not a subclass of [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray), we recommend your
  class define special methods like ``__add__`` and ``__lt__`` that
  delegate to ufuncs just like ndarray does.  An easy way to do this
  is to subclass from [``NDArrayOperatorsMixin``](https://numpy.org/devdocs/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin).
  - If you subclass [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray), we recommend that you put all your
  override logic in ``__array_ufunc__`` and not also override special
  methods. This ensures the class hierarchy is determined in only one
  place rather than separately by the ufunc machinery and by the binary
  operation rules (which gives preference to special methods of
  subclasses; the alternative way to enforce a one-place only hierarchy,
  of setting [``__array_ufunc__``](#numpy.class.__array_ufunc__) to [``None``](https://docs.python.org/dev/library/constants.html#None), would seem very
  unexpected and thus confusing, as then the subclass would not work at
  all with ufuncs).
  - [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) defines its own [``__array_ufunc__``](#numpy.class.__array_ufunc__), which,
  evaluates the ufunc if no arguments have overrides, and returns
  [``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented) otherwise. This may be useful for subclasses
  for which [``__array_ufunc__``](#numpy.class.__array_ufunc__) converts any instances of its own
  class to [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray): it can then pass these on to its
  superclass using ``super().__array_ufunc__(*inputs, **kwargs)``,
  and finally return the results after possible back-conversion. The
  advantage of this practice is that it ensures that it is possible
  to have a hierarchy of subclasses that extend the behaviour. See
  [Subclassing ndarray](https://numpy.org/devdocs/user/basics.subclassing.html#basics-subclassing) for details.

  :::

  ::: tip Note

  If a class defines the [``__array_ufunc__``](#numpy.class.__array_ufunc__) method,
  this disables the [``__array_wrap__``](#numpy.class.__array_wrap__),
  [``__array_prepare__``](#numpy.class.__array_prepare__), [``__array_priority__``](#numpy.class.__array_priority__) mechanism
  described below for ufuncs (which may eventually be deprecated).

  :::


- ``class.__array_function__``(*func*, *types*, *args*, *kwargs*)

  *New in version 1.16.* 

  ::: tip Note

  - In NumPy 1.17, the protocol is enabled by default, but can be disabled
  with ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0``.
  - In NumPy 1.16, you need to set the environment variable
  ``NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1`` before importing NumPy to use
  NumPy function overrides.
  - Eventually, expect to ``__array_function__`` to always be enabled.

  :::

  - ``func`` is an arbitrary callable exposed by NumPy’s public API,
  which was called in the form ``func(*args, **kwargs)``.
  - ``types`` is a [collection](collections.abc.Collection)
  of unique argument types from the original NumPy function call that
  implement ``__array_function__``.
  - The tuple ``args`` and dict ``kwargs`` are directly passed on from the
  original call.

  As a convenience for ``__array_function__`` implementors, ``types``
  provides all argument types with an ``'__array_function__'`` attribute.
  This allows implementors to quickly identify cases where they should defer
  to ``__array_function__`` implementations on other arguments.
  Implementations should not rely on the iteration order of ``types``.

  Most implementations of ``__array_function__`` will start with two
  checks:

  1. Is the given function something that we know how to overload?
  1. Are all arguments of a type that we know how to handle?

  If these conditions hold, ``__array_function__`` should return the result
  from calling its implementation for ``func(*args, **kwargs)``.  Otherwise,
  it should return the sentinel value ``NotImplemented``, indicating that the
  function is not implemented by these types.

  There are no general requirements on the return value from
  ``__array_function__``, although most sensible implementations should
  probably return array(s) with the same type as one of the function’s
  arguments.

  It may also be convenient to define a custom decorators (``implements``
  below) for registering ``__array_function__`` implementations.

  ``` python
  HANDLED_FUNCTIONS = {}

  class MyArray:
      def __array_function__(self, func, types, args, kwargs):
          if func not in HANDLED_FUNCTIONS:
              return NotImplemented
          # Note: this allows subclasses that don't override
          # __array_function__ to handle MyArray objects
          if not all(issubclass(t, MyArray) for t in types):
              return NotImplemented
          return HANDLED_FUNCTIONS[func](*args, **kwargs)

  def implements(numpy_function):
      """Register an __array_function__ implementation for MyArray objects."""
      def decorator(func):
          HANDLED_FUNCTIONS[numpy_function] = func
          return func
      return decorator

  @implements(np.concatenate)
  def concatenate(arrays, axis=0, out=None):
      ...  # implementation of concatenate for MyArray objects

  @implements(np.broadcast_to)
  def broadcast_to(array, shape):
      ...  # implementation of broadcast_to for MyArray objects
  ```

  Note that it is not required for ``__array_function__`` implementations to
  include *all* of the corresponding NumPy function’s optional arguments
  (e.g., ``broadcast_to`` above omits the irrelevant ``subok`` argument).
  Optional arguments are only passed in to ``__array_function__`` if they
  were explicitly used in the NumPy function call.

  Just like the case for builtin special methods like ``__add__``, properly
  written ``__array_function__`` methods should always return
  ``NotImplemented`` when an unknown type is encountered. Otherwise, it will
  be impossible to correctly override NumPy functions from another object
  if the operation also includes one of your objects.

  For the most part, the rules for dispatch with ``__array_function__``
  match those for ``__array_ufunc__``. In particular:

  - NumPy will gather implementations of ``__array_function__`` from all
  specified inputs and call them in order: subclasses before
  superclasses, and otherwise left to right. Note that in some edge cases
  involving subclasses, this differs slightly from the
  [current behavior](https://bugs.python.org/issue30140) of Python.
  - Implementations of ``__array_function__`` indicate that they can
  handle the operation by returning any value other than
  ``NotImplemented``.
  - If all ``__array_function__`` methods return ``NotImplemented``,
  NumPy will raise ``TypeError``.

  If no ``__array_function__`` methods exists, NumPy will default to calling
  its own implementation, intended for use on NumPy arrays. This case arises,
  for example, when all array-like arguments are Python numbers or lists.
  (NumPy arrays do have a ``__array_function__`` method, given below, but it
  always returns ``NotImplemented`` if any argument other than a NumPy array
  subclass implements ``__array_function__``.)

  One deviation from the current behavior of ``__array_ufunc__`` is that
  NumPy will only call ``__array_function__`` on the *first* argument of each
  unique type. This matches Python’s [rule for calling reflected methods](https://docs.python.org/3/reference/datamodel.html#object.__ror__), and
  this ensures that checking overloads has acceptable performance even when
  there are a large number of overloaded arguments.

- ``class.__array_finalize__``(*obj*)

  This method is called whenever the system internally allocates a
  new array from *obj*, where *obj* is a subclass (subtype) of the
  [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray). It can be used to change attributes of *self*
  after construction (so as to ensure a 2-d matrix for example), or
  to update meta-information from the “parent.” Subclasses inherit
  a default implementation of this method that does nothing.


- ``class.__array_prepare__``(*array*, *context=None*)

  At the beginning of every [ufunc](ufuncs.html#ufuncs-output-type), this
  method is called on the input object with the highest array
  priority, or the output object if one was specified. The output
  array is passed in and whatever is returned is passed to the ufunc.
  Subclasses inherit a default implementation of this method which
  simply returns the output array unmodified. Subclasses may opt to
  use this method to transform the output array into an instance of
  the subclass and update metadata before returning the array to the
  ufunc for computation.

  ::: tip Note

  For ufuncs, it is hoped to eventually deprecate this method in
  favour of [``__array_ufunc__``](#numpy.class.__array_ufunc__).

  :::


- ``class.__array_wrap__``(*array*, *context=None*)

  At the end of every [ufunc](ufuncs.html#ufuncs-output-type), this method
  is called on the input object with the highest array priority, or
  the output object if one was specified. The ufunc-computed array
  is passed in and whatever is returned is passed to the user.
  Subclasses inherit a default implementation of this method, which
  transforms the array into a new instance of the object’s class.
  Subclasses may opt to use this method to transform the output array
  into an instance of the subclass and update metadata before
  returning the array to the user.

  ::: tip Note

  For ufuncs, it is hoped to eventually deprecate this method in
  favour of [``__array_ufunc__``](#numpy.class.__array_ufunc__).

  :::

- ``class.__array_priority__``

  The value of this attribute is used to determine what type of
  object to return in situations where there is more than one
  possibility for the Python type of the returned object. Subclasses
  inherit a default value of 0.0 for this attribute.

  ::: tip Note

  For ufuncs, it is hoped to eventually deprecate this method in
  favour of [``__array_ufunc__``](#numpy.class.__array_ufunc__).

  :::

- ``class.__array__``([*dtype*])

  If a class (ndarray subclass or not) having the [``__array__``](#numpy.class.__array__)
  method is used as the output object of an [ufunc](ufuncs.html#ufuncs-output-type), results will be written to the object
  returned by [``__array__``](#numpy.class.__array__). Similar conversion is done on
  input arrays.

## Matrix objects

::: tip Note

It is strongly advised *not* to use the matrix subclass.  As described
below, it makes writing functions that deal consistently with matrices
and regular arrays very difficult. Currently, they are mainly used for
interacting with ``scipy.sparse``. We hope to provide an alternative
for this use, however, and eventually remove the ``matrix`` subclass.

:::

[``matrix``](https://numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix) objects inherit from the ndarray and therefore, they
have the same attributes and methods of ndarrays. There are six
important differences of matrix objects, however, that may lead to
unexpected results when you use matrices but expect them to act like
arrays:

1. Matrix objects can be created using a string notation to allow Matlab-style syntax where spaces separate columns and semicolons (‘;’) separate rows.
1. Matrix objects are always two-dimensional. This has far-reaching implications, in that m.ravel() is still two-dimensional (with a 1 in the first dimension) and item selection returns two-dimensional objects so that sequence behavior is fundamentally different than arrays.
1. Matrix objects over-ride multiplication to be matrix-multiplication. Make sure you understand this for functions that you may want to receive matrices. Especially in light of the fact that asanyarray(m) returns a matrix when m is a matrix.
1. Matrix objects over-ride power to be matrix raised to a power. The same warning about using power inside a function that uses asanyarray(…) to get an array object holds for this fact.
1. The default \_\_array_priority__ of matrix objects is 10.0, and therefore mixed operations with ndarrays always produce matrices.
1. Matrices have special attributes which make calculations easier. These are

    method | description
    ---|---
    matrix.T | Returns the transpose of the matrix.
    matrix.H | Returns the (complex) conjugate transpose of self.
    matrix.I | Returns the (multiplicative) inverse of invertible self.
    matrix.A | Return self as an ndarray object.

::: danger Warning

Matrix objects over-ride multiplication, ‘*’, and power, ‘**’, to
be matrix-multiplication and matrix power, respectively. If your
subroutine can accept sub-classes and you do not convert to base-
class arrays, then you must use the ufuncs multiply and power to
be sure that you are performing the correct operation for all
inputs.

:::

The matrix class is a Python subclass of the ndarray and can be used
as a reference for how to construct your own subclass of the ndarray.
Matrices can be created from other matrices, strings, and anything
else that can be converted to an ``ndarray`` . The name “mat “is an
alias for “matrix “in NumPy.

method | description
---|---
[matrix](https://numpy.org/devdocs/reference/generated/numpy.matrix.html#numpy.matrix)(data[, dtype, copy]) | **Note:** It is no longer recommended to use this class, even for linear
[asmatrix](https://numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | Interpret the input as a matrix.
[bmat](https://numpy.org/devdocs/reference/generated/numpy.bmat.html#numpy.bmat)(obj[, ldict, gdict]) | Build a matrix object from a string, nested sequence, or array.

Example 1: Matrix creation from a string

``` python
>>> a=mat('1 2 3; 4 5 3')
>>> print (a*a.T).I
[[ 0.2924 -0.1345]
 [-0.1345  0.0819]]
```

Example 2: Matrix creation from nested sequence

``` python
>>> mat([[1,5,10],[1.0,3,4j]])
matrix([[  1.+0.j,   5.+0.j,  10.+0.j],
        [  1.+0.j,   3.+0.j,   0.+4.j]])
```

Example 3: Matrix creation from an array

``` python
>>> mat(random.rand(3,3)).T
matrix([[ 0.7699,  0.7922,  0.3294],
        [ 0.2792,  0.0101,  0.9219],
        [ 0.3398,  0.7571,  0.8197]])
```

## Memory-mapped file arrays

Memory-mapped files are useful for reading and/or modifying small
segments of a large file with regular layout, without reading the
entire file into memory. A simple subclass of the ndarray uses a
memory-mapped file for the data buffer of the array. For small files,
the over-head of reading the entire file into memory is typically not
significant, however for large files using memory mapping can save
considerable resources.

Memory-mapped-file arrays have one additional method (besides those
they inherit from the ndarray): [``.flush()``](https://numpy.org/devdocs/reference/generated/numpy.memmap.flush.html#numpy.memmap.flush) which
must be called manually by the user to ensure that any changes to the
array actually get written to disk.

method | description
---|---
[memmap](https://numpy.org/devdocs/reference/generated/numpy.memmap.html#numpy.memmap) | Create a memory-map to an array stored in a binary file on disk.
[memmap.flush](https://numpy.org/devdocs/reference/generated/numpy.memmap.flush.html#numpy.memmap.flush)(self) | Write any changes in the array to the file on disk.

Example:

``` python
>>> a = memmap('newfile.dat', dtype=float, mode='w+', shape=1000)
>>> a[10] = 10.0
>>> a[30] = 30.0
>>> del a
>>> b = fromfile('newfile.dat', dtype=float)
>>> print b[10], b[30]
10.0 30.0
>>> a = memmap('newfile.dat', dtype=float)
>>> print a[10], a[30]
10.0 30.0
```

## Character arrays (``numpy.char``)

::: tip See also

[Creating character arrays (numpy.char)](routines.array-creation.html#routines-array-creation-char)

:::

::: tip Note

The [``chararray``](https://numpy.org/devdocs/reference/generated/numpy.chararray.html#numpy.chararray) class exists for backwards compatibility with
Numarray, it is not recommended for new development. Starting from numpy
1.4, if one needs arrays of strings, it is recommended to use arrays of
[``dtype``](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype) ``object_``, ``string_`` or ``unicode_``, and use the free functions
in the [``numpy.char``](routines.char.html#module-numpy.char) module for fast vectorized string operations.

:::

These are enhanced arrays of either ``string_`` type or
``unicode_`` type.  These arrays inherit from the
[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray), but specially-define the operations ``+``, ``*``,
and ``%`` on a (broadcasting) element-by-element basis.  These
operations are not available on the standard [``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) of
character type. In addition, the [``chararray``](https://numpy.org/devdocs/reference/generated/numpy.chararray.html#numpy.chararray) has all of the
standard [``string``](https://docs.python.org/dev/library/stdtypes.html#str) (and ``unicode``) methods,
executing them on an element-by-element basis. Perhaps the easiest
way to create a chararray is to use [``self.view(chararray)``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.view.html#numpy.ndarray.view) where *self* is an ndarray of str or unicode
data-type. However, a chararray can also be created using the
[``numpy.chararray``](https://numpy.org/devdocs/reference/generated/numpy.chararray.html#numpy.chararray) constructor, or via the
[``numpy.char.array``](https://numpy.org/devdocs/reference/generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array) function:

method | description
---|---
[chararray](https://numpy.org/devdocs/reference/generated/numpy.chararray.html#numpy.chararray)(shape[, itemsize, unicode, …]) | Provides a convenient view on arrays of string and unicode values.
[core.defchararray.array](https://numpy.org/devdocs/reference/generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array)(obj[, itemsize, …]) | Create a chararray.

Another difference with the standard ndarray of str data-type is
that the chararray inherits the feature introduced by Numarray that
white-space at the end of any element in the array will be ignored
on item retrieval and comparison operations.

## Record arrays (``numpy.rec``)

::: tip See also

[Creating record arrays (numpy.rec)](routines.array-creation.html#routines-array-creation-rec), [Data type routines](routines.dtype.html#routines-dtype),
[Data type objects (dtype)](arrays.dtypes.html#arrays-dtypes).

:::

NumPy provides the [``recarray``](https://numpy.org/devdocs/reference/generated/numpy.recarray.html#numpy.recarray) class which allows accessing the
fields of a structured array as attributes, and a corresponding
scalar data type object [``record``](https://numpy.org/devdocs/reference/generated/numpy.record.html#numpy.record).

method | description
---|---
[recarray](https://numpy.org/devdocs/reference/generated/numpy.recarray.html#numpy.recarray) | Construct an ndarray that allows field access using attributes.
[record](https://numpy.org/devdocs/reference/generated/numpy.record.html#numpy.record) | A data-type scalar that allows field access as attribute lookup.

## Masked arrays (``numpy.ma``)

::: tip See also

[Masked arrays](maskedarray.html#maskedarray)

:::

## Standard container class

For backward compatibility and as a standard “container “class, the
UserArray from Numeric has been brought over to NumPy and named
[``numpy.lib.user_array.container``](https://numpy.org/devdocs/reference/generated/numpy.lib.user_array.container.html#numpy.lib.user_array.container) The container class is a
Python class whose self.array attribute is an ndarray. Multiple
inheritance is probably easier with numpy.lib.user_array.container
than with the ndarray itself and so it is included by default. It is
not documented here beyond mentioning its existence because you are
encouraged to use the ndarray class directly if you can.

method | description
---|---
[numpy.lib.user_array.container](https://numpy.org/devdocs/reference/generated/numpy.lib.user_array.container.html#numpy.lib.user_array.container)(data[, …]) | Standard container-class for easy multiple-inheritance.

## Array Iterators

Iterators are a powerful concept for array processing. Essentially,
iterators implement a generalized for-loop. If *myiter* is an iterator
object, then the Python code:

``` python
for val in myiter:
    ...
    some code involving val
    ...
```

calls ``val = next(myiter)`` repeatedly until [``StopIteration``](https://docs.python.org/dev/library/exceptions.html#StopIteration) is
raised by the iterator. There are several ways to iterate over an
array that may be useful: default iteration, flat iteration, and
-dimensional enumeration.

### Default iteration

The default iterator of an ndarray object is the default Python
iterator of a sequence type. Thus, when the array object itself is
used as an iterator. The default behavior is equivalent to:

``` python
for i in range(arr.shape[0]):
    val = arr[i]
```

This default iterator selects a sub-array of dimension 
from the array. This can be a useful construct for defining recursive
algorithms. To loop over the entire array requires  for-loops.

``` python
>>> a = arange(24).reshape(3,2,4)+10
>>> for val in a:
...     print 'item:', val
item: [[10 11 12 13]
 [14 15 16 17]]
item: [[18 19 20 21]
 [22 23 24 25]]
item: [[26 27 28 29]
 [30 31 32 33]]
```

### Flat iteration

method | description
---|---
[ndarray.flat](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | A 1-D iterator over the array.

As mentioned previously, the flat attribute of ndarray objects returns
an iterator that will cycle over the entire array in C-style
contiguous order.

``` python
>>> for i, val in enumerate(a.flat):
...     if i%5 == 0: print i, val
0 10
5 15
10 20
15 25
20 30
```

Here, I’ve used the built-in enumerate iterator to return the iterator
index as well as the value.

### N-dimensional enumeration

method | description
---|---
[ndenumerate](https://numpy.org/devdocs/reference/generated/numpy.ndenumerate.html#numpy.ndenumerate)(arr) | Multidimensional index iterator.

Sometimes it may be useful to get the N-dimensional index while
iterating. The ndenumerate iterator can achieve this.

``` python
>>> for i, val in ndenumerate(a):
...     if sum(i)%5 == 0: print i, val
(0, 0, 0) 10
(1, 1, 3) 25
(2, 0, 3) 29
(2, 1, 2) 32
```

### Iterator for broadcasting

method | description
---|---
[broadcast](https://numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast) | Produce an object that mimics broadcasting.

The general concept of broadcasting is also available from Python
using the [``broadcast``](https://numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast) iterator. This object takes 
objects as inputs and returns an iterator that returns tuples
providing each of the input sequence elements in the broadcasted
result.

``` python
>>> for val in broadcast([[1,0],[2,3]],[0,1]):
...     print val
(1, 0)
(0, 1)
(2, 0)
(3, 1)
```
