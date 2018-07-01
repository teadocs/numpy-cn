# 通用函数相关

# ``ufunc``

## Optional keyword arguments

All ufuncs take optional keyword arguments. Most of these represent advanced usage and will not typically be used.

*out*

> *New in version 1.6.*
> 
> The first output can be provided as either a positional or a keyword parameter. Keyword ‘out’ arguments are incompatible with positional ones.
> 
> *New in version 1.10.*
> 
> The ‘out’ keyword argument is expected to be a tuple with one entry per output (which can be None for arrays to be allocated by the ufunc). For ufuncs with a single output, passing a single array (instead of a tuple holding a single array) is also valid.
> 
> Passing a single array in the ‘out’ keyword argument to a ufunc with multiple outputs is deprecated, and will raise a warning in numpy 1.10, and an error in a future release.

*where*

> *New in version 1.7.*
> 
> Accepts a boolean array which is broadcast together with the operands. Values of True indicate to calculate the ufunc at that position, values of False indicate to leave the value in the output alone. This argument cannot be used for generalized ufuncs as those take non-scalar input.

*casting*

> *New in version 1.6.*
> 
> May be ‘no’, ‘equiv’, ‘safe’, ‘same_kind’, or ‘unsafe’. See can_cast for explanations of the parameter values.
> 
> Provides a policy for what kind of casting is permitted. For compatibility with previous versions of NumPy, this defaults to ‘unsafe’ for numpy < 1.7. In numpy 1.7 a transition to ‘same_kind’ was begun where ufuncs produce a DeprecationWarning for calls which are allowed under the ‘unsafe’ rules, but not under the ‘same_kind’ rules. From numpy 1.10 and onwards, the default is ‘same_kind’.

*order*

> *New in version 1.6.*
> 
> Specifies the calculation iteration order/memory layout of the output array. Defaults to ‘K’. ‘C’ means the output should be C-contiguous, ‘F’ means F-contiguous, ‘A’ means F-contiguous if the inputs are F-contiguous and not also not C-contiguous, C-contiguous otherwise, and ‘K’ means to match the element ordering of the inputs as closely as possible.

*dtype*

> *New in version 1.6.*
> 
> Overrides the dtype of the calculation and output arrays. Similar to signature.

*subok*

> *New in version 1.6.*
> 
> Defaults to true. If set to false, the output will always be a strict array, not a subtype.

*signature*

> Either a data-type, a tuple of data-types, or a special signature string indicating the input and output types of a ufunc. This argument allows you to provide a specific signature for the 1-d loop to use in the underlying calculation. If the loop specified does not exist for the ufunc, then a TypeError is raised. Normally, a suitable loop is found automatically by comparing the input types with what is available and searching for a loop with data-types to which all inputs can be cast safely. This keyword argument lets you bypass that search and choose a particular loop. A list of available signatures is provided by the types attribute of the ufunc object. For backwards compatibility this argument can also be provided as sig, although the long form is preferred. Note that this should not be confused with the generalized ufunc signature that is stored in the signature attribute of the of the ufunc object.

*extobj*

> a list of length 1, 2, or 3 specifying the ufunc buffer-size, the error mode integer, and the error call-back function. Normally, these values are looked up in a thread-specific dictionary. Passing them here circumvents that look up and uses the low-level specification provided for the error mode. This may be useful, for example, as an optimization for calculations requiring many ufunc calls on small arrays in a loop.

## Attributes

There are some informational attributes that universal functions possess. None of the attributes can be set.

- __doc__	A docstring for each ufunc. The first part of the docstring is dynamically generated from the number of outputs, the name, and the number of inputs. The second part of the docstring is provided at creation time and stored with the ufunc.
- __name__	The name of the ufunc.
``ufunc.nin``	The number of inputs.
``ufunc.nout``	The number of outputs.
``ufunc.nargs``	The number of arguments.
``ufunc.ntypes``	The number of types.
``ufunc.types``	Returns a list with types grouped input->output.
``ufunc.identity``	The identity value.
``ufunc.signature``	Definition of the core elements a generalized ufunc operates on.

## Methods

All ufuncs have four methods. However, these methods only make sense on scalar ufuncs that take two input arguments and return one output argument. Attempting to call these methods on other ufuncs will cause a ``ValueError``. The reduce-like methods all take an axis keyword, a dtype keyword, and an out keyword, and the arrays must all have dimension >= 1. The axis keyword specifies the axis of the array over which the reduction will take place (with negative values counting backwards). Generally, it is an integer, though for ``ufunc.reduce``, it can also be a tuple of ``int`` to reduce over several axes at once, or None, to reduce over all axes. The dtype keyword allows you to manage a very common problem that arises when naively using ``ufunc.reduce``. Sometimes you may have an array of a certain data type and wish to add up all of its elements, but the result does not fit into the data type of the array. This commonly happens if you have an array of single-byte integers. The dtype keyword allows you to alter the data type over which the reduction takes place (and therefore the type of the output). Thus, you can ensure that the output is a data type with precision large enough to handle your output. The responsibility of altering the reduce type is mostly up to you. There is one exception: if no dtype is given for a reduction on the “add” or “multiply” operations, then if the input type is an integer (or Boolean) data-type and smaller than the size of the ``int_`` data type, it will be internally upcast to the ``int_`` (or ``uint``) data-type. Finally, the out keyword allows you to provide an output array (for single-output ufuncs, which are currently the only ones supported; for future extension, however, a tuple with a single argument can be passed in). If out is given, the dtype argument is ignored.

Ufuncs also have a fifth method that allows in place operations to be performed using fancy indexing. No buffering is used on the dimensions where fancy indexing is used, so the fancy index can list an item more than once and the operation will be performed on the result of the previous operation for that item.

- ``ufunc.reduce``(a[, axis, dtype, out, keepdims])	Reduces a’s dimension by one, by applying ufunc along one axis.
- ``ufunc.accumulate``(array[, axis, dtype, out])	Accumulate the result of applying the operator to all elements.
- ``ufunc.reduceat``(a, indices[, axis, dtype, out])	Performs a (local) reduce with specified slices over a single axis.
- ``ufunc.outer``(A, B, **kwargs)	Apply the ufunc op to all pairs (a, b) with a in A and b in B.
- ``ufunc.at``(a, indices[, b])	Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’.

<div class="waring-warp">
<div>Warning</b>
<p>A reduce-like operation on an array with a data-type that has a range “too small” to handle the result will silently wrap. One should use <code>dtype</code> to increase the size of the data-type over which reduction takes place.</p>
</div>