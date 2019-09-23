# Data type routines

method | description
---|---
[can_cast](https://numpy.org/devdocs/reference/generated/numpy.can_cast.html#numpy.can_cast)(from_, to[, casting]) | Returns True if cast between data types can occur according to the casting rule.
[promote_types](https://numpy.org/devdocs/reference/generated/numpy.promote_types.html#numpy.promote_types)(type1, type2) | Returns the data type with the smallest size and smallest scalar kind to which both type1 and type2 may be safely cast.
[min_scalar_type](https://numpy.org/devdocs/reference/generated/numpy.min_scalar_type.html#numpy.min_scalar_type)(a) | For scalar a, returns the data type with the smallest size and smallest scalar kind which can hold its value.
[result_type](https://numpy.org/devdocs/reference/generated/numpy.result_type.html#numpy.result_type)(*arrays_and_dtypes) | Returns the type that results from applying the NumPy type promotion rules to the arguments.
[common_type](https://numpy.org/devdocs/reference/generated/numpy.common_type.html#numpy.common_type)(\*arrays) | Return a scalar type which is common to the input arrays.
[obj2sctype](https://numpy.org/devdocs/reference/generated/numpy.obj2sctype.html#numpy.obj2sctype)(rep[, default]) | Return the scalar dtype or NumPy equivalent of Python type of an object.

## Creating data types

method | description
---|---
[dtype](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype)(obj[, align, copy]) | Create a data type object.
[format_parser](https://numpy.org/devdocs/reference/generated/numpy.format_parser.html#numpy.format_parser)(formats, names, titles[, …]) | Class to convert formats, names, titles description to a dtype.

## Data type information

method | description
---|---
[finfo](https://numpy.org/devdocs/reference/generated/numpy.finfo.html#numpy.finfo)(dtype) | Machine limits for floating point types.
[iinfo](https://numpy.org/devdocs/reference/generated/numpy.iinfo.html#numpy.iinfo)(type) | Machine limits for integer types.
[MachAr](https://numpy.org/devdocs/reference/generated/numpy.MachAr.html#numpy.MachAr)([float_conv, int_conv, …]) | Diagnosing machine parameters.

## Data type testing

method | description
---|---
[issctype](https://numpy.org/devdocs/reference/generated/numpy.issctype.html#numpy.issctype)(rep) | Determines whether the given object represents a scalar data-type.
[issubdtype](https://numpy.org/devdocs/reference/generated/numpy.issubdtype.html#numpy.issubdtype)(arg1, arg2) | Returns True if first argument is a typecode lower/equal in type hierarchy.
[issubsctype](https://numpy.org/devdocs/reference/generated/numpy.issubsctype.html#numpy.issubsctype)(arg1, arg2) | Determine if the first argument is a subclass of the second argument.
[issubclass_](https://numpy.org/devdocs/reference/generated/numpy.issubclass_.html#numpy.issubclass_)(arg1, arg2) | Determine if a class is a subclass of a second class.
[find_common_type](https://numpy.org/devdocs/reference/generated/numpy.find_common_type.html#numpy.find_common_type)(array_types, scalar_types) | Determine common type following standard coercion rules.

## Miscellaneous

method | description
---|---
[typename](https://numpy.org/devdocs/reference/generated/numpy.typename.html#numpy.typename)(char) | Return a description for the given data type code.
[sctype2char](https://numpy.org/devdocs/reference/generated/numpy.sctype2char.html#numpy.sctype2char)(sctype) | Return the string representation of a scalar dtype.
[mintypecode](https://numpy.org/devdocs/reference/generated/numpy.mintypecode.html#numpy.mintypecode)(typechars[, typeset, default]) | Return the character for the minimum-size type to which given types can be safely cast.
[maximum_sctype](https://numpy.org/devdocs/reference/generated/numpy.maximum_sctype.html#numpy.maximum_sctype)(t) | Return the scalar type of highest precision of the same kind as the input.