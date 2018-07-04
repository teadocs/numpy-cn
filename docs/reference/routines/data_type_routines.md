# 数据类型操作

- can_cast(from_, to[, casting])	Returns True if cast between data types can occur according to the casting rule.
- promote_types(type1, type2)	Returns the data type with the smallest size and smallest scalar kind to which both type1 and type2 may be safely cast.
- min_scalar_type(a)	For scalar a, returns the data type with the smallest size and smallest scalar kind which can hold its value.
- result_type(*arrays_and_dtypes)	Returns the type that results from applying the NumPy type promotion rules to the arguments.
- common_type(*arrays)	Return a scalar type which is common to the input arrays.
- obj2sctype(rep[, default])	Return the scalar dtype or NumPy equivalent of Python type of an object.

## Creating data types

- dtype(obj[, align, copy])	Create a data type object.
- format_parser(formats, names, titles[, …])	Class to convert formats, names, titles description to a dtype.

## Data type information

- finfo(dtype)	Machine limits for floating point types.
- iinfo(type)	Machine limits for integer types.
- MachAr([float_conv, int_conv, …])	Diagnosing machine parameters.

## Data type testing

- issctype(rep)	Determines whether the given object represents a scalar data-type.
- issubdtype(arg1, arg2)	Returns True if first argument is a typecode lower/equal in type hierarchy.
- issubsctype(arg1, arg2)	Determine if the first argument is a subclass of the second argument.
- issubclass_(arg1, arg2)	Determine if a class is a subclass of a second class.
- find_common_type(array_types, scalar_types)	Determine common type following standard coercion rules.

## Miscellaneous

- typename(char)	Return a description for the given data type code.
- sctype2char(sctype)	Return the string representation of a scalar dtype.
- mintypecode(typechars[, typeset, default])	Return the character for the minimum-size type to which given types can be safely cast.