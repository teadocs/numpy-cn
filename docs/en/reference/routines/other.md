# Miscellaneous routines

## Performance tuning

method | description
---|---
[setbufsize](generated/numpy.setbufsize.html#numpy.setbufsize)(size) | Set the size of the buffer used in ufuncs.
[getbufsize](generated/numpy.getbufsize.html#numpy.getbufsize)() | Return the size of the buffer used in ufuncs.

## Memory ranges

method | description
---|---
[shares_memory](generated/numpy.shares_memory.html#numpy.shares_memory)(a, b[, max_work]) | Determine if two arrays share memory
[may_share_memory](generated/numpy.may_share_memory.html#numpy.may_share_memory)(a, b[, max_work]) | Determine if two arrays might share memory
[byte_bounds](generated/numpy.byte_bounds.html#numpy.byte_bounds)(a) | Returns pointers to the end-points of an array.

## Array mixins

method | description
---|---
[lib.mixins.NDArrayOperatorsMixin](generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin) | Mixin defining all operator special methods using __array_ufunc__.

## NumPy version comparison

method | description
---|---
[lib.NumpyVersion](generated/numpy.lib.NumpyVersion.html#numpy.lib.NumpyVersion)(vstring) | Parse and compare numpy version strings.

## Utility

method | description
---|---
[get_include](generated/numpy.get_include.html#numpy.get_include)() | Return the directory that contains the NumPy *.h header files.
[deprecate](generated/numpy.deprecate.html#numpy.deprecate)(\*args, \*\*kwargs) | Issues a DeprecationWarning, adds warning to old_name’s docstring, rebinds old_name.__name__ and returns the new function object.
[deprecate_with_doc](generated/numpy.deprecate_with_doc.html#numpy.deprecate_with_doc)(msg) | 

## Matlab-like Functions

method | description
---|---
[who](generated/numpy.who.html#numpy.who)([vardict]) | Print the NumPy arrays in the given dictionary.
[disp](generated/numpy.disp.html#numpy.disp)(mesg[, device, linefeed]) | Display a message on a device.