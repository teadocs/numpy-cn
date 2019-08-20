# Floating point error handling

## Setting and getting error handling

method | description
---|---
[seterr](generated/numpy.seterr.html#numpy.seterr)([all, divide, over, under, invalid]) | Set how floating-point errors are handled.
[geterr](generated/numpy.geterr.html#numpy.geterr)() | Get the current way of handling floating-point errors.
[seterrcall](generated/numpy.seterrcall.html#numpy.seterrcall)(func) | Set the floating-point error callback function or log object.
[geterrcall](generated/numpy.geterrcall.html#numpy.geterrcall)() | Return the current callback function used on floating-point errors.
[errstate](generated/numpy.errstate.html#numpy.errstate)(**kwargs) | Context manager for floating-point error handling.

## Internal functions

method | description
---|---
[seterrobj](generated/numpy.seterrobj.html#numpy.seterrobj)(errobj) | Set the object that defines floating-point error handling.
[geterrobj](generated/numpy.geterrobj.html#numpy.geterrobj)() | Return the current object that defines floating-point error handling.