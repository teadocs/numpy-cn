# Binary operations

## Elementwise bit operations

method | description
---|---
[bitwise_and](generated/numpy.bitwise_and.html#numpy.bitwise_and)(x1, x2, /[, out, where, …]) | Compute the bit-wise AND of two arrays element-wise.
[bitwise_or](generated/numpy.bitwise_or.html#numpy.bitwise_or)(x1, x2, /[, out, where, casting, …]) | Compute the bit-wise OR of two arrays element-wise.
[bitwise_xor](generated/numpy.bitwise_xor.html#numpy.bitwise_xor)(x1, x2, /[, out, where, …]) | Compute the bit-wise XOR of two arrays element-wise.
[invert](generated/numpy.invert.html#numpy.invert)(x, /[, out, where, casting, order, …]) | Compute bit-wise inversion, or bit-wise NOT, element-wise.
[left_shift](generated/numpy.left_shift.html#numpy.left_shift)(x1, x2, /[, out, where, casting, …]) | Shift the bits of an integer to the left.
[right_shift](generated/numpy.right_shift.html#numpy.right_shift)(x1, x2, /[, out, where, …]) | Shift the bits of an integer to the right.

## Bit packing

method | description
---|---
[packbits](generated/numpy.packbits.html#numpy.packbits)(a[, axis, bitorder]) | Packs the elements of a binary-valued array into bits in a uint8 array.
[unpackbits](generated/numpy.unpackbits.html#numpy.unpackbits)(a[, axis, count, bitorder]) | Unpacks elements of a uint8 array into a binary-valued output array.

## Output formatting

method | description
---|---
[binary_repr](generated/numpy.binary_repr.html#numpy.binary_repr)(num[, width]) | Return the binary representation of the input number as a string.