# 二进制操作

## Elementwise bit operations

- bitwise_and(x1, x2, /[, out, where, …])	Compute the bit-wise AND of two arrays element-wise.
- bitwise_or(x1, x2, /[, out, where, casting, …])	Compute the bit-wise OR of two arrays element-wise.
- bitwise_xor(x1, x2, /[, out, where, …])	Compute the bit-wise XOR of two arrays element-wise.
- invert(x, /[, out, where, casting, order, …])	Compute bit-wise inversion, or bit-wise NOT, element-wise.
- left_shift(x1, x2, /[, out, where, casting, …])	Shift the bits of an integer to the left.
- right_shift(x1, x2, /[, out, where, …])	Shift the bits of an integer to the right.

## Bit packing

- packbits(myarray[, axis])	Packs the elements of a binary-valued array into bits in a uint8 array.
- unpackbits(myarray[, axis])	Unpacks elements of a uint8 array into a binary-valued output array.

## Output formatting

- binary_repr(num[, width])	Return the binary representation of the input number as a string.