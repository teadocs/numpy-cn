# 杂项API

## Buffer objects

- getbuffer	
- newbuffer	

## Performance tuning

- setbufsize(size)	Set the size of the buffer used in ufuncs.
- getbufsize()	Return the size of the buffer used in ufuncs.

## Memory ranges

- shares_memory(a, b[, max_work])	Determine if two arrays share memory
- may_share_memory(a, b[, max_work])	Determine if two arrays might share memory

## Array mixins

- lib.mixins.NDArrayOperatorsMixin	Mixin defining all operator special methods using __array_ufunc__.

## NumPy version comparison

- lib.NumpyVersion(vstring)	Parse and compare numpy version strings.