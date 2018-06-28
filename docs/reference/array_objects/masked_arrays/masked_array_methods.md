# MaskedArray的方法

另见：

> Array methods

## Conversion

- MaskedArray.__float__()Convert to float.
- MaskedArray.__hex__	
- MaskedArray.__int__()	Convert to int.
- MaskedArray.__long__()	Convert to long.
- MaskedArray.__oct__	
- MaskedArray.view([dtype, type])	New view of array with the same data.
- MaskedArray.astype(newtype)	Returns a copy of the MaskedArray cast to given newtype.
- MaskedArray.byteswap([inplace])	Swap the bytes of the array elements
- MaskedArray.compressed()	Return all the non-masked data as a 1-D array.
- MaskedArray.filled([fill_value])	Return a copy of self, with masked values filled with a given value.

- MaskedArray.tofile(fid[, sep, format])	Save a masked array to a file in binary format.
- MaskedArray.toflex()	Transforms a masked array into a flexible-type array.
- MaskedArray.tolist([fill_value])	Return the data portion of the masked array as a hierarchical Python list.

- MaskedArray.torecords()	Transforms a masked array into a flexible-type array.
- MaskedArray.tostring([fill_value, order])	This function is a compatibility alias for tobytes.
- MaskedArray.tobytes([fill_value, order])	Return the array data as a string containing the raw bytes in the array.

## Shape manipulation

For reshape, resize, and transpose, the single tuple argument may be replaced with n integers which will be interpreted as an n-tuple.

- MaskedArray.flatten([order])	Return a copy of the array collapsed into one dimension.
- MaskedArray.ravel([order])	Returns a 1D version of self, as a view.
- MaskedArray.reshape(*s, **kwargs)	Give a new shape to the array without changing its data.
- MaskedArray.resize(newshape[, refcheck, order])	
- MaskedArray.squeeze([axis])	Remove single-dimensional entries from the shape of a.
- MaskedArray.swapaxes(axis1, axis2)	Return a view of the array with axis1 and axis2 interchanged.
- MaskedArray.transpose(*axes)	Returns a view of the array with axes transposed.
- MaskedArray.T	

## Item selection and manipulation

For array methods that take an axis keyword, it defaults to None. If axis is None, then the array is treated as a 1-D array. Any other value for axis represents the dimension along which the operation should proceed.

- MaskedArray.argmax([axis, fill_value, out])	Returns array of indices of the maximum values along the given axis.
- MaskedArray.argmin([axis, fill_value, out])	Return array of indices to the minimum values along the given axis.
- MaskedArray.argsort([axis, kind, order, …])	Return an ndarray of indices that sort the array along the specified axis.
- MaskedArray.choose(choices[, out, mode])	Use an index array to construct a new array from a set of choices.
- MaskedArray.compress(condition[, axis, out])	Return a where condition is True.
- MaskedArray.diagonal([offset, axis1, axis2])	Return specified diagonals.
- MaskedArray.fill(value)	Fill the array with a scalar value.
- MaskedArray.item(*args)	Copy an element of an array to a standard Python scalar and return it.
- MaskedArray.nonzero()	Return the indices of unmasked elements that are not zero.
- MaskedArray.put(indices, values[, mode])	Set storage-indexed locations to corresponding values.
- MaskedArray.repeat(repeats[, axis])	Repeat elements of an array.
- MaskedArray.searchsorted(v[, side, sorter])	Find indices where elements of v should be inserted in a to maintain order.
- MaskedArray.sort([axis, kind, order, …])	Sort the array, in-place
- MaskedArray.take(indices[, axis, out, mode])	

## Pickling and copy

- MaskedArray.copy([order])	Return a copy of the array.
- MaskedArray.dump(file)	Dump a pickle of the array to the specified file.
- MaskedArray.dumps()	Returns the pickle of the array as a string.

## Calculations

- MaskedArray.all([axis, out, keepdims])	Returns True if all elements evaluate to True.
- MaskedArray.anom([axis, dtype])	Compute the anomalies (deviations from the arithmetic mean) along the given axis.
- MaskedArray.any([axis, out, keepdims])	Returns True if any of the elements of a evaluate to True.
- MaskedArray.clip([min, max, out])	Return an array whose values are limited to [min, max].
- MaskedArray.conj()	Complex-conjugate all elements.
- MaskedArray.conjugate()	Return the complex conjugate, element-wise.
- MaskedArray.cumprod([axis, dtype, out])	Return the cumulative product of the array elements over the given axis.
- MaskedArray.cumsum([axis, dtype, out])	Return the cumulative sum of the array elements over the given axis.
- MaskedArray.max([axis, out, fill_value, …])	Return the maximum along a given axis.
- MaskedArray.mean([axis, dtype, out, keepdims])	Returns the average of the array elements along given axis.
- MaskedArray.min([axis, out, fill_value, …])	Return the minimum along a given axis.
- MaskedArray.prod([axis, dtype, out, keepdims])	Return the product of the array elements over the given axis.
- MaskedArray.product([axis, dtype, out, keepdims])	Return the product of the array elements over the given axis.
- MaskedArray.ptp([axis, out, fill_value])	Return (maximum - minimum) along the given dimension (i.e.
- MaskedArray.round([decimals, out])	Return each element rounded to the given number of decimals.
- MaskedArray.std([axis, dtype, out, ddof, …])	Returns the standard deviation of the array elements along given axis.
- MaskedArray.sum([axis, dtype, out, keepdims])	Return the sum of the array elements over the given axis.
- MaskedArray.trace([offset, axis1, axis2, …])	Return the sum along diagonals of the array.
- MaskedArray.var([axis, dtype, out, ddof, …])	Compute the variance along the specified axis.

## Arithmetic and comparison operations

### Comparison operators:

- MaskedArray.__lt__($self, value, /)	Return self<value.
- MaskedArray.__le__($self, value, /)	Return self<=value.
- MaskedArray.__gt__($self, value, /)	Return self>value.
- MaskedArray.__ge__($self, value, /)	Return self>=value.
- MaskedArray.__eq__(other)	Check whether other equals self elementwise.
- MaskedArray.__ne__(other)	Check whether other does not equal self elementwise.

### Truth value of an array (``bool``):

- MaskedArray.__nonzero__	

### Arithmetic:

- MaskedArray.__abs__(self)	
- MaskedArray.__add__(other)	Add self to other, and return a new masked array.
- MaskedArray.__radd__(other)	Add other to self, and return a new masked array.
- MaskedArray.__sub__(other)	Subtract other from self, and return a new masked array.
- MaskedArray.__rsub__(other)	Subtract self from other, and return a new masked array.
- MaskedArray.__mul__(other)	Multiply self by other, and return a new masked array.
- MaskedArray.__rmul__(other)	Multiply other by self, and return a new masked array.
- MaskedArray.__div__(other)	Divide other into self, and return a new masked array.
- MaskedArray.__rdiv__	
- MaskedArray.__truediv__(other)	Divide other into self, and return a new masked array.
- MaskedArray.__rtruediv__(other)	Divide self into other, and return a new masked array.
- MaskedArray.__floordiv__(other)	Divide other into self, and return a new masked array.
- MaskedArray.__rfloordiv__(other)	Divide self into other, and return a new masked array.
- MaskedArray.__mod__($self, value, /)	Return self%value.
- MaskedArray.__rmod__($self, value, /)	Return value%self.
- MaskedArray.__divmod__($self, value, /)	Return divmod(self, value).
- MaskedArray.__rdivmod__($self, value, /)	Return divmod(value, self).
- MaskedArray.__pow__(other)	Raise self to the power other, masking the potential NaNs/Infs
- MaskedArray.__rpow__(other)	Raise other to the power self, masking the potential NaNs/Infs
- MaskedArray.__lshift__($self, value, /)	Return self<<value.
- MaskedArray.__rlshift__($self, value, /)	Return value<<self.
- MaskedArray.__rshift__($self, value, /)	Return self>>value.
- MaskedArray.__rrshift__($self, value, /)	Return value>>self.
- MaskedArray.__and__($self, value, /)	Return self&value.
- MaskedArray.__rand__($self, value, /)	Return value&self.
- MaskedArray.__or__($self, value, /)	Return self|value.
- MaskedArray.__ror__($self, value, /)	Return value|self.
- MaskedArray.__xor__($self, value, /)	Return self^value.
- MaskedArray.__rxor__($self, value, /)	Return value^self.

### Arithmetic, in-place:

- MaskedArray.__iadd__(other)	Add other to self in-place.
- MaskedArray.__isub__(other)	Subtract other from self in-place.
- MaskedArray.__imul__(other)	Multiply self by other in-place.
- MaskedArray.__idiv__(other)	Divide self by other in-place.
- MaskedArray.__itruediv__(other)	True divide self by other in-place.
- MaskedArray.__ifloordiv__(other)	Floor divide self by other in-place.
- MaskedArray.__imod__($self, value, /)	Return self%=value.
- MaskedArray.__ipow__(other)	Raise self to the power other, in place.
- MaskedArray.__ilshift__($self, value, /)	Return self<<=value.
- MaskedArray.__irshift__($self, value, /)	Return self>>=value.
- MaskedArray.__iand__($self, value, /)	Return self&=value.
- MaskedArray.__ior__($self, value, /)	Return self|=value.
- MaskedArray.__ixor__($self, value, /)	Return self^=value.

## Representation

- MaskedArray.__repr__()	Literal string representation.
- MaskedArray.__str__()	Return str(self).
- MaskedArray.ids()	Return the addresses of the data and mask areas.
- MaskedArray.iscontiguous()	Return a boolean indicating whether the data is contiguous.

## Special methods

For standard library functions:

- MaskedArray.__copy__()	Used if copy.copy is called on an array.
- MaskedArray.__deepcopy__(memo, /)	Used if copy.deepcopy is called on an array.
- MaskedArray.__getstate__()	Return the internal state of the masked array, for pickling purposes.
- MaskedArray.__reduce__()	Return a 3-tuple for pickling a MaskedArray.
- MaskedArray.__setstate__(state)	Restore the internal state of the masked array, for pickling purposes.

Basic customization:

- MaskedArray.__new__([data, mask, dtype, …])	Create a new masked array from scratch.
- MaskedArray.__array__(|dtype)	Returns either a new reference to self if dtype is not given or a new array of provided data type if dtype is different from the current dtype of the array.
- MaskedArray.__array_wrap__(obj[, context])	Special hook for ufuncs.

Container customization: (see Indexing)

- MaskedArray.__len__($self, /)	Return len(self).
- MaskedArray.__getitem__(indx)	x.__getitem__(y) <==> x[y]
- MaskedArray.__setitem__(indx, value)	x.__setitem__(i, y) <==> x[i]=y
- MaskedArray.__delitem__($self, key, /)	Delete self[key].
- MaskedArray.__contains__($self, key, /)	Return key in self.

## Specific methods

### Handling the mask

The following methods can be used to access information about the mask or to manipulate the mask.

- ``MaskedArray.__setmask__``(mask[, copy])	Set the mask.
- ``MaskedArray.harden_mask``()	Force the mask to hard.
- ``MaskedArray.soften_mask``()	Force the mask to soft.
- ``MaskedArray.unshare_mask``()	Copy the mask and set the sharedmask flag to False.
- ``MaskedArray.shrink_mask``()	Reduce a mask to nomask when possible.

### Handling the fill_value

- ``MaskedArray.get_fill_value``()	Return the filling value of the masked array.
- ``MaskedArray.set_fill_value``([value])	Set the filling value of the masked array.

### Counting the missing elements

- ``MaskedArray.count``([axis, keepdims])	Count the non-masked elements of the array along the given axis.