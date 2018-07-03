# 数组操作

## Basic operations

- copyto(dst, src[, casting, where])	Copies values from one array to another, broadcasting as necessary.

## Changing array shape

- reshape(a, newshape[, order])	Gives a new shape to an array without changing its data.
- ravel(a[, order])	Return a contiguous flattened array.
- ndarray.flat	A 1-D iterator over the array.
- ndarray.flatten([order])	Return a copy of the array collapsed into one dimension.

## Transpose-like operations

- moveaxis(a, source, destination)	Move axes of an array to new positions.
- rollaxis(a, axis[, start])	Roll the specified axis backwards, until it lies in a given position.
- swapaxes(a, axis1, axis2)	Interchange two axes of an array.
- ndarray.T	Same as self.transpose(), except that self is returned if self.ndim < 2.
- transpose(a[, axes])	Permute the dimensions of an array.

## Changing number of dimensions

- atleast_1d(*arys)	Convert inputs to arrays with at least one dimension.
- atleast_2d(*arys)	View inputs as arrays with at least two dimensions.
- atleast_3d(*arys)	View inputs as arrays with at least three dimensions.
broadcast	Produce an object that mimics broadcasting.
- broadcast_to(array, shape[, subok])	Broadcast an array to a new shape.
- broadcast_arrays(*args, **kwargs)	Broadcast any number of arrays against each other.
- expand_dims(a, axis)	Expand the shape of an array.
- squeeze(a[, axis])	Remove single-dimensional entries from the shape of an array.

## Changing kind of array

- asarray(a[, dtype, order])	Convert the input to an array.
- asanyarray(a[, dtype, order])	Convert the input to an ndarray, but pass ndarray subclasses through.
- asmatrix(data[, dtype])	Interpret the input as a matrix.
- asfarray(a[, dtype])	Return an array converted to a float type.
- asfortranarray(a[, dtype])	Return an array laid out in Fortran order in memory.
- ascontiguousarray(a[, dtype])	Return a contiguous array in memory (C order).
- asarray_chkfinite(a[, dtype, order])	Convert the input to an array, checking for NaNs or Infs.
asscalar(a)	Convert an array of size 1 to its scalar equivalent.
- require(a[, dtype, requirements])	Return an ndarray of the provided type that satisfies requirements.

## Joining arrays

- concatenate((a1, a2, …)[, axis, out])	Join a sequence of arrays along an existing axis.
- stack(arrays[, axis, out])	Join a sequence of arrays along a new axis.
- column_stack(tup)	Stack 1-D arrays as columns into a 2-D array.
- dstack(tup)	Stack arrays in sequence depth wise (along third axis).
- hstack(tup)	Stack arrays in sequence horizontally (column wise).
- vstack(tup)	Stack arrays in sequence vertically (row wise).
- block(arrays)	Assemble an nd-array from nested lists of blocks.

## Splitting arrays

- split(ary, indices_or_sections[, axis])	Split an array into multiple sub-arrays.
- array_split(ary, indices_or_sections[, axis])	Split an array into multiple sub-arrays.
- dsplit(ary, indices_or_sections)	Split array into multiple sub-arrays along the 3rd axis (depth).
- hsplit(ary, indices_or_sections)	Split an array into multiple sub-arrays horizontally (column-wise).
- vsplit(ary, indices_or_sections)	Split an array into multiple sub-arrays vertically (row-wise).

## Tiling arrays

- tile(A, reps)	Construct an array by repeating A the number of times given by reps.
- repeat(a, repeats[, axis])	Repeat elements of an array.
Adding and removing elements
- delete(arr, obj[, axis])	Return a new array with sub-arrays along an axis deleted.
- insert(arr, obj, values[, axis])	Insert values along the given axis before the given indices.
- append(arr, values[, axis])	Append values to the end of an array.
- resize(a, new_shape)	Return a new array with the specified shape.
- trim_zeros(filt[, trim])	Trim the leading and/or trailing zeros from a 1-D array or sequence.
- unique(ar[, return_index, return_inverse, …])	Find the unique elements of an array.

## Rearranging elements

- flip(m, axis)	Reverse the order of elements in an array along the given axis.
- fliplr(m)	Flip array in the left/right direction.
- flipud(m)	Flip array in the up/down direction.
- reshape(a, newshape[, order])	Gives a new shape to an array without changing its data.
- roll(a, shift[, axis])	Roll array elements along a given axis.
- rot90(m[, k, axes])	Rotate an array by 90 degrees in the plane specified by axes.