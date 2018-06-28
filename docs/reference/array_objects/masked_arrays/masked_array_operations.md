# 操作掩码数组

## Constants

``ma.MaskType``	alias of ``numpy.bool_``

## Creation

### From existing data

- ma.masked_array	alias of numpy.ma.core.MaskedArray
- ma.array(data[, dtype, copy, order, mask, …])	An array class with possibly masked values.
- ma.copy(self, *args, **params) a.copy(order=)	Return a copy of the array.
- ma.frombuffer(buffer[, dtype, count, offset])	Interpret a buffer as a 1-dimensional array.
- ma.fromfunction(function, shape, **kwargs)	Construct an array by executing a function over each coordinate.
- ma.MaskedArray.copy([order])	Return a copy of the array.

### Ones and zeros

- ma.empty(shape[, dtype, order])	Return a new array of given shape and type, without initializing entries.
- ma.empty_like(a[, dtype, order, subok])	Return a new array with the same shape and type as a given array.
- ma.masked_all(shape[, dtype])	Empty masked array with all elements masked.
- ma.masked_all_like(arr)	Empty masked array with the properties of an existing array.
- ma.ones(shape[, dtype, order])	Return a new array of given shape and type, filled with ones.
- ma.zeros(shape[, dtype, order])	Return a new array of given shape and type, filled with zeros.

### Inspecting the array

- ma.all(self[, axis, out, keepdims])	Returns True if all elements evaluate to True.
- ma.any(self[, axis, out, keepdims])	Returns True if any of the elements of a evaluate to True.
- ma.count(self[, axis, keepdims])	Count the non-masked elements of the array along the given axis.
- ma.count_masked(arr[, axis])	Count the number of masked elements along the given axis.
- ma.getmask(a)	Return the mask of a masked array, or nomask.
- ma.getmaskarray(arr)	Return the mask of a masked array, or full boolean array of False.
- ma.getdata(a[, subok])	Return the data of a masked array as an ndarray.
- ma.nonzero(self)	Return the indices of unmasked elements that are not zero.
- ma.shape(obj)	Return the shape of an array.
- ma.size(obj[, axis])	Return the number of elements along a given axis.
- ma.is_masked(x)	Determine whether input has masked values.
- ma.is_mask(m)	Return True if m is a valid, standard mask.
- ma.MaskedArray.data	Return the current data, as a view of the original underlying data.
- ma.MaskedArray.mask	Mask
- ma.MaskedArray.recordmask	Return the mask of the records.
- ma.MaskedArray.all([axis, out, keepdims])	Returns True if all elements evaluate to True.
- ma.MaskedArray.any([axis, out, keepdims])	Returns True if any of the elements of a evaluate to True.
- ma.MaskedArray.count([axis, keepdims])	Count the non-masked elements of the array along the given axis.
- ma.MaskedArray.nonzero()	Return the indices of unmasked elements that are not zero.
- ma.shape(obj)	Return the shape of an array.
- ma.size(obj[, axis])	Return the number of elements along a given axis.

## Manipulating a MaskedArray

### Changing the shape

- ma.ravel(self[, order])	Returns a 1D version of self, as a view.
- ma.reshape(a, new_shape[, order])	Returns an array containing the same data with a new shape.
- ma.resize(x, new_shape)	Return a new masked array with the specified size and shape.
- ma.MaskedArray.flatten([order])	Return a copy of the array collapsed into one dimension.
- ma.MaskedArray.ravel([order])	Returns a 1D version of self, as a view.
- ma.MaskedArray.reshape(*s, **kwargs)	Give a new shape to the array without changing its data.
- ma.MaskedArray.resize(newshape[, refcheck, …])	

### Modifying axes

- ma.swapaxes(self, *args, …)	Return a view of the array with axis1 and axis2 interchanged.
- ma.transpose(a[, axes])	Permute the dimensions of an array.
- ma.MaskedArray.swapaxes(axis1, axis2)	Return a view of the array with axis1 and axis2 interchanged.
- ma.MaskedArray.transpose(*axes)	Returns a view of the array with axes transposed.

### Changing the number of dimensions

- ma.atleast_1d(*arys)	Convert inputs to arrays with at least one dimension.
- ma.atleast_2d(*arys)	View inputs as arrays with at least two dimensions.
- ma.atleast_3d(*arys)	View inputs as arrays with at least three dimensions.
- ma.expand_dims(x, axis)	Expand the shape of an array.
- ma.squeeze(a[, axis])	Remove single-dimensional entries from the shape of an array.
- ma.MaskedArray.squeeze([axis])	Remove single-dimensional entries from the shape of a.
- ma.column_stack(tup)	Stack 1-D arrays as columns into a 2-D array.
- ma.concatenate(arrays[, axis])	Concatenate a sequence of arrays along the given axis.
- ma.dstack(tup)	Stack arrays in sequence depth wise (along third axis).
- ma.hstack(tup)	Stack arrays in sequence horizontally (column wise).
- ma.hsplit(ary, indices_or_sections)	Split an array into multiple sub-arrays horizontally (column-wise).
- ma.mr_	Translate slice objects to concatenation along the first axis.
- ma.row_stack(tup)	Stack arrays in sequence vertically (row wise).
- ma.vstack(tup)	Stack arrays in sequence vertically (row wise).

### Joining arrays

- ma.column_stack(tup)	Stack 1-D arrays as columns into a 2-D array.
- ma.concatenate(arrays[, axis])	Concatenate a sequence of arrays along the given axis.
- ma.append(a, b[, axis])	Append values to the end of an array.
- ma.dstack(tup)	Stack arrays in sequence depth wise (along third axis).
- ma.hstack(tup)	Stack arrays in sequence horizontally (column wise).
- ma.vstack(tup)	Stack arrays in sequence vertically (row wise).

## Operations on masks

### Creating a mask

- ma.make_mask(m[, copy, shrink, dtype])	Create a boolean mask from an array.
- ma.make_mask_none(newshape[, dtype])	Return a boolean mask of the given shape, filled with False.
- ma.mask_or(m1, m2[, copy, shrink])	Combine two masks with the logical_or operator.
- ma.make_mask_descr(ndtype)	Construct a dtype description list from a given dtype.

### Accessing a mask

- ma.getmask(a)	Return the mask of a masked array, or nomask.
- ma.getmaskarray(arr)	Return the mask of a masked array, or full boolean array of False.
- ma.masked_array.mask	Mask

### Finding masked data

- ma.flatnotmasked_contiguous(a)	Find contiguous unmasked data in a masked array along the given axis.
- ma.flatnotmasked_edges(a)	Find the indices of the first and last unmasked values.
- ma.notmasked_contiguous(a[, axis])	Find contiguous unmasked data in a masked array along the given axis.
- ma.notmasked_edges(a[, axis])	Find the indices of the first and last unmasked values along an axis.
- ma.clump_masked(a)	Returns a list of slices corresponding to the masked clumps of a 1-D array.
- ma.clump_unmasked(a)	Return list of slices corresponding to the unmasked clumps of a 1-D array.

### Modifying a mask

- ma.mask_cols(a[, axis])	Mask columns of a 2D array that contain masked values.
- ma.mask_or(m1, m2[, copy, shrink])	Combine two masks with the logical_or operator.
- ma.mask_rowcols(a[, axis])	Mask rows and/or columns of a 2D array that contain masked values.
- ma.mask_rows(a[, axis])	Mask rows of a 2D array that contain masked values.
- ma.harden_mask(self)	Force the mask to hard.
- ma.soften_mask(self)	Force the mask to soft.
- ma.MaskedArray.harden_mask()	Force the mask to hard.
- ma.MaskedArray.soften_mask()	Force the mask to soft.
- ma.MaskedArray.shrink_mask()	Reduce a mask to nomask when possible.
- ma.MaskedArray.unshare_mask()	Copy the mask and set the sharedmask flag to False.

## Conversion operations

### \> to a masked array

- ma.asarray(a[, dtype, order])	Convert the input to a masked array of the given data-type.
- ma.asanyarray(a[, dtype])	Convert the input to a masked array, conserving subclasses.
- ma.fix_invalid(a[, mask, copy, fill_value])	Return input with invalid data masked and replaced by a fill value.
- ma.masked_equal(x, value[, copy])	Mask an array where equal to a given value.
- ma.masked_greater(x, value[, copy])	Mask an array where greater than a given value.
- ma.masked_greater_equal(x, value[, copy])	Mask an array where greater than or equal to a given value.
- ma.masked_inside(x, v1, v2[, copy])	Mask an array inside a given interval.
- ma.masked_invalid(a[, copy])	Mask an array where invalid values occur (NaNs or infs).
- ma.masked_less(x, value[, copy])	Mask an array where less than a given value.
- ma.masked_less_equal(x, value[, copy])	Mask an array where less than or equal to a given value.
- ma.masked_not_equal(x, value[, copy])	Mask an array where not equal to a given value.
- ma.masked_object(x, value[, copy, shrink])	Mask the array x where the data are exactly equal to value.
- ma.masked_outside(x, v1, v2[, copy])	Mask an array outside a given interval.
- ma.masked_values(x, value[, rtol, atol, …])	Mask using floating point equality.
- ma.masked_where(condition, a[, copy])	Mask an array where a condition is met.

### \> to a ndarray

- ma.compress_cols(a)	Suppress whole columns of a 2-D array that contain masked values.
- ma.compress_rowcols(x[, axis])	Suppress the rows and/or columns of a 2-D array that contain masked values.
- ma.compress_rows(a)	Suppress whole rows of a 2-D array that contain masked values.
- ma.compressed(x)	Return all the non-masked data as a 1-D array.
- ma.filled(a[, fill_value])	Return input as an array with masked data replaced by a fill value.
- ma.MaskedArray.compressed()	Return all the non-masked data as a 1-D array.
- ma.MaskedArray.filled([fill_value])	Return a copy of self, with masked values filled with a given value.

### \> to another object

- ma.MaskedArray.tofile(fid[, sep, format])	Save a masked array to a file in binary format.
- ma.MaskedArray.tolist([fill_value])	Return the data portion of the masked array as a hierarchical Python list.
- ma.MaskedArray.torecords()	Transforms a masked array into a flexible-type array.
- ma.MaskedArray.tobytes([fill_value, order])	Return the array data as a string containing the raw bytes in the array.

### Pickling and unpickling

- ma.dump(a, F)	Pickle a masked array to a file.
- ma.dumps(a)	Return a string corresponding to the pickling of a masked array.
- ma.load(F)	Wrapper around cPickle.load which accepts either a file-like object or a filename.
- ma.loads(strg)	Load a pickle from the current string.

### Filling a masked array

- ma.common_fill_value(a, b)	Return the common filling value of two masked arrays, if any.
- ma.default_fill_value(obj)	Return the default fill value for the argument object.
- ma.maximum_fill_value(obj)	Return the minimum value that can be represented by the dtype of an object.
- ma.maximum_fill_value(obj)	Return the minimum value that can be represented by the dtype of an object.
- ma.set_fill_value(a, fill_value)	Set the filling value of a, if a is a masked array.
- ma.MaskedArray.get_fill_value()	Return the filling value of the masked array.
- ma.MaskedArray.set_fill_value([value])	Set the filling value of the masked array.
- ma.MaskedArray.fill_value	Filling value.

## Masked arrays arithmetics

### Arithmetics

- ma.anom(self[, axis, dtype])	Compute the anomalies (deviations from the arithmetic mean) along the given axis.
- ma.anomalies(self[, axis, dtype])	Compute the anomalies (deviations from the arithmetic mean) along the given axis.
- ma.average(a[, axis, weights, returned])	Return the weighted average of array over the given axis.
- ma.conjugate(x, /[, out, where, casting, …])	Return the complex conjugate, element-wise.
- ma.corrcoef(x[, y, rowvar, bias, …])	Return Pearson product-moment correlation coefficients.
- ma.cov(x[, y, rowvar, bias, allow_masked, ddof])	Estimate the covariance matrix.
- ma.cumsum(self[, axis, dtype, out])	Return the cumulative sum of the array elements over the given axis.
- ma.cumprod(self[, axis, dtype, out])	Return the cumulative product of the array elements over the given axis.
- ma.mean(self[, axis, dtype, out, keepdims])	Returns the average of the array elements along given axis.
- ma.median(a[, axis, out, overwrite_input, …])	Compute the median along the specified axis.
- ma.power(a, b[, third])	Returns element-wise base array raised to power from second array.
- ma.prod(self[, axis, dtype, out, keepdims])	Return the product of the array elements over the given axis.
- ma.std(self[, axis, dtype, out, ddof, keepdims])	Returns the standard deviation of the array elements along given axis.
- ma.sum(self[, axis, dtype, out, keepdims])	Return the sum of the array elements over the given axis.
- ma.var(self[, axis, dtype, out, ddof, keepdims])	Compute the variance along the specified axis.
- ma.MaskedArray.anom([axis, dtype])	Compute the anomalies (deviations from the arithmetic mean) along the given axis.
- ma.MaskedArray.cumprod([axis, dtype, out])	Return the cumulative product of the array elements over the given axis.
- ma.MaskedArray.cumsum([axis, dtype, out])	Return the cumulative sum of the array elements over the given axis.
- ma.MaskedArray.mean([axis, dtype, out, keepdims])	Returns the average of the array elements along given axis.
- ma.MaskedArray.prod([axis, dtype, out, keepdims])	Return the product of the array elements over the given axis.
- ma.MaskedArray.std([axis, dtype, out, ddof, …])	Returns the standard deviation of the array elements along given axis.
- ma.MaskedArray.sum([axis, dtype, out, keepdims])	Return the sum of the array elements over the given axis.
- ma.MaskedArray.var([axis, dtype, out, ddof, …])	Compute the variance along the specified axis.

### Minimum/maximum

- ma.argmax(self[, axis, fill_value, out])	Returns array of indices of the maximum values along the given axis.
- ma.argmin(self[, axis, fill_value, out])	Return array of indices to the minimum values along the given axis.
- ma.max(obj[, axis, out, fill_value, keepdims])	Return the maximum along a given axis.
- ma.min(obj[, axis, out, fill_value, keepdims])	Return the minimum along a given axis.
- ma.ptp(obj[, axis, out, fill_value])	Return (maximum - minimum) along the given dimension (i.e.
- ma.MaskedArray.argmax([axis, fill_value, out])	Returns array of indices of the maximum values along the given axis.
- ma.MaskedArray.argmin([axis, fill_value, out])	Return array of indices to the minimum values along the given axis.
- ma.MaskedArray.max([axis, out, fill_value, …])	Return the maximum along a given axis.
- ma.MaskedArray.min([axis, out, fill_value, …])	Return the minimum along a given axis.
- ma.MaskedArray.ptp([axis, out, fill_value])	Return (maximum - minimum) along the given dimension (i.e.

### Sorting

- ma.argsort(a[, axis, kind, order, endwith, …])	Return an ndarray of indices that sort the array along the specified axis.
- ma.sort(a[, axis, kind, order, endwith, …])	Sort the array, in-place
- ma.MaskedArray.argsort([axis, kind, order, …])	Return an ndarray of indices that sort the array along the specified axis.
- ma.MaskedArray.sort([axis, kind, order, …])	Sort the array, in-place

### Algebra

- ma.diag(v[, k])	Extract a diagonal or construct a diagonal array.
- ma.dot(a, b[, strict, out])	Return the dot product of two arrays.
- ma.identity(n[, dtype])	Return the identity array.
- ma.inner(a, b)	Inner product of two arrays.
- ma.innerproduct(a, b)	Inner product of two arrays.
- ma.outer(a, b)	Compute the outer product of two vectors.
- ma.outerproduct(a, b)	Compute the outer product of two vectors.
- ma.trace(self[, offset, axis1, axis2, …])	Return the sum along diagonals of the array.
- ma.transpose(a[, axes])	Permute the dimensions of an array.
- ma.MaskedArray.trace([offset, axis1, axis2, …])	Return the sum along diagonals of the array.
- ma.MaskedArray.transpose(*axes)	Returns a view of the array with axes transposed.
Polynomial fit¶
- ma.vander(x[, n])	Generate a Vandermonde matrix.
- ma.polyfit(x, y, deg[, rcond, full, w, cov])	Least squares polynomial fit.

### Clipping and rounding

- ma.around	Round an array to the given number of decimals.
- ma.clip(a, a_min, a_max[, out])	Clip (limit) the values in an array.
- ma.round(a[, decimals, out])	Return a copy of a, rounded to ‘decimals’ places.
[ ma.MaskedArray.clip([min, max, out])	Return an array whose values are limited to [min, max].
- ma.MaskedArray.round([decimals, out])	Return each element rounded to the given number of 
decimals.

### Miscellanea

- ma.allequal(a, b[, fill_value])	Return True if all entries of a and b are equal, using fill_value as a truth value where either or both are masked.
- ma.allclose(a, b[, masked_equal, rtol, atol])	Returns True if two arrays are element-wise equal within a tolerance.
- ma.apply_along_axis(func1d, axis, arr, …)	Apply a function to 1-D slices along the given axis.
- ma.arange([start,] stop[, step,][, dtype])	Return evenly spaced values within a given interval.
- ma.choose(indices, choices[, out, mode])	Use an index array to construct a new array from a set of choices.
- ma.ediff1d(arr[, to_end, to_begin])	Compute the differences between consecutive elements of an array.
- ma.indices(dimensions[, dtype])	Return an array representing the indices of a grid.
- ma.where(condition[, x, y])	Return a masked array with elements from x or y, depending on condition.