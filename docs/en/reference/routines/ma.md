# Masked array operations

## Constants

method | description
---|---
[ma.MaskType](generated/numpy.ma.MaskType.html#numpy.ma.MaskType) | alias of numpy.bool_

## Creation

### From existing data

method | description
---|---
[ma.masked_array](generated/numpy.ma.masked_array.html#numpy.ma.masked_array) | alias of numpy.ma.core.MaskedArray
[ma.array](generated/numpy.ma.array.html#numpy.ma.array)(data[, dtype, copy, order, mask, …]) | An array class with possibly masked values.
[ma.copy](generated/numpy.ma.copy.html#numpy.ma.copy)(self, *args, **params) a.copy(order=) | Return a copy of the array.
[ma.frombuffer](generated/numpy.ma.frombuffer.html#numpy.ma.frombuffer)(buffer[, dtype, count, offset]) | Interpret a buffer as a 1-dimensional array.
[ma.fromfunction](generated/numpy.ma.fromfunction.html#numpy.ma.fromfunction)(function, shape, **kwargs) | Construct an array by executing a function over each coordinate.
[ma.MaskedArray.copy](generated/numpy.ma.MaskedArray.copy.html#numpy.ma.MaskedArray.copy)([order]) | Return a copy of the array.

### Ones and zeros

method | description
---|---
[ma.empty](generated/numpy.ma.empty.html#numpy.ma.empty)(shape[, dtype, order]) | Return a new array of given shape and type, without initializing entries.
[ma.empty_like](generated/numpy.ma.empty_like.html#numpy.ma.empty_like)(prototype[, dtype, order, …]) | Return a new array with the same shape and type as a given array.
[ma.masked_all](generated/numpy.ma.masked_all.html#numpy.ma.masked_all)(shape[, dtype]) | Empty masked array with all elements masked.
[ma.masked_all_like](generated/numpy.ma.masked_all_like.html#numpy.ma.masked_all_like)(arr) | Empty masked array with the properties of an existing array.
[ma.ones](generated/numpy.ma.ones.html#numpy.ma.ones)(shape[, dtype, order]) | Return a new array of given shape and type, filled with ones.
[ma.zeros](generated/numpy.ma.zeros.html#numpy.ma.zeros)(shape[, dtype, order]) | Return a new array of given shape and type, filled with zeros.

## Inspecting the array

method | description
---|---
[ma.all](generated/numpy.ma.all.html#numpy.ma.all)(self[, axis, out, keepdims]) | Returns True if all elements evaluate to True.
[ma.any](generated/numpy.ma.any.html#numpy.ma.any)(self[, axis, out, keepdims]) | Returns True if any of the elements of a evaluate to True.
[ma.count](generated/numpy.ma.count.html#numpy.ma.count)(self[, axis, keepdims]) | Count the non-masked elements of the array along the given axis.
[ma.count_masked](generated/numpy.ma.count_masked.html#numpy.ma.count_masked)(arr[, axis]) | Count the number of masked elements along the given axis.
[ma.getmask](generated/numpy.ma.getmask.html#numpy.ma.getmask)(a) | Return the mask of a masked array, or nomask.
[ma.getmaskarray](generated/numpy.ma.getmaskarray.html#numpy.ma.getmaskarray)(arr) | Return the mask of a masked array, or full boolean array of False.
[ma.getdata](generated/numpy.ma.getdata.html#numpy.ma.getdata)(a[, subok]) | Return the data of a masked array as an ndarray.
[ma.nonzero](generated/numpy.ma.nonzero.html#numpy.ma.nonzero)(self) | Return the indices of unmasked elements that are not zero.
[ma.shape](generated/numpy.ma.shape.html#numpy.ma.shape)(obj) | Return the shape of an array.
[ma.size](generated/numpy.ma.size.html#numpy.ma.size)(obj[, axis]) | Return the number of elements along a given axis.
[ma.is_masked](generated/numpy.ma.is_masked.html#numpy.ma.is_masked)(x) | Determine whether input has masked values.
[ma.is_mask](generated/numpy.ma.is_mask.html#numpy.ma.is_mask)(m) | Return True if m is a valid, standard mask.
[ma.MaskedArray.all](generated/numpy.ma.MaskedArray.all.html#numpy.ma.MaskedArray.all)(self[, axis, out, keepdims]) | Returns True if all elements evaluate to True.
[ma.MaskedArray.any](generated/numpy.ma.MaskedArray.any.html#numpy.ma.MaskedArray.any)(self[, axis, out, keepdims]) | Returns True if any of the elements of a evaluate to True.
[ma.MaskedArray.count](generated/numpy.ma.MaskedArray.count.html#numpy.ma.MaskedArray.count)(self[, axis, keepdims]) | Count the non-masked elements of the array along the given axis.
[ma.MaskedArray.nonzero](generated/numpy.ma.MaskedArray.nonzero.html#numpy.ma.MaskedArray.nonzero)(self) | Return the indices of unmasked elements that are not zero.
[ma.shape](generated/numpy.ma.shape.html#numpy.ma.shape)(obj) | Return the shape of an array.
[ma.size](generated/numpy.ma.size.html#numpy.ma.size)(obj[, axis]) | Return the number of elements along a given axis.

method | description
---|---
[ma.MaskedArray.data](maskedarray.baseclass.html#numpy.ma.MaskedArray.data) | Returns the underlying data, as a view of the masked array.
[ma.MaskedArray.mask](maskedarray.baseclass.html#numpy.ma.MaskedArray.mask) | Current mask.
[ma.MaskedArray.recordmask](maskedarray.baseclass.html#numpy.ma.MaskedArray.recordmask) | Get or set the mask of the array if it has no named fields.

## Manipulating a MaskedArray

### Changing the shape

method | description
---|---
[ma.ravel](generated/numpy.ma.ravel.html#numpy.ma.ravel)(self[, order]) | Returns a 1D version of self, as a view.
[ma.reshape](generated/numpy.ma.reshape.html#numpy.ma.reshape)(a, new_shape[, order]) | Returns an array containing the same data with a new shape.
[ma.resize](generated/numpy.ma.resize.html#numpy.ma.resize)(x, new_shape) | Return a new masked array with the specified size and shape.
[ma.MaskedArray.flatten](generated/numpy.ma.MaskedArray.flatten.html#numpy.ma.MaskedArray.flatten)([order]) | Return a copy of the array collapsed into one dimension.
[ma.MaskedArray.ravel](generated/numpy.ma.MaskedArray.ravel.html#numpy.ma.MaskedArray.ravel)(self[, order]) | Returns a 1D version of self, as a view.
[ma.MaskedArray.reshape](generated/numpy.ma.MaskedArray.reshape.html#numpy.ma.MaskedArray.reshape)(self, \*s, \*\*kwargs) | Give a new shape to the array without changing its data.
[ma.MaskedArray.resize](generated/numpy.ma.MaskedArray.resize.html#numpy.ma.MaskedArray.resize)(self, newshape[, …]) | 

### Modifying axes

method | description
---|---
[ma.swapaxes](generated/numpy.ma.swapaxes.html#numpy.ma.swapaxes)(self, *args, …) | Return a view of the array with axis1 and axis2 interchanged.
[ma.transpose](generated/numpy.ma.transpose.html#numpy.ma.transpose)(a[, axes]) | Permute the dimensions of an array.
[ma.MaskedArray.swapaxes](generated/numpy.ma.MaskedArray.swapaxes.html#numpy.ma.MaskedArray.swapaxes)(axis1, axis2) | Return a view of the array with axis1 and axis2 interchanged.
[ma.MaskedArray.transpose](generated/numpy.ma.MaskedArray.transpose.html#numpy.ma.MaskedArray.transpose)(*axes) | Returns a view of the array with axes transposed.

### Changing the number of dimensions

method | description
---|---
[ma.atleast_1d](generated/numpy.ma.atleast_1d.html#numpy.ma.atleast_1d)(*args, **kwargs) | Convert inputs to arrays with at least one dimension.
[ma.atleast_2d](generated/numpy.ma.atleast_2d.html#numpy.ma.atleast_2d)(*args, **kwargs) | View inputs as arrays with at least two dimensions.
[ma.atleast_3d](generated/numpy.ma.atleast_3d.html#numpy.ma.atleast_3d)(*args, **kwargs) | View inputs as arrays with at least three dimensions.
[ma.expand_dims](generated/numpy.ma.expand_dims.html#numpy.ma.expand_dims)(a, axis) | Expand the shape of an array.
[ma.squeeze](generated/numpy.ma.squeeze.html#numpy.ma.squeeze)(a[, axis]) | Remove single-dimensional entries from the shape of an array.
[ma.MaskedArray.squeeze](generated/numpy.ma.MaskedArray.squeeze.html#numpy.ma.MaskedArray.squeeze)([axis]) | Remove single-dimensional entries from the shape of a.
[ma.stack](generated/numpy.ma.stack.html#numpy.ma.stack)(*args, **kwargs) | Join a sequence of arrays along a new axis.
[ma.column_stack](generated/numpy.ma.column_stack.html#numpy.ma.column_stack)(*args, **kwargs) | Stack 1-D arrays as columns into a 2-D array.
[ma.concatenate](generated/numpy.ma.concatenate.html#numpy.ma.concatenate)(arrays[, axis]) | Concatenate a sequence of arrays along the given axis.
[ma.dstack](generated/numpy.ma.dstack.html#numpy.ma.dstack)(*args, **kwargs) | Stack arrays in sequence depth wise (along third axis).
[ma.hstack](generated/numpy.ma.hstack.html#numpy.ma.hstack)(*args, **kwargs) | Stack arrays in sequence horizontally (column wise).
[ma.hsplit](generated/numpy.ma.hsplit.html#numpy.ma.hsplit)(*args, **kwargs) | Split an array into multiple sub-arrays horizontally (column-wise).
[ma.mr_](generated/numpy.ma.mr_.html#numpy.ma.mr_) | Translate slice objects to concatenation along the first axis.
[ma.row_stack](generated/numpy.ma.row_stack.html#numpy.ma.row_stack)(*args, **kwargs) | Stack arrays in sequence vertically (row wise).
[ma.vstack](generated/numpy.ma.vstack.html#numpy.ma.vstack)(*args, **kwargs) | Stack arrays in sequence vertically (row wise).

### Joining arrays

method | description
---|---
[ma.stack](generated/numpy.ma.stack.html#numpy.ma.stack)(*args, **kwargs) | Join a sequence of arrays along a new axis.
[ma.column_stack](generated/numpy.ma.column_stack.html#numpy.ma.column_stack)(*args, **kwargs) | Stack 1-D arrays as columns into a 2-D array.
[ma.concatenate](generated/numpy.ma.concatenate.html#numpy.ma.concatenate)(arrays[, axis]) | Concatenate a sequence of arrays along the given axis.
[ma.append](generated/numpy.ma.append.html#numpy.ma.append)(a, b[, axis]) | Append values to the end of an array.
[ma.dstack](generated/numpy.ma.dstack.html#numpy.ma.dstack)(*args, **kwargs) | Stack arrays in sequence depth wise (along third axis).
[ma.hstack](generated/numpy.ma.hstack.html#numpy.ma.hstack)(*args, **kwargs) | Stack arrays in sequence horizontally (column wise).
[ma.vstack](generated/numpy.ma.vstack.html#numpy.ma.vstack)(*args, **kwargs) | Stack arrays in sequence vertically (row wise).

## Operations on masks

### Creating a mask

method | description
---|---
[ma.make_mask](generated/numpy.ma.make_mask.html#numpy.ma.make_mask)(m[, copy, shrink, dtype]) | Create a boolean mask from an array.
[ma.make_mask_none](generated/numpy.ma.make_mask_none.html#numpy.ma.make_mask_none)(newshape[, dtype]) | Return a boolean mask of the given shape, filled with False.
[ma.mask_or](generated/numpy.ma.mask_or.html#numpy.ma.mask_or)(m1, m2[, copy, shrink]) | Combine two masks with the logical_or operator.
[ma.make_mask_descr](generated/numpy.ma.make_mask_descr.html#numpy.ma.make_mask_descr)(ndtype) | Construct a dtype description list from a given dtype.

### Accessing a mask

method | description
---|---
[ma.getmask](generated/numpy.ma.getmask.html#numpy.ma.getmask)(a) | Return the mask of a masked array, or nomask.
[ma.getmaskarray](generated/numpy.ma.getmaskarray.html#numpy.ma.getmaskarray)(arr) | Return the mask of a masked array, or full boolean array of False.
[ma.masked_array.mask](generated/numpy.ma.masked_array.mask.html#numpy.ma.masked_array.mask) | Current mask.

### Finding masked data

method | description
---|---
[ma.flatnotmasked_contiguous](generated/numpy.ma.flatnotmasked_contiguous.html#numpy.ma.flatnotmasked_contiguous)(a) | Find contiguous unmasked data in a masked array along the given axis.
[ma.flatnotmasked_edges](generated/numpy.ma.flatnotmasked_edges.html#numpy.ma.flatnotmasked_edges)(a) | Find the indices of the first and last unmasked values.
[ma.notmasked_contiguous](generated/numpy.ma.notmasked_contiguous.html#numpy.ma.notmasked_contiguous)(a[, axis]) | Find contiguous unmasked data in a masked array along the given axis.
[ma.notmasked_edges](generated/numpy.ma.notmasked_edges.html#numpy.ma.notmasked_edges)(a[, axis]) | Find the indices of the first and last unmasked values along an axis.
[ma.clump_masked](generated/numpy.ma.clump_masked.html#numpy.ma.clump_masked)(a) | Returns a list of slices corresponding to the masked clumps of a 1-D array.
[ma.clump_unmasked](generated/numpy.ma.clump_unmasked.html#numpy.ma.clump_unmasked)(a) | Return list of slices corresponding to the unmasked clumps of a 1-D array.

### Modifying a mask

method | description
---|---
[ma.mask_cols](generated/numpy.ma.mask_cols.html#numpy.ma.mask_cols)(a[, axis]) | Mask columns of a 2D array that contain masked values.
[ma.mask_or](generated/numpy.ma.mask_or.html#numpy.ma.mask_or)(m1, m2[, copy, shrink]) | Combine two masks with the logical_or operator.
[ma.mask_rowcols](generated/numpy.ma.mask_rowcols.html#numpy.ma.mask_rowcols)(a[, axis]) | Mask rows and/or columns of a 2D array that contain masked values.
[ma.mask_rows](generated/numpy.ma.mask_rows.html#numpy.ma.mask_rows)(a[, axis]) | Mask rows of a 2D array that contain masked values.
[ma.harden_mask](generated/numpy.ma.harden_mask.html#numpy.ma.harden_mask)(self) | Force the mask to hard.
[ma.soften_mask](generated/numpy.ma.soften_mask.html#numpy.ma.soften_mask)(self) | Force the mask to soft.
[ma.MaskedArray.harden_mask](generated/numpy.ma.MaskedArray.harden_mask.html#numpy.ma.MaskedArray.harden_mask)(self) | Force the mask to hard.
[ma.MaskedArray.soften_mask](generated/numpy.ma.MaskedArray.soften_mask.html#numpy.ma.MaskedArray.soften_mask)(self) | Force the mask to soft.
[ma.MaskedArray.shrink_mask](generated/numpy.ma.MaskedArray.shrink_mask.html#numpy.ma.MaskedArray.shrink_mask)(self) | Reduce a mask to nomask when possible.
[ma.MaskedArray.unshare_mask](generated/numpy.ma.MaskedArray.unshare_mask.html#numpy.ma.MaskedArray.unshare_mask)(self) | Copy the mask and set the sharedmask flag to False.

## Conversion operations

### > to a masked array

method | description
---|---
[ma.asarray](generated/numpy.ma.asarray.html#numpy.ma.asarray)(a[, dtype, order]) | Convert the input to a masked array of the given data-type.
[ma.asanyarray](generated/numpy.ma.asanyarray.html#numpy.ma.asanyarray)(a[, dtype]) | Convert the input to a masked array, conserving subclasses.
[ma.fix_invalid](generated/numpy.ma.fix_invalid.html#numpy.ma.fix_invalid)(a[, mask, copy, fill_value]) | Return input with invalid data masked and replaced by a fill value.
[ma.masked_equal](generated/numpy.ma.masked_equal.html#numpy.ma.masked_equal)(x, value[, copy]) | Mask an array where equal to a given value.
[ma.masked_greater](generated/numpy.ma.masked_greater.html#numpy.ma.masked_greater)(x, value[, copy]) | Mask an array where greater than a given value.
[ma.masked_greater_equal](generated/numpy.ma.masked_greater_equal.html#numpy.ma.masked_greater_equal)(x, value[, copy]) | Mask an array where greater than or equal to a given value.
[ma.masked_inside](generated/numpy.ma.masked_inside.html#numpy.ma.masked_inside)(x, v1, v2[, copy]) | Mask an array inside a given interval.
[ma.masked_invalid](generated/numpy.ma.masked_invalid.html#numpy.ma.masked_invalid)(a[, copy]) | Mask an array where invalid values occur (NaNs or infs).
[ma.masked_less](generated/numpy.ma.masked_less.html#numpy.ma.masked_less)(x, value[, copy]) | Mask an array where less than a given value.
[ma.masked_less_equal](generated/numpy.ma.masked_less_equal.html#numpy.ma.masked_less_equal)(x, value[, copy]) | Mask an array where less than or equal to a given value.
[ma.masked_not_equal](generated/numpy.ma.masked_not_equal.html#numpy.ma.masked_not_equal)(x, value[, copy]) | Mask an array where not equal to a given value.
[ma.masked_object](generated/numpy.ma.masked_object.html#numpy.ma.masked_object)(x, value[, copy, shrink]) | Mask the array x where the data are exactly equal to value.
[ma.masked_outside](generated/numpy.ma.masked_outside.html#numpy.ma.masked_outside)(x, v1, v2[, copy]) | Mask an array outside a given interval.
[ma.masked_values](generated/numpy.ma.masked_values.html#numpy.ma.masked_values)(x, value[, rtol, atol, …]) | Mask using floating point equality.
[ma.masked_where](generated/numpy.ma.masked_where.html#numpy.ma.masked_where)(condition, a[, copy]) | Mask an array where a condition is met.

### > to a ndarray

method | description
---|---
[ma.compress_cols](generated/numpy.ma.compress_cols.html#numpy.ma.compress_cols)(a) | Suppress whole columns of a 2-D array that contain masked values.
[ma.compress_rowcols](generated/numpy.ma.compress_rowcols.html#numpy.ma.compress_rowcols)(x[, axis]) | Suppress the rows and/or columns of a 2-D array that contain masked values.
[ma.compress_rows](generated/numpy.ma.compress_rows.html#numpy.ma.compress_rows)(a) | Suppress whole rows of a 2-D array that contain masked values.
[ma.compressed](generated/numpy.ma.compressed.html#numpy.ma.compressed)(x) | Return all the non-masked data as a 1-D array.
[ma.filled](generated/numpy.ma.filled.html#numpy.ma.filled)(a[, fill_value]) | Return input as an array with masked data replaced by a fill value.
[ma.MaskedArray.compressed](generated/numpy.ma.MaskedArray.compressed.html#numpy.ma.MaskedArray.compressed)(self) | Return all the non-masked data as a 1-D array.
[ma.MaskedArray.filled](generated/numpy.ma.MaskedArray.filled.html#numpy.ma.MaskedArray.filled)(self[, fill_value]) | Return a copy of self, with masked values filled with a given value.

### > to another object

method | description
---|---
[ma.MaskedArray.tofile](generated/numpy.ma.MaskedArray.tofile.html#numpy.ma.MaskedArray.tofile)(self, fid[, sep, format]) | Save a masked array to a file in binary format.
[ma.MaskedArray.tolist](generated/numpy.ma.MaskedArray.tolist.html#numpy.ma.MaskedArray.tolist)(self[, fill_value]) | Return the data portion of the masked array as a hierarchical Python list.
[ma.MaskedArray.torecords](generated/numpy.ma.MaskedArray.torecords.html#numpy.ma.MaskedArray.torecords)(self) | Transforms a masked array into a flexible-type array.
[ma.MaskedArray.tobytes](generated/numpy.ma.MaskedArray.tobytes.html#numpy.ma.MaskedArray.tobytes)(self[, fill_value, order]) | Return the array data as a string containing the raw bytes in the array.

### Pickling and unpickling

method | description
---|---
[ma.dump](generated/numpy.ma.dump.html#numpy.ma.dump)(a, F) | Pickle a masked array to a file.
[ma.dumps](generated/numpy.ma.dumps.html#numpy.ma.dumps)(a) | Return a string corresponding to the pickling of a masked array.
[ma.load](generated/numpy.ma.load.html#numpy.ma.load)(F) | Wrapper around cPickle.load which accepts either a file-like object or a filename.
[ma.loads](generated/numpy.ma.loads.html#numpy.ma.loads)(strg) | Load a pickle from the current string.

### Filling a masked array

method | description
---|---
[ma.common_fill_value](generated/numpy.ma.common_fill_value.html#numpy.ma.common_fill_value)(a, b) | Return the common filling value of two masked arrays, if any.
[ma.default_fill_value](generated/numpy.ma.default_fill_value.html#numpy.ma.default_fill_value)(obj) | Return the default fill value for the argument object.
[ma.maximum_fill_value](generated/numpy.ma.maximum_fill_value.html#numpy.ma.maximum_fill_value)(obj) | Return the minimum value that can be represented by the dtype of an object.
[ma.maximum_fill_value](generated/numpy.ma.maximum_fill_value.html#numpy.ma.maximum_fill_value)(obj) | Return the minimum value that can be represented by the dtype of an object.
[ma.set_fill_value](generated/numpy.ma.set_fill_value.html#numpy.ma.set_fill_value)(a, fill_value) | Set the filling value of a, if a is a masked array.
[ma.MaskedArray.get_fill_value](generated/numpy.ma.MaskedArray.get_fill_value.html#numpy.ma.MaskedArray.get_fill_value)(self) | The filling value of the masked array is a scalar.
[ma.MaskedArray.set_fill_value](generated/numpy.ma.MaskedArray.set_fill_value.html#numpy.ma.MaskedArray.set_fill_value)(self[, value]) | 

method | description
---|---
[ma.MaskedArray.fill_value](maskedarray.baseclass.html#numpy.ma.MaskedArray.fill_value) | The filling value of the masked array is a scalar.

## Masked arrays arithmetics

### Arithmetics

method | description
---|---
[ma.anom](generated/numpy.ma.anom.html#numpy.ma.anom)(self[, axis, dtype]) | Compute the anomalies (deviations from the arithmetic mean) along the given axis.
[ma.anomalies](generated/numpy.ma.anomalies.html#numpy.ma.anomalies)(self[, axis, dtype]) | Compute the anomalies (deviations from the arithmetic mean) along the given axis.
[ma.average](generated/numpy.ma.average.html#numpy.ma.average)(a[, axis, weights, returned]) | Return the weighted average of array over the given axis.
[ma.conjugate](generated/numpy.ma.conjugate.html#numpy.ma.conjugate)(x, /[, out, where, casting, …]) | Return the complex conjugate, element-wise.
[ma.corrcoef](generated/numpy.ma.corrcoef.html#numpy.ma.corrcoef)(x[, y, rowvar, bias, …]) | Return Pearson product-moment correlation coefficients.
[ma.cov](generated/numpy.ma.cov.html#numpy.ma.cov)(x[, y, rowvar, bias, allow_masked, ddof]) | Estimate the covariance matrix.
[ma.cumsum](generated/numpy.ma.cumsum.html#numpy.ma.cumsum)(self[, axis, dtype, out]) | Return the cumulative sum of the array elements over the given axis.
[ma.cumprod](generated/numpy.ma.cumprod.html#numpy.ma.cumprod)(self[, axis, dtype, out]) | Return the cumulative product of the array elements over the given axis.
[ma.mean](generated/numpy.ma.mean.html#numpy.ma.mean)(self[, axis, dtype, out, keepdims]) | Returns the average of the array elements along given axis.
[ma.median](generated/numpy.ma.median.html#numpy.ma.median)(a[, axis, out, overwrite_input, …]) | Compute the median along the specified axis.
[ma.power](generated/numpy.ma.power.html#numpy.ma.power)(a, b[, third]) | Returns element-wise base array raised to power from second array.
[ma.prod](generated/numpy.ma.prod.html#numpy.ma.prod)(self[, axis, dtype, out, keepdims]) | Return the product of the array elements over the given axis.
[ma.std](generated/numpy.ma.std.html#numpy.ma.std)(self[, axis, dtype, out, ddof, keepdims]) | Returns the standard deviation of the array elements along given axis.
[ma.sum](generated/numpy.ma.sum.html#numpy.ma.sum)(self[, axis, dtype, out, keepdims]) | Return the sum of the array elements over the given axis.
[ma.var](generated/numpy.ma.var.html#numpy.ma.var)(self[, axis, dtype, out, ddof, keepdims]) | Compute the variance along the specified axis.
[ma.MaskedArray.anom](generated/numpy.ma.MaskedArray.anom.html#numpy.ma.MaskedArray.anom)(self[, axis, dtype]) | Compute the anomalies (deviations from the arithmetic mean) along the given axis.
[ma.MaskedArray.cumprod](generated/numpy.ma.MaskedArray.cumprod.html#numpy.ma.MaskedArray.cumprod)(self[, axis, dtype, out]) | Return the cumulative product of the array elements over the given axis.
[ma.MaskedArray.cumsum](generated/numpy.ma.MaskedArray.cumsum.html#numpy.ma.MaskedArray.cumsum)(self[, axis, dtype, out]) | Return the cumulative sum of the array elements over the given axis.
[ma.MaskedArray.mean](generated/numpy.ma.MaskedArray.mean.html#numpy.ma.MaskedArray.mean)(self[, axis, dtype, …]) | Returns the average of the array elements along given axis.
[ma.MaskedArray.prod](generated/numpy.ma.MaskedArray.prod.html#numpy.ma.MaskedArray.prod)(self[, axis, dtype, …]) | Return the product of the array elements over the given axis.
[ma.MaskedArray.std](generated/numpy.ma.MaskedArray.std.html#numpy.ma.MaskedArray.std)(self[, axis, dtype, out, …]) | Returns the standard deviation of the array elements along given axis.
[ma.MaskedArray.sum](generated/numpy.ma.MaskedArray.sum.html#numpy.ma.MaskedArray.sum)(self[, axis, dtype, out, …]) | Return the sum of the array elements over the given axis.
[ma.MaskedArray.var](generated/numpy.ma.MaskedArray.var.html#numpy.ma.MaskedArray.var)(self[, axis, dtype, out, …]) | Compute the variance along the specified axis.

### Minimum/maximum

method | description
---|---
[ma.argmax](generated/numpy.ma.argmax.html#numpy.ma.argmax)(self[, axis, fill_value, out]) | Returns array of indices of the maximum values along the given axis.
[ma.argmin](generated/numpy.ma.argmin.html#numpy.ma.argmin)(self[, axis, fill_value, out]) | Return array of indices to the minimum values along the given axis.
[ma.max](generated/numpy.ma.max.html#numpy.ma.max)(obj[, axis, out, fill_value, keepdims]) | Return the maximum along a given axis.
[ma.min](generated/numpy.ma.min.html#numpy.ma.min)(obj[, axis, out, fill_value, keepdims]) | Return the minimum along a given axis.
[ma.ptp](generated/numpy.ma.ptp.html#numpy.ma.ptp)(obj[, axis, out, fill_value, keepdims]) | Return (maximum - minimum) along the given dimension (i.e.
[ma.MaskedArray.argmax](generated/numpy.ma.MaskedArray.argmax.html#numpy.ma.MaskedArray.argmax)(self[, axis, …]) | Returns array of indices of the maximum values along the given axis.
[ma.MaskedArray.argmin](generated/numpy.ma.MaskedArray.argmin.html#numpy.ma.MaskedArray.argmin)(self[, axis, …]) | Return array of indices to the minimum values along the given axis.
[ma.MaskedArray.max](generated/numpy.ma.MaskedArray.max.html#numpy.ma.MaskedArray.max)(self[, axis, out, …]) | Return the maximum along a given axis.
[ma.MaskedArray.min](generated/numpy.ma.MaskedArray.min.html#numpy.ma.MaskedArray.min)(self[, axis, out, …]) | Return the minimum along a given axis.
[ma.MaskedArray.ptp](generated/numpy.ma.MaskedArray.ptp.html#numpy.ma.MaskedArray.ptp)(self[, axis, out, …]) | Return (maximum - minimum) along the given dimension (i.e.

### Sorting

method | description
---|---
[ma.argsort](generated/numpy.ma.argsort.html#numpy.ma.argsort)(a[, axis, kind, order, endwith, …]) | Return an ndarray of indices that sort the array along the specified axis.
[ma.sort](generated/numpy.ma.sort.html#numpy.ma.sort)(a[, axis, kind, order, endwith, …]) | Sort the array, in-place
[ma.MaskedArray.argsort](generated/numpy.ma.MaskedArray.argsort.html#numpy.ma.MaskedArray.argsort)(self[, axis, kind, …]) | Return an ndarray of indices that sort the array along the specified axis.
[ma.MaskedArray.sort](generated/numpy.ma.MaskedArray.sort.html#numpy.ma.MaskedArray.sort)(self[, axis, kind, …]) | Sort the array, in-place

### Algebra

method | description
---|---
[ma.diag](generated/numpy.ma.diag.html#numpy.ma.diag)(v[, k]) | Extract a diagonal or construct a diagonal array.
[ma.dot](generated/numpy.ma.dot.html#numpy.ma.dot)(a, b[, strict, out]) | Return the dot product of two arrays.
[ma.identity](generated/numpy.ma.identity.html#numpy.ma.identity)(n[, dtype]) | Return the identity array.
[ma.inner](generated/numpy.ma.inner.html#numpy.ma.inner)(a, b) | Inner product of two arrays.
[ma.innerproduct](generated/numpy.ma.innerproduct.html#numpy.ma.innerproduct)(a, b) | Inner product of two arrays.
[ma.outer](generated/numpy.ma.outer.html#numpy.ma.outer)(a, b) | Compute the outer product of two vectors.
[ma.outerproduct](generated/numpy.ma.outerproduct.html#numpy.ma.outerproduct)(a, b) | Compute the outer product of two vectors.
[ma.trace](generated/numpy.ma.trace.html#numpy.ma.trace)(self[, offset, axis1, axis2, …]) | Return the sum along diagonals of the array.
[ma.transpose](generated/numpy.ma.transpose.html#numpy.ma.transpose)(a[, axes]) | Permute the dimensions of an array.
[ma.MaskedArray.trace](generated/numpy.ma.MaskedArray.trace.html#numpy.ma.MaskedArray.trace)([offset, axis1, axis2, …]) | Return the sum along diagonals of the array.
[ma.MaskedArray.transpose](generated/numpy.ma.MaskedArray.transpose.html#numpy.ma.MaskedArray.transpose)(*axes) | Returns a view of the array with axes transposed.

### Polynomial fit

method | description
---|---
[ma.vander](generated/numpy.ma.vander.html#numpy.ma.vander)(x[, n]) | Generate a Vandermonde matrix.
[ma.polyfit](generated/numpy.ma.polyfit.html#numpy.ma.polyfit)(x, y, deg[, rcond, full, w, cov]) | Least squares polynomial fit.

### Clipping and rounding

method | description
---|---
[ma.around](generated/numpy.ma.around.html#numpy.ma.around)(a, \*args, \*\*kwargs) | Round an array to the given number of decimals.
[ma.clip](generated/numpy.ma.clip.html#numpy.ma.clip)(a, a_min, a_max[, out]) | Clip (limit) the values in an array.
[ma.round](generated/numpy.ma.round.html#numpy.ma.round)(a[, decimals, out]) | Return a copy of a, rounded to ‘decimals’ places.
[ma.MaskedArray.clip](generated/numpy.ma.MaskedArray.clip.html#numpy.ma.MaskedArray.clip)([min, max, out]) | Return an array whose values are limited to [min, max].
[ma.MaskedArray.round](generated/numpy.ma.MaskedArray.round.html#numpy.ma.MaskedArray.round)(self[, decimals, out]) | Return each element rounded to the given number of decimals.

### Miscellanea

method | description
---|---
[ma.allequal](generated/numpy.ma.allequal.html#numpy.ma.allequal)(a, b[, fill_value]) | Return True if all entries of a and b are equal, using fill_value as a truth value where either or both are masked.
[ma.allclose](generated/numpy.ma.allclose.html#numpy.ma.allclose)(a, b[, masked_equal, rtol, atol]) | Returns True if two arrays are element-wise equal within a tolerance.
[ma.apply_along_axis](generated/numpy.ma.apply_along_axis.html#numpy.ma.apply_along_axis)(func1d, axis, arr, …) | Apply a function to 1-D slices along the given axis.
[ma.arange](generated/numpy.ma.arange.html#numpy.ma.arange)([start,] stop[, step,][, dtype]) | Return evenly spaced values within a given interval.
[ma.choose](generated/numpy.ma.choose.html#numpy.ma.choose)(indices, choices[, out, mode]) | Use an index array to construct a new array from a set of choices.
[ma.ediff1d](generated/numpy.ma.ediff1d.html#numpy.ma.ediff1d)(arr[, to_end, to_begin]) | Compute the differences between consecutive elements of an array.
[ma.indices](generated/numpy.ma.indices.html#numpy.ma.indices)(dimensions[, dtype, sparse]) | Return an array representing the indices of a grid.
[ma.where](generated/numpy.ma.where.html#numpy.ma.where)(condition[, x, y]) | Return a masked array with elements from x or y, depending on condition.
