# Array manipulation routines

## Basic operations

method | description
---|---
[copyto](https://numpy.org/devdocs/reference/generated/numpy.copyto.html#numpy.copyto)(dst, src[, casting, where]) | Copies values from one array to another, broadcasting as necessary.

## Changing array shape

method | description
---|---
[reshape](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape)(a, newshape[, order]) | Gives a new shape to an array without changing its data.
[ravel](https://numpy.org/devdocs/reference/generated/numpy.ravel.html#numpy.ravel)(a[, order]) | Return a contiguous flattened array.
[ndarray.flat](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | A 1-D iterator over the array.
[ndarray.flatten](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten)([order]) | Return a copy of the array collapsed into one dimension.

## Transpose-like operations

method | description
---|---
[moveaxis](https://numpy.org/devdocs/reference/generated/numpy.moveaxis.html#numpy.moveaxis)(a, source, destination) | Move axes of an array to new positions.
[rollaxis](https://numpy.org/devdocs/reference/generated/numpy.rollaxis.html#numpy.rollaxis)(a, axis[, start]) | Roll the specified axis backwards, until it lies in a given position.
[swapaxes](https://numpy.org/devdocs/reference/generated/numpy.swapaxes.html#numpy.swapaxes)(a, axis1, axis2) | Interchange two axes of an array.
[ndarray.T](https://numpy.org/devdocs/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T) | The [transpose](https://numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose)d array.
transpose(a[, axes]) | Permute the dimensions of an array.

## Changing number of dimensions

method | description
---|---
[atleast_1d](https://numpy.org/devdocs/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d)(\*arys) | Convert inputs to arrays with at least one dimension.
[atleast_2d](https://numpy.org/devdocs/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d)(\*arys) | View inputs as arrays with at least two dimensions.
[atleast_3d](https://numpy.org/devdocs/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d)(\*arys) | View inputs as arrays with at least three dimensions.
[broadcast](https://numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast) | Produce an object that mimics broadcasting.
[broadcast_to](https://numpy.org/devdocs/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to)(array, shape[, subok]) | Broadcast an array to a new shape.
[broadcast_arrays](https://numpy.org/devdocs/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays)(\*args, \*\*kwargs) | Broadcast any number of arrays against each other.
[expand_dims](https://numpy.org/devdocs/reference/generated/numpy.expand_dims.html#numpy.expand_dims)(a, axis) | Expand the shape of an array.
[squeeze](https://numpy.org/devdocs/reference/generated/numpy.squeeze.html#numpy.squeeze)(a[, axis]) | Remove single-dimensional entries from the shape of an array.

## Changing kind of array

method | description
---|---
[asarray](https://numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray)(a[, dtype, order]) | Convert the input to an array.
[asanyarray](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray)(a[, dtype, order]) | Convert the input to an ndarray, but pass ndarray subclasses through.
[asmatrix](https://numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | Interpret the input as a matrix.
[asfarray](https://numpy.org/devdocs/reference/generated/numpy.asfarray.html#numpy.asfarray)(a[, dtype]) | Return an array converted to a float type.
[asfortranarray](https://numpy.org/devdocs/reference/generated/numpy.asfortranarray.html#numpy.asfortranarray)(a[, dtype]) | Return an array (ndim >= 1) laid out in Fortran order in memory.
[ascontiguousarray](https://numpy.org/devdocs/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray)(a[, dtype]) | Return a contiguous array (ndim >= 1) in memory (C order).
[asarray_chkfinite](https://numpy.org/devdocs/reference/generated/numpy.asarray_chkfinite.html#numpy.asarray_chkfinite)(a[, dtype, order]) | Convert the input to an array, checking for NaNs or Infs.
[asscalar](https://numpy.org/devdocs/reference/generated/numpy.asscalar.html#numpy.asscalar)(a) | Convert an array of size 1 to its scalar equivalent.
[require](https://numpy.org/devdocs/reference/generated/numpy.require.html#numpy.require)(a[, dtype, requirements]) | Return an ndarray of the provided type that satisfies requirements.

## Joining arrays

method | description
---|---
[concatenate](https://numpy.org/devdocs/reference/generated/numpy.concatenate.html#numpy.concatenate)((a1, a2, …)[, axis, out]) | Join a sequence of arrays along an existing axis.
[stack](https://numpy.org/devdocs/reference/generated/numpy.stack.html#numpy.stack)(arrays[, axis, out]) | Join a sequence of arrays along a new axis.
[column_stack](https://numpy.org/devdocs/reference/generated/numpy.column_stack.html#numpy.column_stack)(tup) | Stack 1-D arrays as columns into a 2-D array.
[dstack](https://numpy.org/devdocs/reference/generated/numpy.dstack.html#numpy.dstack)(tup) | Stack arrays in sequence depth wise (along third axis).
[hstack](https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack)(tup) | Stack arrays in sequence horizontally (column wise).
[vstack](https://numpy.org/devdocs/reference/generated/numpy.vstack.html#numpy.vstack)(tup) | Stack arrays in sequence vertically (row wise).
[block](https://numpy.org/devdocs/reference/generated/numpy.block.html#numpy.block)(arrays) | Assemble an nd-array from nested lists of blocks.

## Splitting arrays

method | description
---|---
[split](https://numpy.org/devdocs/reference/generated/numpy.split.html#numpy.split)(ary, indices_or_sections[, axis]) | Split an array into multiple sub-arrays.
[array_split](https://numpy.org/devdocs/reference/generated/numpy.array_split.html#numpy.array_split)(ary, indices_or_sections[, axis]) | Split an array into multiple sub-arrays.
[dsplit](https://numpy.org/devdocs/reference/generated/numpy.dsplit.html#numpy.dsplit)(ary, indices_or_sections) | Split array into multiple sub-arrays along the 3rd axis (depth).
[hsplit](https://numpy.org/devdocs/reference/generated/numpy.hsplit.html#numpy.hsplit)(ary, indices_or_sections) | Split an array into multiple sub-arrays horizontally (column-wise).
[vsplit](https://numpy.org/devdocs/reference/generated/numpy.vsplit.html#numpy.vsplit)(ary, indices_or_sections) | Split an array into multiple sub-arrays vertically (row-wise).

## Tiling arrays

method | description
---|---
[tile](https://numpy.org/devdocs/reference/generated/numpy.tile.html#numpy.tile)(A, reps) | Construct an array by [repeat](https://numpy.org/devdocs/reference/generated/numpy.repeat.html#numpy.repeat)ing A the number of times given by reps.
repeat(a, repeats[, axis]) | Repeat elements of an array.

## Adding and removing elements

method | description
---|---
[delete](https://numpy.org/devdocs/reference/generated/numpy.delete.html#numpy.delete)(arr, obj[, axis]) | Return a new array with sub-arrays along an axis deleted.
[insert](https://numpy.org/devdocs/reference/generated/numpy.insert.html#numpy.insert)(arr, obj, values[, axis]) | Insert values along the given axis before the given indices.
[append](https://numpy.org/devdocs/reference/generated/numpy.append.html#numpy.append)(arr, values[, axis]) | Append values to the end of an array.
[resize](https://numpy.org/devdocs/reference/generated/numpy.resize.html#numpy.resize)(a, new_shape) | Return a new array with the specified shape.
[trim_zeros](https://numpy.org/devdocs/reference/generated/numpy.trim_zeros.html#numpy.trim_zeros)(filt[, trim]) | Trim the leading and/or trailing zeros from a 1-D array or sequence.
[unique](https://numpy.org/devdocs/reference/generated/numpy.unique.html#numpy.unique)(ar[, return_index, return_inverse, …]) | Find the unique elements of an array.

## Rearranging elements

method | description
---|---
[flip](https://numpy.org/devdocs/reference/generated/numpy.flip.html#numpy.flip)(m[, axis]) | Reverse the order of elements in an array along the given axis.
[fliplr](https://numpy.org/devdocs/reference/generated/numpy.fliplr.html#numpy.fliplr)(m) | Flip array in the left/right direction.
[flipud](https://numpy.org/devdocs/reference/generated/numpy.flipud.html#numpy.flipud)(m) | Flip array in the up/down direction.
[reshape](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape)(a, newshape[, order]) | Gives a new shape to an array without changing its data.
[roll](https://numpy.org/devdocs/reference/generated/numpy.roll.html#numpy.roll)(a, shift[, axis]) | Roll array elements along a given axis.
[rot90](https://numpy.org/devdocs/reference/generated/numpy.rot90.html#numpy.rot90)(m[, k, axes]) | Rotate an array by 90 degrees in the plane specified by axes.
