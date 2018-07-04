# 索引相关API

另见：
> Indexing

## Generating index arrays

- c_	Translates slice objects to concatenation along the second axis.
- r_	Translates slice objects to concatenation along the first axis.
- s_	A nicer way to build up index tuples for arrays.
- nonzero(a)	Return the indices of the elements that are non-zero.
- where(condition, [x, y])	Return elements, either from x or y, depending on condition.
- indices(dimensions[, dtype])	Return an array representing the indices of a grid.
- ix_(*args)	Construct an open mesh from multiple sequences.
- ogrid	nd_grid instance which returns an open multi-dimensional “meshgrid”.
- ravel_multi_index(multi_index, dims[, mode, …])	Converts a tuple of index arrays into an array of flat indices, applying boundary modes to the multi-index.
- unravel_index(indices, dims[, order])	Converts a flat index or array of flat indices into a tuple of coordinate arrays.
- diag_indices(n[, ndim])	Return the indices to access the main diagonal of an array.
- diag_indices_from(arr)	Return the indices to access the main diagonal of an n-dimensional array.
- mask_indices(n, mask_func[, k])	Return the indices to access (n, n) arrays, given a masking function.
- tril_indices(n[, k, m])	Return the indices for the lower-triangle of an (n, m) array.
- tril_indices_from(arr[, k])	Return the indices for the lower-triangle of arr.
- triu_indices(n[, k, m])	Return the indices for the upper-triangle of an (n, m) array.
- triu_indices_from(arr[, k])	Return the indices for the upper-triangle of arr.

## Indexing-like operations

- take(a, indices[, axis, out, mode])	Take elements from an array along an axis.
- choose(a, choices[, out, mode])	Construct an array from an index array and a set of arrays to choose from.
- compress(condition, a[, axis, out])	Return selected slices of an array along given axis.
- diag(v[, k])	Extract a diagonal or construct a diagonal array.
- diagonal(a[, offset, axis1, axis2])	Return specified diagonals.
- select(condlist, choicelist[, default])	Return an array drawn from elements in choicelist, depending on conditions.
- lib.stride_tricks.as_strided(x[, shape, …])	Create a view into the array with the given shape and strides.

## Inserting data into arrays

- place(arr, mask, vals)	Change elements of an array based on conditional and input values.
- put(a, ind, v[, mode])	Replaces specified elements of an array with given values.
- putmask(a, mask, values)	Changes elements of an array based on conditional and input values.
- fill_diagonal(a, val[, wrap])	Fill the main diagonal of the given array of any dimensionality.

## Iterating over arrays

- nditer	Efficient multi-dimensional iterator object to iterate over arrays.
- ndenumerate(arr)	Multidimensional index iterator.
- ndindex(*shape)	An N-dimensional iterator object to index arrays.
- flatiter	Flat iterator object to iterate over arrays.
- lib.Arrayterator(var[, buf_size])	Buffered iterator for big arrays.