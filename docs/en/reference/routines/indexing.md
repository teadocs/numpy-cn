# Indexing routines

::: tip See also

[Indexing](arrays.indexing.html#arrays-indexing)

:::

## Generating index arrays

method | description
---|---
[c_](https://numpy.org/devdocs/reference/generated/numpy.c_.html#numpy.c_) | Translates slice objects to concatenation along the second axis.
[r_](https://numpy.org/devdocs/reference/generated/numpy.r_.html#numpy.r_) | Translates slice objects to concatenation along the first axis.
[s_](https://numpy.org/devdocs/reference/generated/numpy.s_.html#numpy.s_) | A nicer way to build up index tuples for arrays.
[nonzero](https://numpy.org/devdocs/reference/generated/numpy.nonzero.html#numpy.nonzero)(a) | Return the [indices](https://numpy.org/devdocs/reference/generated/numpy.indices.html#numpy.indices) of the elements that are non-zero.
[where](https://numpy.org/devdocs/reference/generated/numpy.where.html#numpy.where)(condition, [x, y]) | Return elements chosen from x or y depending on condition.
indices(dimensions[, dtype, sparse]) | Return an array representing the indices of a grid.
[ix_](https://numpy.org/devdocs/reference/generated/numpy.ix_.html#numpy.ix_)(\*args) | Construct an open mesh from multiple sequences.
[ogrid](https://numpy.org/devdocs/reference/generated/numpy.ogrid.html#numpy.ogrid) | nd_grid instance which returns an open multi-dimensional “meshgrid”.
[ravel_multi_index](https://numpy.org/devdocs/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index)(multi_index, dims[, mode, …]) | Converts a tuple of index arrays into an array of flat indices, applying boundary modes to the multi-index.
[unravel_index](https://numpy.org/devdocs/reference/generated/numpy.unravel_index.html#numpy.unravel_index)(indices, shape[, order]) | Converts a flat index or array of flat indices into a tuple of coordinate arrays.
[diag_indices](https://numpy.org/devdocs/reference/generated/numpy.diag_indices.html#numpy.diag_indices)(n[, ndim]) | Return the indices to access the main diagonal of an array.
[diag_indices_from](https://numpy.org/devdocs/reference/generated/numpy.diag_indices_from.html#numpy.diag_indices_from)(arr) | Return the indices to access the main diagonal of an n-dimensional array.
[mask_indices](https://numpy.org/devdocs/reference/generated/numpy.mask_indices.html#numpy.mask_indices)(n, mask_func[, k]) | Return the indices to access (n, n) arrays, given a masking function.
[tril_indices](https://numpy.org/devdocs/reference/generated/numpy.tril_indices.html#numpy.tril_indices)(n[, k, m]) | Return the indices for the lower-triangle of an (n, m) array.
[tril_indices_from](https://numpy.org/devdocs/reference/generated/numpy.tril_indices_from.html#numpy.tril_indices_from)(arr[, k]) | Return the indices for the lower-triangle of arr.
[triu_indices](https://numpy.org/devdocs/reference/generated/numpy.triu_indices.html#numpy.triu_indices)(n[, k, m]) | Return the indices for the upper-triangle of an (n, m) array.
[triu_indices_from](https://numpy.org/devdocs/reference/generated/numpy.triu_indices_from.html#numpy.triu_indices_from)(arr[, k]) | Return the indices for the upper-triangle of arr.

## Indexing-like operations

method | description
---|---
[take](https://numpy.org/devdocs/reference/generated/numpy.take.html#numpy.take)(a, indices[, axis, out, mode]) | Take elements from an array along an axis.
[take_along_axis](https://numpy.org/devdocs/reference/generated/numpy.take_along_axis.html#numpy.take_along_axis)(arr, indices, axis) | Take values from the input array by matching 1d index and data slices.
[choose](https://numpy.org/devdocs/reference/generated/numpy.choose.html#numpy.choose)(a, choices[, out, mode]) | Construct an array from an index array and a set of arrays to choose from.
[compress](https://numpy.org/devdocs/reference/generated/numpy.compress.html#numpy.compress)(condition, a[, axis, out]) | Return [select](https://numpy.org/devdocs/reference/generated/numpy.select.html#numpy.select)ed slices of an array along given axis.
[diag](https://numpy.org/devdocs/reference/generated/numpy.diag.html#numpy.diag)(v[, k]) | Extract a [diagonal](https://numpy.org/devdocs/reference/generated/numpy.diagonal.html#numpy.diagonal) or construct a diagonal array.
[diagonal](https://numpy.org/devdocs/reference/generated/numpy.diagonal.html#numpy.diagonal)(a[, offset, axis1, axis2]) | Return specified diagonals.
[select](https://numpy.org/devdocs/reference/generated/numpy.select.html#numpy.select)(condlist, choicelist[, default]) | Return an array drawn from elements in choicelist, depending on conditions.
[lib.stride_tricks.as_strided](https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.as_strided.html#numpy.lib.stride_tricks.as_strided)(x[, shape, …]) | Create a view into the array with the given shape and strides.

## Inserting data into arrays

method | description
---|---
[place](https://numpy.org/devdocs/reference/generated/numpy.place.html#numpy.place)(arr, mask, vals) | Change elements of an array based on conditional and in[put](https://numpy.org/devdocs/reference/generated/numpy.put.html#numpy.put) values.
[put](https://numpy.org/devdocs/reference/generated/numpy.put.html#numpy.put)(a, ind, v[, mode]) | Replaces specified elements of an array with given values.
[put_along_axis](https://numpy.org/devdocs/reference/generated/numpy.put_along_axis.html#numpy.put_along_axis)(arr, indices, values, axis) | Put values into the destination array by matching 1d index and data slices.
[putmask](https://numpy.org/devdocs/reference/generated/numpy.putmask.html#numpy.putmask)(a, mask, values) | Changes elements of an array based on conditional and input values.
[fill_diagonal](https://numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html#numpy.fill_diagonal)(a, val[, wrap]) | Fill the main diagonal of the given array of any dimensionality.

## Iterating over arrays

method | description
---|---
[nditer](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer) | Efficient multi-dimensional iterator object to iterate over arrays.
[ndenumerate](https://numpy.org/devdocs/reference/generated/numpy.ndenumerate.html#numpy.ndenumerate)(arr) | Multidimensional index iterator.
[ndindex](https://numpy.org/devdocs/reference/generated/numpy.ndindex.html#numpy.ndindex)(*shape) | An N-dimensional iterator object to index arrays.
[nested_iters](https://numpy.org/devdocs/reference/generated/numpy.nested_iters.html#numpy.nested_iters)() | Create nditers for use in nested loops
[flatiter](https://numpy.org/devdocs/reference/generated/numpy.flatiter.html#numpy.flatiter) | Flat iterator object to iterate over arrays.
[lib.Arrayterator](https://numpy.org/devdocs/reference/generated/numpy.lib.Arrayterator.html#numpy.lib.Arrayterator)(var[, buf_size]) | Buffered iterator for big arrays.