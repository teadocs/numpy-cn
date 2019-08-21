# Sorting, searching, and counting

## Sorting

method | description
---|---
[sort](generated/numpy.sort.html#numpy.sort)(a[, axis, kind, order]) | Return a sorted copy of an array.
[lexsort](generated/numpy.lexsort.html#numpy.lexsort)(keys[, axis]) | Perform an indirect stable sort using a sequence of keys.
[argsort](generated/numpy.argsort.html#numpy.argsort)(a[, axis, kind, order]) | Returns the indices that would sort an array.
[ndarray.sort](generated/numpy.ndarray.sort.html#numpy.ndarray.sort)([axis, kind, order]) | Sort an array in-place.
[msort](generated/numpy.msort.html#numpy.msort)(a) | Return a copy of an array sorted along the first axis.
[sort_complex](generated/numpy.sort_complex.html#numpy.sort_complex)(a) | Sort a complex array using the real part first, then the imaginary part.
[partition](generated/numpy.partition.html#numpy.partition)(a, kth[, axis, kind, order]) | Return a partitioned copy of an array.
[argpartition](generated/numpy.argpartition.html#numpy.argpartition)(a, kth[, axis, kind, order]) | Perform an indirect partition along the given axis using the algorithm specified by the kind keyword.

## Searching

method | description
---|---
[argmax](generated/numpy.argmax.html#numpy.argmax)(a[, axis, out]) | Returns the indices of the maximum values along an axis.
[nanargmax](generated/numpy.nanargmax.html#numpy.nanargmax)(a[, axis]) | Return the indices of the maximum values in the specified axis ignoring NaNs.
[argmin](generated/numpy.argmin.html#numpy.argmin)(a[, axis, out]) | Returns the indices of the minimum values along an axis.
[nanargmin](generated/numpy.nanargmin.html#numpy.nanargmin)(a[, axis]) | Return the indices of the minimum values in the specified axis ignoring NaNs.
[argwhere](generated/numpy.argwhere.html#numpy.argwhere)(a) | Find the indices of array elements that are non-zero, grouped by element.
[nonzero](generated/numpy.nonzero.html#numpy.nonzero)(a) | Return the indices of the elements that are non-zero.
[flatnonzero](generated/numpy.flatnonzero.html#numpy.flatnonzero)(a) | Return indices that are non-zero in the flattened version of a.
[where](generated/numpy.where.html#numpy.where)(condition, [x, y]) | Return elements chosen from x or y depending on condition.
[searchsorted](generated/numpy.searchsorted.html#numpy.searchsorted)(a, v[, side, sorter]) | Find indices where elements should be inserted to maintain order.
[extract](generated/numpy.extract.html#numpy.extract)(condition, arr) | Return the elements of an array that satisfy some condition.

## Counting

method | description
---|---
[count_nonzero](generated/numpy.count_nonzero.html#numpy.count_nonzero)(a[, axis]) | Counts the number of non-zero values in the array a.