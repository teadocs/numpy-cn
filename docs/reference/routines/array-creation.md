# Array creation routines

::: tip See also

[Array creation](https://numpy.org/devdocs/user/basics.creation.html#arrays-creation)

:::

## Ones and zeros

[empty](generated/numpy.empty.html#numpy.empty)(shape[, dtype, order]) | Return a new array of given shape and type, without initializing entries.
---|---
[empty_like](generated/numpy.empty_like.html#numpy.empty_like)(prototype[, dtype, order, subok, …]) | Return a new array with the same shape and type as a given array.
[eye](generated/numpy.eye.html#numpy.eye)(N[, M, k, dtype, order]) | Return a 2-D array with [ones](generated/numpy.ones.html#numpy.ones) on the diagonal and [zeros](generated/numpy.zeros.html#numpy.zeros) elsewhere.
[identity](generated/numpy.identity.html#numpy.identity)(n[, dtype]) | Return the identity array.
ones(shape[, dtype, order]) | Return a new array of given shape and type, filled with ones.
[ones_like](generated/numpy.ones_like.html#numpy.ones_like)(a[, dtype, order, subok, shape]) | Return an array of ones with the same shape and type as a given array.
zeros(shape[, dtype, order]) | Return a new array of given shape and type, filled with zeros.
[zeros_like](generated/numpy.zeros_like.html#numpy.zeros_like)(a[, dtype, order, subok, shape]) | Return an array of zeros with the same shape and type as a given array.
[full](generated/numpy.full.html#numpy.full)(shape, fill_value[, dtype, order]) | Return a new array of given shape and type, filled with fill_value.
[full_like](generated/numpy.full_like.html#numpy.full_like)(a, fill_value[, dtype, order, …]) | Return a full array with the same shape and type as a given array.

## From existing data

[array](generated/numpy.array.html#numpy.array)(object[, dtype, [copy](generated/numpy.copy.html#numpy.copy), order, subok, ndmin]) | Create an array.
---|---
[asarray](generated/numpy.asarray.html#numpy.asarray)(a[, dtype, order]) | Convert the input to an array.
[asanyarray](generated/numpy.asanyarray.html#numpy.asanyarray)(a[, dtype, order]) | Convert the input to an ndarray, but pass ndarray subclasses through.
[ascontiguousarray](generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray)(a[, dtype]) | Return a contiguous array (ndim >= 1) in memory (C order).
[asmatrix](generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | Interpret the input as a matrix.
copy(a[, order]) | Return an array copy of the given object.
[frombuffer](generated/numpy.frombuffer.html#numpy.frombuffer)(buffer[, dtype, count, offset]) | Interpret a buffer as a 1-dimensional array.
[fromfile](generated/numpy.fromfile.html#numpy.fromfile)(file[, dtype, count, sep, offset]) | Construct an array from data in a text or binary file.
[fromfunction](generated/numpy.fromfunction.html#numpy.fromfunction)(function, shape, \*\*kwargs) | Construct an array by executing a function over each coordinate.
[fromiter](generated/numpy.fromiter.html#numpy.fromiter)(iterable, dtype[, count]) | Create a new 1-dimensional array from an iterable object.
[fromstring](generated/numpy.fromstring.html#numpy.fromstring)(string[, dtype, count, sep]) | A new 1-D array initialized from text data in a string.
[loadtxt](generated/numpy.loadtxt.html#numpy.loadtxt)(fname[, dtype, comments, delimiter, …]) | Load data from a text file.

## Creating record arrays (``numpy.rec``)

::: tip Note

``numpy.rec`` is the preferred alias for
``numpy.core.records``.

:::

[core.records.array](generated/numpy.core.records.array.html#numpy.core.records.array)(obj[, dtype, shape, …]) | Construct a record array from a wide-variety of objects.
---|---
[core.records.fromarrays](generated/numpy.core.records.fromarrays.html#numpy.core.records.fromarrays)(arrayList[, dtype, …]) | create a record array from a (flat) list of arrays
[core.records.fromrecords](generated/numpy.core.records.fromrecords.html#numpy.core.records.fromrecords)(recList[, dtype, …]) | create a recarray from a list of records in text form
[core.records.fromstring](generated/numpy.core.records.fromstring.html#numpy.core.records.fromstring)(datastring[, dtype, …]) | create a (read-only) record array from binary data contained in a string
[core.records.fromfile](generated/numpy.core.records.fromfile.html#numpy.core.records.fromfile)(fd[, dtype, shape, …]) | Create an array from binary file data

## Creating character arrays (``numpy.char``)

::: tip Note

[``numpy.char``](routines.char.html#module-numpy.char) is the preferred alias for
``numpy.core.defchararray``.

:::

[core.def[[chararray](generated/numpy.chararray.html#numpy.chararray)](generated/numpy.chararray.html#numpy.chararray).array](generated/numpy.core.defchararray.array.html#numpy.core.defchararray.array)(obj[, itemsize, …]) | Create a chararray.
---|---
[core.defchararray.asarray](generated/numpy.core.defchararray.asarray.html#numpy.core.defchararray.asarray)(obj[, itemsize, …]) | Convert the input to a chararray, copying the data only if necessary.

## Numerical ranges

[arange](generated/numpy.arange.html#numpy.arange)([start,] stop[, step,][, dtype]) | Return evenly spaced values within a given interval.
---|---
[linspace](generated/numpy.linspace.html#numpy.linspace)(start, stop[, num, endpoint, …]) | Return evenly spaced numbers over a specified interval.
[logspace](generated/numpy.logspace.html#numpy.logspace)(start, stop[, num, endpoint, base, …]) | Return numbers spaced evenly on a log scale.
[geomspace](generated/numpy.geomspace.html#numpy.geomspace)(start, stop[, num, endpoint, …]) | Return numbers spaced evenly on a log scale (a geometric progression).
[meshgrid](generated/numpy.meshgrid.html#numpy.meshgrid)(\*xi, \*\*kwargs) | Return coordinate matrices from coordinate vectors.
[mgrid](generated/numpy.mgrid.html#numpy.mgrid) | nd_grid instance which returns a dense multi-dimensional “meshgrid”.
[ogrid](generated/numpy.ogrid.html#numpy.ogrid) | nd_grid instance which returns an open multi-dimensional “meshgrid”.

## Building matrices

[diag](generated/numpy.diag.html#numpy.diag)(v[, k]) | Extract a diagonal or construct a diagonal array.
---|---
[diagflat](generated/numpy.diagflat.html#numpy.diagflat)(v[, k]) | Create a two-dimensional array with the flattened input as a diagonal.
[tri](generated/numpy.tri.html#numpy.tri)(N[, M, k, dtype]) | An array with ones at and below the given diagonal and zeros elsewhere.
[tril](generated/numpy.tril.html#numpy.tril)(m[, k]) | Lower triangle of an array.
[triu](generated/numpy.triu.html#numpy.triu)(m[, k]) | Upper triangle of an array.
[vander](generated/numpy.vander.html#numpy.vander)(x[, N, increasing]) | Generate a Vandermonde matrix.

## The Matrix class

[mat](generated/numpy.mat.html#numpy.mat)(data[, dtype]) | Interpret the input as a matrix.
---|---
[bmat](generated/numpy.bmat.html#numpy.bmat)(obj[, ldict, gdict]) | Build a matrix object from a string, nested sequence, or array.
