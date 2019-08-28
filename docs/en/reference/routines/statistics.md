# Statistics

## Order statistics

method | description
---|---
[amin](generated/numpy.amin.html#numpy.amin)(a[, axis, out, keepdims, initial, where]) | Return the minimum of an array or minimum along an axis.
[amax](generated/numpy.amax.html#numpy.amax)(a[, axis, out, keepdims, initial, where]) | Return the maximum of an array or maximum along an axis.
[nanmin](generated/numpy.nanmin.html#numpy.nanmin)(a[, axis, out, keepdims]) | Return minimum of an array or minimum along an axis, ignoring any NaNs.
[nanmax](generated/numpy.nanmax.html#numpy.nanmax)(a[, axis, out, keepdims]) | Return the maximum of an array or maximum along an axis, ignoring any NaNs.
[ptp](generated/numpy.ptp.html#numpy.ptp)(a[, axis, out, keepdims]) | Range of values (maximum - minimum) along an axis.
[percentile](generated/numpy.percentile.html#numpy.percentile)(a, q[, axis, out, …]) | Compute the q-th percentile of the data along the specified axis.
[nanpercentile](generated/numpy.nanpercentile.html#numpy.nanpercentile)(a, q[, axis, out, …]) | Compute the qth percentile of the data along the specified axis, while ignoring nan values.
[quantile](generated/numpy.quantile.html#numpy.quantile)(a, q[, axis, out, overwrite_input, …]) | Compute the q-th quantile of the data along the specified axis.
[nanquantile](generated/numpy.nanquantile.html#numpy.nanquantile)(a, q[, axis, out, …]) | Compute the qth quantile of the data along the specified axis, while ignoring nan values.

## Averages and variances

method | description
---|---
[median](generated/numpy.median.html#numpy.median)(a[, axis, out, overwrite_input, keepdims]) | Compute the median along the specified axis.
[average](generated/numpy.average.html#numpy.average)(a[, axis, weights, returned]) | Compute the weighted average along the specified axis.
[mean](generated/numpy.mean.html#numpy.mean)(a[, axis, dtype, out, keepdims]) | Compute the arithmetic mean along the specified axis.
[std](generated/numpy.std.html#numpy.std)(a[, axis, dtype, out, ddof, keepdims]) | Compute the standard deviation along the specified axis.
[var](generated/numpy.var.html#numpy.var)(a[, axis, dtype, out, ddof, keepdims]) | Compute the variance along the specified axis.
[nanmedian](generated/numpy.nanmedian.html#numpy.nanmedian)(a[, axis, out, overwrite_input, …]) | Compute the median along the specified axis, while ignoring NaNs.
[nanmean](generated/numpy.nanmean.html#numpy.nanmean)(a[, axis, dtype, out, keepdims]) | Compute the arithmetic mean along the specified axis, ignoring NaNs.
[nanstd](generated/numpy.nanstd.html#numpy.nanstd)(a[, axis, dtype, out, ddof, keepdims]) | Compute the standard deviation along the specified axis, while ignoring NaNs.
[nanvar](generated/numpy.nanvar.html#numpy.nanvar)(a[, axis, dtype, out, ddof, keepdims]) | Compute the variance along the specified axis, while ignoring NaNs.

## Correlating

method | description
---|---
[corrcoef](generated/numpy.corrcoef.html#numpy.corrcoef)(x[, y, rowvar, bias, ddof]) | Return Pearson product-moment correlation coefficients.
[correlate](generated/numpy.correlate.html#numpy.correlate)(a, v[, mode]) | Cross-correlation of two 1-dimensional sequences.
[cov](generated/numpy.cov.html#numpy.cov)(m[, y, rowvar, bias, ddof, fweights, …]) | Estimate a covariance matrix, given data and weights.

## Histograms

method | description
---|---
[histogram](generated/numpy.histogram.html#numpy.histogram)(a[, bins, range, normed, weights, …]) | Compute the histogram of a set of data.
[histogram2d](generated/numpy.histogram2d.html#numpy.histogram2d)(x, y[, bins, range, normed, …]) | Compute the bi-dimensional histogram of two data samples.
[histogramdd](generated/numpy.histogramdd.html#numpy.histogramdd)(sample[, bins, range, normed, …]) | Compute the multidimensional histogram of some data.
[bincount](generated/numpy.bincount.html#numpy.bincount)(x[, weights, minlength]) | Count number of occurrences of each value in array of non-negative ints.
[histogram_bin_edges](generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges)(a[, bins, range, weights]) | Function to calculate only the edges of the bins used by the histogram function.
[digitize](generated/numpy.digitize.html#numpy.digitize)(x, bins[, right]) | Return the indices of the bins to which each value in input array belongs.
