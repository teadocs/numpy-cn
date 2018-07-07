# 统计学

## Order statistics

- amin(a[, axis, out, keepdims])	Return the minimum of an array or minimum along an axis.
- amax(a[, axis, out, keepdims])	Return the maximum of an array or maximum along an axis.
- nanmin(a[, axis, out, keepdims])	Return minimum of an array or minimum along an axis, ignoring any NaNs.
- nanmax(a[, axis, out, keepdims])	Return the maximum of an array or maximum along an axis, ignoring any NaNs.
- ptp(a[, axis, out])	Range of values (maximum - minimum) along an axis.
- percentile(a, q[, axis, out, …])	Compute the qth percentile of the data along the specified axis.
- nanpercentile(a, q[, axis, out, …])	Compute the qth percentile of the data along the specified axis, while ignoring nan values.

## Averages and variances

- median(a[, axis, out, overwrite_input, keepdims])	Compute the median along the specified axis.
- average(a[, axis, weights, returned])	Compute the weighted average along the specified axis.
- mean(a[, axis, dtype, out, keepdims])	Compute the arithmetic mean along the specified axis.
- std(a[, axis, dtype, out, ddof, keepdims])	Compute the standard deviation along the specified axis.
- var(a[, axis, dtype, out, ddof, keepdims])	Compute the variance along the specified axis.
- nanmedian(a[, axis, out, overwrite_input, …])	Compute the median along the specified axis, while ignoring NaNs.
- nanmean(a[, axis, dtype, out, keepdims])	Compute the arithmetic mean along the specified axis, ignoring NaNs.
- nanstd(a[, axis, dtype, out, ddof, keepdims])	Compute the standard deviation along the specified axis, while ignoring NaNs.
- nanvar(a[, axis, dtype, out, ddof, keepdims])	Compute the variance along the specified axis, while ignoring NaNs.

## Correlating

- corrcoef(x[, y, rowvar, bias, ddof])	Return Pearson product-moment correlation coefficients.
- correlate(a, v[, mode])	Cross-correlation of two 1-dimensional sequences.
- cov(m[, y, rowvar, bias, ddof, fweights, …])	Estimate a covariance matrix, given data and weights.

## Histograms

- histogram(a[, bins, range, normed, weights, …])	Compute the histogram of a set of data.
- histogram2d(x, y[, bins, range, normed, weights])	Compute the bi-dimensional histogram of two data samples.
- histogramdd(sample[, bins, range, normed, …])	Compute the multidimensional histogram of some data.
- bincount(x[, weights, minlength])	Count number of occurrences of each value in array of non-negative ints.
- digitize(x, bins[, right])	Return the indices of the bins to which each value in input array belongs.