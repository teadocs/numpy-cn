# 统计学

## 顺序统计

- amin(a[, axis, out, keepdims])	返回数组的最小值或沿轴的最小值。
- amax(a[, axis, out, keepdims])	返回数组的最大值或沿轴的最大值。
- nanmin(a[, axis, out, keepdims])	返回数组的最小值或沿轴的最小值，忽略任何NAS。
- nanmax(a[, axis, out, keepdims])	返回数组的最大值或沿轴方向的最大值，忽略任何NAS。
- ptp(a[, axis, out])	沿轴的值的范围(最大值-最小值)。
- percentile(a, q[, axis, out, …])	计算数据沿指定轴的第qth百分位数。
- nanpercentile(a, q[, axis, out, …])	在忽略NaN值的情况下，沿着指定的轴计算数据的第qth百分位数。

## 平均数和差异

- median(a[, axis, out, overwrite_input, keepdims])	沿指定轴计算中值。
- average(a[, axis, weights, returned])	计算沿指定轴的加权平均。
- mean(a[, axis, dtype, out, keepdims])	沿指定的轴计算算术平均值。
- std(a[, axis, dtype, out, ddof, keepdims])	计算沿指定轴的标准偏差。
- var(a[, axis, dtype, out, ddof, keepdims])	计算沿指定轴的方差。
- nanmedian(a[, axis, out, overwrite_input, …])	在忽略NAS的情况下，沿指定的轴计算中值。
- nanmean(a[, axis, dtype, out, keepdims])	计算沿指定轴的算术平均值，忽略NAS。
- nanstd(a[, axis, dtype, out, ddof, keepdims])	计算指定轴上的标准偏差，而忽略NAS。
- nanvar(a[, axis, dtype, out, ddof, keepdims])	计算指定轴上的方差，同时忽略NAS。

## 关联

- corrcoef(x[, y, rowvar, bias, ddof])	返回Pearson乘积矩相关系数。
- correlate(a, v[, mode]) 返回两个一维序列的交叉关系。
- cov(m[, y, rowvar, bias, ddof, fweights, …])	估计协方差矩阵，给定数据和权重。

## 直方图📊

- histogram(a[, bins, range, normed, weights, …])	计算一组数据的直方图。
- histogram2d(x, y[, bins, range, normed, weights])	计算两个数据样本的二维直方图。
- histogramdd(sample[, bins, range, normed, …])	计算某些数据的多维直方图。
- bincount(x[, weights, minlength])	计算非负INT数组中每个值出现的次数。
- digitize(x, bins[, right])	返回输入数组中每个值所属的bins的索引。