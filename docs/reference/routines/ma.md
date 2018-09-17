# 掩码数组操作

## 常量

- ma.MaskType numpy.bool_ 的别名

## 创造

### 根据现有数据

- ma.masked_array numpy.ma.core.MaskedArray 的别名
- ma.array(data[, dtype, copy, order, mask, …])	可能带有掩码值的数组类。
- ma.copy(self, *args, **params) a.copy(order=)	返回数组的拷贝。
- ma.frombuffer(buffer[, dtype, count, offset])	将缓冲区解释为一维数组。
- ma.fromfunction(function, shape, **kwargs)	通过在每个坐标上执行函数来构造数组。
- ma.MaskedArray.copy([order])	返回数组的副本。

### Ones 和 zeros 方法

- ma.empty(shape[, dtype, order])	返回给定形状和类型的新数组，而不初始化条目。
- ma.empty_like(a[, dtype, order, subok])	返回一个与给定数组具有相同形状和类型的新数组。
- ma.masked_all(shape[, dtype])	带所有元素的空掩码数组。
- ma.masked_all_like(arr)	空掩码数组，具有现有数组的属性。
- ma.ones(shape[, dtype, order])	返回一个给定形状和类型的新数组，其中填充了这些数组。
- ma.zeros(shape[, dtype, order])	返回一个给定形状和类型的新数组，其中填充了零。

### 检察数组

- ma.all(self[, axis, out, keepdims])	如果所有元素都计算为True，则返回True。
- ma.any(self[, axis, out, keepdims])	如果求值的任何元素为True，则返回True。
- ma.count(self[, axis, keepdims])	沿给定轴计算数组的非掩码元素。
- ma.count_masked(arr[, axis]) 计算沿给定轴的遮罩元素数。
- ma.getmask(a)	返回掩码数组或nomask的掩码。
- ma.getmaskarray(arr)	返回掩码数组的掩码，或False的完整布尔数组。
- ma.getdata(a[, subok])	将掩码数组的数据作为ndarray返回。
- ma.nonzero(self)	返回非零的未屏蔽元素的索引。
- ma.shape(obj)	返回数组的形状。
- ma.size(obj[, axis])	返回给定轴上的元素数。
- ma.is_masked(x)	确定输入是否具有掩码值。
- ma.is_mask(m)	如果m是有效的标准掩码，则返回True。
- ma.MaskedArray.data	返回当前数据，作为原始基础数据的视图。
- ma.MaskedArray.mask	Mask
- ma.MaskedArray.recordmask	返回记录的掩码。
- ma.MaskedArray.all([axis, out, keepdims])	如果所有元素都计算为True，则返回True。
- ma.MaskedArray.any([axis, out, keepdims])	果求值的任何元素为True，则返回True。
- ma.MaskedArray.count([axis, keepdims])	沿给定轴计算数组的非掩码元素。
- ma.MaskedArray.nonzero()	返回非零的未屏蔽元素的索引。
- ma.shape(obj)	返回数组的形状。
- ma.size(obj[, axis])	返回给定轴上的元素数。

## 操作掩码数组

### 改变形状

- ma.ravel(self[, order])	以视图的形式返回Self的一维版本。
- ma.reshape(a, new_shape[, order])	返回一个数组，该数组包含具有新形状的相同数据。
- ma.resize(x, new_shape)	返回具有指定大小和形状的新掩码数组。
- ma.MaskedArray.flatten([order])	返回折叠成一维的数组的副本。
- ma.MaskedArray.ravel([order])	以视图的形式返回Self的一维版本。
- ma.MaskedArray.reshape(*s, **kwargs)	给数组一个新的形状，而不改变它的数据。
- ma.MaskedArray.resize(newshape[, refcheck, …])	

### 修改轴

- ma.swapaxes(self, *args, …)	返回axis1和axis2互换后的数组视图。
- ma.transpose(a[, axes])	排列数组的大小。
- ma.MaskedArray.swapaxes(axis1, axis2)	返回axis1和axis2互换后的数组视图。
- ma.MaskedArray.transpose(*axes)	返回已移置轴的数组视图。

### 改变维数

- ma.atleast_1d(*arys)	将输入转换为至少具有一维的数组。
- ma.atleast_2d(*arys)	将输入视为至少具有两个维度的数组。
- ma.atleast_3d(*arys)	将输入视为至少具有三个维度的数组。
- ma.expand_dims(x, axis)	展开数组的形状。
- ma.squeeze(a[, axis])	从数组的形状中移除一维项。
- ma.MaskedArray.squeeze([axis])	对象的形状中移除一维项。
- ma.column_stack(tup)	将一维数组作为列堆栈到二维数组中.
- ma.concatenate(arrays[, axis])	沿着给定的轴连接数组序列。
- ma.dstack(tup)	按序列深度排列数组(沿第三轴)。
- ma.hstack(tup)	以水平顺序(列方式)将数组堆叠。
- ma.hsplit(ary, indices_or_sections)	横向(按列)将数组拆分为多个子数组。
- ma.mr_	沿第一轴将切片对象转换为串联。
- ma.row_stack(tup)	按顺序垂直(行)排列数组。
- ma.vstack(tup)	按顺序垂直(行)排列数组。
- Joining arrays
- ma.column_stack(tup)	将一维数组作为列堆栈到二维数组中.
- ma.concatenate(arrays[, axis])	沿着给定的轴连接数组序列。
- ma.append(a, b[, axis])	将值追加到数组的末尾。
- ma.dstack(tup)	按序列深度排列数组(沿第三轴)。
- ma.hstack(tup)	以水平顺序(列方式)将数组堆叠。
- ma.vstack(tup)	按顺序垂直(行)排列数组。

## 掩码操作

### 创建掩码

- ma.make_mask(m[, copy, shrink, dtype])	从数组创建布尔掩码。
- ma.make_mask_none(newshape[, dtype])	返回给定形状的布尔掩码，填充False。
- ma.mask_or(m1, m2[, copy, shrink])	使用logical_or运算符组合两个掩码。
- ma.make_mask_descr(ndtype)	从给定的dtype构造一个dtype描述列表。

### 访问掩码

- ma.getmask(a)	返回蒙版数组或nomask的掩码。
- ma.getmaskarray(arr)	返回掩码数组的掩码，或False的完整布尔数组。
- ma.masked_array.mask	Mask

### 查找掩码数据

- ma.flatnotmasked_contiguous(a)	沿给定轴在掩码数组中查找连续的未屏蔽数据。
- ma.flatnotmasked_edges(a)	查找第一个和最后一个未屏蔽值的索引。
- ma.notmasked_contiguous(a[, axis])	沿给定轴在掩码数组中查找连续的未屏蔽数据。
- ma.notmasked_edges(a[, axis])	查找沿轴的第一个和最后一个未屏蔽值的索引。
- ma.clump_masked(a)	返回与1-D数组的掩码块对应的切片列表。
- ma.clump_unmasked(a)	返回与1-D阵列的未掩蔽块相对应的切片列表。

### 修改掩码

- ma.mask_cols(a[, axis])	屏蔽包含屏蔽值的2D数组的列。
- ma.mask_or(m1, m2[, copy, shrink])	使用logical_or运算符组合两个掩码。
- ma.mask_rowcols(a[, axis])	屏蔽包含屏蔽值的2D数组的行和/或列。
- ma.mask_rows(a[, axis])	屏蔽包含屏蔽值的2D数组的行。
- ma.harden_mask(self)	硬化掩码。
- ma.soften_mask(self)	软化掩码。
- ma.MaskedArray.harden_mask()	硬化掩码。
- ma.MaskedArray.soften_mask()	软化掩码。
- ma.MaskedArray.shrink_mask()	尽可能将掩码减少到nomask。
- ma.MaskedArray.unshare_mask()	复制掩码并将sharedmask标志设置为False。

## 转换操作

### \> 转化为掩码数组

- ma.asarray(a[, dtype, order])	将输入转换为给定数据类型的掩码数组。
- ma.asanyarray(a[, dtype])	将输入转换为掩码数组，保留子类。
- ma.fix_invalid(a[, mask, copy, fill_value])	返回带有无效数据的输入，并用填充值替换。
- ma.masked_equal(x, value[, copy])	掩盖一个等于给定值的数组。
- ma.masked_greater(x, value[, copy])	掩盖大于给定值的数组。
- ma.masked_greater_equal(x, value[, copy])	掩盖大于或等于给定值的数组。
- ma.masked_inside(x, v1, v2[, copy])	在给定间隔内掩盖数组。
- ma.masked_invalid(a[, copy])	掩盖出现无效值的数组（NaN或infs）。
- ma.masked_less(x, value[, copy])	掩盖小于给定值的数组。
- ma.masked_less_equal(x, value[, copy])	掩盖小于或等于给定值的数组。
- ma.masked_not_equal(x, value[, copy])	掩盖不等于给定值的数组。
- ma.masked_object(x, value[, copy, shrink])	掩盖数据正好等于值的数组x。
- ma.masked_outside(x, v1, v2[, copy])	在给定间隔之外屏蔽数组。
- ma.masked_values(x, value[, rtol, atol, …])	掩盖浮点数相等的数组。
- ma.masked_where(condition, a[, copy])	掩盖满足条件的数组。

### \> 转化为一个numpy数组

- ma.compress_cols(a)	取消包含掩码值的二维数组的整列。
- ma.compress_rowcols(x[, axis])	取消二维数组中包含掩码值的行和/或列。
- ma.compress_rows(a)	取消包含掩码值的二维数组的整行数据。
- ma.compressed(x)	以一维数组的形式返回所有非掩码数据。
- ma.filled(a[, fill_value])	以数组的形式返回输入，用填充值替换掩码数据。
- ma.MaskedArray.compressed()	以一维数组的形式返回所有非掩码数据。
- ma.MaskedArray.filled([fill_value])	返回Self的副本，并使用给定的值填充掩码值。

### \> 转化为其他对象

- ma.MaskedArray.tofile(fid[, sep, format])	将掩码数组保存到二进制格式的文件中。
- ma.MaskedArray.tolist([fill_value])	以层次化Python列表的形式返回掩码数组的数据部分。
- ma.MaskedArray.torecords()	将掩码数组转换为灵活类型的数组。
- ma.MaskedArray.tobytes([fill_value, order])	将数组数据作为包含数组中原始字节的字符串返回。

### 腌制和反腌制

- ma.dump(a, F)	腌制一个掩码数组并写入文件
- ma.dumps(a)	返回与掩码数组的腌制相对应的字符串。
- ma.load(F)	cPickle.load的包装器，它接受类似文件的对象或文件名。
- ma.loads(strg)	从当前字符串加载腌制后的数组。

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

### Polynomial fit

- ma.vander(x[, n])	Generate a Vandermonde matrix.
- ma.polyfit(x, y, deg[, rcond, full, w, cov])	Least squares polynomial fit.

### Clipping and rounding

- ma.around	Round an array to the given number of decimals.
- ma.clip(a, a_min, a_max[, out])	Clip (limit) the values in an array.
- ma.round(a[, decimals, out])	Return a copy of a, rounded to ‘decimals’ places.
- ma.MaskedArray.clip([min, max, out])	Return an array whose values are limited to [min, max].
- ma.MaskedArray.round([decimals, out])	Return each element rounded to the given number of decimals.

### Miscellanea

- ma.allequal(a, b[, fill_value])	Return True if all entries of a and b are equal, using fill_value as a truth value where either or both are masked.
- ma.allclose(a, b[, masked_equal, rtol, atol])	Returns True if two arrays are element-wise equal within a tolerance.
- ma.apply_along_axis(func1d, axis, arr, …)	Apply a function to 1-D slices along the given axis.
- ma.arange([start,] stop[, step,][, dtype])	Return evenly spaced values within a given interval.
- ma.choose(indices, choices[, out, mode])	Use an index array to construct a new array from a set of choices.
- ma.ediff1d(arr[, to_end, to_begin])	Compute the differences between consecutive elements of an array.
- ma.indices(dimensions[, dtype])	Return an array representing the indices of a grid.
- ma.where(condition[, x, y])	Return a masked array with elements from x or y, depending on condition.