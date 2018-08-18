# 操作掩码数组

## 常量

 ``numpy.bool_`` 的别名是 ``ma.MaskType``。

## 创建

### 从现有数据

- ma.masked_array numpy.ma.core.MaskedArray 的别名
- ma.array(data[, dtype, copy, order, mask, …])	具有可能掩盖值的数组类。
- ma.copy(self, *args, **params) a.copy(order=)	返回一个数组的拷贝.
- ma.frombuffer(buffer[, dtype, count, offset])	将缓冲区解释为一维数组。
- ma.fromfunction(function, shape, **kwargs) 通过在每个坐标上执行函数来构造数组。
- ma.MaskedArray.copy([order]) 返回数组的副本。

### 一个零和一个零

- ma.empty(shape[, dtype, order]) 返回给定形状和类型的新数组，而不初始化条目。
- ma.empty_like(a[, dtype, order, subok]) 返回一个与给定数组具有相同形状和类型的新数组。
- ma.masked_all(shape[, dtype])	空掩码数组，所有元素都被屏蔽。
- ma.masked_all_like(arr) 具有现有数组属性的空掩码数组。
- ma.ones(shape[, dtype, order]) 返回一个给定形状和类型的新数组，用一个填充。
- ma.zeros(shape[, dtype, order]) 返回给定形状和类型的新数组，用零填充。

### 检查数组

- ma.all(self[, axis, out, keepdims])	如果所有元素都计算为True，则返回True。
- ma.any(self[, axis, out, keepdims])	如果求值的任何元素为True，则返回True。
- ma.count(self[, axis, keepdims])	沿给定轴计算数组的非掩码元素。
- ma.count_masked(arr[, axis])	计算沿给定轴的遮罩元素数。
- ma.getmask(a)	返回掩码数组或nomask的掩码。
- ma.getmaskarray(arr)	返回掩码数组的掩码，或False的完整布尔数组。
- ma.getdata(a[, subok])	将掩码数组的数据作为ndarray返回。
- ma.nonzero(self)	返回非零的未屏蔽元素的索引。
- ma.shape(obj)	返回非零的未屏蔽元素的索引。
- ma.size(obj[, axis])	返回给定轴上的元素数。
- ma.is_masked(x)	确定输入是否具有掩码值。
- ma.is_mask(m)	如果m是有效的标准掩码，则返回True。
- ma.MaskedArray.data	返回当前数据，作为原始基础数据的视图。
- ma.MaskedArray.mask	掩盖
- ma.MaskedArray.recordmask	返回记录的掩码。
- ma.MaskedArray.all([axis, out, keepdims])	如果所有元素都计算为True，则返回True。
- ma.MaskedArray.any([axis, out, keepdims])	如果求值的任何元素为True，则返回True。
- ma.MaskedArray.count([axis, keepdims])	沿给定轴计算数组的非掩码元素。
- ma.MaskedArray.nonzero()	返回非零的未屏蔽元素的索引。
- ma.shape(obj) 返回数组的形状。
- ma.size(obj[, axis])	返回给定轴上的元素数。

## 操纵MaskedArray

### 改变形状

- ma.ravel(self[, order])	作为视图返回self的1维版本。
- ma.reshape(a, new_shape[, order])	返回包含具有新形状的相同数据的数组。
- ma.resize(x, new_shape)	返回具有指定大小和形状的新蒙版数组。
- ma.MaskedArray.flatten([order])	将折叠的数组的副本返回到一个维度。
- ma.MaskedArray.ravel([order])	作为视图返回self的1D版本。
- ma.MaskedArray.reshape(*s, **kwargs)	为数组赋予新形状而不更改其数据。
- ma.MaskedArray.resize(newshape[, refcheck, …])	

### 修改轴

- ma.swapaxes(self, *args, …)	返回数组的视图，其中axis1和axis2互换。
- ma.transpose(a[, axes])	置换数组的维度。
- ma.MaskedArray.swapaxes(axis1, axis2)	返回数组的视图，其中axis1和axis2互换。
- ma.MaskedArray.transpose(*axes)	返回轴转置的数组视图。

### 更改维度数

- ma.atleast_1d(*arys)	将输入转换为至少具有一个维度的数组。
- ma.atleast_2d(*arys)	将输入视为具有至少两个维度的数组。
- ma.atleast_3d(*arys)	将输入视为具有至少三维的数组。
- ma.expand_dims(x, axis)	展开数组的形状。
- ma.squeeze(a[, axis])	从数组的形状中删除一维条目。
- ma.MaskedArray.squeeze([axis])	从a的形状中删除一维条目。
- ma.column_stack(tup)	将1-D阵列作为列堆叠成2-D阵列。
- ma.concatenate(arrays[, axis])	沿给定轴连接一系列数组。
- ma.dstack(tup)	按顺序深度堆叠阵列（沿第三轴）。
- ma.hstack(tup)	按顺序堆叠数组（列式）。
- ma.hsplit(ary, indices_or_sections)	将数组水平拆分为多个子数组（按列）。
- ma.mr_	将切片对象转换为沿第一轴的连接。
- ma.row_stack(tup)	垂直堆叠数组（行方式）。
- ma.vstack(tup)	垂直堆叠数组（行方式）。

### 组合数组

- ma.column_stack(tup)	1-D阵列作为列堆叠成2-D阵列。
- ma.concatenate(arrays[, axis])	沿给定轴连接一系列数组。
- ma.append(a, b[, axis])	将值附加到数组的末尾。
- ma.dstack(tup)	按顺序深度堆叠阵列（沿第三轴）。
- ma.hstack(tup)	按顺序堆叠数组（列式）。
- ma.vstack(tup)	垂直堆叠数组（行方式）。

## 掩码操作

### 创建掩码

- ma.make_mask(m[, copy, shrink, dtype])	从数组创建布尔掩码。
- ma.make_mask_none(newshape[, dtype])	返回给定形状的布尔掩码，填充False。
- ma.mask_or(m1, m2[, copy, shrink])	使用logical_or运算符组合两个掩码。
- ma.make_mask_descr(ndtype)	从给定的dtype构造一个dtype描述列表。

### 访问掩码

- ma.getmask(a)	返回掩码数组或nomask的掩码。
- ma.getmaskarray(arr)	返回掩码数组的掩码，或False的完整布尔数组。
- ma.masked_array.mask Mask

### 查找掩码数据

- ma.flatnotmasked_contiguous(a)	沿给定轴在掩码数组中查找连续的未屏蔽数据。
- ma.flatnotmasked_edges(a)	查找第一个和最后一个未屏蔽值的索引。
- ma.notmasked_contiguous(a[, axis])	沿给定轴在掩码数组中查找连续的未屏蔽数据。
- ma.notmasked_edges(a[, axis])	查找沿轴的第一个和最后一个未屏蔽值的索引。
- ma.clump_masked(a)	返回与1-D数组的掩码块对应的切片列表。
- ma.clump_unmasked(a)	返回与1-D阵列的未掩蔽块相对应的切片列表。

### 修改掩码

- ma.mask_cols(a[, axis])	屏蔽包含掩码值的2D数组的列。
- ma.mask_or(m1, m2[, copy, shrink])	使用logical_or运算符组合两个掩码。
- ma.mask_rowcols(a[, axis])	屏蔽包含掩码值的2D数组的行和/或列。
- ma.mask_row···s(a[, axis])	屏蔽包含掩码值的2D数组的行。
- ma.harden_mask(self) 强制掩码变坚硬。
- ma.soften_mask(self)	强制掩码变柔软。
- ma.MaskedArray.harden_mask()	强制掩码变坚硬。
- ma.MaskedArray.soften_mask()	强制掩码变柔软。
- ma.MaskedArray.shrink_mask() 尽可能将掩码减少到nomask。
- ma.MaskedArray.unshare_mask()	复制掩码并将sharedmask标志设置为False。

## 转换操作

### \> 变成掩码数组

- ma.asarray(a[, dtype, order])	将输入转换为给定数据类型的掩码数组。
- ma.asanyarray(a[, dtype])	将输入转换为掩码数组，保留子类。
- ma.fix_invalid(a[, mask, copy, fill_value])	返回带有无效数据的输入，并用填充值替换。
- ma.masked_equal(x, value[, copy])	屏蔽一个等于给定值的数组。
- ma.masked_greater(x, value[, copy])	屏蔽大于给定值的数组
- ma.masked_greater_equal(x, value[, copy])	屏蔽大于或等于给定值的数组。
- ma.masked_inside(x, v1, v2[, copy])	给定间隔内屏蔽数组。
- ma.masked_invalid(a[, copy])	屏蔽出现无效值的数组（NaN或infs）。
- ma.masked_less(x, value[, copy])	屏蔽小于给定值的数组。
- ma.masked_less_equal(x, value[, copy])	屏蔽小于或等于给定值的数组。
- ma.masked_not_equal(x, value[, copy])	屏蔽不等于给定值的数组。
- ma.masked_object(x, value[, copy, shrink])	屏蔽数组x，其中数据正好等于value。
- ma.masked_outside(x, v1, v2[, copy])	在给定间隔之外屏蔽数组。
- ma.masked_values(x, value[, rtol, atol, …])	掩码使用浮点相等。
    - ma.masked_where(condition, a[, copy])	屏蔽满足条件的数组。

### \> 变成一个numpy的数组

- ma.compress_cols(a)	抑制包含掩码值的二维数组的整列。
- ma.compress_rowcols(x[, axis])	禁止包含掩码值的二维数组的行和/或列。
- ma.compress_rows(a)	抑制包含掩码值的2-D数组的整行。
- ma.compressed(x)	将所有未屏蔽的数据作为1-D数组返回。
- ma.filled(a[, fill_value])	将输入作为数组返回，屏蔽数据替换为填充值。
- ma.MaskedArray.compressed()	将所有未屏蔽的数据作为1-D数组返回。
- ma.MaskedArray.filled([fill_value])	返回self的副本，其中屏蔽值填充给定值。

### \> 变成别的对象

- ma.MaskedArray.tofile(fid[, sep, format])	将掩码数组以二进制格式保存到文件中。
- ma.MaskedArray.tolist([fill_value])	将掩码数组的数据部分作为分层Python列表返回。
- ma.MaskedArray.torecords()	将掩码数组转换为灵活类型数组。
- ma.MaskedArray.tobytes([fill_value, order])	将数组数据作为包含数组中原始字节的字符串返回。

### 颜值和去皮

- ma.dump(a, F)	将掩码数组腌制到文件中
- ma.dumps(a)	返回与掩码数组的腌制相对应的字符串。
- ma.load(F)	cPickle.load周围的包装器，它接受类似文件的对象或文件名。
- ma.loads(strg)	从当前字符串加载腌制数组。

### 填充蒙面数组

- ma.common_fill_value(a, b)	返回两个蒙版数组的公共填充值（如果有）。
- ma.default_fill_value(obj)	返回参数对象的默认填充值。
- ma.maximum_fill_value(obj)	返回可以由对象的dtype表示的最小值。
- ma.set_fill_value(a, fill_value)	如果a是掩码数组，则设置a的填充值。
- ma.MaskedArray.get_fill_value()	返回蒙版掩码的填充值。
- ma.MaskedArray.set_fill_value([value])	设置掩码数组的填充值。
- ma.MaskedArray.fill_value	填充值。

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
Polynomial fit¶
- ma.vander(x[, n])	Generate a Vandermonde matrix.
- ma.polyfit(x, y, deg[, rcond, full, w, cov])	Least squares polynomial fit.

### Clipping and rounding

- ma.around	Round an array to the given number of decimals.
- ma.clip(a, a_min, a_max[, out])	Clip (limit) the values in an array.
- ma.round(a[, decimals, out])	Return a copy of a, rounded to ‘decimals’ places.
[ ma.MaskedArray.clip([min, max, out])	Return an array whose values are limited to [min, max].
- ma.MaskedArray.round([decimals, out])	Return each element rounded to the given number of 
decimals.

### Miscellanea

- ma.allequal(a, b[, fill_value])	Return True if all entries of a and b are equal, using fill_value as a truth value where either or both are masked.
- ma.allclose(a, b[, masked_equal, rtol, atol])	Returns True if two arrays are element-wise equal within a tolerance.
- ma.apply_along_axis(func1d, axis, arr, …)	Apply a function to 1-D slices along the given axis.
- ma.arange([start,] stop[, step,][, dtype])	Return evenly spaced values within a given interval.
- ma.choose(indices, choices[, out, mode])	Use an index array to construct a new array from a set of choices.
- ma.ediff1d(arr[, to_end, to_begin])	Compute the differences between consecutive elements of an array.
- ma.indices(dimensions[, dtype])	Return an array representing the indices of a grid.
- ma.where(condition[, x, y])	Return a masked array with elements from x or y, depending on condition.