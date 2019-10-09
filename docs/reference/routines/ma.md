# 操作掩码数组

## 常量

方法 | 描述
---|---
[ma.MaskType](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskType.html#numpy.ma.MaskType) | numpy.bool_的别名

## 创建

### 从现有数据

方法 | 描述
---|---
[ma.masked_array](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_array.html#numpy.ma.masked_array) | numpy.ma.core.MaskedArray 的别名
[ma.array](https://numpy.org/devdocs/reference/generated/numpy.ma.array.html#numpy.ma.array)(data[, dtype, copy, order, mask, …]) | 具有可能被屏蔽的值的数组类。
[ma.copy](https://numpy.org/devdocs/reference/generated/numpy.ma.copy.html#numpy.ma.copy)(self, *args, **params) a.copy(order=) | 返回数组的副本。
[ma.frombuffer](https://numpy.org/devdocs/reference/generated/numpy.ma.frombuffer.html#numpy.ma.frombuffer)(buffer[, dtype, count, offset]) | 将缓冲区解释为一维数组。
[ma.fromfunction](https://numpy.org/devdocs/reference/generated/numpy.ma.fromfunction.html#numpy.ma.fromfunction)(function, shape, **kwargs) | 通过在每个坐标上执行一个函数来构造一个数组。
[ma.MaskedArray.copy](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.copy.html#numpy.ma.MaskedArray.copy)([order]) | 返回数组的副本。

### Ones 和 zeros

方法 | 描述
---|---
[ma.empty](https://numpy.org/devdocs/reference/generated/numpy.ma.empty.html#numpy.ma.empty)(shape[, dtype, order]) | 返回给定形状和类型的新数组，而无需初始化条目。
[ma.empty_like](https://numpy.org/devdocs/reference/generated/numpy.ma.empty_like.html#numpy.ma.empty_like)(prototype[, dtype, order, …]) | 返回形状和类型与给定数组相同的新数组。
[ma.masked_all](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_all.html#numpy.ma.masked_all)(shape[, dtype]) | 清空所有元素都被屏蔽的掩码数组。
[ma.masked_all_like](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_all_like.html#numpy.ma.masked_all_like)(arr) | 清空具有现有数组属性的掩码数组。
[ma.ones](https://numpy.org/devdocs/reference/generated/numpy.ma.ones.html#numpy.ma.ones)(shape[, dtype, order]) | 返回给定形状和类型的新数组，并填充为1。
[ma.zeros](https://numpy.org/devdocs/reference/generated/numpy.ma.zeros.html#numpy.ma.zeros)(shape[, dtype, order]) | 返回给定形状和类型的新数组，并用0填充。

## 检查数组

方法 | 描述
---|---
[ma.all](https://numpy.org/devdocs/reference/generated/numpy.ma.all.html#numpy.ma.all)(self[, axis, out, keepdims]) | 如果所有元素的评估结果为True，则返回True。
[ma.any](https://numpy.org/devdocs/reference/generated/numpy.ma.any.html#numpy.ma.any)(self[, axis, out, keepdims]) | 如果评估的任何元素为True，则返回True。
[ma.count](https://numpy.org/devdocs/reference/generated/numpy.ma.count.html#numpy.ma.count)(self[, axis, keepdims]) | 沿给定轴计算数组的未遮罩元素。
[ma.count_masked](https://numpy.org/devdocs/reference/generated/numpy.ma.count_masked.html#numpy.ma.count_masked)(arr[, axis]) | 计算沿给定轴的掩码元素的数量。
[ma.getmask](https://numpy.org/devdocs/reference/generated/numpy.ma.getmask.html#numpy.ma.getmask)(a) | 返回掩码数组的掩码或nomask。
[ma.getmaskarray](https://numpy.org/devdocs/reference/generated/numpy.ma.getmaskarray.html#numpy.ma.getmaskarray)(arr) | 返回掩码数组的掩码或False的完整布尔数组。
[ma.getdata](https://numpy.org/devdocs/reference/generated/numpy.ma.getdata.html#numpy.ma.getdata)(a[, subok]) | 将被掩码数组的数据作为ndarray返回。
[ma.nonzero](https://numpy.org/devdocs/reference/generated/numpy.ma.nonzero.html#numpy.ma.nonzero)(self) | 返回不为零的未屏蔽元素的索引。
[ma.shape](https://numpy.org/devdocs/reference/generated/numpy.ma.shape.html#numpy.ma.shape)(obj) | 返回数组的形状。
[ma.size](https://numpy.org/devdocs/reference/generated/numpy.ma.size.html#numpy.ma.size)(obj[, axis]) | 返回沿给定轴的元素数。
[ma.is_masked](https://numpy.org/devdocs/reference/generated/numpy.ma.is_masked.html#numpy.ma.is_masked)(x) | 确定输入是否具有掩码值。
[ma.is_mask](https://numpy.org/devdocs/reference/generated/numpy.ma.is_mask.html#numpy.ma.is_mask)(m) | 如果m是有效的标准掩码，则返回True。
[ma.MaskedArray.all](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.all.html#numpy.ma.MaskedArray.all)(self[, axis, out, keepdims]) | 如果所有元素的评估结果为True，则返回True。
[ma.MaskedArray.any](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.any.html#numpy.ma.MaskedArray.any)(self[, axis, out, keepdims]) | 如果评估的任何元素为True，则返回True。
[ma.MaskedArray.count](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.count.html#numpy.ma.MaskedArray.count)(self[, axis, keepdims]) | 沿给定轴计算数组的非掩码元素。
[ma.MaskedArray.nonzero](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.nonzero.html#numpy.ma.MaskedArray.nonzero)(self) | 返回不为零的未屏蔽元素的索引。
[ma.shape](https://numpy.org/devdocs/reference/generated/numpy.ma.shape.html#numpy.ma.shape)(obj) | 返回数组的形状。
[ma.size](https://numpy.org/devdocs/reference/generated/numpy.ma.size.html#numpy.ma.size)(obj[, axis]) | 返回沿给定轴的元素数。

方法 | 描述
---|---
[ma.MaskedArray.data](maskedarray.baseclass.html#numpy.ma.MaskedArray.data) | 返回基础数据，作为掩码数组的视图。
[ma.MaskedArray.mask](maskedarray.baseclass.html#numpy.ma.MaskedArray.mask) | 当前的掩码。
[ma.MaskedArray.recordmask](maskedarray.baseclass.html#numpy.ma.MaskedArray.recordmask) | 如果没有命名字段，则获取或设置数组的掩码。

## 操作 MaskedArray

### 改变形状

方法 | 描述
---|---
[ma.ravel](https://numpy.org/devdocs/reference/generated/numpy.ma.ravel.html#numpy.ma.ravel)(self[, order]) | 返回self的一维版本，作为视图。
[ma.reshape](https://numpy.org/devdocs/reference/generated/numpy.ma.reshape.html#numpy.ma.reshape)(a, new_shape[, order]) | 返回一个数组，其中包含具有相同形状的相同数据。
[ma.resize](https://numpy.org/devdocs/reference/generated/numpy.ma.resize.html#numpy.ma.resize)(x, new_shape) | 返回具有指定大小和形状的新的掩码数组。
[ma.MaskedArray.flatten](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.flatten.html#numpy.ma.MaskedArray.flatten)([order]) | 返回折叠成一维的数组副本。
[ma.MaskedArray.ravel](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.ravel.html#numpy.ma.MaskedArray.ravel)(self[, order]) | 返回self的一维版本，作为视图。
[ma.MaskedArray.reshape](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.reshape.html#numpy.ma.MaskedArray.reshape)(self, \*s, \*\*kwargs) | 在不更改数据的情况下为数组赋予新的形状。
[ma.MaskedArray.resize](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.resize.html#numpy.ma.MaskedArray.resize)(self, newshape[, …]) | 

### 修改轴

方法 | 描述
---|---
[ma.swapaxes](https://numpy.org/devdocs/reference/generated/numpy.ma.swapaxes.html#numpy.ma.swapaxes)(self, *args, …) | 返回轴1和轴2互换的数组视图。
[ma.transpose](https://numpy.org/devdocs/reference/generated/numpy.ma.transpose.html#numpy.ma.transpose)(a[, axes]) | 排列数组的尺寸。
[ma.MaskedArray.swapaxes](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.swapaxes.html#numpy.ma.MaskedArray.swapaxes)(axis1, axis2) | 返回轴1和轴2互换的数组视图。
[ma.MaskedArray.transpose](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.transpose.html#numpy.ma.MaskedArray.transpose)(*axes) | 返回轴已转置的数组视图。

### 更改尺寸数量

方法 | 描述
---|---
[ma.atleast_1d](https://numpy.org/devdocs/reference/generated/numpy.ma.atleast_1d.html#numpy.ma.atleast_1d)(*args, **kwargs) | 将输入转换为至少一维的数组。
[ma.atleast_2d](https://numpy.org/devdocs/reference/generated/numpy.ma.atleast_2d.html#numpy.ma.atleast_2d)(*args, **kwargs) | 将输入视为至少具有二维的数组。
[ma.atleast_3d](https://numpy.org/devdocs/reference/generated/numpy.ma.atleast_3d.html#numpy.ma.atleast_3d)(*args, **kwargs) | 将输入查看为至少具有三个维度的数组。
[ma.expand_dims](https://numpy.org/devdocs/reference/generated/numpy.ma.expand_dims.html#numpy.ma.expand_dims)(a, axis) | 扩展数组的形状。
[ma.squeeze](https://numpy.org/devdocs/reference/generated/numpy.ma.squeeze.html#numpy.ma.squeeze)(a[, axis]) | 从数组形状中删除一维条目。
[ma.MaskedArray.squeeze](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.squeeze.html#numpy.ma.MaskedArray.squeeze)([axis]) | 从a的形状中删除一维条目。
[ma.stack](https://numpy.org/devdocs/reference/generated/numpy.ma.stack.html#numpy.ma.stack)(*args, **kwargs) | 沿新轴连接一系列数组。
[ma.column_stack](https://numpy.org/devdocs/reference/generated/numpy.ma.column_stack.html#numpy.ma.column_stack)(*args, **kwargs) | 将一维数组作为列堆叠到二维数组中。
[ma.concatenate](https://numpy.org/devdocs/reference/generated/numpy.ma.concatenate.html#numpy.ma.concatenate)(arrays[, axis]) | 沿给定轴连接一系列数组。
[ma.dstack](https://numpy.org/devdocs/reference/generated/numpy.ma.dstack.html#numpy.ma.dstack)(*args, **kwargs) | 沿深度方向（沿第三轴）按顺序堆叠数组。
[ma.hstack](https://numpy.org/devdocs/reference/generated/numpy.ma.hstack.html#numpy.ma.hstack)(*args, **kwargs) | 水平（按列）顺序堆叠数组。
[ma.hsplit](https://numpy.org/devdocs/reference/generated/numpy.ma.hsplit.html#numpy.ma.hsplit)(*args, **kwargs) | 水平（按列）将一个数组拆分为多个子数组。
[ma.mr_](https://numpy.org/devdocs/reference/generated/numpy.ma.mr_.html#numpy.ma.mr_) | 沿第一个轴将切片对象平移为串联。
[ma.row_stack](https://numpy.org/devdocs/reference/generated/numpy.ma.row_stack.html#numpy.ma.row_stack)(*args, **kwargs) | 垂直（行）按顺序堆叠数组。
[ma.vstack](https://numpy.org/devdocs/reference/generated/numpy.ma.vstack.html#numpy.ma.vstack)(*args, **kwargs) | 垂直（行）按顺序堆叠数组。

### 连接数组

方法 | 描述
---|---
[ma.stack](https://numpy.org/devdocs/reference/generated/numpy.ma.stack.html#numpy.ma.stack)(*args, **kwargs) | 沿新轴连接一系列数组。
[ma.column_stack](https://numpy.org/devdocs/reference/generated/numpy.ma.column_stack.html#numpy.ma.column_stack)(*args, **kwargs) | 将一维数组作为列堆叠到二维数组中。
[ma.concatenate](https://numpy.org/devdocs/reference/generated/numpy.ma.concatenate.html#numpy.ma.concatenate)(arrays[, axis]) | 沿给定轴连接一系列数组。
[ma.append](https://numpy.org/devdocs/reference/generated/numpy.ma.append.html#numpy.ma.append)(a, b[, axis]) | 将值附加到数组的末尾。
[ma.dstack](https://numpy.org/devdocs/reference/generated/numpy.ma.dstack.html#numpy.ma.dstack)(*args, **kwargs) | 沿深度方向（沿第三轴）按顺序堆叠数组。
[ma.hstack](https://numpy.org/devdocs/reference/generated/numpy.ma.hstack.html#numpy.ma.hstack)(*args, **kwargs) | 水平（按列）顺序堆叠数组。
[ma.vstack](https://numpy.org/devdocs/reference/generated/numpy.ma.vstack.html#numpy.ma.vstack)(*args, **kwargs) | 垂直（行）按顺序堆叠数组。

## 在掩码数组上操作

### 创建一个“掩面”

方法 | 描述
---|---
[ma.make_mask](https://numpy.org/devdocs/reference/generated/numpy.ma.make_mask.html#numpy.ma.make_mask)(m[, copy, shrink, dtype]) | 从数组创建布尔掩码。
[ma.make_mask_none](https://numpy.org/devdocs/reference/generated/numpy.ma.make_mask_none.html#numpy.ma.make_mask_none)(newshape[, dtype]) | 返回给定形状的布尔蒙版，填充为False。
[ma.mask_or](https://numpy.org/devdocs/reference/generated/numpy.ma.mask_or.html#numpy.ma.mask_or)(m1, m2[, copy, shrink]) | 将两个掩码与logical_or运算符结合使用。
[ma.make_mask_descr](https://numpy.org/devdocs/reference/generated/numpy.ma.make_mask_descr.html#numpy.ma.make_mask_descr)(ndtype) | 从给定的dtype构造一个dtype描述列表。

### 访问掩码

方法 | 描述
---|---
[ma.getmask](https://numpy.org/devdocs/reference/generated/numpy.ma.getmask.html#numpy.ma.getmask)(a) | 返回掩码数组的掩码或nomask。
[ma.getmaskarray](https://numpy.org/devdocs/reference/generated/numpy.ma.getmaskarray.html#numpy.ma.getmaskarray)(arr) | 返回掩码数组的掩码或False的完整布尔数组。
[ma.masked_array.mask](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_array.mask.html#numpy.ma.masked_array.mask) | 当前掩码。

### 查找被遮掩的数据

方法 | 描述
---|---
[ma.flatnotmasked_contiguous](https://numpy.org/devdocs/reference/generated/numpy.ma.flatnotmasked_contiguous.html#numpy.ma.flatnotmasked_contiguous)(a) | 沿给定轴在掩码数组中查找连续的未掩码数据。
[ma.flatnotmasked_edges](https://numpy.org/devdocs/reference/generated/numpy.ma.flatnotmasked_edges.html#numpy.ma.flatnotmasked_edges)(a) | 查找第一个和最后一个未屏蔽值的索引。
[ma.notmasked_contiguous](https://numpy.org/devdocs/reference/generated/numpy.ma.notmasked_contiguous.html#numpy.ma.notmasked_contiguous)(a[, axis]) | 沿给定轴在掩码数组中查找连续的未掩码数据。
[ma.notmasked_edges](https://numpy.org/devdocs/reference/generated/numpy.ma.notmasked_edges.html#numpy.ma.notmasked_edges)(a[, axis]) | 查找沿轴的第一个和最后一个未被遮掩的值索引。
[ma.clump_masked](https://numpy.org/devdocs/reference/generated/numpy.ma.clump_masked.html#numpy.ma.clump_masked)(a) | 返回与一维数组的掩码块相对应的切片列表。
[ma.clump_unmasked](https://numpy.org/devdocs/reference/generated/numpy.ma.clump_unmasked.html#numpy.ma.clump_unmasked)(a) | 返回与一维数组的未掩码块对应的切片列表。

### 修改掩码

方法 | 描述
---|---
[ma.mask_cols](https://numpy.org/devdocs/reference/generated/numpy.ma.mask_cols.html#numpy.ma.mask_cols)(a[, axis]) | 对包含掩码值的2D数组的掩码列进行掩码。
[ma.mask_or](https://numpy.org/devdocs/reference/generated/numpy.ma.mask_or.html#numpy.ma.mask_or)(m1, m2[, copy, shrink]) | 将两个掩码与logical_or运算符结合使用。
[ma.mask_rowcols](https://numpy.org/devdocs/reference/generated/numpy.ma.mask_rowcols.html#numpy.ma.mask_rowcols)(a[, axis]) | 对包含掩码值的2D数组的行和/或列进行掩码。
[ma.mask_rows](https://numpy.org/devdocs/reference/generated/numpy.ma.mask_rows.html#numpy.ma.mask_rows)(a[, axis]) | 对包含掩码值的2D数组的行进行掩码。
[ma.harden_mask](https://numpy.org/devdocs/reference/generated/numpy.ma.harden_mask.html#numpy.ma.harden_mask)(self) | 强制让掩码变硬。
[ma.soften_mask](https://numpy.org/devdocs/reference/generated/numpy.ma.soften_mask.html#numpy.ma.soften_mask)(self) | 强制让掩码变软。
[ma.MaskedArray.harden_mask](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.harden_mask.html#numpy.ma.MaskedArray.harden_mask)(self) | 强制让掩码变硬。
[ma.MaskedArray.soften_mask](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.soften_mask.html#numpy.ma.MaskedArray.soften_mask)(self) | 强制让掩码变软。
[ma.MaskedArray.shrink_mask](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.shrink_mask.html#numpy.ma.MaskedArray.shrink_mask)(self) | 尽可能将掩码减小为无掩码。
[ma.MaskedArray.unshare_mask](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.unshare_mask.html#numpy.ma.MaskedArray.unshare_mask)(self) | 复制掩码，并将sharedmask标志设置为False。

## 转换操作

### 转换为一个掩码数组

方法 | 描述
---|---
[ma.asarray](https://numpy.org/devdocs/reference/generated/numpy.ma.asarray.html#numpy.ma.asarray)(a[, dtype, order]) | 将输入转换为给定数据类型的掩码数组。
[ma.asanyarray](https://numpy.org/devdocs/reference/generated/numpy.ma.asanyarray.html#numpy.ma.asanyarray)(a[, dtype]) | 将输入转换为掩码数组，从而保留子类。
[ma.fix_invalid](https://numpy.org/devdocs/reference/generated/numpy.ma.fix_invalid.html#numpy.ma.fix_invalid)(a[, mask, copy, fill_value]) | 返回带有无效数据的输入，该数据被掩盖并替换为填充值。
[ma.masked_equal](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_equal.html#numpy.ma.masked_equal)(x, value[, copy]) | 掩码数组等于给定值。
[ma.masked_greater](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_greater.html#numpy.ma.masked_greater)(x, value[, copy]) | Mask an array where greater than a given value.
[ma.masked_greater_equal](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_greater_equal.html#numpy.ma.masked_greater_equal)(x, value[, copy]) | Mask an array where greater than or equal to a given value.
[ma.masked_inside](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_inside.html#numpy.ma.masked_inside)(x, v1, v2[, copy]) | Mask an array inside a given interval.
[ma.masked_invalid](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_invalid.html#numpy.ma.masked_invalid)(a[, copy]) | Mask an array where invalid values occur (NaNs or infs).
[ma.masked_less](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_less.html#numpy.ma.masked_less)(x, value[, copy]) | Mask an array where less than a given value.
[ma.masked_less_equal](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_less_equal.html#numpy.ma.masked_less_equal)(x, value[, copy]) | Mask an array where less than or equal to a given value.
[ma.masked_not_equal](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_not_equal.html#numpy.ma.masked_not_equal)(x, value[, copy]) | Mask an array where not equal to a given value.
[ma.masked_object](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_object.html#numpy.ma.masked_object)(x, value[, copy, shrink]) | Mask the array x where the data are exactly equal to value.
[ma.masked_outside](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_outside.html#numpy.ma.masked_outside)(x, v1, v2[, copy]) | Mask an array outside a given interval.
[ma.masked_values](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_values.html#numpy.ma.masked_values)(x, value[, rtol, atol, …]) | Mask using floating point equality.
[ma.masked_where](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_where.html#numpy.ma.masked_where)(condition, a[, copy]) | Mask an array where a condition is met.

### 输出为一个ndarray

方法 | 描述
---|---
[ma.compress_cols](https://numpy.org/devdocs/reference/generated/numpy.ma.compress_cols.html#numpy.ma.compress_cols)(a) | Suppress whole columns of a 2-D array that contain masked values.
[ma.compress_rowcols](https://numpy.org/devdocs/reference/generated/numpy.ma.compress_rowcols.html#numpy.ma.compress_rowcols)(x[, axis]) | Suppress the rows and/or columns of a 2-D array that contain masked values.
[ma.compress_rows](https://numpy.org/devdocs/reference/generated/numpy.ma.compress_rows.html#numpy.ma.compress_rows)(a) | Suppress whole rows of a 2-D array that contain masked values.
[ma.compressed](https://numpy.org/devdocs/reference/generated/numpy.ma.compressed.html#numpy.ma.compressed)(x) | Return all the non-masked data as a 1-D array.
[ma.filled](https://numpy.org/devdocs/reference/generated/numpy.ma.filled.html#numpy.ma.filled)(a[, fill_value]) | Return input as an array with masked data replaced by a fill value.
[ma.MaskedArray.compressed](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.compressed.html#numpy.ma.MaskedArray.compressed)(self) | Return all the non-masked data as a 1-D array.
[ma.MaskedArray.filled](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.filled.html#numpy.ma.MaskedArray.filled)(self[, fill_value]) | Return a copy of self, with masked values filled with a given value.

### 输出到其他对象

方法 | 描述
---|---
[ma.MaskedArray.tofile](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.tofile.html#numpy.ma.MaskedArray.tofile)(self, fid[, sep, format]) | Save a masked array to a file in binary format.
[ma.MaskedArray.tolist](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.tolist.html#numpy.ma.MaskedArray.tolist)(self[, fill_value]) | Return the data portion of the masked array as a hierarchical Python list.
[ma.MaskedArray.torecords](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.torecords.html#numpy.ma.MaskedArray.torecords)(self) | Transforms a masked array into a flexible-type array.
[ma.MaskedArray.tobytes](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.tobytes.html#numpy.ma.MaskedArray.tobytes)(self[, fill_value, order]) | Return the array data as a string containing the raw bytes in the array.

### 酸洗和解酸

方法 | 描述
---|---
[ma.dump](https://numpy.org/devdocs/reference/generated/numpy.ma.dump.html#numpy.ma.dump)(a, F) | Pickle a masked array to a file.
[ma.dumps](https://numpy.org/devdocs/reference/generated/numpy.ma.dumps.html#numpy.ma.dumps)(a) | Return a string corresponding to the pickling of a masked array.
[ma.load](https://numpy.org/devdocs/reference/generated/numpy.ma.load.html#numpy.ma.load)(F) | Wrapper around cPickle.load which accepts either a file-like object or a filename.
[ma.loads](https://numpy.org/devdocs/reference/generated/numpy.ma.loads.html#numpy.ma.loads)(strg) | Load a pickle from the current string.

### 填充掩码数组

方法 | 描述
---|---
[ma.common_fill_value](https://numpy.org/devdocs/reference/generated/numpy.ma.common_fill_value.html#numpy.ma.common_fill_value)(a, b) | Return the common filling value of two masked arrays, if any.
[ma.default_fill_value](https://numpy.org/devdocs/reference/generated/numpy.ma.default_fill_value.html#numpy.ma.default_fill_value)(obj) | Return the default fill value for the argument object.
[ma.maximum_fill_value](https://numpy.org/devdocs/reference/generated/numpy.ma.maximum_fill_value.html#numpy.ma.maximum_fill_value)(obj) | Return the minimum value that can be represented by the dtype of an object.
[ma.maximum_fill_value](https://numpy.org/devdocs/reference/generated/numpy.ma.maximum_fill_value.html#numpy.ma.maximum_fill_value)(obj) | Return the minimum value that can be represented by the dtype of an object.
[ma.set_fill_value](https://numpy.org/devdocs/reference/generated/numpy.ma.set_fill_value.html#numpy.ma.set_fill_value)(a, fill_value) | Set the filling value of a, if a is a masked array.
[ma.MaskedArray.get_fill_value](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.get_fill_value.html#numpy.ma.MaskedArray.get_fill_value)(self) | The filling value of the masked array is a scalar.
[ma.MaskedArray.set_fill_value](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.set_fill_value.html#numpy.ma.MaskedArray.set_fill_value)(self[, value]) | 

方法 | 描述
---|---
[ma.MaskedArray.fill_value](maskedarray.baseclass.html#numpy.ma.MaskedArray.fill_value) | The filling value of the masked array is a scalar.

## 掩码数组算法

### 算术运算

方法 | 描述
---|---
[ma.anom](https://numpy.org/devdocs/reference/generated/numpy.ma.anom.html#numpy.ma.anom)(self[, axis, dtype]) | 沿给定轴计算异常（与算术平均值的偏差）。
[ma.anomalies](https://numpy.org/devdocs/reference/generated/numpy.ma.anomalies.html#numpy.ma.anomalies)(self[, axis, dtype]) | 沿给定轴计算异常（与算术平均值的偏差）。
[ma.average](https://numpy.org/devdocs/reference/generated/numpy.ma.average.html#numpy.ma.average)(a[, axis, weights, returned]) | 返回给定轴上数组的加权平均值。
[ma.conjugate](https://numpy.org/devdocs/reference/generated/numpy.ma.conjugate.html#numpy.ma.conjugate)(x, /[, out, where, casting, …]) | 逐元素返回复共轭。
[ma.corrcoef](https://numpy.org/devdocs/reference/generated/numpy.ma.corrcoef.html#numpy.ma.corrcoef)(x[, y, rowvar, bias, …]) | 返回皮尔逊积矩相关系数。
[ma.cov](https://numpy.org/devdocs/reference/generated/numpy.ma.cov.html#numpy.ma.cov)(x[, y, rowvar, bias, allow_masked, ddof]) | 估计协方差矩阵。
[ma.cumsum](https://numpy.org/devdocs/reference/generated/numpy.ma.cumsum.html#numpy.ma.cumsum)(self[, axis, dtype, out]) | 返回给定轴上数组元素的累积和。
[ma.cumprod](https://numpy.org/devdocs/reference/generated/numpy.ma.cumprod.html#numpy.ma.cumprod)(self[, axis, dtype, out]) | 返回给定轴上数组元素的累积乘积。
[ma.mean](https://numpy.org/devdocs/reference/generated/numpy.ma.mean.html#numpy.ma.mean)(self[, axis, dtype, out, keepdims]) | 返回沿给定轴的数组元素的平均值。
[ma.median](https://numpy.org/devdocs/reference/generated/numpy.ma.median.html#numpy.ma.median)(a[, axis, out, overwrite_input, …]) | 计算沿指定轴的中位数。
[ma.power](https://numpy.org/devdocs/reference/generated/numpy.ma.power.html#numpy.ma.power)(a, b[, third]) | 返回从第二个数组提升为幂的逐元素基本数组。
[ma.prod](https://numpy.org/devdocs/reference/generated/numpy.ma.prod.html#numpy.ma.prod)(self[, axis, dtype, out, keepdims]) | Return the product of the array elements over the given axis.
[ma.std](https://numpy.org/devdocs/reference/generated/numpy.ma.std.html#numpy.ma.std)(self[, axis, dtype, out, ddof, keepdims]) | Returns the standard deviation of the array elements along given axis.
[ma.sum](https://numpy.org/devdocs/reference/generated/numpy.ma.sum.html#numpy.ma.sum)(self[, axis, dtype, out, keepdims]) | Return the sum of the array elements over the given axis.
[ma.var](https://numpy.org/devdocs/reference/generated/numpy.ma.var.html#numpy.ma.var)(self[, axis, dtype, out, ddof, keepdims]) | Compute the variance along the specified axis.
[ma.MaskedArray.anom](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.anom.html#numpy.ma.MaskedArray.anom)(self[, axis, dtype]) | Compute the anomalies (deviations from the arithmetic mean) along the given axis.
[ma.MaskedArray.cumprod](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.cumprod.html#numpy.ma.MaskedArray.cumprod)(self[, axis, dtype, out]) | Return the cumulative product of the array elements over the given axis.
[ma.MaskedArray.cumsum](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.cumsum.html#numpy.ma.MaskedArray.cumsum)(self[, axis, dtype, out]) | Return the cumulative sum of the array elements over the given axis.
[ma.MaskedArray.mean](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.mean.html#numpy.ma.MaskedArray.mean)(self[, axis, dtype, …]) | Returns the average of the array elements along given axis.
[ma.MaskedArray.prod](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.prod.html#numpy.ma.MaskedArray.prod)(self[, axis, dtype, …]) | Return the product of the array elements over the given axis.
[ma.MaskedArray.std](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.std.html#numpy.ma.MaskedArray.std)(self[, axis, dtype, out, …]) | Returns the standard deviation of the array elements along given axis.
[ma.MaskedArray.sum](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.sum.html#numpy.ma.MaskedArray.sum)(self[, axis, dtype, out, …]) | Return the sum of the array elements over the given axis.
[ma.MaskedArray.var](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.var.html#numpy.ma.MaskedArray.var)(self[, axis, dtype, out, …]) | Compute the variance along the specified axis.

### 最小/最大

方法 | 描述
---|---
[ma.argmax](https://numpy.org/devdocs/reference/generated/numpy.ma.argmax.html#numpy.ma.argmax)(self[, axis, fill_value, out]) | Returns array of indices of the maximum values along the given axis.
[ma.argmin](https://numpy.org/devdocs/reference/generated/numpy.ma.argmin.html#numpy.ma.argmin)(self[, axis, fill_value, out]) | Return array of indices to the minimum values along the given axis.
[ma.max](https://numpy.org/devdocs/reference/generated/numpy.ma.max.html#numpy.ma.max)(obj[, axis, out, fill_value, keepdims]) | Return the maximum along a given axis.
[ma.min](https://numpy.org/devdocs/reference/generated/numpy.ma.min.html#numpy.ma.min)(obj[, axis, out, fill_value, keepdims]) | Return the minimum along a given axis.
[ma.ptp](https://numpy.org/devdocs/reference/generated/numpy.ma.ptp.html#numpy.ma.ptp)(obj[, axis, out, fill_value, keepdims]) | Return (maximum - minimum) along the given dimension (i.e.
[ma.MaskedArray.argmax](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.argmax.html#numpy.ma.MaskedArray.argmax)(self[, axis, …]) | Returns array of indices of the maximum values along the given axis.
[ma.MaskedArray.argmin](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.argmin.html#numpy.ma.MaskedArray.argmin)(self[, axis, …]) | Return array of indices to the minimum values along the given axis.
[ma.MaskedArray.max](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.max.html#numpy.ma.MaskedArray.max)(self[, axis, out, …]) | Return the maximum along a given axis.
[ma.MaskedArray.min](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.min.html#numpy.ma.MaskedArray.min)(self[, axis, out, …]) | Return the minimum along a given axis.
[ma.MaskedArray.ptp](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.ptp.html#numpy.ma.MaskedArray.ptp)(self[, axis, out, …]) | Return (maximum - minimum) along the given dimension (i.e.

### 排序

方法 | 描述
---|---
[ma.argsort](https://numpy.org/devdocs/reference/generated/numpy.ma.argsort.html#numpy.ma.argsort)(a[, axis, kind, order, endwith, …]) | Return an ndarray of indices that sort the array along the specified axis.
[ma.sort](https://numpy.org/devdocs/reference/generated/numpy.ma.sort.html#numpy.ma.sort)(a[, axis, kind, order, endwith, …]) | Sort the array, in-place
[ma.MaskedArray.argsort](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.argsort.html#numpy.ma.MaskedArray.argsort)(self[, axis, kind, …]) | Return an ndarray of indices that sort the array along the specified axis.
[ma.MaskedArray.sort](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.sort.html#numpy.ma.MaskedArray.sort)(self[, axis, kind, …]) | Sort the array, in-place

### 代数

方法 | 描述
---|---
[ma.diag](https://numpy.org/devdocs/reference/generated/numpy.ma.diag.html#numpy.ma.diag)(v[, k]) | Extract a diagonal or construct a diagonal array.
[ma.dot](https://numpy.org/devdocs/reference/generated/numpy.ma.dot.html#numpy.ma.dot)(a, b[, strict, out]) | Return the dot product of two arrays.
[ma.identity](https://numpy.org/devdocs/reference/generated/numpy.ma.identity.html#numpy.ma.identity)(n[, dtype]) | Return the identity array.
[ma.inner](https://numpy.org/devdocs/reference/generated/numpy.ma.inner.html#numpy.ma.inner)(a, b) | Inner product of two arrays.
[ma.innerproduct](https://numpy.org/devdocs/reference/generated/numpy.ma.innerproduct.html#numpy.ma.innerproduct)(a, b) | Inner product of two arrays.
[ma.outer](https://numpy.org/devdocs/reference/generated/numpy.ma.outer.html#numpy.ma.outer)(a, b) | Compute the outer product of two vectors.
[ma.outerproduct](https://numpy.org/devdocs/reference/generated/numpy.ma.outerproduct.html#numpy.ma.outerproduct)(a, b) | Compute the outer product of two vectors.
[ma.trace](https://numpy.org/devdocs/reference/generated/numpy.ma.trace.html#numpy.ma.trace)(self[, offset, axis1, axis2, …]) | Return the sum along diagonals of the array.
[ma.transpose](https://numpy.org/devdocs/reference/generated/numpy.ma.transpose.html#numpy.ma.transpose)(a[, axes]) | Permute the dimensions of an array.
[ma.MaskedArray.trace](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.trace.html#numpy.ma.MaskedArray.trace)([offset, axis1, axis2, …]) | Return the sum along diagonals of the array.
[ma.MaskedArray.transpose](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.transpose.html#numpy.ma.MaskedArray.transpose)(*axes) | Returns a view of the array with axes transposed.

### 多项式拟合

方法 | 描述
---|---
[ma.vander](https://numpy.org/devdocs/reference/generated/numpy.ma.vander.html#numpy.ma.vander)(x[, n]) | Generate a Vandermonde matrix.
[ma.polyfit](https://numpy.org/devdocs/reference/generated/numpy.ma.polyfit.html#numpy.ma.polyfit)(x, y, deg[, rcond, full, w, cov]) | Least squares polynomial fit.

### 修剪和舍入

方法 | 描述
---|---
[ma.around](https://numpy.org/devdocs/reference/generated/numpy.ma.around.html#numpy.ma.around)(a, \*args, \*\*kwargs) | Round an array to the given number of decimals.
[ma.clip](https://numpy.org/devdocs/reference/generated/numpy.ma.clip.html#numpy.ma.clip)(a, a_min, a_max[, out]) | Clip (limit) the values in an array.
[ma.round](https://numpy.org/devdocs/reference/generated/numpy.ma.round.html#numpy.ma.round)(a[, decimals, out]) | Return a copy of a, rounded to ‘decimals’ places.
[ma.MaskedArray.clip](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.clip.html#numpy.ma.MaskedArray.clip)([min, max, out]) | Return an array whose values are limited to [min, max].
[ma.MaskedArray.round](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.round.html#numpy.ma.MaskedArray.round)(self[, decimals, out]) | Return each element rounded to the given number of decimals.

### 杂项

方法 | 描述
---|---
[ma.allequal](https://numpy.org/devdocs/reference/generated/numpy.ma.allequal.html#numpy.ma.allequal)(a, b[, fill_value]) | Return True if all entries of a and b are equal, using fill_value as a truth value where either or both are masked.
[ma.allclose](https://numpy.org/devdocs/reference/generated/numpy.ma.allclose.html#numpy.ma.allclose)(a, b[, masked_equal, rtol, atol]) | Returns True if two arrays are element-wise equal within a tolerance.
[ma.apply_along_axis](https://numpy.org/devdocs/reference/generated/numpy.ma.apply_along_axis.html#numpy.ma.apply_along_axis)(func1d, axis, arr, …) | Apply a function to 1-D slices along the given axis.
[ma.arange](https://numpy.org/devdocs/reference/generated/numpy.ma.arange.html#numpy.ma.arange)([start,] stop[, step,][, dtype]) | Return evenly spaced values within a given interval.
[ma.choose](https://numpy.org/devdocs/reference/generated/numpy.ma.choose.html#numpy.ma.choose)(indices, choices[, out, mode]) | Use an index array to construct a new array from a set of choices.
[ma.ediff1d](https://numpy.org/devdocs/reference/generated/numpy.ma.ediff1d.html#numpy.ma.ediff1d)(arr[, to_end, to_begin]) | Compute the differences between consecutive elements of an array.
[ma.indices](https://numpy.org/devdocs/reference/generated/numpy.ma.indices.html#numpy.ma.indices)(dimensions[, dtype, sparse]) | Return an array representing the indices of a grid.
[ma.where](https://numpy.org/devdocs/reference/generated/numpy.ma.where.html#numpy.ma.where)(condition[, x, y]) | Return a masked array with elements from x or y, depending on condition.
