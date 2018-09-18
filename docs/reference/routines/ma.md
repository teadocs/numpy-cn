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

### 填充掩码数组

- ma.common_fill_value(a, b)	返回两个掩码数组的公共填充值(如果有的话)。
- ma.default_fill_value(obj)	返回参数对象的默认填充值。
- ma.maximum_fill_value(obj)	返回可以由对象的dtype表示的最小值。
- ma.maximum_fill_value(obj)	返回可以由对象的dtype表示的最小值。
- ma.set_fill_value(a, fill_value)	如果a是一个掩码数组，则设置a的填充值。
- ma.MaskedArray.get_fill_value()	返回掩码数组的填充值。
- ma.MaskedArray.set_fill_value([value])	设置掩码数组的填充值。
- ma.MaskedArray.fill_value	Filling value.

## 掩码数组算法

### 算法

- ma.anom(self[, axis, dtype])	沿着给定的轴计算异常(与算术平均值的偏差)。
- ma.anomalies(self[, axis, dtype])	沿着给定的轴计算异常(与算术平均值的偏差)。
- ma.average(a[, axis, weights, returned])	返回给定轴上数组的加权平均值。
- ma.conjugate(x, /[, out, where, casting, …]) 返回复共轭元素。
- ma.corrcoef(x[, y, rowvar, bias, …])	返回皮尔逊乘积-矩相关系数。
- ma.cov(x[, y, rowvar, bias, allow_masked, ddof])	估计协方差矩阵。
- ma.cumsum(self[, axis, dtype, out])	返回给定轴上数组元素的累积和。
- ma.cumprod(self[, axis, dtype, out])	返回给定轴上数组元素的累积乘积。
- ma.mean(self[, axis, dtype, out, keepdims])	返回沿给定轴排列的数组元素的平均值。
- ma.median(a[, axis, out, overwrite_input, …])	沿指定轴计算中值。
- ma.power(a, b[, third])	返回从第二个数组提升到幂的按元素划分的基数组。
- ma.prod(self[, axis, dtype, out, keepdims])	返回给定轴上数组元素的乘积。
- ma.std(self[, axis, dtype, out, ddof, keepdims])	返回数组元素沿给定轴的标准差。
- ma.sum(self[, axis, dtype, out, keepdims])	返回给定轴上数组元素的和。
- ma.var(self[, axis, dtype, out, ddof, keepdims])	计算沿指定轴的方差。
- ma.MaskedArray.anom([axis, dtype])	沿着给定的轴计算异常(与算术平均值的偏差)。
- ma.MaskedArray.cumprod([axis, dtype, out])	返回给定轴上数组元素的累积乘积。
- ma.MaskedArray.cumsum([axis, dtype, out])	返回给定轴上数组元素的累积和。
- ma.MaskedArray.mean([axis, dtype, out, keepdims])	返回沿给定轴排列的数组元素的平均值。
- ma.MaskedArray.prod([axis, dtype, out, keepdims])	返回给定轴上数组元素的乘积。
- ma.MaskedArray.std([axis, dtype, out, ddof, …])	返回数组元素沿给定轴的标准差。
- ma.MaskedArray.sum([axis, dtype, out, keepdims])	返回给定轴上数组元素的和。
- ma.MaskedArray.var([axis, dtype, out, ddof, …])	计算沿指定轴的方差。

### 最小/最大

- ma.argmax(self[, axis, fill_value, out])	返回沿给定轴的最大值的索引数组。
- ma.argmin(self[, axis, fill_value, out])	沿着给定的轴将索引数组返回到最小值。
- ma.max(obj[, axis, out, fill_value, keepdims])	沿着给定的轴返回最大值。
- ma.min(obj[, axis, out, fill_value, keepdims])	沿着给定的轴返回最小值。
- ma.ptp(obj[, axis, out, fill_value])	沿着给定的维数返回(最大值-最小值)
- ma.MaskedArray.argmax([axis, fill_value, out])	返回沿给定轴的最大值的索引数组。
- ma.MaskedArray.argmin([axis, fill_value, out])	沿着给定的轴将索引数组返回到最小值。
- ma.MaskedArray.max([axis, out, fill_value, …])	沿着给定的轴返回最大值。
- ma.MaskedArray.min([axis, out, fill_value, …])	沿着给定的轴返回最小值。
- ma.MaskedArray.ptp([axis, out, fill_value])	沿着给定的维数返回(最大值-最小值)。

### 分拣

- ma.argsort(a[, axis, kind, order, endwith, …])	返回按指定轴对数组进行排序的索引的ndarray。
- ma.sort(a[, axis, kind, order, endwith, …])	就地对数组进行排序。
- ma.MaskedArray.argsort([axis, kind, order, …])	返回按指定轴对数组进行排序的索引的ndarray。
- ma.MaskedArray.sort([axis, kind, order, …])	就地对数组进行排序。

### 代数

- ma.diag(v[, k])	提取对角线或构造对角线数组。
- ma.dot(a, b[, strict, out])	返回两个数组的点积。
- ma.identity(n[, dtype])	返回标识数组。
- ma.inner(a, b)	两个数组的内积。
- ma.innerproduct(a, b)	两个数组的内积。
- ma.outer(a, b)	计算两个向量的外积。
- ma.outerproduct(a, b)	计算两个向量的外积。
- ma.trace(self[, offset, axis1, axis2, …])	沿着数组的对角线返回和。
- ma.transpose(a[, axes])	排列数组的大小。
- ma.MaskedArray.trace([offset, axis1, axis2, …])	沿着数组的对角线返回和。
- ma.MaskedArray.transpose(*axes)	返回已移置轴的数组视图。

### 多项式拟合

- ma.vander(x[, n])	生成Vandermonde矩阵。
- ma.polyfit(x, y, deg[, rcond, full, w, cov]) 最小二乘多项式拟合

### 裁剪和舍入

- ma.around	将数组四舍五入到给定的小数位数。
- ma.clip(a, a_min, a_max[, out])	剪辑(限制)数组中的值。
- ma.round(a[, decimals, out])	返回a的副本，将其四舍五入为“小数”位。
- ma.MaskedArray.clip([min, max, out])	返回值限制为[min，max]的数组。
- ma.MaskedArray.round([decimals, out])	返回每个元素四舍五入到给定的小数位数。

### 杂项

- ma.allequal(a, b[, fill_value])	如果a和b的所有条目都相等，则返回True，使用Fill_VALUE作为真值，其中任一项或两者都被屏蔽。
- ma.allclose(a, b[, masked_equal, rtol, atol])	如果两个数组在公差内按元素大小相等，则返回True。
- ma.apply_along_axis(func1d, axis, arr, …)	沿着给定的轴向一维切片应用函数.
- ma.arange([start,] stop[, step,][, dtype])	在给定的间隔内返回均匀间隔的值。
- ma.choose(indices, choices[, out, mode])	使用索引数组从一组选择中构造新数组。
- ma.ediff1d(arr[, to_end, to_begin])	计算数组中连续元素之间的差异。
- ma.indices(dimensions[, dtype])	返回一个表示网格索引的数组。
- ma.where(condition[, x, y])	根据条件返回包含x或y元素的掩码数组。