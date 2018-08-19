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

## 掩码数组

### 算术

- ma.anom(self[, axis, dtype])	计算沿给定轴的异常（与算术平均值的偏差）。
- ma.anomalies(self[, axis, dtype])	计算沿给定轴的异常（与算术平均值的偏差）。
- ma.average(a[, axis, weights, returned])	返回给定轴上的数组的加权平均值。
- ma.conjugate(x, /[, out, where, casting, …])	以元素方式返回复共轭。
- ma.corrcoef(x[, y, rowvar, bias, …])	返回Pearson乘积矩相关系数。
- ma.cov(x[, y, rowvar, bias, allow_masked, ddof])	估计协方差矩阵。
- ma.cumsum(self[, axis, dtype, out])	返回给定轴上的数组元素的累积和。
- ma.cumprod(self[, axis, dtype, out])	返回给定轴上的数组元素的累积乘积。
- ma.mean(self[, axis, dtype, out, keepdims])	返回给定轴上数组元素的平均值。
- ma.median(a[, axis, out, overwrite_input, …])	计算沿指定轴的中位数。
- ma.power(a, b[, third])	返回从第二个数组提升到幂的基于元素的基本数组。
- ma.prod(self[, axis, dtype, out, keepdims])	返回给定轴上的数组元素的乘积。
- ma.std(self[, axis, dtype, out, ddof, keepdims])	返回给定轴的数组元素的标准偏差。
- ma.sum(self[, axis, dtype, out, keepdims])	返回给定轴上的数组元素的总和。
- ma.var(self[, axis, dtype, out, ddof, keepdims])	计算沿指定轴的方差。
- ma.MaskedArray.anom([axis, dtype])	计算沿给定轴的异常（与算术平均值的偏差）。
- ma.MaskedArray.cumprod([axis, dtype, out])	返回给定轴上的数组元素的累积乘积。
- ma.MaskedArray.cumsum([axis, dtype, out])	返回给定轴上的数组元素的累积和。
- ma.MaskedArray.mean([axis, dtype, out, keepdims])	返回给定轴上数组元素的平均值。
- ma.MaskedArray.prod([axis, dtype, out, keepdims])	返回给定轴上的数组元素的乘积。
- ma.MaskedArray.std([axis, dtype, out, ddof, …])	返回给定轴的数组元素的标准偏差。
- ma.MaskedArray.sum([axis, dtype, out, keepdims])	返回给定轴上的数组元素的总和。
- ma.MaskedArray.var([axis, dtype, out, ddof, …])	计算沿指定轴的方差。

### 最小/最大

- ma.argmax(self[, axis, fill_value, out])	返回沿给定轴的最大值的索引数组。
- ma.argmin(self[, axis, fill_value, out])	将索引数组返回到给定轴的最小值。
- ma.max(obj[, axis, out, fill_value, keepdims])	沿给定轴返回最大值。
- ma.min(obj[, axis, out, fill_value, keepdims])	沿给定轴返回最小值。
- ma.ptp(obj[, axis, out, fill_value])	沿给定维度返回（最大 - 最小）。
- ma.MaskedArray.argmax([axis, fill_value, out])	返回沿给定轴的最大值的索引数组。
- ma.MaskedArray.argmin([axis, fill_value, out])	将索引数组返回到给定轴的最小值。
- ma.MaskedArray.max([axis, out, fill_value, …])	沿给定轴返回最大值。
- ma.MaskedArray.min([axis, out, fill_value, …])	沿给定轴返回最小值。
- ma.MaskedArray.ptp([axis, out, fill_value])	沿给定维度返回（最大 - 最小）。

### 排序

- ma.argsort(a[, axis, kind, order, endwith, …])	返回一个索引的ndarray，它沿着指定的轴对数组进行排序。
- ma.sort(a[, axis, kind, order, endwith, …])	就地对阵列进行排序。
- ma.MaskedArray.argsort([axis, kind, order, …])	对数组进行排序，in-Return一个索引的ndarray，它沿着指定的轴对数组进行排序。
- ma.MaskedArray.sort([axis, kind, order, …])	就地对阵列进行排序。

### 代数

- ma.diag(v[, k])	提取对角线或构造对角线阵列。
- ma.dot(a, b[, strict, out])	返回两个数组的点积。
- ma.identity(n[, dtype])	返回标识数组。
- ma.inner(a, b)	两个数组的内积。
- ma.innerproduct(a, b)	两个数组的内积。
- ma.outer(a, b)	计算两个向量的外积。
- ma.outerproduct(a, b)	计算两个向量的外积。
- ma.trace(self[, offset, axis1, axis2, …])	返回数组对角线的总和。
- ma.transpose(a[, axes])	置换数组的维度。
- ma.MaskedArray.trace([offset, axis1, axis2, …])	返回数组对角线的总和。
- ma.MaskedArray.transpose(*axes)	返回轴转置的数组视图。

### 多项式拟合

- ma.vander(x[, n])	生成Vandermonde矩阵。
- ma.polyfit(x, y, deg[, rcond, full, w, cov])	最小二乘多项式拟合。

### 剪裁和舍入

- ma.around	将数组舍入到给定的小数位数。
- ma.clip(a, a_min, a_max[, out])	剪辑（限制）数组中的值。
- ma.round(a[, decimals, out])	返回a的副本，四舍五入到“小数”位置。
[ ma.MaskedArray.clip([min, max, out])	返回其值限制为[min，max]的数组。
- ma.MaskedArray.round([decimals, out])	返回舍入到给定小数位数的每个元素。

### 杂项

- ma.allequal(a, b[, fill_value])	如果a和b的所有条目相等，则返回True，使用fill_value作为掩盖其中一个或两个的真值。
- ma.allclose(a, b[, masked_equal, rtol, atol])	如果两个数组在容差范围内在元素方面相等，则返回True。
- ma.apply_along_axis(func1d, axis, arr, …)	将函数应用于沿给定轴的1-D切片。
- ma.arange([start,] stop[, step,][, dtype])	在给定间隔内返回均匀间隔的值。
- ma.choose(indices, choices[, out, mode])	使用索引数组从一组选项中构造新数组。
- ma.ediff1d(arr[, to_end, to_begin])	计算数组的连续元素之间的差异。
- ma.indices(dimensions[, dtype])	返回表示网格索引的数组。
- ma.where(condition[, x, y])	返回带有x或y元素的蒙版数组，具体取决于条件。