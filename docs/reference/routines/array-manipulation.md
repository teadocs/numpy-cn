---
meta:
  - name: keywords
    content: numpy数组,数组处理
  - name: description
    content: NumPy的数组处理程序，它的api包含以下内容。
---

# 数组处理程序

## 基本操作

方法 | 描述
---|---
[copyto](https://numpy.org/devdocs/reference/generated/numpy.copyto.html#numpy.copyto)(dst, src[, casting, where]) | 将值从一个数组复制到另一个数组，并根据需要进行广播。

## 改变数组形状

方法 | 描述
---|---
[reshape](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape)(a, newshape[, order]) | 在不更改数据的情况下为数组赋予新的形状。
[ravel](https://numpy.org/devdocs/reference/generated/numpy.ravel.html#numpy.ravel)(a[, order]) | 返回一个连续的扁平数组。
[ndarray.flat](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | 数组上的一维迭代器。
[ndarray.flatten](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten)([order]) | 返回折叠成一维的数组副本。

## 类转置操作

方法 | 描述
---|---
[moveaxis](https://numpy.org/devdocs/reference/generated/numpy.moveaxis.html#numpy.moveaxis)(a, source, destination) | 将数组的轴移到新位置。
[rollaxis](https://numpy.org/devdocs/reference/generated/numpy.rollaxis.html#numpy.rollaxis)(a, axis[, start]) | 向后滚动指定的轴，直到其位于给定的位置。
[swapaxes](https://numpy.org/devdocs/reference/generated/numpy.swapaxes.html#numpy.swapaxes)(a, axis1, axis2) | 互换数组的两个轴。
[ndarray.T](https://numpy.org/devdocs/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T) | 转置数组。
[transpose](https://numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose)(a[, axes]) | 排列数组的尺寸。

## 更改维度数

方法 | 描述
---|---
[atleast_1d](https://numpy.org/devdocs/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d)(\*arys) | 将输入转换为至少一维的数组。
[atleast_2d](https://numpy.org/devdocs/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d)(\*arys) | 将输入视为至少具有二维的数组。
[atleast_3d](https://numpy.org/devdocs/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d)(\*arys) | 以至少三个维度的数组形式查看输入。
[broadcast](https://numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast) | 产生模仿广播的对象。
[broadcast_to](https://numpy.org/devdocs/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to)(array, shape[, subok]) | 将数组广播为新形状。
[broadcast_arrays](https://numpy.org/devdocs/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays)(\*args, \*\*kwargs) | 互相广播任意数量的阵列。
[expand_dims](https://numpy.org/devdocs/reference/generated/numpy.expand_dims.html#numpy.expand_dims)(a, axis) | 扩展数组的形状。
[squeeze](https://numpy.org/devdocs/reference/generated/numpy.squeeze.html#numpy.squeeze)(a[, axis]) | 从数组形状中删除一维条目。

## 改变数组的种类

方法 | 描述
---|---
[asarray](https://numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray)(a[, dtype, order]) | 将输入转换为数组。
[asanyarray](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray)(a[, dtype, order]) | 将输入转换为ndarray，但通过ndarray子类。
[asmatrix](https://numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | 将输入解释为矩阵。
[asfarray](https://numpy.org/devdocs/reference/generated/numpy.asfarray.html#numpy.asfarray)(a[, dtype]) | 返回转换为浮点类型的数组。
[asfortranarray](https://numpy.org/devdocs/reference/generated/numpy.asfortranarray.html#numpy.asfortranarray)(a[, dtype]) | 返回以Fortran顺序排列在内存中的数组（ndim> = 1）。
[ascontiguousarray](https://numpy.org/devdocs/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray)(a[, dtype]) | 返回内存中的连续数组（ndim> = 1）（C顺序）。
[asarray_chkfinite](https://numpy.org/devdocs/reference/generated/numpy.asarray_chkfinite.html#numpy.asarray_chkfinite)(a[, dtype, order]) | 将输入转换为数组，检查NaN或Infs。
[asscalar](https://numpy.org/devdocs/reference/generated/numpy.asscalar.html#numpy.asscalar)(a) | 将大小为1的数组转换为其等效的标量。
[require](https://numpy.org/devdocs/reference/generated/numpy.require.html#numpy.require)(a[, dtype, requirements]) | 返回提供的类型满足要求的ndarray。

## 组合数组

方法 | 描述
---|---
[concatenate](https://numpy.org/devdocs/reference/generated/numpy.concatenate.html#numpy.concatenate)((a1, a2, …) | 沿现有轴连接一系列数组。
[stack](https://numpy.org/devdocs/reference/generated/numpy.stack.html#numpy.stack)(arrays[, axis, out]) | 沿新轴连接一系列数组。
[column_stack](https://numpy.org/devdocs/reference/generated/numpy.column_stack.html#numpy.column_stack)(tup) | 将一维数组作为列堆叠到二维数组中。
[dstack](https://numpy.org/devdocs/reference/generated/numpy.dstack.html#numpy.dstack)(tup) | 沿深度方向（沿第三轴）按顺序堆叠数组。
[hstack](https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack)(tup) | 水平（按列）顺序堆叠数组。
[vstack](https://numpy.org/devdocs/reference/generated/numpy.vstack.html#numpy.vstack)(tup) | 垂直（行）按顺序堆叠数组。
[block](https://numpy.org/devdocs/reference/generated/numpy.block.html#numpy.block)(arrays) | 从块的嵌套列表中组装一个nd数组。

## 拆分数组

方法 | 描述
---|---
[split](https://numpy.org/devdocs/reference/generated/numpy.split.html#numpy.split)(ary, indices_or_sections[, axis]) | 将数组拆分为多个子数组，作为ary的视图。
[array_split](https://numpy.org/devdocs/reference/generated/numpy.array_split.html#numpy.array_split)(ary, indices_or_sections[, axis]) | 将一个数组拆分为多个子数组。
[dsplit](https://numpy.org/devdocs/reference/generated/numpy.dsplit.html#numpy.dsplit)(ary, indices_or_sections) | 沿第3轴（深度）将数组拆分为多个子数组。
[hsplit](https://numpy.org/devdocs/reference/generated/numpy.hsplit.html#numpy.hsplit)(ary, indices_or_sections) | 水平（按列）将一个数组拆分为多个子数组。
[vsplit](https://numpy.org/devdocs/reference/generated/numpy.vsplit.html#numpy.vsplit)(ary, indices_or_sections) | 垂直（行）将数组拆分为多个子数组。

## 平铺数组

方法 | 描述
---|---
[tile](https://numpy.org/devdocs/reference/generated/numpy.tile.html#numpy.tile)(A, reps) | 通过重复A代表次数来构造一个数组。
[repeat](https://numpy.org/devdocs/reference/generated/numpy.repeat.html#numpy.repeat)(a, repeats[, axis]) | 重复数组的元素。

## 添加和删除元素

方法 | 描述
---|---
[delete](https://numpy.org/devdocs/reference/generated/numpy.delete.html#numpy.delete)(arr, obj[, axis]) | 返回一个新的数组，该数组具有沿删除的轴的子数组。
[insert](https://numpy.org/devdocs/reference/generated/numpy.insert.html#numpy.insert)(arr, obj, values[, axis]) | 沿给定轴在给定索引之前插入值。
[append](https://numpy.org/devdocs/reference/generated/numpy.append.html#numpy.append)(arr, values[, axis]) | 将值附加到数组的末尾。
[resize](https://numpy.org/devdocs/reference/generated/numpy.resize.html#numpy.resize)(a, new_shape) | 返回具有指定形状的新数组。
[trim_zeros](https://numpy.org/devdocs/reference/generated/numpy.trim_zeros.html#numpy.trim_zeros)s(filt[, trim]) | 修剪一维数组或序列中的前导和/或尾随零。
[unique](https://numpy.org/devdocs/reference/generated/numpy.unique.html#numpy.unique)(ar[, return_index, return_inverse, …]) | 查找数组的唯一元素。

## 重新排列元素

方法 | 描述
---|---
[flip](https://numpy.org/devdocs/reference/generated/numpy.flip.html#numpy.flip)(m[, axis]) | 沿给定轴颠倒数组中元素的顺序。
[fliplr](https://numpy.org/devdocs/reference/generated/numpy.fliplr.html#numpy.fliplr)(m) | 左右翻转数组。
[flipud](https://numpy.org/devdocs/reference/generated/numpy.flipud.html#numpy.flipud)(m) | 上下翻转阵列。
[reshape](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape)(a, newshape[, order])	| 在不更改数据的情况下为数组赋予新的形状。
[roll](https://numpy.org/devdocs/reference/generated/numpy.roll.html#numpy.roll)(a, shift[, axis]) | 沿给定轴滚动数组元素。
[rot90](https://numpy.org/devdocs/reference/generated/numpy.rot90.html#numpy.rot90)(m[, k, axes]) | 在轴指定的平面中将阵列旋转90度。
