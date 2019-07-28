# 数组操作

## 基本操作

- copyto(dst, src[, casting, where]) 将值从一个数组复制到另一个数组，并根据需要进行广播。

## 改变数组形状

- reshape(a, newshape[, order])	为数组提供新形状而不更改其数据。
- ravel(a[, order])	返回一个连续的扁平数组。
- ndarray.flat	数组上的一维迭代器.
- ndarray.flatten([order])	返回折叠成一维的数组的副本。
ss
## 转置式运算

- moveaxis(a, source, destination)	将数组的轴移动到新位置。
- rollaxis(a, axis[, start])	向后滚动指定的轴，直到它位于给定位置。
- swapaxes(a, axis1, axis2)	交换数组的两个轴。
- ndarray.T	与self.transpose() 相同，只是如果self.ndim < 2 则返回self。
- transpose(a[, axes])	置换数组的维度。

## 更改尺寸数量

- atleast_1d(*arys)	将输入转换为至少具有一个维度的数组。
- atleast_2d(*arys)	将输入视为具有至少两个维度的数组。
- atleast_3d(*arys)	将输入视为具有至少三维的数组。
- broadcast	制作一个模仿广播的对象。
- broadcast_to(array, shape[, subok])	将数组广播到新形状。
- broadcast_arrays(*args, **kwargs)	相互广播任意数量的数组。
- expand_dims(a, axis)	展开数组的形状。
- squeeze(a[, axis])	展开数组的形状。

## 改变阵列的种类

- asarray(a[, dtype, order])	将输入转换为数组。
- asanyarray(a[, dtype, order])	将输入转换为ndarray，但通过ndarray子类。
- asmatrix(data[, dtype])	将输入解释为矩阵。
- asfarray(a[, dtype])	返回转换为float类型的数组。
- asfortranarray(a[, dtype])	返回在内存中以Fortran顺序布局的数组。
- ascontiguousarray(a[, dtype])	在内存中返回一个连续的数组（C顺序）。
- asarray_chkfinite(a[, dtype, order])	将输入转换为数组，检查NaN或Infs。
- asscalar(a) 将大小为1的数组转换为标量等效数组。
- require(a[, dtype, requirements])	返回满足要求的提供类型的ndarray。

## 加入数组

- concatenate((a1, a2, …)[, axis, out])	沿现有轴加入一系列数组。
- stack(arrays[, axis, out])	沿新轴加入一系列数组。
- column_stack(tup)	将1-D阵列作为列堆叠成2-D阵列。
- dstack(tup)	按顺序深度堆叠阵列（沿第三轴）。
- hstack(tup)	按顺序堆叠数组（列式）。
- vstack(tup)	垂直堆叠数组（行方式）。
- block(arrays)	从嵌套的块列表中组装nd数组。

## 拆分数组

- split(ary, indices_or_sections[, axis])	将数组拆分为多个子数组。
- array_split(ary, indices_or_sections[, axis])	将数组拆分为多个子数组。
- dsplit(ary, indices_or_sections)	沿第3轴（深度）将数组拆分为多个子数组。
- hsplit(ary, indices_or_sections)	将数组水平拆分为多个子数组（按列）。
- vsplit(ary, indices_or_sections)	将数组垂直拆分为多个子数组（逐行）。

## 平铺阵列

- tile(A, reps)	通过重复A重复给出的次数来构造数组。
- repeat(a, repeats[, axis])	重复数组的元素。

## Adding and removing elements

- delete(arr, obj[, axis])	返回一个新数组，其子轴数组沿轴被删除。
- insert(arr, obj, values[, axis])	在给定索引之前沿给定轴插入值。
- append(arr, values[, axis])	将值附加到数组的末尾。
- resize(a, new_shape)	返回具有指定形状的新数组。
- trim_zeros(filt[, trim])	从1-D数组或序列中修剪前导和/或尾随零。
- unique(ar[, return_index, return_inverse, …])	找到数组的唯一元素。

## 重新排列元素

- flip(m, axis)	沿给定轴反转数组中元素的顺序。
- fliplr(m)	向左/向右翻转阵列。
- flipud(m)	向上/向下翻转阵列。
- reshape(a, newshape[, order])	为数组提供新形状而不更改其数据。
- roll(a, shift[, axis])	沿给定轴滚动数组元素。
- rot90(m[, k, axes])	在轴指定的平面中将数组旋转90度。
