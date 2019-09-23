# 索引相关

::: tip 另见

[Indexing](/reference/arrays/indexing.html)

:::

## 生成索引数组

方法 | 描述
---|---
[c_](https://numpy.org/devdocs/reference/generated/numpy.c_.html#numpy.c_) | 将切片对象平移到第二个轴上并置。
[r_](https://numpy.org/devdocs/reference/generated/numpy.r_.html#numpy.r_) | 将切片对象平移到沿第一个轴的连接。
[s_](https://numpy.org/devdocs/reference/generated/numpy.s_.html#numpy.s_) | 为数组建立索引元组的更好方法。
[nonzero](https://numpy.org/devdocs/reference/generated/numpy.nonzero.html#numpy.nonzero)(a) | 返回非零元素的[索引](https://numpy.org/devdocs/reference/generated/numpy.indices.html#numpy.indices)。
[where](https://numpy.org/devdocs/reference/generated/numpy.where.html#numpy.where)(condition, [x, y]) | 根据条件返回从x或y中选择的元素。
indices(dimensions[, dtype, sparse]) | 返回表示网格索引的数组。
[ix_](https://numpy.org/devdocs/reference/generated/numpy.ix_.html#numpy.ix_)(\*args) | 从多个序列构造一个开放的网格。
[ogrid](https://numpy.org/devdocs/reference/generated/numpy.ogrid.html#numpy.ogrid) | nd_grid实例，它返回一个开放的多维“ meshgrid”。
[ravel_multi_index](https://numpy.org/devdocs/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index)(multi_index, dims[, mode, …]) | 将边界模式应用于多索引，将索引数组的元组转换为平面索引的数组。
[unravel_index](https://numpy.org/devdocs/reference/generated/numpy.unravel_index.html#numpy.unravel_index)(indices, shape[, order]) | 将平面索引或平面索引数组转换为坐标数组的元组。
[diag_indices](https://numpy.org/devdocs/reference/generated/numpy.diag_indices.html#numpy.diag_indices)(n[, ndim]) | 返回索引以访问数组的主对角线。
[diag_indices_from](https://numpy.org/devdocs/reference/generated/numpy.diag_indices_from.html#numpy.diag_indices_from)(arr) | 返回索引以访问n维数组的主对角线。
[mask_indices](https://numpy.org/devdocs/reference/generated/numpy.mask_indices.html#numpy.mask_indices)(n, mask_func[, k]) | 给定掩码函数，将索引返回到访问（n，n）数组。
[tril_indices](https://numpy.org/devdocs/reference/generated/numpy.tril_indices.html#numpy.tril_indices)(n[, k, m]) | 返回 (n, m) 数组下三角的索引。
[tril_indices_from](https://numpy.org/devdocs/reference/generated/numpy.tril_indices_from.html#numpy.tril_indices_from)(arr[, k]) | 返回arr下三角的索引。
[triu_indices](https://numpy.org/devdocs/reference/generated/numpy.triu_indices.html#numpy.triu_indices)(n[, k, m]) | 返回 (n, m) 数组上三角的索引。
[triu_indices_from](https://numpy.org/devdocs/reference/generated/numpy.triu_indices_from.html#numpy.triu_indices_from)(arr[, k]) | 返回arr的上三角的索引。

## 类似于索引的操作

方法 | 描述
---|---
[take](https://numpy.org/devdocs/reference/generated/numpy.take.html#numpy.take)(a, indices[, axis, out, mode]) | 沿轴从数组中获取元素。
[take_along_axis](https://numpy.org/devdocs/reference/generated/numpy.take_along_axis.html#numpy.take_along_axis)(arr, indices, axis) | 通过匹配1d索引和数据切片从输入数组中获取值。
[choose](https://numpy.org/devdocs/reference/generated/numpy.choose.html#numpy.choose)(a, choices[, out, mode]) | 从索引数组和一组数组中构造一个数组以供选择。
[compress](https://numpy.org/devdocs/reference/generated/numpy.compress.html#numpy.compress)(condition, a[, axis, out]) | 沿给定轴返回数组的[选定](https://numpy.org/devdocs/reference/generated/numpy.select.html#numpy.select)切片。
[diag](https://numpy.org/devdocs/reference/generated/numpy.diag.html#numpy.diag)(v[, k]) | 提取[对角线](https://numpy.org/devdocs/reference/generated/numpy.diagonal.html#numpy.diagonal)或构造对角线阵列。
[diagonal](https://numpy.org/devdocs/reference/generated/numpy.diagonal.html#numpy.diagonal)(a[, offset, axis1, axis2]) | 返回指定的对角线。
[select](https://numpy.org/devdocs/reference/generated/numpy.select.html#numpy.select)(condlist, choicelist[, default]) | 根据条件返回从Choicelist中的元素中提取的数组。
[lib.stride_tricks.as_strided](https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.as_strided.html#numpy.lib.stride_tricks.as_strided)(x[, shape, …]) | 使用给定的形状和步幅在阵列中创建视图。

## 将数据插入数组

方法 | 描述
---|---
[place](https://numpy.org/devdocs/reference/generated/numpy.place.html#numpy.place)(arr, mask, vals) | 基于条件值和[输入](https://numpy.org/devdocs/reference/generated/numpy.put.html#numpy.put)值更改数组的元素。
[put](https://numpy.org/devdocs/reference/generated/numpy.put.html#numpy.put)(a, ind, v[, mode]) | 用给定值替换数组的指定元素。
[put_along_axis](https://numpy.org/devdocs/reference/generated/numpy.put_along_axis.html#numpy.put_along_axis)(arr, indices, values, axis) | 通过匹配1D索引和数据切片将值放入目标数组中。
[putmask](https://numpy.org/devdocs/reference/generated/numpy.putmask.html#numpy.putmask)(a, mask, values) | 基于条件值和输入值更改数组的元素。
[fill_diagonal](https://numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html#numpy.fill_diagonal)(a, val[, wrap]) | 填充任意维数的给定数组的主对角线。

## 迭代数组

方法 | 描述
---|---
[nditer](https://numpy.org/devdocs/reference/generated/numpy.nditer.html#numpy.nditer) | 高效的多维迭代器对象来迭代数组。
[ndenumerate](https://numpy.org/devdocs/reference/generated/numpy.ndenumerate.html#numpy.ndenumerate)(arr) | 多维索引迭代器。
[ndindex](https://numpy.org/devdocs/reference/generated/numpy.ndindex.html#numpy.ndindex)(\*shape) | 用于索引数组的N维迭代器对象。
[nested_iters](https://numpy.org/devdocs/reference/generated/numpy.nested_iters.html#numpy.nested_iters)() | 创建用于嵌套循环的nditer
[flatiter](https://numpy.org/devdocs/reference/generated/numpy.flatiter.html#numpy.flatiter) | 要迭代数组的平面迭代器对象。
[lib.Arrayterator](https://numpy.org/devdocs/reference/generated/numpy.lib.Arrayterator.html#numpy.lib.Arrayterator)(var[, buf_size]) | 大数组的缓冲迭代器。
