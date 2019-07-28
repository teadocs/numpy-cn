# 索引相关API

另见：
> Indexing

## 生成索引数组

- c_	将切片对象转换为沿第二轴的连接。
- r_	将切片对象转换为沿第一轴的连接。
- s_	为数组构建索引元组的更好方法。
- nonzero(a)	返回非零元素的索引。
- where(condition, [x, y])	返回元素，可以是x或y，具体取决于条件。
- indices(dimensions[, dtype])	返回表示网格索引的数组。
- ix_(*args)	从多个序列构造一个开放网格。
- ogrid	nd_grid实例，它返回一个开放的多维“meshgrid”。
- ravel_multi_index(multi_index, dims[, mode, …])	将索引数组的元组转换为平面索引数组，将边界模式应用于多索引。
- unravel_index(indices, dims[, order])	将平面索引或平面索引数组转换为坐标数组的元组。
- diag_indices(n[, ndim])	返回索引以访问数组的主对角线。
- diag_indices_from(arr)	返回索引以访问n维数组的主对角线。
- mask_indices(n, mask_func[, k])	给定掩蔽函数，将索引返回到访问（n，n）数组。
- tril_indices(n[, k, m])	返回（n，m）数组的下三角形的索引。
- tril_indices_from(arr[, k])	返回arr的下三角形的索引。
- triu_indices(n[, k, m])	返回（n，m）数组的上三角形的索引。
- triu_indices_from(arr[, k])	返回arr上三角的索引。

## 类似索引的操作

- take(a, indices[, axis, out, mode])	从轴上获取数组中的元素。
- choose(a, choices[, out, mode])	从索引数组和一组数组构造一个数组以供选择。
- compress(condition, a[, axis, out])	沿给定轴返回数组的选定切片。
- diag(v[, k])	提取对角线或构造对角线阵列。
- diagonal(a[, offset, axis1, axis2])	返回指定的对角线。
- select(condlist, choicelist[, default])	返回从选择列表中的元素绘制的数组，具体取决于条件。
- lib.stride_tricks.as_strided(x[, shape, …])	使用给定的形状和步幅创建数组视图。

## 将数据插入数组

- place(arr, mask, vals)	根据条件和输入值更改数组的元素。
- put(a, ind, v[, mode])	用给定值替换数组的指定元素。
- putmask(a, mask, values)	根据条件和输入值更改数组的元素。
- fill_diagonal(a, val[, wrap])	填充任何维度的给定数组的主对角线。

## 迭代数组

- nditer	用于迭代数组的高效多维迭代器对象。
- ndenumerate(arr)	多维索引迭代器。
- ndindex(*shape)	用于索引数组的N维迭代器对象。
- flatiter	平面迭代器对象迭代数组。
- lib.Arrayterator(var[, buf_size])	用于大型数组的缓冲迭代器。