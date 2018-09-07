# 排序、搜索和计数

## 排序

- sort(a[, axis, kind, order])	返回数组的排序副本。
- lexsort(keys[, axis])	使用一系列键执行间接排序。
- argsort(a[, axis, kind, order])	返回对数组进行排序的索引。
- ndarray.sort([axis, kind, order])	就地对数组进行排序。
- msort(a)	返回沿第一个轴排序的数组副本。
- sort_complex(a)	首先使用实部对复杂数组进行排序，然后使用虚部进行排序。
- partition(a, kth[, axis, kind, order])	返回数组的分区副本。
- argpartition(a, kth[, axis, kind, order])	使用kind关键字指定的算法沿给定轴执行间接分区。

## 搜索

- argmax(a[, axis, out])	返回沿轴的最大值的索引。
- nanargmax(a[, axis])	返回指定轴上最大值的索引，忽略NAS。
- argmin(a[, axis, out])	返回沿轴的最小值的索引。
- nanargmin(a[, axis])	返回指定轴上的最小值的索引，忽略NAS。
- argwhere(a)	查找按元素分组的非零数组元素的索引。
- nonzero(a)	返回非零元素的索引。
- flatnonzero(a)	返回a的展平版本中非零的索引。
- where(condition, [x, y])	返回元素，可以是x或y，具体取决于条件。
- searchsorted(a, v[, side, sorter])	查找应插入元素以维护顺序的索引。
- extract(condition, arr)	返回满足某些条件的数组元素。

## 计数

- count_nonzero(a[, axis])	计算数组a中的非零值的数量。
