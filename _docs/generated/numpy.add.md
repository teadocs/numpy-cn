# numpy.add

## 调用概览

```python
numpy.add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'add'>
```
## 方法说明

逐个元素添加参数。

- **参数**: 
    - **x1, x2 : array_like**
        要添加的数组。如果 ``x1.shape != x2.shape``，它们必须可以广播到一个共同的形状（可能是一个或另一个的形状）。
    - **out : ndarray，None或ndarray和None的元组，可选**
        存储结果的位置。如果提供，它必须具有输入广播的形状。如果未提供或None则返回新分配的数组。元组（仅可作为关键字参数）的长度必须等于输出的数量。
    - **where : array_like, 可选**
        值为True表示计算该位置的ufunc，值为False表示仅在输出中保留该值。
    - **\*\*kwargs**
        对于其他关键字参数，请参阅[ufunc docs](/reference/ufuncs/index.html).
- **返回**:
    - **add : ndarray or scalar**
        x1和x2之和，元素

## 注意

在阵列广播方面相当于x1 + x2。

## 例子

```python
>>> np.add(1.0, 4.0)
5.0
>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = np.arange(3.0)
>>> np.add(x1, x2)
array([[  0.,   2.,   4.],
       [  3.,   5.,   7.],
       [  6.,   8.,  10.]])
```