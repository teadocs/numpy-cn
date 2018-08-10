# 使用numpy.ma

## 构造掩码数组

有几种构造掩码数组的方法。

- 第一种可能性是直接调用``MastedArray‘类。

- 第二种可能性是使用两个带掩码的数组构造函数，``array`` 和 ``msued_array``。
    - ``数组`` (数据[、dtype、复制、顺序、掩码、…]) 可能带有掩码值的数组类。
    - ``mabed_array`` 别名 ``numpy.ma.core.MastedArray``

- 第三种选择是获取现有数组的视图。在这种情况下，如果数组没有命名字段，则视图的掩码设置为 ``nomask``，否则设置为与数组结构相同的布尔数组。

    ```python
    >>> x = np.array([1, 2, 3])
    >>> x.view(ma.MaskedArray)
    masked_array(data = [1 2 3],
                mask = False,
        fill_value = 999999)
    >>> x = np.array([(1, 1.), (2, 2.)], dtype=[('a',int), ('b', float)])
    >>> x.view(ma.MaskedArray)
    masked_array(data = [(1, 1.0) (2, 2.0)],
                mask = [(False, False) (False, False)],
        fill_value = (999999, 1e+20),
                dtype = [('a', '<i4'), ('b', '<f8')])
    ```

- 另一种可能性是使用以下任何功能：
    - ``asarray``(a[, dtype, order]) 将输入转换为给定数据类型的掩码数组。
    - ``asanyarray``(a[, dtype])	将输入转换为掩码数组，保留子类。
    - ``fix_invalid``(a[, mask, copy, fill_value]) 返回带有无效数据的输入，并用填充值替换。
    - ``masked_equal``(x, value[, copy])	掩盖一个等于给定值的数组。
    - ``masked_greater``(x, value[, copy])	掩盖大于给定值的数组。
    - ``masked_greater_equal``(x, value[, copy])	掩盖大于或等于给定值的数组。
    - ``masked_inside``(x, v1, v2[, copy])	在给定间隔内掩盖数组。
    - ``masked_invalid``(a[, copy])	Mask an array 无效值出现的地方（NaN或infs）。
    - ``masked_less``(x, value[, copy])	掩盖小于给定值的数组。
    - ``masked_less_equal``(x, value[, copy])	掩盖小于或等于给定值的数组。
    - ``masked_not_equal``(x, value[, copy])	掩盖不等于给定值的数组。
    - ``masked_object``(x, value[, copy, shrink])	掩盖数组x，其中数据正好等于value。
    - ``masked_outside``(x, v1, v2[, copy])	掩盖给定的数组之外的数组interval.
    - ``masked_values``(x, value[, rtol, atol, copy, …])	掩码使用浮点相等。
    - ``masked_where``(condition, a[, copy])	掩盖满足条件的数组。

## 访问数据

可以通过多种方式访问掩码数组的基础数据：

- 通过数据属性。 输出是数组的视图，作为numpy.ndarray或其子类之一，具体取决于掩码数组创建时基础数据的类型。
- 通过__array__方法。 输出然后是numpy.ndarray。
- 通过直接将掩盖的数组视图作为numpy.ndarray或其子类之一（实际上是使用data属性的那个）。
- 通过使用getdata函数。

如果某些条目被标记为无效，则这些方法都不是完全令人满意的。作为一般规则，在不需要任何屏蔽条目的情况下需要表示数组时，建议使用填充方法填充数组。

## 访问 mask

掩码数组的掩码可通过其``mask``属性访问。 我们必须记住，掩码中的True条目表示无效数据。

另一种可能性是使用``getmask``和``getmaskarray``函数。 如果x是一个掩码数组，``getmask（x）``输出x的掩码，否则输出特殊值``nomask``。 如果``x``是一个掩码数组，``getmaskarray（x）``输出``x``的掩码。 如果x没有无效条目或者不是掩码数组，则该函数输出一个“False”的布尔数组，其元素与x一样多。

## 仅访问有效条目

要仅检索有效条目，我们可以使用掩码的反转作为索引。 掩码的逆可以用``numpy.logical_not``函数计算，或者只用〜运算符计算：

```python
>>> x = ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
>>> x[~x.mask]
masked_array(data = [1 4],
             mask = [False False],
       fill_value = 999999)
```

另一种检索有效数据的方法是使用``compressed``方法，该方法返回一维``ndarray``（或其子类之一，具体取决于``baseclass``属性的值）：

```python
>>> x.compressed()
array([1, 4])
```

请注意，``compressed``的输出始终为1D。

## 修改 mask

### 掩盖一个条目

将掩码数组的一个或多个特定条目标记为无效的推荐方法是为它们分配特殊值``masked``：

```python
>>> x = ma.array([1, 2, 3])
>>> x[0] = ma.masked
>>> x
masked_array(data = [-- 2 3],
             mask = [ True False False],
       fill_value = 999999)
>>> y = ma.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> y[(0, 1, 2), (1, 2, 0)] = ma.masked
>>> y
masked_array(data =
 [[1 -- 3]
  [4 5 --]
  [-- 8 9]],
             mask =
 [[False  True False]
  [False False  True]
  [ True False False]],
       fill_value = 999999)
>>> z = ma.array([1, 2, 3, 4])
>>> z[:-2] = ma.masked
>>> z
masked_array(data = [-- -- 3 4],
             mask = [ True  True False False],
       fill_value = 999999)
```

第二种可能性是直接修改``mask``，但不鼓励这种用法。

> **注意**
> 当使用简单的非结构化数据类型创建新的掩码数组时，掩码最初设置为特殊值 ``nomask``，大致对应于布尔值``False``。尝试设置``nomask``元素将失败并出现``TypeError``异常，因为布尔值不支持项目赋值。

通过为掩码指定 ``True``，可以立即屏蔽数组的所有条目：

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x.mask = True
>>> x
masked_array(data = [-- -- --],
             mask = [ True  True  True],
       fill_value = 999999)
```

Finally, specific entries can be masked and/or unmasked by assigning to the mask a sequence of booleans:

```python
>>> x = ma.array([1, 2, 3])
>>> x.mask = [0, 1, 0]
>>> x
masked_array(data = [1 -- 3],
             mask = [False  True False],
       fill_value = 999999)
```

### Unmasking an entry

To unmask one or several specific entries, we can just assign one or several new valid values to them:

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x
masked_array(data = [1 2 --],
             mask = [False False  True],
       fill_value = 999999)
>>> x[-1] = 5
>>> x
masked_array(data = [1 2 5],
             mask = [False False False],
       fill_value = 999999)
```

> **Note**
> Unmasking an entry by direct assignment will silently fail if the masked array has a hard mask, as shown by the ``hardmask`` attribute. This feature was introduced to prevent overwriting the mask. To force the unmasking of an entry where the array has a hard mask, the mask must first to be softened using the ``soften_mask`` method before the allocation. It can be re-hardened with ``harden_mask``:

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1], hard_mask=True)
>>> x
masked_array(data = [1 2 --],
             mask = [False False  True],
       fill_value = 999999)
>>> x[-1] = 5
>>> x
masked_array(data = [1 2 --],
             mask = [False False  True],
       fill_value = 999999)
>>> x.soften_mask()
>>> x[-1] = 5
>>> x
masked_array(data = [1 2 5],
             mask = [False False  False],
       fill_value = 999999)
>>> x.harden_mask()
```

To unmask all masked entries of a masked array (provided the mask isn’t a hard mask), the simplest solution is to assign the constant ``nomask`` to the mask:

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x
masked_array(data = [1 2 --],
             mask = [False False  True],
       fill_value = 999999)
>>> x.mask = ma.nomask
>>> x
masked_array(data = [1 2 3],
             mask = [False False False],
       fill_value = 999999)
```

## Indexing and slicing

As a ``MaskedArray`` is a subclass of ``numpy.ndarray``, it inherits its mechanisms for indexing and slicing.

When accessing a single entry of a masked array with no named fields, the output is either a scalar (if the corresponding entry of the mask is ``False``) or the special value ``masked`` (if the corresponding entry of the mask is True):

```python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x[0]
1
>>> x[-1]
masked_array(data = --,
             mask = True,
       fill_value = 1e+20)
>>> x[-1] is ma.masked
True
```

If the masked array has named fields, accessing a single entry returns a ``numpy.void`` object if none of the fields are masked, or a 0d masked array with the same dtype as the initial array if at least one of the fields is masked.

```python
>>> y = ma.masked_array([(1,2), (3, 4)],
...                mask=[(0, 0), (0, 1)],
...               dtype=[('a', int), ('b', int)])
>>> y[0]
(1, 2)
>>> y[-1]
masked_array(data = (3, --),
             mask = (False, True),
       fill_value = (999999, 999999),
            dtype = [('a', '<i4'), ('b', '<i4')])
```

When accessing a slice, the output is a masked array whose ``data`` attribute is a view of the original data, and whose mask is either ``nomask`` (if there was no invalid entries in the original array) or a view of the corresponding slice of the original mask. The view is required to ensure propagation of any modification of the mask to the original.

```python
>>> x = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 1])
>>> mx = x[:3]
>>> mx
masked_array(data = [1 -- 3],
             mask = [False  True False],
       fill_value = 999999)
>>> mx[1] = -1
>>> mx
masked_array(data = [1 -1 3],
             mask = [False False False],
       fill_value = 999999)
>>> x.mask
array([False,  True, False, False,  True])
>>> x.data
array([ 1, -1,  3,  4,  5])
```

Accessing a field of a masked array with structured datatype returns a MaskedArray.

## Operations on masked arrays

Arithmetic and comparison operations are supported by masked arrays. As much as possible, invalid entries of a masked array are not processed, meaning that the corresponding ``data`` entries should be the same before and after the operation.

<div class="warning-warp">
<b>Warning</b>

<p>We need to stress that this behavior may not be systematic, that masked data may be affected by the operation in some cases and therefore users should not rely on this data remaining unchanged.</p>
</div>

The numpy.ma module comes with a specific implementation of most ufuncs. Unary and binary functions that have a validity domain (such as log or divide) return the masked constant whenever the input is masked or falls outside the validity domain:

```python
>>> ma.log([-1, 0, 1, 2])
masked_array(data = [-- -- 0.0 0.69314718056],
             mask = [ True  True False False],
       fill_value = 1e+20)
```

Masked arrays also support standard numpy ufuncs. The output is then a masked array. The result of a unary ufunc is masked wherever the input is masked. The result of a binary ufunc is masked wherever any of the input is masked. If the ufunc also returns the optional context output (a 3-element tuple containing the name of the ufunc, its arguments and its domain), the context is processed and entries of the output masked array are masked wherever the corresponding input fall outside the validity domain:

```python
>>> x = ma.array([-1, 1, 0, 2, 3], mask=[0, 0, 0, 0, 1])
>>> np.log(x)
masked_array(data = [-- -- 0.0 0.69314718056 --],
             mask = [ True  True False False  True],
       fill_value = 1e+20)
```