# 掩码数组

### 理由

掩码数组是包含了丢失或无效条目的数组。
[numpy.ma](#numpy-ma模块) 模块为numpy提供了几乎类似工作的替代方案，
支持带掩码的数据矩阵。

### 什么是掩码数组？

在许多情况下，数据集可能不完整或因无效数据的存在而受到污染。
例如，传感器可能无法记录数据或记录无效值。
[numpy.ma](#numpy-ma模块) 模块通过引入掩码数组提供了一种解决此问题的便捷方法。

掩码数组是标准 [``numpy.ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 和掩码的组合。
掩码或者是 [``nomask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.nomask)，
指示关联数组的任何值都是无效的，或者是布尔数组的数组，用于确定关联数组的每个元素的值是否有效。
当掩码的元素为 ``False`` 时，关联数组的相应元素是有效的，并且被称为未掩码。
当掩码的元素为 ``True`` 时，关联数组的相应元素称为掩码（无效）。

该包确保在计算中不使用被掩码的条目。

作为示例，让我们考虑以下数据集：

``` python
>>> import numpy as np
>>> import numpy.ma as ma
>>> x = np.array([1, 2, 3, -1, 5])
```

我们希望将第四个条目标记为无效。最简单的方法是创建一个掩码数组：

``` python
>>> mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0])
```

我们现在可以计算数据集的平均值，而无需考虑无效数据：

``` python
>>> mx.mean()
2.75
```

## ``numpy.ma``模块

[numpy.ma](#numpy-ma模块) 模块的主要特性是[``MaskedArray``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray)
类，它是的子类[``numpy.ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)。
在[MaskedArray类](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#maskedarray-baseclass)部分中更详细地描述了类、其属性和方法。

[numpy.ma](#numpy-ma模块) 模块可以用作 [``numpy``](index.html) 的补充：

``` python
>>> import numpy as np
>>> import numpy.ma as ma
```

要创建第二个元素掩码数组，我们会这样做：

``` python
>>> y = ma.array([1, 2, 3], mask = [0, 1, 0])
```

要创建一个掩码数组，其中所有接近1.e20的值都无效，我们会这样做：

``` python
>>> z = masked_values([1.0, 1.e20, 3.0, 4.0], 1.e20)
```

有关掩码数组创建方法的完整讨论，请参阅[构造掩码数组](#创建掩码数组)一节。

## 使用 numpy.ma 模块

### 创建掩码数组

有几种方法可以创建一个掩码数组。

- 第一种可能性是直接调用类：[``MaskedArray``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray)。
- 第二种可能性是使用两个掩码数组构造函数，
 [``array``](https://numpy.org/devdocs/reference/generated/numpy.ma.array.html#numpy.ma.array)和[``masked_array``](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_array.html#numpy.ma.masked_array)。

  方法 | 描述
  ---|---
  [array](https://numpy.org/devdocs/reference/generated/numpy.ma.array.html#numpy.ma.array)(data[, dtype, copy, order, mask, …]) | 具有可能掩码值的数组类。
  [masked_array](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_array.html#numpy.ma.masked_array) | 别名 numpy.ma.core.MaskedArray

- 第三种选择是获取现有数组的视图。在这种情况下，[``nomask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.nomask)如果数组没有命名字段，则视图的掩码设置为，否则设置为与数组具有相同结构的布尔数组。

  ``` python
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

  方法 | 描述
  ---|---
  [asarray](https://numpy.org/devdocs/reference/generated/numpy.ma.asarray.html#numpy.ma.asarray)(a[, dtype, order]) | 将输入转换为给定数据类型的掩码数组。
  [asanyarray](https://numpy.org/devdocs/reference/generated/numpy.ma.asanyarray.html#numpy.ma.asanyarray)(a[, dtype]) | 将输入转换为掩码数组，保留子类。
  [fix_invalid](https://numpy.org/devdocs/reference/generated/numpy.ma.fix_invalid.html#numpy.ma.fix_invalid)(a[, mask, copy, fill_value]) | 返回带有无效数据的输入，并用填充值替换。
  [masked_equal](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_equal.html#numpy.ma.masked_equal)(x, value[, copy]) | 掩码一个等于给定值的数组。
  [masked_greater](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_greater.html#numpy.ma.masked_greater)(x, value[, copy]) | 掩码大于给定值的数组。
  [masked_greater_equal](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_greater_equal.html#numpy.ma.masked_greater_equal)(x, value[, copy]) | 掩码大于或等于给定值的数组。
  [masked_inside](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_inside.html#numpy.ma.masked_inside)(x, v1, v2[, copy]) | 在给定间隔内掩码数组。
  [masked_invalid](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_invalid.html#numpy.ma.masked_invalid)(a[, copy]) | 掩码出现无效值的数组（NaN或infs）。
  [masked_less](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_less.html#numpy.ma.masked_less)(x, value[, copy]) | 掩码小于给定值的数组。
  [masked_less_equal](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_less_equal.html#numpy.ma.masked_less_equal)(x, value[, copy]) | 掩码小于或等于给定值的数组。
  [masked_not_equal](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_not_equal.html#numpy.ma.masked_not_equal)(x, value[, copy]) | 掩码不等于给定值的数组。
  [masked_object](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_object.html#numpy.ma.masked_object)(x, value[, copy, shrink]) | 掩码数组x，其中数据完全等于值。
  [masked_outside](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_outside.html#numpy.ma.masked_outside)(x, v1, v2[, copy]) | 在给定间隔之外掩码数组。
  [masked_values](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_values.html#numpy.ma.masked_values)(x, value[, rtol, atol, copy, …]) | 掩码使用浮点相等。
  [masked_where](https://numpy.org/devdocs/reference/generated/numpy.ma.masked_where.html#numpy.ma.masked_where)(condition, a[, copy]) | 掩码满足条件的数组。

### 访问数据

可以通过多种方式访问​​掩码数组的基础数据：

- 通过[``data``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.data)属性。输出是数组的视图，作为[``numpy.ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)其子类之一，具体取决于掩码数组创建时基础数据的类型。
- 通过 [``__array__``](https://numpy.org/devdocs/reference/generated/numpy.ma.MaskedArray.__array__.html#numpy.ma.MaskedArray.__array__) 方法。然后输出为[``numpy.ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)。
- 通过直接将掩码数组视为 [``numpy.ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray) 或其子类之一 （这实际上是使用 [``data``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.data) 属性所做的）。
- 通过使用[``getdata``](https://numpy.org/devdocs/reference/generated/numpy.ma.getdata.html#numpy.ma.getdata)函数。

如果某些条目被标记为无效，则这些方法都不是完全令人满意的。作为一般规则，在需要不带任何掩码条目的数组表示的情况下，建议使用该[``filled``](https://numpy.org/devdocs/reference/generated/numpy.ma.filled.html#numpy.ma.filled)方法填充数组。

### 访问掩码

掩码数组的掩码可通过其[``mask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.mask)
属性访问。我们必须记住，``True``掩码中的条目表示
  *无效* 数据。

另一种可能性是使用[``getmask``](https://numpy.org/devdocs/reference/generated/numpy.ma.getmask.html#numpy.ma.getmask)和[``getmaskarray``](https://numpy.org/devdocs/reference/generated/numpy.ma.getmaskarray.html#numpy.ma.getmaskarray)
函数。``getmask(x)``输出``x``if 的掩码``x``是掩码数组，[``nomask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.nomask)否则输出特殊值。``getmaskarray(x)``
输出``x``if 的掩码``x``是掩码数组。如果``x``没有无效条目或不是掩码数组，则该函数输出一个``False``具有尽可能多的元素的布尔数组
 ``x``。

### 仅访问有效条目

要仅检索有效条目，我们可以使用掩码的反转作为索引。掩码的反转可以使用[``numpy.logical_not``](https://numpy.org/devdocs/reference/generated/numpy.logical_not.html#numpy.logical_not)函数计算，也可以
 使用``~``运算符计算：

``` python
>>> x = ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
>>> x[~x.mask]
masked_array(data = [1 4],
             mask = [False False],
       fill_value = 999999)
```

检索有效数据的另一种方法是使用该[``compressed``](https://numpy.org/devdocs/reference/generated/numpy.ma.compressed.html#numpy.ma.compressed)
方法，该方法返回一维[``ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)(或其子类之一，具体取决于[``baseclass``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.baseclass)
属性的值）：

``` python
>>> x.compressed()
array([1, 4])
```

请注意，输出[``compressed``](https://numpy.org/devdocs/reference/generated/numpy.ma.compressed.html#numpy.ma.compressed)始终为1D。

### 修改掩码

#### 掩码条目

将掩码数组的一个或多个特定条目标记为无效的推荐方法是[``masked``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.masked)为它们分配特殊值：

``` python
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

第二种可能性是[``mask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.mask)直接修改，但不鼓励这种用法。

::: tip 注意

使用简单的非结构化数据类型创建新的掩码数组时，掩码最初设置为特殊value[``nomask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.nomask)，该值大致对应于布尔值``False``。尝试设置元素
 [``nomask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.nomask)将失败并出现[``TypeError``](https://docs.python.org/dev/library/exceptions.html#TypeError)异常，因为布尔值不支持项目分配。

:::

通过分配掩码，可以立即掩码数组的所有条目``True``：

``` python
>>> x = ma.array([1, 2, 3], mask=[0, 0, 1])
>>> x.mask = True
>>> x
masked_array(data = [-- -- --],
             mask = [ True  True  True],
       fill_value = 999999)
```

最后，通过为掩码分配一系列布尔值，可以掩码和/或取消掩码特定条目：

``` python
>>> x = ma.array([1, 2, 3])
>>> x.mask = [0, 1, 0]
>>> x
masked_array(data = [1 -- 3],
             mask = [False  True False],
       fill_value = 999999)
```

#### 取消掩码条目

要取消掩码一个或多个特定条目，我们可以为它们分配一个或多个新的有效值：

``` python
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

::: tip 注意

如果掩码数组具有 *硬* 掩码，则通过直接分配取消掩码条目将无声地失败，如``hardmask``属性所示。引入此功能是为了防止覆盖掩码。要强制取消掩码数组具有硬掩码的条目，必须首先使用[``soften_mask``](https://numpy.org/devdocs/reference/generated/numpy.ma.soften_mask.html#numpy.ma.soften_mask)分配前的方法软化掩码。可以通过以下方式重新强化[``harden_mask``](https://numpy.org/devdocs/reference/generated/numpy.ma.harden_mask.html#numpy.ma.harden_mask)：

``` python
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

:::

要取消掩码掩码数组的所有掩码条目（假设掩码不是硬掩码），最简单的解决方案是将常量赋value[``nomask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.nomask)给掩码：

``` python
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

### 索引和切片

作为a [``MaskedArray``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray)的子类[``numpy.ndarray``](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray)，它继承了索引和切片的机制。

当访问没有命名字段的掩码数组的单个条目时，输出是标量（如果掩码的相应条目是
 ``False``）或特殊value[``masked``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.masked)(如果掩码的相应条目是``True``）：

``` python
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

如果掩码数组具有命名字段，则访问单个条目（``numpy.void``如果没有字段被掩码则返回对象），或者如果至少有一个字段被掩码，则返回
 与初始数组具有相同dtype的0d掩码数组。

``` python
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

访问切片时，输出是一个掩码数组，其
 [``data``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray.data)属性是原始数据的视图，其掩码是[``nomask``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.nomask)(如果原始数组中没有无效条目）或原始掩码的相应切片视图。视图是确保将掩模的任何修改传播到原始视图所必需的。

``` python
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

访问具有结构化数据类型的掩码数组的字段将返回一个[``MaskedArray``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray)。

### 掩码数组的操作

掩码数组支持算术和比较操作。尽可能不处理掩码数组的无效条目，这意味着操作之前和之后相应的``data``条目 *应该* 相同。

::: danger 警告

我们需要强调的是，这种行为可能不是系统性的，在某些情况下，掩码数据可能会受到操作的影响，因此用户不应该依赖这些数据保持不变。

:::

该[``numpy.ma``](#module-numpy.ma)模块附带了大多数ufunc的特定实现。
只要输入被掩码或超出有效域，具有有效域（例如[``log``](https://numpy.org/devdocs/reference/generated/numpy.log.html#numpy.log)或[``divide``](https://numpy.org/devdocs/reference/generated/numpy.divide.html#numpy.divide)）的一元和二元函数
 [``masked``](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#numpy.ma.masked)就会返回常量：

``` python
>>> ma.log([-1, 0, 1, 2])
masked_array(data = [-- -- 0.0 0.69314718056],
             mask = [ True  True False False],
       fill_value = 1e+20)
```

掩码数组也支持标准的numpy ufunc。然后输出是一个掩码数组。在掩码输入的任何地方都会掩码一元ufunc的结果。只要掩码了任何输入，就会掩码二进制ufunc的结果。如果ufunc还返回可选的上下文输出（包含ufunc名称，其参数及其域的3元素元组），则处理上下文，并且只要相应的输入超出有效性，任何地方都会掩码输出掩码数组的条目域：

``` python
>>> x = ma.array([-1, 1, 0, 2, 3], mask=[0, 0, 0, 0, 1])
>>> np.log(x)
masked_array(data = [-- -- 0.0 0.69314718056 --],
             mask = [ True  True False False  True],
       fill_value = 1e+20)
```
## 示例

### 具有表示缺失数据的给定值的数据

让我们考虑一个元素列表``x``，其中值为-9999。代表缺失的数据。我们希望计算数据的平均值和异常矢量（偏离平均值）：

``` python
>>> import numpy.ma as ma
>>> x = [0.,1.,-9999.,3.,4.]
>>> mx = ma.masked_values (x, -9999.)
>>> print mx.mean()
2.0
>>> print mx - mx.mean()
[-2.0 -1.0 -- 1.0 2.0]
>>> print mx.anom()
[-2.0 -1.0 -- 1.0 2.0]
```

### 填写缺失的数据

现在假设我们希望打印相同的数据，但缺失值被平均值替换。

``` python
>>> print mx.filled(mx.mean())
[ 0.  1.  2.  3.  4.]
```

### 数值运算

数值运算可以轻松执行，无需担心缺失值，除以零，负数的平方根等：

``` python
>>> import numpy as np, numpy.ma as ma
>>> x = ma.array([1., -1., 3., 4., 5., 6.], mask=[0,0,0,0,1,0])
>>> y = ma.array([1., 2., 0., 4., 5., 6.], mask=[0,0,0,0,0,1])
>>> print np.sqrt(x/y)
[1.0 -- -- 1.0 -- --]
```

输出的四个值是无效的：第一个值来自取负数的平方根，第二个来自除以零，以及最后两个输入被掩码的位置。

### 忽略极值

让我们考虑一个``d``介于0和1之间的随机浮点数组。我们希望计算值的平均值，``d``同时忽略范围之外的任何数据：``[0.1, 0.9]``

``` python
>>> print ma.masked_outside(d, 0.1, 0.9).mean()
```
