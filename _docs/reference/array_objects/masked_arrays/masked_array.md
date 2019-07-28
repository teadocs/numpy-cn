# MaskedArray类

## ``numpy.ma.MaskedArray`` 类

> ndarray的一个子类，用于处理缺少数据的数值数组。

MaskedArray的一个实例可以被认为是几个元素的组合：

- 数据，作为任何形状或数据类型（数据）的常规numpy.ndarray。
- 与数据具有相同形状的布尔掩码，其中True值表示数据的相应元素无效。 对于没有命名字段的数组，特殊值nomask也是可接受的，并表示没有数据无效。
- fill_value，可用于替换无效条目以返回标准numpy.ndarray的值。

## 掩码数组的属性和属性

另见：

> 数组属性

### ``MaskedArray.data``

返回基础数据，作为掩码数组的视图。如果底层数据是numpy.ndarray的子类，则返回它。

```python
>>> x = ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
>>> x.data
matrix([[1, 2],
        [3, 4]])
```

可以通过``baseclass``属性访问数据类型。

### ``MaskedArray.mask``

返回底层掩码，作为与数据具有相同形状和结构的数组，但所有字段都是原子布尔值。 值 ``True`` 表示无效条目。

### ``MaskedArray.recordmask``

如果没有命名字段，则返回数组的掩码。 对于结构化数组，返回一个布尔的ndarray，其中如果所有字段都被屏蔽，则条目为“True”。否则为“False”。

```python
>>> x = ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
...        dtype=[('a', int), ('b', int)])
>>> x.recordmask
array([False, False,  True, False, False])
```

### ``MaskedArray.fill_value``

返回用于填充掩码数组的无效条目的值。该值可以是标量（如果掩码数组没有命名字段），也可以是具有与掩码数组相同的``dtype``的0-D ndarray（如果它具有命名字段）。

默认填充值取决于数组的数据类型：

数据类型 | 默认值
---|---
bool | True
int | 999999
float | 1.e20
complex | 1.e20+0j
object | ‘?’
string | ‘N/A’

### ``MaskedArray.baseclass``

返回基础数据的类。

```python
>>> x =  ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 0], [1, 0]])
>>> x.baseclass
<class 'numpy.matrixlib.defmatrix.matrix'>
```

### ``MaskedArray.sharedmask``

返回数组掩码是否在多个掩码数组之间共享。 如果是这种情况，对一个阵列的掩码的任何修改都将传播到其他阵列。

### ``MaskedArray.hardmask``

返回蒙版是硬（True）还是软（False）。 当掩码很难时，掩码条目不能被掩盖。

由于``MaskedArray``是``ndarray``的子类，因此掩码数组也继承了ndarray实例的所有属性和属性。

方法 | 描述
---|---
MaskedArray.base | 如果内存来自某个其他对象，则为基础对象。
MaskedArray.ctypes | 一个简化数组与ctypes模块交互的对象。
MaskedArray.dtype | 数组元素的数据类型。
MaskedArray.flags | 有关数组内存布局的信息。
MaskedArray.itemsize | 一个数组元素的长度，以字节为单位
MaskedArray.nbytes | 数组元素消耗的总字节数。
MaskedArray.ndim | 数组维数。
MaskedArray.shape | 数组维度的元组。
MaskedArray.size | 数组中的元素数。
MaskedArray.strides | 遍历数组时，每个维度中的字节元组。
MaskedArray.imag | 虚构的一部分。
MaskedArray.real | 真实的部分
MaskedArray.flat | 数组的平面版本。
MaskedArray.__array_priority__ | -