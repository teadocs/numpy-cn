# MaskedArray的方法

另见：

> Array methods

## 转换

- MaskedArray.__float__() 转换为浮点类型。
- MaskedArray.__hex__	
- MaskedArray.__int__()	转换为int类型。
- MaskedArray.__long__()	转换为long类型.
- MaskedArray.__oct__	
- MaskedArray.view([dtype, type])	具有相同数据的数组的新视图。
- MaskedArray.astype(newtype)	返回MaskedArray强制转换为给定newtype的副本。
- MaskedArray.byteswap([inplace])	交换数组元素的字节。
- MaskedArray.compressed()	将所有未屏蔽的数据作为1-D数组返回。
- MaskedArray.filled([fill_value])	返回self的副本，其中屏蔽值填充给定值。
- MaskedArray.tofile(fid[, sep, format]) 将掩码数组以二进制格式保存到文件中。
- MaskedArray.toflex()	将掩码数组转换为灵活类型数组。
- MaskedArray.tolist([fill_value]) 将掩码数组的数据部分作为分层Python列表返回。

- MaskedArray.torecords() 将掩码数组转换为灵活类型数组。
- MaskedArray.tostring([fill_value, order])	此函数是tobytes的兼容性别名。
- MaskedArray.tobytes([fill_value, order])	将数组数据作为包含数组中原始字节的字符串返回。

## 形状操纵

对于重新整形，调整大小和转置，单个元组参数可以用n个整数替换，这些整数将被解释为n元组。

- MaskedArray.flatten([order])	将折叠的数组的副本返回到一个维度。
- MaskedArray.ravel([order])	作为视图返回self的1D版本。
- MaskedArray.reshape(*s, **kwargs)	为数组赋予新形状而不更改其数据。
- MaskedArray.resize(newshape[, refcheck, order]) 
- MaskedArray.squeeze([axis])	从a的形状中删除一维条目。
- MaskedArray.swapaxes(axis1, axis2)	返回数组的视图，其中axis1和axis2互换。
- MaskedArray.transpose(*axes)	返回轴转置的数组视图。
- MaskedArray.T	

## 项目选择和操作

对于采用axis关键字的数组方法，默认为None。 如果axis为None，则将数组视为1-D数组。 轴的任何其他值表示操作应继续进行的维度。

- MaskedArray.argmax([axis, fill_value, out])	返回沿给定轴的最大值的索引数组。
- MaskedArray.argmin([axis, fill_value, out])	将索引数组返回到给定轴的最小值。
- MaskedArray.argsort([axis, kind, order, …])	返回一个索引的ndarray，它沿着指定的轴对数组进行排序。
- MaskedArray.choose(choices[, out, mode])	使用索引数组从一组选项中构造新数组。
- MaskedArray.compress(condition[, axis, out])	返回where条件为True。
- MaskedArray.diagonal([offset, axis1, axis2])	返回指定的对角线。
- MaskedArray.fill(value)	使用标量值填充数组。
- MaskedArray.item(*args)	将数组元素复制到标准Python标量并返回它。
- MaskedArray.nonzero()	返回非零的未屏蔽元素的索引。
- MaskedArray.put(indices, values[, mode])	将存储索引位置设置为相应的值。
- MaskedArray.repeat(repeats[, axis])	重复数组的元素。
- MaskedArray.searchsorted(v[, side, sorter])	查找应在其中插入v的元素以维护顺序的索引。
- MaskedArray.sort([axis, kind, order, …]) 就地对阵列进行排序
- MaskedArray.take(indices[, axis, out, mode])	

## 腌制和拷贝

- MaskedArray.copy([order])	返回数组的拷贝。
- MaskedArray.dump(file)	将数组的腌制转储到指定的文件。
- MaskedArray.dumps()	以字符串形式返回数组的腌制。

## Calculations

- MaskedArray.all([axis, out, keepdims])	如果所有元素都计算为True，则返回True。
- MaskedArray.anom([axis, dtype])	计算沿给定轴的异常（与算术平均值的偏差）。
- MaskedArray.any([axis, out, keepdims])	如果求值的任何元素为True，则返回True。
- MaskedArray.clip([min, max, out])	返回其值限制为[min, max]的数组。
- MaskedArray.conj()	复合共轭所有元素。
- MaskedArray.conjugate()	以元素方式返回复共轭。
- MaskedArray.cumprod([axis, dtype, out]) 返回给定轴上的数组元素的累积乘积。
- MaskedArray.cumsum([axis, dtype, out]) 返回给定轴上的数组元素的累积和。
- MaskedArray.max([axis, out, fill_value, …]) 沿给定轴返回最大值。
- MaskedArray.mean([axis, dtype, out, keepdims]) 返回给定轴上数组元素的平均值。
- MaskedArray.min([axis, out, fill_value, …])	沿给定轴返回最小值。
- MaskedArray.prod([axis, dtype, out, keepdims]) 返回给定轴上的数组元素的乘积。
- MaskedArray.product([axis, dtype, out, keepdims])	返回给定轴上的数组元素的乘积。
- MaskedArray.ptp([axis, out, fill_value])	沿着给定的维数返回(最大值 - 最小值)。
- MaskedArray.round([decimals, out])	返回舍入到给定小数位数的每个元素。
- MaskedArray.std([axis, dtype, out, ddof, …])	返回给定轴的数组元素的标准偏差。
- MaskedArray.sum([axis, dtype, out, keepdims])	返回给定轴上的数组元素的总和。
- MaskedArray.trace([offset, axis1, axis2, …])	返回数组对角线的总和。
- MaskedArray.var([axis, dtype, out, ddof, …])	计算沿指定轴的方差。

## 算术和比较运算

### 比较运算符：

- MaskedArray.__lt__($self, value, /)	返回 self<value.
- MaskedArray.__le__($self, value, /)	返回 self<=value.
- MaskedArray.__gt__($self, value, /)	返回 self>value.
- MaskedArray.__ge__($self, value, /)	返回 self>=value.
- MaskedArray.__eq__(other)	检查其他是否等于self elementwise。
- MaskedArray.__ne__(other)	检查其他元素是否与元素相等。

### 数组的真假值(``bool``):

- MaskedArray.__nonzero__	

### 算术运算符:

- MaskedArray.__abs__(self)	
- MaskedArray.__add__(other)	将self添加到其他，并返回一个新的掩码数组。
- MaskedArray.__radd__(other)	将其他内容添加到self，并返回一个新的掩码数组。
- MaskedArray.__sub__(other)	从self中减去其他值，并返回一个新的掩码数组。
- MaskedArray.__rsub__(other)	从其他减去self，并返回一个新的掩码数组。
- MaskedArray.__mul__(other)	由其他人乘以自我，并返回一个新的掩码数组。
- MaskedArray.__rmul__(other)	将其他自身相乘，并返回一个新的掩码数组。
- MaskedArray.__div__(other)	将其他分为self，并返回一个新的掩码数组。
- MaskedArray.__rdiv__	
- MaskedArray.__truediv__(other)	将其他分为self，并返回一个新的掩码数组。
- MaskedArray.__rtruediv__(other)	将self划分为其他，并返回一个新的掩码数组。
- MaskedArray.__floordiv__(other)	将其他分为self，并返回一个新的掩码数组。
- MaskedArray.__rfloordiv__(other)	将self划分为其他，并返回一个新的掩码数组。
- MaskedArray.__mod__($self, value, /)	返回 self％value。
- MaskedArray.__rmod__($self, value, /)	返回 value%self。
- MaskedArray.__divmod__($self, value, /)	返回 divmod(self, value)。
- MaskedArray.__rdivmod__($self, value, /)	返回 divmod(value, self)。
- MaskedArray.__pow__(other)	将自我提升到其他权力，掩盖潜在的NaNs / Infs。
- MaskedArray.__rpow__(other)	提升其他权力自我，掩盖潜在的NaNs / Infs。
- MaskedArray.__lshift__($self, value, /)	返回 self<<value.
- MaskedArray.__rlshift__($self, value, /)	返回 value<<self.
- MaskedArray.__rshift__($self, value, /)	返回 self>>value.
- MaskedArray.__rrshift__($self, value, /)	返回 value>>self.
- MaskedArray.__and__($self, value, /)	返回 self&value.
- MaskedArray.__rand__($self, value, /)	返回 value&self.
- MaskedArray.__or__($self, value, /)	返回 self|value.
- MaskedArray.__ror__($self, value, /)	返回 value|self.
- MaskedArray.__xor__($self, value, /)	返回 self^value.
- MaskedArray.__rxor__($self, value, /)	返回 value^self.

### 就地算术运算符：

- MaskedArray.__iadd__(other)	将其他内容添加到就地。
- MaskedArray.__isub__(other)	就地从self中减去其他的。
- MaskedArray.__imul__(other)	就地的从slef中乘以其他。
- MaskedArray.__idiv__(other)	就地从self中划分其他。
- MaskedArray.__itruediv__(other)	就地的从self中划分True值。
- MaskedArray.__ifloordiv__(other)	就地的从self中划分Floor divide self by other in-place.
- MaskedArray.__imod__($self, value, /)	返回 self%=value.
- MaskedArray.__ipow__(other)  就地将自self提升到其他权限。
- MaskedArray.__ilshift__($self, value, /)	返回 self<<=value.
- MaskedArray.__irshift__($self, value, /)	返回 self>>=value.
- MaskedArray.__iand__($self, value, /)	返回 self&=value.
- MaskedArray.__ior__($self, value, /)	返回 self|=value.
- MaskedArray.__ixor__($self, value, /)	返回 self^=value.

## 表示

- MaskedArray.__repr__() 文字字符串表示。
- MaskedArray.__str__()	返回 str(self).
- MaskedArray.ids()	返回数据的地址和掩码区域。
- MaskedArray.iscontiguous() 返回一个布尔值，指示数据是否连续。

## 特殊方法

对于标准库函数：

- MaskedArray.__copy__()	如果在数组上调用copy.copy，则使用此方法。
- MaskedArray.__deepcopy__(memo, /) 如果在数组上调用copy.deepcopy，则使用此方法
- MaskedArray.__getstate__()	返回被掩码数组的内部状态，用于腌制目的。
- MaskedArray.__reduce__()	返回一个3元组来腌制MaskedArray。
- MaskedArray.__setstate__(state)	恢复被遮罩数组的内部状态，用于腌制目的。

基本定制：

- MaskedArray.__new__([data, mask, dtype, …])	从头开始创建一个新的掩码数组。
- MaskedArray.__array__(|dtype)	如果没有给出dtype，则返回对self的新引用;如果dtype与数组的当前dtype不同，则返回提供的数据类型的新数组。
- MaskedArray.__array_wrap__(obj[, context])	ufuncs的特殊钩子。

Container customization: (see Indexing)

- MaskedArray.__len__($self, /)	Return len(self).
- MaskedArray.__getitem__(indx)	x.__getitem__(y) <==> x[y]
- MaskedArray.__setitem__(indx, value)	x.__setitem__(i, y) <==> x[i]=y
- MaskedArray.__delitem__($self, key, /)	Delete self[key].
- MaskedArray.__contains__($self, key, /)	Return key in self.

## 特殊方法

### 掩码处理方法

可以使用以下方法访问有关掩码的信息或操作掩码。

- ``MaskedArray.__setmask__``(mask[, copy])	设置掩码.
- ``MaskedArray.harden_mask``()	强制掩码变坚硬。
- ``MaskedArray.soften_mask``()	强制掩码变柔软.
- ``MaskedArray.unshare_mask``() 复制掩码并将sharedmask标志设置为False。
- ``MaskedArray.shrink_mask``()	尽可能将掩码减少到nomask。

### 处理fill_value

- ``MaskedArray.get_fill_value``()	返回掩码数组的填充值。
- ``MaskedArray.set_fill_value``([value])	设置掩码数组的填充值。

### 计算缺失的元素

- ``MaskedArray.count``([axis, keepdims])	沿给定轴计算数组的非掩码元素。