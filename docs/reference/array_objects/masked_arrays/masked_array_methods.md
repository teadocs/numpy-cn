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
- MaskedArray.__add__(other)	Add self to other, and return a new masked array.
- MaskedArray.__radd__(other)	Add other to self, and return a new masked array.
- MaskedArray.__sub__(other)	Subtract other from self, and return a new masked array.
- MaskedArray.__rsub__(other)	Subtract self from other, and return a new masked array.
- MaskedArray.__mul__(other)	Multiply self by other, and return a new masked array.
- MaskedArray.__rmul__(other)	Multiply other by self, and return a new masked array.
- MaskedArray.__div__(other)	Divide other into self, and return a new masked array.
- MaskedArray.__rdiv__	
- MaskedArray.__truediv__(other)	Divide other into self, and return a new masked array.
- MaskedArray.__rtruediv__(other)	Divide self into other, and return a new masked array.
- MaskedArray.__floordiv__(other)	Divide other into self, and return a new masked array.
- MaskedArray.__rfloordiv__(other)	Divide self into other, and return a new masked array.
- MaskedArray.__mod__($self, value, /)	Return self%value.
- MaskedArray.__rmod__($self, value, /)	Return value%self.
- MaskedArray.__divmod__($self, value, /)	Return divmod(self, value).
- MaskedArray.__rdivmod__($self, value, /)	Return divmod(value, self).
- MaskedArray.__pow__(other)	Raise self to the power other, masking the potential NaNs/Infs
- MaskedArray.__rpow__(other)	Raise other to the power self, masking the potential NaNs/Infs
- MaskedArray.__lshift__($self, value, /)	Return self<<value.
- MaskedArray.__rlshift__($self, value, /)	Return value<<self.
- MaskedArray.__rshift__($self, value, /)	Return self>>value.
- MaskedArray.__rrshift__($self, value, /)	Return value>>self.
- MaskedArray.__and__($self, value, /)	Return self&value.
- MaskedArray.__rand__($self, value, /)	Return value&self.
- MaskedArray.__or__($self, value, /)	Return self|value.
- MaskedArray.__ror__($self, value, /)	Return value|self.
- MaskedArray.__xor__($self, value, /)	Return self^value.
- MaskedArray.__rxor__($self, value, /)	Return value^self.

### Arithmetic, in-place:

- MaskedArray.__iadd__(other)	Add other to self in-place.
- MaskedArray.__isub__(other)	Subtract other from self in-place.
- MaskedArray.__imul__(other)	Multiply self by other in-place.
- MaskedArray.__idiv__(other)	Divide self by other in-place.
- MaskedArray.__itruediv__(other)	True divide self by other in-place.
- MaskedArray.__ifloordiv__(other)	Floor divide self by other in-place.
- MaskedArray.__imod__($self, value, /)	Return self%=value.
- MaskedArray.__ipow__(other)	Raise self to the power other, in place.
- MaskedArray.__ilshift__($self, value, /)	Return self<<=value.
- MaskedArray.__irshift__($self, value, /)	Return self>>=value.
- MaskedArray.__iand__($self, value, /)	Return self&=value.
- MaskedArray.__ior__($self, value, /)	Return self|=value.
- MaskedArray.__ixor__($self, value, /)	Return self^=value.

## Representation

- MaskedArray.__repr__()	Literal string representation.
- MaskedArray.__str__()	Return str(self).
- MaskedArray.ids()	Return the addresses of the data and mask areas.
- MaskedArray.iscontiguous()	Return a boolean indicating whether the data is contiguous.

## Special methods

For standard library functions:

- MaskedArray.__copy__()	Used if copy.copy is called on an array.
- MaskedArray.__deepcopy__(memo, /)	Used if copy.deepcopy is called on an array.
- MaskedArray.__getstate__()	Return the internal state of the masked array, for pickling purposes.
- MaskedArray.__reduce__()	Return a 3-tuple for pickling a MaskedArray.
- MaskedArray.__setstate__(state)	Restore the internal state of the masked array, for pickling purposes.

Basic customization:

- MaskedArray.__new__([data, mask, dtype, …])	Create a new masked array from scratch.
- MaskedArray.__array__(|dtype)	Returns either a new reference to self if dtype is not given or a new array of provided data type if dtype is different from the current dtype of the array.
- MaskedArray.__array_wrap__(obj[, context])	Special hook for ufuncs.

Container customization: (see Indexing)

- MaskedArray.__len__($self, /)	Return len(self).
- MaskedArray.__getitem__(indx)	x.__getitem__(y) <==> x[y]
- MaskedArray.__setitem__(indx, value)	x.__setitem__(i, y) <==> x[i]=y
- MaskedArray.__delitem__($self, key, /)	Delete self[key].
- MaskedArray.__contains__($self, key, /)	Return key in self.

## Specific methods

### Handling the mask

The following methods can be used to access information about the mask or to manipulate the mask.

- ``MaskedArray.__setmask__``(mask[, copy])	Set the mask.
- ``MaskedArray.harden_mask``()	Force the mask to hard.
- ``MaskedArray.soften_mask``()	Force the mask to soft.
- ``MaskedArray.unshare_mask``()	Copy the mask and set the sharedmask flag to False.
- ``MaskedArray.shrink_mask``()	Reduce a mask to nomask when possible.

### Handling the fill_value

- ``MaskedArray.get_fill_value``()	Return the filling value of the masked array.
- ``MaskedArray.set_fill_value``([value])	Set the filling value of the masked array.

### Counting the missing elements

- ``MaskedArray.count``([axis, keepdims])	Count the non-masked elements of the array along the given axis.