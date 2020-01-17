# 杂项(`Miscellaneous routines`)

## 性能调优

method | description
---|---
[setbufsize](https://numpy.org/devdocs/reference/generated/numpy.setbufsize.html#numpy.setbufsize)(size) | 设置用于ufuncs的缓冲区的大小。
[getbufsize](https://numpy.org/devdocs/reference/generated/numpy.getbufsize.html#numpy.getbufsize)() | 返回用于ufuncs的缓冲区的大小。

## 内存区间

method | description
---|---
[shares_memory](https://numpy.org/devdocs/reference/generated/numpy.shares_memory.html#numpy.shares_memory)(a, b[, max_work]) | 确定两个阵列是否共享内存
[may_share_memory](https://numpy.org/devdocs/reference/generated/numpy.may_share_memory.html#numpy.may_share_memory)(a, b[, max_work]) | 确定两个阵列是否可以共享内存
[byte_bounds](https://numpy.org/devdocs/reference/generated/numpy.byte_bounds.html#numpy.byte_bounds)(a) | 返回指向数组端点的指针。

## 数组Mixin

method | description
---|---
[lib.mixins.NDArrayOperatorsMixin](https://numpy.org/devdocs/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin) | Mixin使用__array_ufunc__定义所有运算符的特殊方法。

## NumPy版本比较

method | description
---|---
[lib.NumpyVersion](https://numpy.org/devdocs/reference/generated/numpy.lib.NumpyVersion.html#numpy.lib.NumpyVersion)(vstring) | 解析并比较numpy版本字符串。

## 效用

method | description
---|---
[get_include](https://numpy.org/devdocs/reference/generated/numpy.get_include.html#numpy.get_include)() | 返回包含NumPy * .h头文件的目录。
[deprecate](https://numpy.org/devdocs/reference/generated/numpy.deprecate.html#numpy.deprecate)(\*args, \*\*kwargs) | 发出DeprecationWarning，向old_name的文档字符串添加警告，重新绑定old_name .__ name__并返回新的函数对象。
[deprecate_with_doc](https://numpy.org/devdocs/reference/generated/numpy.deprecate_with_doc.html#numpy.deprecate_with_doc)(msg) |

## 类Matlab函数

method | description
---|---
[who](https://numpy.org/devdocs/reference/generated/numpy.who.html#numpy.who)([vardict]) | 在给定的字典中打印NumPy数组。
[disp](https://numpy.org/devdocs/reference/generated/numpy.disp.html#numpy.disp)(mesg[, device, linefeed]) | 在设备上显示消息。
