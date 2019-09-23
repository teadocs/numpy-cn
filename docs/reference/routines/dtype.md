# 数据类型相关

方法 | 描述
---|---
[can_cast](https://numpy.org/devdocs/reference/generated/numpy.can_cast.html#numpy.can_cast)(from_, to[, casting]) | 如果根据强制转换规则可以在数据类型之间进行强制转换，则返回True。
[promote_types](https://numpy.org/devdocs/reference/generated/numpy.promote_types.html#numpy.promote_types)(type1, type2) | 返回Type1和Type2都可以安全强制转换为的最小大小和最小标量种类的数据类型。
[min_scalar_type](https://numpy.org/devdocs/reference/generated/numpy.min_scalar_type.html#numpy.min_scalar_type)(a) | 对于标量a，返回具有最小大小和可以保存其值的最小标量种类的数据类型。
[result_type](https://numpy.org/devdocs/reference/generated/numpy.result_type.html#numpy.result_type)(*arrays_and_dtypes) | 返回将NumPy类型提升规则应用于参数而得到的类型。
[common_type](https://numpy.org/devdocs/reference/generated/numpy.common_type.html#numpy.common_type)(\*arrays) | 返回输入数组通用的标量类型。
[obj2sctype](https://numpy.org/devdocs/reference/generated/numpy.obj2sctype.html#numpy.obj2sctype)(rep[, default]) | 返回对象的Python类型的标量dtype或NumPy等效值。

## 创建数据类型

方法 | 描述
---|---
[dtype](https://numpy.org/devdocs/reference/generated/numpy.dtype.html#numpy.dtype)(obj[, align, copy]) | 创建数据类型对象。
[format_parser](https://numpy.org/devdocs/reference/generated/numpy.format_parser.html#numpy.format_parser)(formats, names, titles[, …]) | 类将格式、名称、标题说明转换为dtype。

## 数据类型信息

方法 | 描述
---|---
[finfo](https://numpy.org/devdocs/reference/generated/numpy.finfo.html#numpy.finfo)(dtype) | 浮点类型的机器限制。
[iinfo](https://numpy.org/devdocs/reference/generated/numpy.iinfo.html#numpy.iinfo)(type) | 整数类型的机器限制。
[MachAr](https://numpy.org/devdocs/reference/generated/numpy.MachAr.html#numpy.MachAr)([float_conv, int_conv, …]) | 诊断机器参数。

## 数据类型测试

方法 | 描述
---|---
[issctype](https://numpy.org/devdocs/reference/generated/numpy.issctype.html#numpy.issctype)(rep) | 确定给定对象是否表示标量数据类型。
[issubdtype](https://numpy.org/devdocs/reference/generated/numpy.issubdtype.html#numpy.issubdtype)(arg1, arg2) | 如果第一个参数是类型层次结构中较低/等于的类型码，则返回True。
[issubsctype](https://numpy.org/devdocs/reference/generated/numpy.issubsctype.html#numpy.issubsctype)(arg1, arg2) | 确定第一个参数是否是第二个参数的子类。
[issubclass_](https://numpy.org/devdocs/reference/generated/numpy.issubclass_.html#numpy.issubclass_)(arg1, arg2) | 确定一个类是否是第二个类的子类。
[find_common_type](https://numpy.org/devdocs/reference/generated/numpy.find_common_type.html#numpy.find_common_type)(array_types, scalar_types) | 按照标准强制规则确定常见类型。

## 杂项

方法 | 描述
---|---
[typename](https://numpy.org/devdocs/reference/generated/numpy.typename.html#numpy.typename)(char) | 返回给定数据类型代码的说明。
[sctype2char](https://numpy.org/devdocs/reference/generated/numpy.sctype2char.html#numpy.sctype2char)(sctype) | 返回标量dtype的字符串表示形式。
[mintypecode](https://numpy.org/devdocs/reference/generated/numpy.mintypecode.html#numpy.mintypecode)(typechars[, typeset, default]) | 返回给定类型可以安全强制转换到的最小大小类型的字符。
[maximum_sctype](https://numpy.org/devdocs/reference/generated/numpy.maximum_sctype.html#numpy.maximum_sctype)(t) | 返回与输入类型相同精度最高的标量类型。
