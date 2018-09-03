# 数据类型操作

- can_cast(from_, to[, casting])	如果根据强制转换规则在数据类型之间进行转换，则返回True。
- promote_types(type1, type2)	返回具有最小大小和最小标量种类的数据类型，可以安全地转换type1和type2。
- min_scalar_type(a)	对于标量a，返回具有最小大小和最小标量类型的数据类型，该类型可以保存其值。
- result_type(*arrays_and_dtypes)	返回将NumPy类型提升规则应用于参数所产生的类型。
- common_type(*arrays)	返回输入数组所共有的标量类型。
- obj2sctype(rep[, default])	返回对象的标量dtype或与Python类型等效的NumPy。

## 创建数据类型

- dtype(obj[, align, copy])	创建数据类型对象。
- format_parser(formats, names, titles[, …])	类将格式、名称、标题说明转换为dtype。

## 数据类型信息

- finfo(dtype)	浮点类型的机器限制。
- iinfo(type)	整数类型的机器限制。
- MachAr([float_conv, int_conv, …])	诊断机器参数。

## 数据类型测试

- issctype(rep)	确定给定对象是否表示标量数据类型。
- issubdtype(arg1, arg2)	如果第一个参数是类型层次结构中的类型码较低/相等的类型，则返回True。
- issubsctype(arg1, arg2)	确定第一个参数是否是第二个参数的子类。
- issubclass_(arg1, arg2)	确定一个类是否是第二类的子类。
- find_common_type(array_types, scalar_types)	根据标准强制规则确定常见类型。

## 杂项

- typename(char)	返回给定数据类型代码的描述。
- sctype2char(sctype)	返回标量dtype的字符串表示形式。
- mintypecode(typechars[, typeset, default])	返回给定类型可以安全转换到的最小大小类型的字符。