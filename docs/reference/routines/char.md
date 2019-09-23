# 字符串操作

``numpy.char`` 模块为类型 ``numpy.string_`` 或的数组提供了一组向量化的字符串操作``numpy.unicode_``。
它们全部基于Python标准库中的字符串方法。

## 字符串操作

方法 | 描述
---|---
[add](https://numpy.org/devdocs/reference/generated/numpy.char.add.html#numpy.char.add)(x1, x2) | 返回两个str或unicode数组的按元素的字符串连接。
[multiply](https://numpy.org/devdocs/reference/generated/numpy.char.multiply.html#numpy.char.multiply)(a, i) | 返回（a * i），即按元素方式的字符串多重连接。
[mod](https://numpy.org/devdocs/reference/generated/numpy.char.mod.html#numpy.char.mod)(a, values) | 返回（a％i），这是Python 2.6之前的字符串格式（迭代），针对一对str或unicode的array_likes元素。
[capitalize](https://numpy.org/devdocs/reference/generated/numpy.char.capitalize.html#numpy.char.capitalize)(a) | 返回的拷贝一个只有资本的每个元素的第一个字符。
[center](https://numpy.org/devdocs/reference/generated/numpy.char.center.html#numpy.char.center)(a, width[, fillchar]) | 返回的拷贝一与在长度的字符串居中其元素宽度。
[decode](https://numpy.org/devdocs/reference/generated/numpy.char.decode.html#numpy.char.decode)(a[, encoding, errors]) | 按元素调用str.decode。
[encode](https://numpy.org/devdocs/reference/generated/numpy.char.encode.html#numpy.char.encode)(a[, encoding, errors]) | 逐元素调用str.encode。
[expandtabs](https://numpy.org/devdocs/reference/generated/numpy.char.expandtabs.html#numpy.char.expandtabs)(a[, tabsize]) | 返回每个字符串元素的副本，其中所有制表符都被一个或多个空格替换。
[join](https://numpy.org/devdocs/reference/generated/numpy.char.join.html#numpy.char.join)(sep, seq) | 返回一个字符串，该字符串是序列seq中字符串的串联。
[ljust](https://numpy.org/devdocs/reference/generated/numpy.char.ljust.html#numpy.char.ljust)(a, width[, fillchar]) | 返回与元素的数组一个左对齐的长度的字符串宽度。
[lower](https://numpy.org/devdocs/reference/generated/numpy.char.lower.html#numpy.char.lower)(a) | 返回一个将元素转换为小写的数组。
[lstrip](https://numpy.org/devdocs/reference/generated/numpy.char.lstrip.html#numpy.char.lstrip)(a[, chars]) | 对于每一个元素一个，去除了主角返回副本。
[partition](https://numpy.org/devdocs/reference/generated/numpy.char.partition.html#numpy.char.partition)(a) | 将每个元素划分为大约sep。
[replace](https://numpy.org/devdocs/reference/generated/numpy.char.replace.html#numpy.char.replace)e(a, old, new[, count]) | 对于每一个元素一个，返回字符串的副本串中出现的所有旧的换成新的。
[rjust](https://numpy.org/devdocs/reference/generated/numpy.char.rjust.html#numpy.char.rjust)(a, width[, fillchar]) | 返回与元素的数组一个右对齐的长度的字符串宽度。
[rpartition](https://numpy.org/devdocs/reference/generated/numpy.char.rpartition.html#numpy.char.rpartition)(a, sep) | 在最右边的分隔符周围分割（分割）每个元素。
[rsplit](https://numpy.org/devdocs/reference/generated/numpy.char.rsplit.html#numpy.char.rsplit)(a[, sep, maxsplit]) | 在每个元件一个，返回的词的列表字符串中，使用月作为分隔符串。
[rstrip](https://numpy.org/devdocs/reference/generated/numpy.char.rstrip.html#numpy.char.rstrip)(a[, chars]) | 在每个元件一个，返回与除去尾部字符的副本。
[split](https://numpy.org/devdocs/reference/generated/numpy.char.split.html#numpy.char.split)(a[, sep, maxsplit]) | 在每个元件一个，返回的词的列表字符串中，使用月作为分隔符串。
[splitlines](https://numpy.org/devdocs/reference/generated/numpy.char.splitlines.html#numpy.char.splitlines)(a[, keepends]) | 在每个元件一个，返回行的列表中的元素，在断裂线边界。
[strip](https://numpy.org/devdocs/reference/generated/numpy.char.strip.html#numpy.char.strip)(a[, chars]) | 在每个元件一个，其中移除了前缘和后字符返回副本。
[swapcase](https://numpy.org/devdocs/reference/generated/numpy.char.swapcase.html#numpy.char.swapcase)(a) | 按元素返回字符串的副本，该字符串的大写字符转换为小写，反之亦然。
[title](https://numpy.org/devdocs/reference/generated/numpy.char.title.html#numpy.char.title)(a) | 返回字符串或unicode的逐元素标题区分大小写的版本。
[translate](https://numpy.org/devdocs/reference/generated/numpy.char.translate.html#numpy.char.translate)(a, table[, deletechars]) | 对于每一个元素一个，返回其中可选的参数中出现的所有字符的字符串拷贝deletechars被删除，而剩余的字符已经通过给定的转换表映射。
[upper](https://numpy.org/devdocs/reference/generated/numpy.char.upper.html#numpy.char.upper)(a) | 返回一个数组，其中元素转换为大写。
[zfill](https://numpy.org/devdocs/reference/generated/numpy.char.zfill.html#numpy.char.zfill)(a, width) | 返回用零填充的数字字符串

## 比较

与标准numpy比较运算符不同，*char* 模块中的运算符在执行比较之前会删除结尾的空白字符。

方法 | 描述
---|---
[equal](https://numpy.org/devdocs/reference/generated/numpy.char.equal.html#numpy.char.equal)(x1, x2)	| 按元素返回（x1 == x2）。
[not_equal](https://numpy.org/devdocs/reference/generated/numpy.char.not_equal.html#numpy.char.not_equal)(x1, x2)	| 按元素返回（x1！= x2）。
[greater_equal](https://numpy.org/devdocs/reference/generated/numpy.char.greater_equal.html#numpy.char.greater_equal)(x1, x2) | 按元素返回（x1> = x2）。
[less_equal](https://numpy.org/devdocs/reference/generated/numpy.char.less_equal.html#numpy.char.less_equal)(x1, x2) | 按元素返回（x1 <= x2）。
[greater](https://numpy.org/devdocs/reference/generated/numpy.char.greater.html#numpy.char.greater)(x1, x2) | 按元素返回（x1> x2）。
[less](https://numpy.org/devdocs/reference/generated/numpy.char.less.html#numpy.char.less)(x1, x2) | 按元素返回（x1 < x2）。
[compare_chararrays](https://numpy.org/devdocs/reference/generated/numpy.char.compare_chararrays.html#numpy.char.compare_chararrays)(a, b, cmp_op, rstrip) | 使用cmp_op指定的比较运算符对两个字符串数组执行逐元素比较。

## 字符串信息

方法 | 描述
---|---
[count](https://numpy.org/devdocs/reference/generated/numpy.char.count.html#numpy.char.count)(a, sub[, start, end]) | 返回一个数组，该数组的子字符串sub的不重叠出现次数在[ start，end ] 范围内。
[endswith](https://numpy.org/devdocs/reference/generated/numpy.char.endswith.html#numpy.char.endswith)(a, suffix[, start, end]) | 返回一个布尔值阵列，其是真正的，其中在字符串元素一个结尾的后缀，否则假。
[find](https://numpy.org/devdocs/reference/generated/numpy.char.find.html#numpy.char.find)(a, sub[, start, end]) | 对于每个元素，返回找到子字符串sub的字符串中的最低索引。
[index](https://numpy.org/devdocs/reference/generated/numpy.char.index.html#numpy.char.index)(a, sub[, start, end]) | 与相似find，但是在未找到子字符串时引发ValueError。
[isalpha](https://numpy.org/devdocs/reference/generated/numpy.char.isalpha.html#numpy.char.isalpha)(a) | 如果字符串中的所有字符都是字母并且至少包含一个字符，则对每个元素返回true，否则返回false。
[isalnum](https://numpy.org/devdocs/reference/generated/numpy.char.isalnum.html#numpy.char.isalnum)(a) | 如果字符串中的所有字符都是字母数字并且至少包含一个字符，则对每个元素返回true，否则返回false。
[isdecimal](https://numpy.org/devdocs/reference/generated/numpy.char.isdecimal.html#numpy.char.isdecimal)(a) | 对于每个元素，如果元素中只有十进制字符，则返回True。
[isdigit](https://numpy.org/devdocs/reference/generated/numpy.char.isdigit.html#numpy.char.isdigit)(a) | 如果字符串中的所有字符都是数字并且至少有一个字符，则对每个元素返回true，否则返回false。
[islower](https://numpy.org/devdocs/reference/generated/numpy.char.islower.html#numpy.char.islower)(a) | 如果字符串中所有大小写的字符均为小写且至少有一个大小写的字符，则对每个元素返回true，否则返回false。
[isnumeric](https://numpy.org/devdocs/reference/generated/numpy.char.isnumeric.html#numpy.char.isnumeric)(a) | 对于每个元素，如果元素中仅包含数字字符，则返回True。
[isspace](https://numpy.org/devdocs/reference/generated/numpy.char.isspace.html#numpy.char.isspace)(a) | 如果字符串中只有空格字符且至少有一个字符，则对每个元素返回true，否则返回false。
[istitle](https://numpy.org/devdocs/reference/generated/numpy.char.istitle.html#numpy.char.istitle)(a) | 如果该元素是一个带标题的字符串并且至少包含一个字符，则为每个元素返回true，否则返回false。
[isupper](https://numpy.org/devdocs/reference/generated/numpy.char.isupper.html#numpy.char.isupper)(a) | 如果字符串中所有大小写的字符均为大写且至少有一个字符，则为每个元素返回true，否则返回false。
[rfind](https://numpy.org/devdocs/reference/generated/numpy.char.rfind.html#numpy.char.rfind)(a, sub[, start, end]) | 在每个元件一个，在字符串中返回指数最高，其中子子被发现，使得子包含[内开始，结束。
[rindex](https://numpy.org/devdocs/reference/generated/numpy.char.rindex.html#numpy.char.rindex)(a, sub[, start, end]) | 与相似rfind，但是在未找到子字符串sub时引发ValueError。
[startswith](https://numpy.org/devdocs/reference/generated/numpy.char.startswith.html#numpy.char.startswith)（一个，前缀[，开始，结束]） | 返回一个布尔数组是真正的地方在字符串元素一个开头前缀，否则假。
[str_len](https://numpy.org/devdocs/reference/generated/numpy.char.str_len.html#numpy.char.str_len)(a) | 按元素返回len（a）。

## 便利构造函数

方法 | 描述
---|---
[array](https://numpy.org/devdocs/reference/generated/numpy.char.array.html#numpy.char.array)(obj[, itemsize, copy, unicode, order]) | 创建一个 [chararray](https://numpy.org/devdocs/reference/generated/numpy.char.chararray.html#numpy.char.chararray)。
[asarray](https://numpy.org/devdocs/reference/generated/numpy.char.asarray.html#numpy.char.asarray)(obj[, itemsize, unicode, order]) | 将输入转换为 [chararray](https://numpy.org/devdocs/reference/generated/numpy.char.chararray.html#numpy.char.chararray) ，仅在必要时复制数据。
[chararray](https://numpy.org/devdocs/reference/generated/numpy.char.chararray.html#numpy.char.chararray)(shape[, itemsize, unicode, …]) | 提供有关字符串和Unicode值数组的便捷视图。
