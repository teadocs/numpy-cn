# 字符串操作

此模块为numpy.string_或numpy.unicode_类型的数组提供一组矢量化字符串操作。所有这些都基于Python标准库中的字符串方法。

## String operations

- add(x1, x2)	返回两个str或unicode数组的逐元素字符串连接。
- multiply(a, i)  返回(a * i)， 即字符串多个连接，逐个元素。
- mod(a, values) 返回（a％i），即Python之前的2.6字符串格式化（插值），对于str或unicode等一对数组的元素。
- capitalize(a)	返回a的副本，其中只有每个元素的第一个字符大写。
- center(a, width[, fillchar])	返回a的副本，其元素以长度为一的字符串为中心。
- decode(a[, encoding, errors])	逐元素方式调用str.decode。
- encode(a[, encoding, errors])	逐元素方式调用str.encode。
- join(sep, seq)	返回一个字符串，它是序列seq中字符串的串联。
- ljust(a, width[, fillchar])	返回一个数组，其中包含左对齐的元素，长度为宽度的字符串。
- lower(a)	返回一个数组，其元素转换为小写。  
- lstrip(a[, chars])	对于a中的每个元素，返回删除了前导字符的副本。
- partition(a, sep)	将每个元素分成一个周围的sep。
- replace(a, old, new[, count])	对于a中的每个元素，返回一个字符串的副本，其中所有出现的substring old都替换为new。
- rjust(a, width[, fillchar])	返回一个数组，其中右对齐元素的长度为宽度。
- rpartition(a, sep)	对最右边的分隔符周围的每个元素进行分区（拆分）。
- rsplit(a[, sep, maxsplit])	对于a中的每个元素，使用sep作为分隔符字符串，返回字符串中单词的列表。
- rstrip(a[, chars])	对于a中的每个元素，返回一个删除了尾随字符的副本。
- split(a[, sep, maxsplit])	对于a中的每个元素，使用sep作为分隔符字符串，返回字符串中单词的列表。
- splitlines(a[, keepends])	对于a中的每个元素，返回元素中的行列表，在行边界处断开。
- strip(a[, chars])	对于a中的每个元素，返回一个删除了前导和尾随字符的副本。
- swapcase(a)	返回元素的字符串副本，大写字符转换为小写，反之亦然。
- title(a)	返回元素字符串的字符串或unicode的版本。
- translate(a, table[, deletechars]) 对于a中的每个元素，返回字符串的副本，其中删除可选参数deletechars中出现的所有字符，并通过给定的转换表映射其余字符。
- upper(a)	返回一个数组，其元素转换为大写。
- zfill(a, width)	返回左边用零填充的数字字符串

## 对照

与标准的numpy比较运算符不同，char模块中的那些运算符在执行比较之前剥离尾随空白字符。

- equal(x1, x2)	返回 (x1 == x2) 逐元素。
- not_equal(x1, x2)	返回 (x1 != x2) 逐元素。
- greater_equal(x1, x2)	返回 (x1 >= x2) 逐元素。
- less_equal(x1, x2) 返回 (x1 <= x2) 逐元素。
- greater(x1, x2) 返回 (x1 > x2) 逐元素。
- less(x1, x2) 返回 (x1 < x2) 逐元素。

## 字符串信息

- count(a, sub[, start, end])	返回一个数组，其中包含[start, end]范围内substring sub的非重叠出现次数。
- find(a, sub[, start, end])	对于每个元素，返回找到substring sub的字符串中的最低索引。
- index(a, sub[, start, end])	与find一样，但在找不到子字符串时会引发ValueError。
- isalpha(a)	如果字符串中的所有字符都是字母并且至少有一个字符，则返回每个元素的true，否则返回false。
- isdecimal(a)	对于每个元素，如果元素中只有十进制字符，则返回True。
- isdigit(a)	如果字符串中的所有字符都是数字并且至少有一个字符，则返回每个元素的true，否则返回false。
- islower(a)	如果字符串中的所有外壳字符都是小写且至少有一个外壳字符，则为每个元素返回true，否则返回false。
- isnumeric(a)	对于每个元素，如果元素中只有数字字符，则返回True。
- isspace(a)	如果字符串中只有空格字符并且至少有一个字符，则为每个元素返回true，否则返回false。
- istitle(a)	如果元素是一个带有标题的字符串并且至少有一个字符，则为每个元素返回true，否则返回false。
- isupper(a)	如果字符串中的所有外壳字符都是大写且至少有一个字符，则为每个元素返回true，否则返回false。
- rfind(a, sub[, start, end])	对于a中的每个元素，返回找到substring sub的字符串中的最高索引，使得sub包含在[start, end]中。
- rindex(a, sub[, start, end])	和rfind一样，但是当找不到substring sub时会引发ValueError。
- startswith(a, prefix[, start, end]) 返回一个布尔数组，该数组为True，其中a中的字符串元素以prefix开头，否则为False。

## 便捷的类

- chararray(shape[, itemsize, unicode, …]) 提供有关字符串和unicode值数组的便捷视图。