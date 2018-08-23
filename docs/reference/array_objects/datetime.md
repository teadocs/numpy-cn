# 日期时间和时间增量

*版本1.7.0中的新功能。*

从NumPy 1.7开始，有核心数组数据类型本身支持日期时间功能。 数据类型称为“datetime64”，因为“datetime”已被Python中包含的日期时间库占用。

> **注意**
> datetime API在1.7.0中是实验性的，并且可能在未来版本的NumPy中进行更改。

## 基本的日期时间

创建日期时间的最基本方法是使用ISO 8601日期或日期时间格式的字符串。 内部存储单元自动从字符串的形式中选择，可以是日期单位或时间单位。 日期单位是年（'Y'），月（'M'），周（'W'）和天（'D'），而时间单位是小时（'h'），分钟（'m'） ），秒（'s'），毫秒（'ms'）和一些额外的SI前缀基于秒的单位。

**例子**

一个简单的ISO日期：

```python
>>> np.datetime64('2005-02-25')
numpy.datetime64('2005-02-25')
```

使用月份为单位：

```python
>>> np.datetime64('2005-02')
numpy.datetime64('2005-02')
```

仅指定月份，但强制“天”单位：

```python
>>> np.datetime64('2005-02', 'D')
numpy.datetime64('2005-02-01')
```

从日期和时间：

```python
>>> np.datetime64('2005-02-25T03:30')
numpy.datetime64('2005-02-25T03:30')
```

从字符串创建日期时间数组时，仍然可以通过使用带有通用单位的日期时间类型从输入中自动选择单位。

**例子**

```python
>>> np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')
array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64[D]')
```

```python
>>> np.array(['2001-01-01T12:00', '2002-02-03T13:56:03.172'], dtype='datetime64')
array(['2001-01-01T12:00:00.000-0600', '2002-02-03T13:56:03.172-0600'], dtype='datetime64[ms]')
```

datetime类型适用于许多常见的NumPy函数，例如，可以使用arange来生成日期范围。

**例子**

所有日期为一个月：

```python
>>> np.arange('2005-02', '2005-03', dtype='datetime64[D]')
array(['2005-02-01', '2005-02-02', '2005-02-03', '2005-02-04',
       '2005-02-05', '2005-02-06', '2005-02-07', '2005-02-08',
       '2005-02-09', '2005-02-10', '2005-02-11', '2005-02-12',
       '2005-02-13', '2005-02-14', '2005-02-15', '2005-02-16',
       '2005-02-17', '2005-02-18', '2005-02-19', '2005-02-20',
       '2005-02-21', '2005-02-22', '2005-02-23', '2005-02-24',
       '2005-02-25', '2005-02-26', '2005-02-27', '2005-02-28'],
       dtype='datetime64[D]')
```

datetime对象表示单个时刻。 如果两个日期时间具有不同的单位，它们可能仍然代表相同的时刻，并且从较大的单位（如月份）转换为较小的单位（如天数）被视为“安全”演员，因为时刻仍在准确表示。

**例子**

```python
>>> np.datetime64('2005') == np.datetime64('2005-01-01')
True
```

```python
>>> np.datetime64('2010-03-14T15Z') == np.datetime64('2010-03-14T15:00:00.00Z')
True
```

## Datetime和Timedelta算术运算

NumPy允许减去两个Datetime值，这个操作产生一个带有时间单位的数字。 由于NumPy的核心没有物理量系统，因此创建了timedelta64数据类型以补充datetime64。

Datetimes和Timedeltas一起工作，为简单的日期时间计算提供方法。

**例子**

```python
>>> np.datetime64('2009-01-01') - np.datetime64('2008-01-01')
numpy.timedelta64(366,'D')
```

```python
>>> np.datetime64('2009') + np.timedelta64(20, 'D')
numpy.datetime64('2009-01-21')
```

```python
>>> np.datetime64('2011-06-15T00:00') + np.timedelta64(12, 'h')
numpy.datetime64('2011-06-15T12:00-0500')
>>> np.timedelta64(1,'W') / np.timedelta64(1,'D')
7.0
```

有两个Timedelta单位（'Y'，年和'M'，几个月）被特别处理，因为他们代表的时间根据使用时间而变化。 虽然timedelta日单位相当于24小时，但无法将月份单位转换为天数，因为不同的月份具有不同的天数。

**例子**

```python
>>> a = np.timedelta64(1, 'Y')
```

```python
>>> np.timedelta64(a, 'M')
numpy.timedelta64(12,'M')

```python
>>> np.timedelta64(a, 'D')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Cannot cast NumPy timedelta64 scalar from metadata [Y] to [D] according to the rule 'same_kind'
```

## 日期时间单位

Datetime和Timedelta数据类型支持大量时间单位，以及可以根据输入数据强制转换为任何其他单位的通用单位。

始终基于POSIX时间存储日期时间（尽管具有允许计算闰秒的TAI模式），具有1970-01-01T00：00Z的时期。 这意味着支持的日期总是围绕时期的对称间隔，在下表中称为“时间跨度”。

跨度的长度是64位整数乘以日期或单位长度的范围。 例如，'W'（周）的时间跨度恰好是'D'（日）的时间跨度的7倍，'D'（日）的时间跨度恰好是时间跨度的24倍 为'h'（小时）。

以下是日期单位：

代码 | 含义 | 时间跨度（相对） | 时间跨度（绝对）
---|---|---|---
Y | 年 | +/- 9.2e18 years | [9.2e18 BC, 9.2e18 AD]
M | 月 | +/- 7.6e17 years | [7.6e17 BC, 7.6e17 AD]
W | 周 | +/- 1.7e17 years | [1.7e17 BC, 1.7e17 AD]
D | 日 | +/- 2.5e16 years | [2.5e16 BC, 2.5e16 AD]

以下是时间单位：

代码 | 含义 | 时间跨度（相对） | 时间跨度（绝对）
---|---|---|---
h | 小时 | +/- 1.0e15 years | [1.0e15 BC, 1.0e15 AD]
m | 分钟 | +/- 1.7e13 years | [1.7e13 BC, 1.7e13 AD]
s | 秒 | +/- 2.9e11 years | [2.9e11 BC, 2.9e11 AD]
ms | 毫秒 | +/- 2.9e8 years | [ 2.9e8 BC, 2.9e8 AD]
us | 微秒 | +/- 2.9e5 years | [290301 BC, 294241 AD]
ns | 纳秒 | +/- 292 years | [ 1678 AD, 2262 AD]
ps | 皮秒 | +/- 106 days | [ 1969 AD, 1970 AD]
fs | 飞秒 | +/- 2.6 hours | [ 1969 AD, 1970 AD]
as | 阿秒 | +/- 9.2 seconds | [ 1969 AD, 1970 AD]

## 工作日功能

为了允许在只有一周中某些日子有效的上下文中使用日期时间，NumPy包含一组“busday”（工作日）功能。

busday功能的默认值是唯一有效的日期是周一到周五（通常的工作日）。 该实现基于一个“weekmask”，包含7个布尔标志，用于指示有效天数; 可以指定其他有效天数集。

“busday”功能还可以检查“假日”日期列表，特定日期是无效日期。

函数``busday_offset``允许您将工作日中指定的偏移量应用于日期时间，单位为“D”（天）。

**例子**

```python
>>> np.busday_offset('2011-06-23', 1)
numpy.datetime64('2011-06-24')
```

```python
>>> np.busday_offset('2011-06-23', 2)
numpy.datetime64('2011-06-27')
```

当输入日期落在周末或假日时，``busday_offset``首先应用规则将日期滚动到有效的工作日，然后应用偏移量。 默认规则是'raise'，它只会引发异常。 最常用的规则是“前进”和“后退”。

**例子**

```python
>>> np.busday_offset('2011-06-25', 2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Non-business day date in busday_offset
```

```python
>>> np.busday_offset('2011-06-25', 0, roll='forward')
numpy.datetime64('2011-06-27')
```

```python
>>> np.busday_offset('2011-06-25', 2, roll='forward')
numpy.datetime64('2011-06-29')
```

```python
>>> np.busday_offset('2011-06-25', 0, roll='backward')
numpy.datetime64('2011-06-24')
```

```python
>>> np.busday_offset('2011-06-25', 2, roll='backward')
numpy.datetime64('2011-06-28')
```

在某些情况下，需要适当使用滚动和偏移以获得所需的答案。

**例子**

日期或之后的第一个工作日：

```python
>>> np.busday_offset('2011-03-20', 0, roll='forward')
numpy.datetime64('2011-03-21','D')
>>> np.busday_offset('2011-03-22', 0, roll='forward')
numpy.datetime64('2011-03-22','D')
```

严格遵守日期后的第一个工作日：

```python
>>> np.busday_offset('2011-03-20', 1, roll='backward')
numpy.datetime64('2011-03-21','D')
>>> np.busday_offset('2011-03-22', 1, roll='backward')
numpy.datetime64('2011-03-23','D')
```

该功能对于计算某些日子如假期也很有用。 在加拿大和美国，母亲节是在5月的第二个星期天，可以使用自定义的周掩码来计算。

**例子**

```python
>>> np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
numpy.datetime64('2012-05-13','D')
```

当性能对于使用一个特定选择的周掩码和假日来操纵许多业务日期很重要时，有一个对象``busdaycalendar``，它以优化的形式存储必要的数据。

### np.is_busday():

要测试datetime64值以查看它是否为有效日期，请使用is_busday。

**例子**

```python
>>> np.is_busday(np.datetime64('2011-07-15'))  # a Friday
True
>>> np.is_busday(np.datetime64('2011-07-16')) # a Saturday
False
>>> np.is_busday(np.datetime64('2011-07-16'), weekmask="Sat Sun")
True
>>> a = np.arange(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
>>> np.is_busday(a)
array([ True,  True,  True,  True,  True, False, False], dtype='bool')
```
### np.busday_count():

要查找指定日期时间64日期范围内有效天数，请使用busday_count：

**例子**

```python
>>> np.busday_count(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
5
>>> np.busday_count(np.datetime64('2011-07-18'), np.datetime64('2011-07-11'))
-5
```

如果您有一组datetime64天值，并且您想要计算其中有多少是有效日期，则可以执行以下操作：

**例子**

```python
>>> a = np.arange(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
>>> np.count_nonzero(np.is_busday(a))
5
```

#### 自定义Weekmasks

以下是自定义周掩码值的几个示例。 这些示例指定周一至周五的“busday”默认值为有效天数。

一些例子:

```python
# Positional sequences; positions are Monday through Sunday.
# Length of the sequence must be exactly 7.
weekmask = [1, 1, 1, 1, 1, 0, 0]
# list or other sequence; 0 == invalid day, 1 == valid day
weekmask = "1111100"
# string '0' == invalid day, '1' == valid day

# string abbreviations from this list: Mon Tue Wed Thu Fri Sat Sun
weekmask = "Mon Tue Wed Thu Fri"
# any amount of whitespace is allowed; abbreviations are case-sensitive.
weekmask = "MonTue Wed  Thu\tFri"
```

## 使用NumPy 1.11进行更改

在NumPy的早期版本中，datetime64类型始终以UTC格式存储。 默认情况下，从字符串创建datetime64对象或打印它将从或转换为本地时间：

```python
# old behavior
>>>> np.datetime64('2000-01-01T00:00:00')
numpy.datetime64('2000-01-01T00:00:00-0800')  # note the timezone offset -08:00
```

datetime64用户的共识认为这种行为是不可取的，并且与通常使用datetime64的方式不一致（例如，通过pandas）。 对于大多数用例，首选的是时区天真日期时间类型，类似于Python标准库中的datetime.datetime类型。 因此，datetime64不再假定输入是在本地时间，也不是打印本地时间：

```python
>>>> np.datetime64('2000-01-01T00:00:00')
numpy.datetime64('2000-01-01T00:00:00')
```

为了向后兼容，datetime64仍解析时区偏移，它通过转换为UTC来处理。 但是，生成的日期时间是时区的天真：

```python
>>> np.datetime64('2000-01-01T00:00:00-08')
DeprecationWarning: parsing timezone aware datetimes is deprecated; this will raise an error in the future
numpy.datetime64('2000-01-01T08:00:00')
```

作为此更改的必然结果，我们不再禁止在日期时间与日期单位和日期时间与时间单位之间进行转换。 对于时区天真的日期时间，从日期到时间的投射规则不再模糊。

## 1.6和1.7日期时间之间的差异

NumPy 1.6版本包含比1.7更原始的日期时间数据类型。 本节介绍了许多已发生的变化。

### 字符串解析

NumPy 1.6中的日期时间字符串解析器在它接受的内容中非常自由，并且默默地允许无效输入而不会引发错误。 NumPy 1.7中的解析器对于仅接受ISO 8601日期非常严格，只有一些便利扩展。 1.6默认情况下始终创建微秒（us）单位，而1.7则根据字符串的格式检测单位。 这是一个比较：

```python
# NumPy 1.6.1
>>> np.datetime64('1979-03-22')
1979-03-22 00:00:00
# NumPy 1.7.0
>>> np.datetime64('1979-03-22')
numpy.datetime64('1979-03-22')

# NumPy 1.6.1, unit default microseconds
>>> np.datetime64('1979-03-22').dtype
dtype('datetime64[us]')
# NumPy 1.7.0, unit of days detected from string
>>> np.datetime64('1979-03-22').dtype
dtype('<M8[D]')

# NumPy 1.6.1, ignores invalid part of string
>>> np.datetime64('1979-03-2corruptedstring')
1979-03-02 00:00:00
# NumPy 1.7.0, raises error for invalid input
>>> np.datetime64('1979-03-2corruptedstring')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Error parsing datetime string "1979-03-2corruptedstring" at position 8

# NumPy 1.6.1, 'nat' produces today's date
>>> np.datetime64('nat')
2012-04-30 00:00:00
# NumPy 1.7.0, 'nat' produces not-a-time
>>> np.datetime64('nat')
numpy.datetime64('NaT')

# NumPy 1.6.1, 'garbage' produces today's date
>>> np.datetime64('garbage')
2012-04-30 00:00:00
# NumPy 1.7.0, 'garbage' raises an exception
>>> np.datetime64('garbage')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Error parsing datetime string "garbage" at position 0

# NumPy 1.6.1, can't specify unit in scalar constructor
>>> np.datetime64('1979-03-22T19:00', 'h')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: function takes at most 1 argument (2 given)
# NumPy 1.7.0, unit in scalar constructor
>>> np.datetime64('1979-03-22T19:00', 'h')
numpy.datetime64('1979-03-22T19:00-0500','h')

# NumPy 1.6.1, reads ISO 8601 strings w/o TZ as UTC
>>> np.array(['1979-03-22T19:00'], dtype='M8[h]')
array([1979-03-22 19:00:00], dtype=datetime64[h])
# NumPy 1.7.0, reads ISO 8601 strings w/o TZ as local (ISO specifies this)
>>> np.array(['1979-03-22T19:00'], dtype='M8[h]')
array(['1979-03-22T19-0500'], dtype='datetime64[h]')

# NumPy 1.6.1, doesn't parse all ISO 8601 strings correctly
>>> np.array(['1979-03-22T12'], dtype='M8[h]')
array([1979-03-22 00:00:00], dtype=datetime64[h])
>>> np.array(['1979-03-22T12:00'], dtype='M8[h]')
array([1979-03-22 12:00:00], dtype=datetime64[h])
# NumPy 1.7.0, handles this case correctly
>>> np.array(['1979-03-22T12'], dtype='M8[h]')
array(['1979-03-22T12-0500'], dtype='datetime64[h]')
>>> np.array(['1979-03-22T12:00'], dtype='M8[h]')
array(['1979-03-22T12-0500'], dtype='datetime64[h]')
```

### 单位转换

日期时间的1.6实现不能正确转换单位：

```python
# NumPy 1.6.1, the representation value is untouched
>>> np.array(['1979-03-22'], dtype='M8[D]')
array([1979-03-22 00:00:00], dtype=datetime64[D])
>>> np.array(['1979-03-22'], dtype='M8[D]').astype('M8[M]')
array([2250-08-01 00:00:00], dtype=datetime64[M])
# NumPy 1.7.0, the representation is scaled accordingly
>>> np.array(['1979-03-22'], dtype='M8[D]')
array(['1979-03-22'], dtype='datetime64[D]')
>>> np.array(['1979-03-22'], dtype='M8[D]').astype('M8[M]')
array(['1979-03'], dtype='datetime64[M]')
```

### 日期算术运算

日期时间的1.6实现仅适用于一小部分算术运算。这里我们展示一些简单的案例：

```python
# NumPy 1.6.1, produces invalid results if units are incompatible
>>> a = np.array(['1979-03-22T12'], dtype='M8[h]')
>>> b = np.array([3*60], dtype='m8[m]')
>>> a + b
array([1970-01-01 00:00:00.080988], dtype=datetime64[us])
# NumPy 1.7.0, promotes to higher-resolution unit
>>> a = np.array(['1979-03-22T12'], dtype='M8[h]')
>>> b = np.array([3*60], dtype='m8[m]')
>>> a + b
array(['1979-03-22T15:00-0500'], dtype='datetime64[m]')

# NumPy 1.6.1, arithmetic works if everything is microseconds
>>> a = np.array(['1979-03-22T12:00'], dtype='M8[us]')
>>> b = np.array([3*60*60*1000000], dtype='m8[us]')
>>> a + b
array([1979-03-22 15:00:00], dtype=datetime64[us])
# NumPy 1.7.0
>>> a = np.array(['1979-03-22T12:00'], dtype='M8[us]')
>>> b = np.array([3*60*60*1000000], dtype='m8[us]')
>>> a + b
array(['1979-03-22T15:00:00.000000-0500'], dtype='datetime64[us]')
```