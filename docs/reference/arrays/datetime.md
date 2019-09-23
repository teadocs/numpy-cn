# 日期时间和时间增量

*1.7.0版中的新功能。* 

从NumPy 1.7开始，有核心数组数据类型本身支持日期时间功能。数据类型称为 “datetime64” ，
因为  “datetime” 已被Python中包含的日期时间库所占用。

::: tip 注意

datetime API 在1.7.0中是 *实验性* 的，并且可能会在未来版本的NumPy中进行更改。

:::

## 基本日期时间

创建日期时间的最基本方法是使用ISO 8601日期或日期时间格式的字符串。内部存储单元自动从字符串的形式中选择，
可以是``日期单位``或``时间单位``。
日期单位是年（'Y'），月（'M'），周（'W'）和天（'D'），
而时间单位是小时（'h'），分钟（'m'） ），秒（'s'），
毫秒（'ms'）和一些额外的SI前缀基于秒的单位。
对于“非时间”值，datetime64数据类型还接受字符串“NAT”，
以小写/大写字母的任意组合。

**示例：**

一个简单的ISO日期：

``` python
>>> np.datetime64('2005-02-25')
numpy.datetime64('2005-02-25')
```

使用月份为单位：

``` python
>>> np.datetime64('2005-02')
numpy.datetime64('2005-02')
```

仅指定月份，但强制使用 “天” 单位：

**示例：**

``` python
>>> np.datetime64('2005-02', 'D')
numpy.datetime64('2005-02-01')
```

从日期和时间：

``` python
>>> np.datetime64('2005-02-25T03:30')
numpy.datetime64('2005-02-25T03:30')
```

NAT（不是时间）：

``` python
>>> numpy.datetime64('nat')
numpy.datetime64('NaT')
```

从字符串创建日期时间数组时，仍然可以通过使用具有通用单位的日期时间类型从输入中自动选择单位。

``` python
>>> np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')
array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64[D]')
```

``` python
>>> np.array(['2001-01-01T12:00', '2002-02-03T13:56:03.172'], dtype='datetime64')
array(['2001-01-01T12:00:00.000-0600', '2002-02-03T13:56:03.172-0600'], dtype='datetime64[ms]')
```

datetime类型适用于许多常见的NumPy函数，例如[``arange``](https://numpy.org/devdocs/reference/generated/numpy.arange.html#numpy.arange)可用于生成日期范围。

**示例：**

所有日期为一个月：

``` python
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

datetime对象表示单个时刻。如果两个日期时间具有不同的单位，它们可能仍然代表相同的时刻，并且从较大的单位（如月份）转换为较小的单位（如天数）被视为“安全”投射，因为时刻仍然正好表示。

**示例：**

``` python
>>> np.datetime64('2005') == np.datetime64('2005-01-01')
True
```

``` python
>>> np.datetime64('2010-03-14T15Z') == np.datetime64('2010-03-14T15:00:00.00Z')
True
```

## Datetime 和 Timedelta 算法

NumPy允许减去两个Datetime值，这个操作产生一个带有时间单位的数字。由于NumPy的核心没有物理量系统，因此创建了timedelta64数据类型以补充datetime64。timedelta64的参数是一个数字，用于表示单位数，以及日期/时间单位，如 (D)ay, (M)onth, (Y)ear, (h)ours, (m)inutes, 或者 (s)econds。timedelta64数据类型也接受字符串“NAT”代替“非时间”值的数字。

**示例：**

``` python
>>> numpy.timedelta64(1, 'D')
numpy.timedelta64(1,'D')
```

``` python
>>> numpy.timedelta64(4, 'h')
numpy.timedelta64(4,'h')
```

``` python
>>> numpy.timedelta64('nAt')
numpy.timedelta64('NaT')
```

Datetimes 和 Timedeltas 一起工作，为简单的日期时间计算提供方法。

**示例：**

``` python
>>> np.datetime64('2009-01-01') - np.datetime64('2008-01-01')
numpy.timedelta64(366,'D')
```

``` python
>>> np.datetime64('2009') + np.timedelta64(20, 'D')
numpy.datetime64('2009-01-21')
```

``` python
>>> np.datetime64('2011-06-15T00:00') + np.timedelta64(12, 'h')
numpy.datetime64('2011-06-15T12:00-0500')
```

``` python
>>> np.timedelta64(1,'W') / np.timedelta64(1,'D')
7.0
```

``` python
>>> np.timedelta64(1,'W') % np.timedelta64(10,'D')
numpy.timedelta64(7,'D')
```

``` python
>>> numpy.datetime64('nat') - numpy.datetime64('2009-01-01')
numpy.timedelta64('NaT','D')
```

``` python
>>> numpy.datetime64('2009-01-01') + numpy.timedelta64('nat')
numpy.datetime64('NaT')
```

有两个 Timedelta 单位（'Y'，年和'M'，几个月）被特别处理，因为它们代表的时间根据使用时间而变化。虽然timedelta日单位相当于24小时，但无法将月份单位转换为天数，因为不同的月份具有不同的天数。

**示例：**

``` python
>>> a = np.timedelta64(1, 'Y')
```

``` python
>>> np.timedelta64(a, 'M')
numpy.timedelta64(12,'M')
```

``` python
>>> np.timedelta64(a, 'D')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Cannot cast NumPy timedelta64 scalar from metadata [Y] to [D] according to the rule 'same_kind'
```

## 日期时间单位

Datetime和Timedelta数据类型支持大量时间单位，以及可以根据输入数据强制转换为任何其他单位的通用单位。

始终基于POSIX时间存储日期时间（尽管具有允许计算闰秒的TAI模式），具有1970-01-01T00：00Z的纪元。这意味着支持的日期总是围绕时期的对称间隔，在下表中称为“时间跨度”。

跨度的长度是64位整数乘以日期或单位长度的范围。例如，'W'（周）的时间跨度恰好是'D'（日）的时间跨度的7倍，'D'（日）的时间跨度恰好是时间跨度的24倍为'h'（小时）。

以下是日期单位：

代码 | 含义 | 时间跨度（相对） | 时间跨度（绝对）
---|---|---|---
Y | 年 | +/-9.2e18年 | [公元前9.2e18，公元9.2e18]
M | 月 | +/-7.6e17年 | [公元前7.6e17，公元7.6e17]
W | 周 | +/-1.7e17年 | [公元前1.7e17，公元17e17]
D | 天 | +/-2.5e16年 | [公元前2.5e16，公元2.5e16]

以下是时间单位：

代码 | 含义 | 时间跨度（相对） | 时间跨度（绝对）
---|---|---|---
h | 小时 | +/-1.0e15年 | [公元前1.0e15，公元1.0e15]
m | 分钟 | +/-1.7e13年 | [公元前1.7e13，公元1.7e13]
s | 第二 | +/-2.9e11年 | [公元前2.9e11，公元2.9e11]
ms | 毫秒 | +/-2.9e8年 | [公元前2.9e8，公元2.9e8]
us | 微秒 | +/-2.9e5年 | [290301 BC，294241 AD]
ns | 纳秒 | +/- 292年 | [公元1678年，公元2262年]
ps | 皮秒 | +/- 106天 | [公元1969年，公元1970年]
fs | 飞秒 | +/- 2.6小时 | [公元1969年，公元1970年]
as | 阿秒 | +/- 9.2秒 | [公元1969年，公元1970年]

## Datetime 功能

为了允许在只有一周中某些日子有效的上下文中使用日期时间，NumPy包含一组“busday”（工作日）功能。

busday功能的默认值是唯一有效的日期是周一到周五（通常的工作日）。该实现基于一个“weekmask”，包含7个布尔标志，用于指示有效天数; 可以指定其他有效天数集的自定义工资单。

“busday”功能还可以检查“假日”日期列表，特定日期是无效日期。

该功能[``busday_offset``](https://numpy.org/devdocs/reference/generated/numpy.busday_offset.html#numpy.busday_offset)允许您将工作日中指定的偏移量应用于日期时间，单位为“D”（天）。

**示例：**

``` python
>>> np.busday_offset('2011-06-23', 1)
numpy.datetime64('2011-06-24')
```

``` python
>>> np.busday_offset('2011-06-23', 2)
numpy.datetime64('2011-06-27')
```

当输入日期落在周末或假日时，
 [``busday_offset``](https://numpy.org/devdocs/reference/generated/numpy.busday_offset.html#numpy.busday_offset)首先应用规则将日期滚动到有效的工作日，然后应用偏移量。默认规则是'raise'，它只会引发异常。最常用的规则是“前进”和“后退”。

**示例：**

``` python
>>> np.busday_offset('2011-06-25', 2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Non-business day date in busday_offset
```

``` python
>>> np.busday_offset('2011-06-25', 0, roll='forward')
numpy.datetime64('2011-06-27')
```

``` python
>>> np.busday_offset('2011-06-25', 2, roll='forward')
numpy.datetime64('2011-06-29')
```

``` python
>>> np.busday_offset('2011-06-25', 0, roll='backward')
numpy.datetime64('2011-06-24')
```

``` python
>>> np.busday_offset('2011-06-25', 2, roll='backward')
numpy.datetime64('2011-06-28')
```

在某些情况下，需要适当使用滚动和偏移以获得所需的答案。

**示例：**

日期或之后的第一个工作日：

``` python
>>> np.busday_offset('2011-03-20', 0, roll='forward')
numpy.datetime64('2011-03-21','D')
>>> np.busday_offset('2011-03-22', 0, roll='forward')
numpy.datetime64('2011-03-22','D')
```

严格遵守日期后的第一个工作日：

``` python
>>> np.busday_offset('2011-03-20', 1, roll='backward')
numpy.datetime64('2011-03-21','D')
>>> np.busday_offset('2011-03-22', 1, roll='backward')
numpy.datetime64('2011-03-23','D')
```

该功能对于计算某些日子如假期也很有用。在加拿大和美国，母亲节是在5月的第二个星期天，可以使用自定义的周掩码来计算。

**示例：**

``` python
>>> np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
numpy.datetime64('2012-05-13','D')
```

当性能对于使用一个特定选择的周工具和假期来操纵许多业务日期很重要时，有一个对象[``busdaycalendar``](https://numpy.org/devdocs/reference/generated/numpy.busdaycalendar.html#numpy.busdaycalendar)以优化的形式存储必要的数据。

### np.is_busday() 方法

要测试datetime64值以查看它是否为有效日期，请使用[``is_busday``](https://numpy.org/devdocs/reference/generated/numpy.is_busday.html#numpy.is_busday)。

``` python
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

### np.busday_count() 方法

要查找指定日期时间64日期范围内有效天数，请使用[``busday_count``](https://numpy.org/devdocs/reference/generated/numpy.busday_count.html#numpy.busday_count)：

**示例：**

``` python
>>> np.busday_count(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
5
>>> np.busday_count(np.datetime64('2011-07-18'), np.datetime64('2011-07-11'))
-5
```

如果您有一组datetime64天值，并且您希望计算其中有多少是有效日期，则可以执行以下操作：

**示例：**

``` python
>>> a = np.arange(np.datetime64('2011-07-11'), np.datetime64('2011-07-18'))
>>> np.count_nonzero(np.is_busday(a))
5
```

#### 自定义周掩码

以下是自定义周掩码值的几个示例。这些示例指定周一至周五的 “busday” 默认值为有效天数。

**一些例子：**

``` python
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

## NumPy 1.11 的更改

在NumPy的早期版本中，datetime64类型始终以UTC格式存储。默认情况下，从字符串创建datetime64对象或打印它将从或转换为本地时间：

``` python
# old behavior
>>>> np.datetime64('2000-01-01T00:00:00')
numpy.datetime64('2000-01-01T00:00:00-0800')  # note the timezone offset -08:00
```

datetime64用户一致认为这种行为是不可取的，并且与datetime64通常的使用方式（例如，[pandas](https://pandas.pydata.org)）不一致。对于大多数用例，首选时区朴素的datetime类型，类似于Python标准库中的 ``datetime.datetime`` 类型。因此，datetime64 不再假设输入为本地时间，也不会打印本地时间：

``` python
>>>> np.datetime64('2000-01-01T00:00:00')
numpy.datetime64('2000-01-01T00:00:00')
```

为了向后兼容，datetime64仍然解析时区偏移，它通过转换为UTC来处理。但是，生成的日期时间是时区的天真：

``` python
>>> np.datetime64('2000-01-01T00:00:00-08')
DeprecationWarning: parsing timezone aware datetimes is deprecated; this will raise an error in the future
numpy.datetime64('2000-01-01T08:00:00')
```

作为此更改的必然结果，我们不再禁止在日期时间与日期单位和日期时间与时间单位之间进行转换。对于时区天真的日期时间，从日期到时间的投射规则不再模糊。