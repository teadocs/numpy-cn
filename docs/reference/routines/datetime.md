# 时间日期相关

方法 | 描述
---|---
[datetime_as_string](https://numpy.org/devdocs/reference/generated/numpy.datetime_as_string.html#numpy.datetime_as_string)(arr[, unit, timezone, …]) | 将日期时间数组转换为字符串数组。
[datetime_data](https://numpy.org/devdocs/reference/generated/numpy.datetime_data.html#numpy.datetime_data)(dtype, /) | 获取有关日期或时间类型的步长的信息。

## 工作日函数

方法 | 描述
---|---
[busdaycalendar](https://numpy.org/devdocs/reference/generated/numpy.busdaycalendar.html#numpy.busdaycalendar)([weekmask, holidays]) | 一个工作日日历对象，它有效地存储定义Busday系列函数的有效天数的信息。
[is_busday](https://numpy.org/devdocs/reference/generated/numpy.is_busday.html#numpy.is_busday)(dates[, weekmask, holidays, …]) | 计算给定日期中哪些是有效日期，哪些不是。
[busday_offset](https://numpy.org/devdocs/reference/generated/numpy.busday_offset.html#numpy.busday_offset)(dates, offsets[, roll, …]) | 首先根据滚动规则将日期调整为属于有效日期，然后将偏移量应用于有效日期中计算的给定日期。
[busday_count](https://numpy.org/devdocs/reference/generated/numpy.busday_count.html#numpy.busday_count)(begindates, enddates[, …]) | 计算介于beginupdates和结束日期之间的有效天数，不包括结束日期。