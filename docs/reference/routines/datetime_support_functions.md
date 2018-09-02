# 时间日期操作

- datetime_as_string(arr[, unit, timezone, …])	将日期时间数组转换为字符串数组。
- datetime_data(dtype, /)	获取有关日期或时间类型的步长的信息。

## 营业日功能

- busdaycalendar([weekmask, holidays])	工作日日历对象，可有效存储定义busday系列函数的有效天数的信息。
- is_busday(dates[, weekmask, holidays, …])	计算哪个给定日期是有效日期，哪些日期不是。
- busday_offset(dates, offsets[, roll, …])	首先根据滚动规则将日期调整为有效日期，然后将偏移应用于有效日期内计算的给定日期。
- busday_count(begindates, enddates[, …])	计算beginupdates和结束日期之间的有效天数，不包括enddates的日期。