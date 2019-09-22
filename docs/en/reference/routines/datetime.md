# Datetime Support Functions

method | description
---|---
[datetime_as_string](https://numpy.org/devdocs/reference/generated/numpy.datetime_as_string.html#numpy.datetime_as_string)(arr[, unit, timezone, …]) | Convert an array of datetimes into an array of strings.
[datetime_data](https://numpy.org/devdocs/reference/generated/numpy.datetime_data.html#numpy.datetime_data)(dtype, /) | Get information about the step size of a date or time type.

## Business Day Functions

method | description
---|---
[busdaycalendar](https://numpy.org/devdocs/reference/generated/numpy.busdaycalendar.html#numpy.busdaycalendar)([weekmask, holidays]) | A business day calendar object that efficiently stores information defining valid days for the busday family of functions.
[is_busday](https://numpy.org/devdocs/reference/generated/numpy.is_busday.html#numpy.is_busday)(dates[, weekmask, holidays, …]) | Calculates which of the given dates are valid days, and which are not.
[busday_offset](https://numpy.org/devdocs/reference/generated/numpy.busday_offset.html#numpy.busday_offset)(dates, offsets[, roll, …]) | First adjusts the date to fall on a valid day according to the roll rule, then applies offsets to the given dates counted in valid days.
[busday_count](https://numpy.org/devdocs/reference/generated/numpy.busday_count.html#numpy.busday_count)(begindates, enddates[, …]) | Counts the number of valid days between begindates and enddates, not including the day of enddates.