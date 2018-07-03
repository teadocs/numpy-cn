# 字符串操作

This module provides a set of vectorized string operations for arrays of type numpy.string_ or numpy.unicode_. All of them are based on the string methods in the Python standard library.

## String operations

- add(x1, x2)	Return element-wise string concatenation for two arrays of str or unicode.
- multiply(a, i)	Return (a * i), that is string multiple concatenation, element-wise.
- mod(a, values)	Return (a % i), that is pre-Python 2.6 string formatting (iterpolation), element-wise for a pair of array_likes of str or unicode.
- capitalize(a)	Return a copy of a with only the first character of each element capitalized.
- center(a, width[, fillchar])	Return a copy of a with its elements centered in a string of length width.
- decode(a[, encoding, errors])	Calls str.decode element-wise.
- encode(a[, encoding, errors])	Calls str.encode element-wise.
- join(sep, seq)	Return a string which is the concatenation of the strings in the sequence seq.
- ljust(a, width[, fillchar])	Return an array with the elements of a left-justified in a string of length width.
- lower(a)	Return an array with the elements converted to lowercase.
- lstrip(a[, chars])	For each element in a, return a copy with the leading characters removed.
- partition(a, sep)	Partition each element in a around sep.
- replace(a, old, new[, count])	For each element in a, return a copy of the string with all occurrences of substring old replaced by new.
- rjust(a, width[, fillchar])	Return an array with the elements of a right-justified in a string of length width.
- rpartition(a, sep)	Partition (split) each element around the right-most separator.
- rsplit(a[, sep, maxsplit])	For each element in a, return a list of the words in the string, using sep as the delimiter string.
- rstrip(a[, chars])	For each element in a, return a copy with the trailing characters removed.
- split(a[, sep, maxsplit])	For each element in a, return a list of the words in the string, using sep as the delimiter string.
- splitlines(a[, keepends])	For each element in a, return a list of the lines in the element, breaking at line boundaries.
- strip(a[, chars])	For each element in a, return a copy with the leading and trailing characters removed.
- swapcase(a)	Return element-wise a copy of the string with uppercase characters converted to lowercase and vice versa.
- title(a)	Return element-wise title cased version of string or unicode.
- translate(a, table[, deletechars])	For each element in a, return a copy of the string where all characters occurring in the optional argument deletechars are removed, and the remaining characters have been mapped through the given translation table.
- upper(a)	Return an array with the elements converted to uppercase.
- zfill(a, width)	Return the numeric string left-filled with zeros

## Comparison

Unlike the standard numpy comparison operators, the ones in the char module strip trailing whitespace characters before performing the comparison.

- equal(x1, x2)	Return (x1 == x2) element-wise.
- not_equal(x1, x2)	Return (x1 != x2) element-wise.
- greater_equal(x1, x2)	Return (x1 >= x2) element-wise.
- less_equal(x1, x2)	Return (x1 <= x2) element-wise.
- greater(x1, x2)	Return (x1 > x2) element-wise.
- less(x1, x2)	Return (x1 < x2) element-wise.

## String information

- count(a, sub[, start, end])	Returns an array with the number of non-overlapping occurrences of substring sub in the range [start, end].
- find(a, sub[, start, end])	For each element, return the lowest index in the string where substring sub is found.
- index(a, sub[, start, end])	Like find, but raises ValueError when the substring is not found.
- isalpha(a)	Returns true for each element if all characters in the string are alphabetic and there is at least one character, false otherwise.
- isdecimal(a)	For each element, return True if there are only decimal characters in the element.
- isdigit(a)	Returns true for each element if all characters in the string are digits and there is at least one character, false otherwise.
- islower(a)	Returns true for each element if all cased characters in the string are lowercase and there is at least one cased character, false otherwise.
- isnumeric(a)	For each element, return True if there are only numeric characters in the element.
- isspace(a)	Returns true for each element if there are only whitespace characters in the string and there is at least one character, false otherwise.
- istitle(a)	Returns true for each element if the element is a titlecased string and there is at least one character, false otherwise.
- isupper(a)	Returns true for each element if all cased characters in the string are uppercase and there is at least one character, false otherwise.
- rfind(a, sub[, start, end])	For each element in a, return the highest index in the string where substring sub is found, such that sub is contained within [start, end].
- rindex(a, sub[, start, end])	Like rfind, but raises ValueError when the substring sub is not found.
- startswith(a, prefix[, start, end])	Returns a boolean array which is True where the string element in a starts with prefix, otherwise False.

## Convenience class

- chararray(shape[, itemsize, unicode, …])	Provides a convenient view on arrays of string and unicode values.