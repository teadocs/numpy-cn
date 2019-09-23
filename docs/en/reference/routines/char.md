# String operations

The [``numpy.char``](#module-numpy.char) module provides a set of vectorized string
operations for arrays of type ``numpy.string_`` or ``numpy.unicode_``.
All of them are based on the string methods in the Python standard library.

## String operations

method | description
---|---
[add](https://numpy.org/devdocs/reference/generated/numpy.char.add.html#numpy.char.add)(x1, x2) | Return element-wise string concatenation for two arrays of str or unicode.
[multiply](https://numpy.org/devdocs/reference/generated/numpy.char.multiply.html#numpy.char.multiply)(a, i) | Return (a * i), that is string multiple concatenation, element-wise.
[mod](https://numpy.org/devdocs/reference/generated/numpy.char.mod.html#numpy.char.mod)(a, values) | Return (a % i), that is pre-Python 2.6 string formatting (iterpolation), element-wise for a pair of array_likes of str or unicode.
[capitalize](https://numpy.org/devdocs/reference/generated/numpy.char.capitalize.html#numpy.char.capitalize)(a) | Return a copy of a with only the first character of each element capitalized.
[center](https://numpy.org/devdocs/reference/generated/numpy.char.center.html#numpy.char.center)(a, width[, fillchar]) | Return a copy of a with its elements centered in a string of length width.
[decode](https://numpy.org/devdocs/reference/generated/numpy.char.decode.html#numpy.char.decode)(a[, encoding, errors]) | Calls str.decode element-wise.
[encode](https://numpy.org/devdocs/reference/generated/numpy.char.encode.html#numpy.char.encode)(a[, encoding, errors]) | Calls str.encode element-wise.
[expandtabs](https://numpy.org/devdocs/reference/generated/numpy.char.expandtabs.html#numpy.char.expandtabs)(a[, tabsize]) | Return a copy of each string element where all tab characters are [replace](https://numpy.org/devdocs/reference/generated/numpy.char.replace.html#numpy.char.replace)d by one or more spaces.
[join](https://numpy.org/devdocs/reference/generated/numpy.char.join.html#numpy.char.join)(sep, seq) | Return a string which is the concatenation of the strings in the sequence seq.
[ljust](https://numpy.org/devdocs/reference/generated/numpy.char.ljust.html#numpy.char.ljust)(a, width[, fillchar]) | Return an array with the elements of a left-justified in a string of length width.
[lower](https://numpy.org/devdocs/reference/generated/numpy.char.lower.html#numpy.char.lower)(a) | Return an array with the elements converted to lowercase.
[lstrip](https://numpy.org/devdocs/reference/generated/numpy.char.lstrip.html#numpy.char.lstrip)(a[, chars]) | For each element in a, return a copy with the leading characters removed.
[partition](https://numpy.org/devdocs/reference/generated/numpy.char.partition.html#numpy.char.partition)(a, sep) | Partition each element in a around sep.
[replace](https://numpy.org/devdocs/reference/generated/numpy.char.replace.html#numpy.char.replace)(a, old, new[, count]) | For each element in a, return a copy of the string with all occurrences of substring old replaced by new.
[rjust](https://numpy.org/devdocs/reference/generated/numpy.char.rjust.html#numpy.char.rjust)(a, width[, fillchar]) | Return an array with the elements of a right-justified in a string of length width.
[rpartition](https://numpy.org/devdocs/reference/generated/numpy.char.rpartition.html#numpy.char.rpartition)(a, sep) | Partition ([split](https://numpy.org/devdocs/reference/generated/numpy.char.split.html#numpy.char.split)) each element around the right-most separator.
[rsplit](https://numpy.org/devdocs/reference/generated/numpy.char.rsplit.html#numpy.char.rsplit)(a[, sep, maxsplit]) | For each element in a, return a list of the words in the string, using sep as the delimiter string.
[rstrip](https://numpy.org/devdocs/reference/generated/numpy.char.rstrip.html#numpy.char.rstrip)(a[, chars]) | For each element in a, return a copy with the trailing characters removed.
[split](https://numpy.org/devdocs/reference/generated/numpy.char.rsplit.html#numpy.char.rsplit)(a[, sep, maxsplit]) | For each element in a, return a list of the words in the string, using sep as the delimiter string.
[splitlines](https://numpy.org/devdocs/reference/generated/numpy.char.splitlines.html#numpy.char.splitlines)(a[, keepends]) | For each element in a, return a list of the lines in the element, breaking at line boundaries.
[strip](https://numpy.org/devdocs/reference/generated/numpy.char.strip.html#numpy.char.strip)(a[, chars]) | For each element in a, return a copy with the leading and trailing characters removed.
[swapcase](https://numpy.org/devdocs/reference/generated/numpy.char.swapcase.html#numpy.char.swapcase)(a) | Return element-wise a copy of the string with [upper](https://numpy.org/devdocs/reference/generated/numpy.char.upper.html#numpy.char.upper)case characters converted to lowercase and vice versa.
[title](https://numpy.org/devdocs/reference/generated/numpy.char.title.html#numpy.char.title)(a) | Return element-wise title cased version of string or unicode.
[translate](https://numpy.org/devdocs/reference/generated/numpy.char.translate.html#numpy.char.translate)(a, table[, deletechars]) | For each element in a, return a copy of the string where all characters occurring in the optional argument deletechars are removed, and the remaining characters have been mapped through the given translation table.
[upper](https://numpy.org/devdocs/reference/generated/numpy.char.upper.html#numpy.char.upper)(a) | Return an array with the elements converted to uppercase.
[zfill](https://numpy.org/devdocs/reference/generated/numpy.char.zfill.html#numpy.char.zfill)(a, width) | Return the numeric string left-filled with zeros

## Comparison

Unlike the standard numpy comparison operators, the ones in the *char*
module strip trailing whitespace characters before performing the
comparison.

method | description
---|---
[equal](https://numpy.org/devdocs/reference/generated/numpy.char.equal.html#numpy.char.equal)(x1, x2) | Return (x1 == x2) element-wise.
[not_equal](https://numpy.org/devdocs/reference/generated/numpy.char.not_equal.html#numpy.char.not_equal)(x1, x2) | Return (x1 != x2) element-wise.
[greater_equal](https://numpy.org/devdocs/reference/generated/numpy.char.greater_equal.html#numpy.char.greater_equal)(x1, x2) | Return (x1 >= x2) element-wise.
[less_equal](https://numpy.org/devdocs/reference/generated/numpy.char.less_equal.html#numpy.char.less_equal)(x1, x2) | Return (x1 <= x2) element-wise.
[greater](https://numpy.org/devdocs/reference/generated/numpy.char.greater.html#numpy.char.greater)(x1, x2) | Return (x1 > x2) element-wise.
[less](https://numpy.org/devdocs/reference/generated/numpy.char.less.html#numpy.char.less)(x1, x2) | Return (x1 < x2) element-wise.
[compare_chararrays](https://numpy.org/devdocs/reference/generated/numpy.char.compare_chararrays.html#numpy.char.compare_chararrays)(a, b, cmp_op, rstrip) | Performs element-wise comparison of two string arrays using the comparison operator specified by cmp_op.

## String information

method | description
---|---
[count](https://numpy.org/devdocs/reference/generated/numpy.char.count.html#numpy.char.count)(a, sub[, start, end]) | Returns an array with the number of non-overlapping occurrences of substring sub in the range [start, end].
[endswith](https://numpy.org/devdocs/reference/generated/numpy.char.endswith.html#numpy.char.endswith)(a, suffix[, start, end]) | Returns a boolean array which is True where the string element in a ends with suffix, otherwise False.
[find](https://numpy.org/devdocs/reference/generated/numpy.char.find.html#numpy.char.find)(a, sub[, start, end]) | For each element, return the lowest [index](https://numpy.org/devdocs/reference/generated/numpy.char.index.html#numpy.char.index) in the string where substring sub is found.
[index](https://numpy.org/devdocs/reference/generated/numpy.char.index.html#numpy.char.index)(a, sub[, start, end]) | Like find, but raises ValueError when the substring is not found.
[isalpha](https://numpy.org/devdocs/reference/generated/numpy.char.isalpha.html#numpy.char.isalpha)(a) | Returns true for each element if all characters in the string are alphabetic and there is at least one character, false otherwise.
[isalnum](https://numpy.org/devdocs/reference/generated/numpy.char.isalnum.html#numpy.char.isalnum)(a) | Returns true for each element if all characters in the string are alphanumeric and there is at least one character, false otherwise.
[isdecimal](https://numpy.org/devdocs/reference/generated/numpy.char.isdecimal.html#numpy.char.isdecimal)(a) | For each element, return True if there are only decimal characters in the element.
[isdigit](https://numpy.org/devdocs/reference/generated/numpy.char.isdigit.html#numpy.char.isdigit)(a) | Returns true for each element if all characters in the string are digits and there is at least one character, false otherwise.
[islower](https://numpy.org/devdocs/reference/generated/numpy.char.islower.html#numpy.char.islower)(a) | Returns true for each element if all cased characters in the string are lowercase and there is at least one cased character, false otherwise.
[isnumeric](https://numpy.org/devdocs/reference/generated/numpy.char.isnumeric.html#numpy.char.isnumeric)(a) | For each element, return True if there are only numeric characters in the element.
[isspace](https://numpy.org/devdocs/reference/generated/numpy.char.isspace.html#numpy.char.isspace)(a) | Returns true for each element if there are only whitespace characters in the string and there is at least one character, false otherwise.
[istitle](https://numpy.org/devdocs/reference/generated/numpy.char.istitle.html#numpy.char.istitle)(a) | Returns true for each element if the element is a titlecased string and there is at least one character, false otherwise.
[isupper](https://numpy.org/devdocs/reference/generated/numpy.char.isupper.html#numpy.char.isupper)(a) | Returns true for each element if all cased characters in the string are uppercase and there is at least one character, false otherwise.
[rfind](https://numpy.org/devdocs/reference/generated/numpy.char.rfind.html#numpy.char.rfind)(a, sub[, start, end]) | For each element in a, return the highest index in the string where substring sub is found, such that sub is contained within [start, end].
[rindex](https://numpy.org/devdocs/reference/generated/numpy.char.rindex.html#numpy.char.rindex)(a, sub[, start, end]) | Like rfind, but raises ValueError when the substring sub is not found.
[startswith](https://numpy.org/devdocs/reference/generated/numpy.char.startswith.html#numpy.char.startswith)(a, prefix[, start, end]) | Returns a boolean array which is True where the string element in a starts with prefix, otherwise False.
[str_len](https://numpy.org/devdocs/reference/generated/numpy.char.str_len.html#numpy.char.str_len)(a) | Return len(a) element-wise.

## Convenience class

method | description
---|---
[array](https://numpy.org/devdocs/reference/generated/numpy.char.array.html#numpy.char.array)(obj[, itemsize, copy, unicode, order]) | Create a [chararray](https://numpy.org/devdocs/reference/generated/numpy.char.chararray.html#numpy.char.chararray).
[asarray](https://numpy.org/devdocs/reference/generated/numpy.char.asarray.html#numpy.char.asarray)(obj[, itemsize, unicode, order]) | Convert the input to a [chararray](https://numpy.org/devdocs/reference/generated/numpy.char.chararray.html#numpy.char.chararray), copying the data only if necessary.
[chararray](https://numpy.org/devdocs/reference/generated/numpy.char.chararray.html#numpy.char.chararray)(shape[, itemsize, unicode, …]) | Provides a convenient view on arrays of string and unicode values.
