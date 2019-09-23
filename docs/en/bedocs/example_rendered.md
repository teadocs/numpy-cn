# Example Rendered

This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.

Every module should have a docstring at the very top of the file.  The
module’s docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line.

- doc.example.``foo``( *var1* ,  *var2* ,  *long_var_name='hi'* )[[source]](https://github.com/numpy/numpy/blob/master/numpy/../../../../../doc/sphinxext/doc/example.py#L37-L123)

    A one-line summary that does not use variable names or the function name.

    Several sentences providing an extended description. Refer to variables using back-ticks, e.g.  *var* .

    **Parameters:**

    type | description
    ---|---
    var1 : array_like | Array_like means all those objects – lists, nested lists, etc. – that can be converted to an array. We can also refer to variables like var1.
    var2 : int | The type above can either refer to an actual Python type (e.g. int), or describe the type of the variable in more detail, e.g. (N,) ndarray or array_like.
    long_var_name : {‘hi’, ‘ho’}, optional | Choices in brackets, default first when optional.

    **Returns:**
    type | description
    ---|---
    type | Explanation of anonymous return value of type type.
    describe : type | Explanation of return value named describe.
    out : type | Explanation of out.
    type_without_description | 

    **Other Parameters:**
    type | description
    ---|---
    only_seldom_used_keywords : type | Explanation
    common_parameters_listed_above : type | Explanation

    **Raises:**

    type | description
    ---|---
    BadException | Because you shouldn’t have done that.

::: tip See also

``otherfunc``

``newfunc``

``thirdfunc``, ``fourthfunc``, ``fifthfunc``

:::

**Notes**

Notes about the implementation algorithm (if needed).

This can have multiple paragraphs.

You may include some math:

![math](/static/images/math/003f271cc4b6ba7e6fb8c6b30c851c95ea8038ba.svg)

And even use a Greek symbol like  inline.

**References**

Cite the relevant literature, e.g. You may also cite these
references in the notes section above.

O. McNoleg, “The integration of GIS, remote sensing, expert systems and adaptive co-kriging for environmental habitat modelling of the Highland Haggis using object-oriented, fuzzy-logic and neural-network techniques,” Computers & Geosciences, vol. 22, pp. 585-588, 1996.

**Examples**

These are written in doctest format, and should illustrate how to use the function.

``` python
>>> a = [1, 2, 3]
>>> print [x + 3 for x in a]
[4, 5, 6]
>>> print "a\n\nb"
a
b
```
