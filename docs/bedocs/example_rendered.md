---
meta:
  - name: keywords
    content: NumPy 渲染示例
  - name: description
    content: 这是 example.py 模块的 docstring。模块名称应具有简短的全小写名称。
---

# 渲染示例

这是 example.py 模块的 docstring。模块名称应具有简短的全小写名称。
如果这提高了可读性，则模块名称可能具有下划线。

每个模块都应该在文件的最顶部有一个docstring。
模块的文档字符串可以扩展到多行。
如果您的文档字符串确实延伸到多行，则结束三个引号必须单独在一行上，最好在前面加一个空行。

- doc.example.``foo``( *var1* ,  *var2* ,  *long_var_name='hi'* )[[source]](https://github.com/numpy/numpy/blob/master/numpy/../../../../../doc/sphinxext/doc/example.py#L37-L123)

    不使用变量名称或函数名称的单行摘要。

    几个句子提供了扩展的描述。使用反向标记引用变量，例如：*var* 。

    **参数：**

    类型 | 描述
    ---|---
    var1 : array_like | Array_like 表示可以转换为数组的所有对象、列表、嵌套列表等。我们也可以参考像var1这样的变量。
    var2 : int | 上面的类型可以引用实际的Python类型（例如int），或者更详细地描述变量的类型，例如： (n, ) ndarray 或 array_like。
    long_var_name : {‘hi’, ‘ho’}, optional | 括号中的选项，在可选时默认为默认值。

    **返回值：**

    类型 | 描述
    ---|---
    type | 类型的匿名返回值的说明。
    describe : type | 名为Describe的返回值的说明。
    out : type | 输出说明
    type_without_description | 无

    **其他参数：**

    类型 | 描述
    ---|---
    only_seldom_used_keywords : type | 说明
    common_parameters_listed_above : type | 说明

    **额外：**

    类型 | 描述
    ---|---
    BadException | 因为你不该那么做。

::: tip 另见

``otherfunc``

``newfunc``

``thirdfunc``, ``fourthfunc``, ``fifthfunc``

:::

**注解**

关于实现算法的注解（如果需要）。

这可以有多个段落。

可以包括一些数学公式：

![math](/static/images/math/003f271cc4b6ba7e6fb8c6b30c851c95ea8038ba.svg)

甚至使用像内联的希腊符号。

**参考**

引用相关文献，例如 您也可以在上面的注释部分引用这些参考文献。

O. McNoleg, “The integration of GIS, remote sensing, expert systems and adaptive co-kriging for environmental habitat modelling of the Highland Haggis using object-oriented, fuzzy-logic and neural-network techniques,” Computers & Geosciences, vol. 22, pp. 585-588, 1996.

**示例**

这些是以doctest格式编写的，应该说明如何使用该函数。

``` python
>>> a = [1, 2, 3]
>>> print [x + 3 for x in a]
[4, 5, 6]
>>> print "a\n\nb"
a
b
```
