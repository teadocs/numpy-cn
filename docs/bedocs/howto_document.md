---
meta:
  - name: keywords
    content: 一份给NumPy/SciPy的文档做贡献的指南
  - name: description
    content: 将Sphinx与numpy约定结合使用时，应使用numpydoc扩展名，以便正确处理文档字符串。
---

# 一份给NumPy/SciPy的文档做贡献的指南

将[Sphinx](https://www.sphinx-doc.org/)与numpy约定结合使用时，应使用``numpydoc``扩展名，以便正确处理文档字符串。例如，Sphinx ``Parameters``将从您的docstring中提取该
 部分并将其转换为字段列表。使用``numpydoc``也将避免普通Sphinx在遇到``-------------``像sphinx期望在文档字符串中找到的节标题（例如）之类的numpy docstring约定时产生的reStructuredText错误。

本文档中描述的某些功能需要最新版本的
 ``numpydoc``。例如，**Yields**部分以``numpydoc``0.6 加入
 。

它可以从：

- [PyPI上的numpydoc](https://pypi.python.org/pypi/numpydoc)
- [GitHub上的numpydoc](https://github.com/numpy/numpydoc/)

请注意，对于numpy中的文档，没有必要
 在示例的开头进行。但是，某些子模块（例如，默认情况下不会导入），您必须明确包含它们：``import numpy as np``fft``

``` python
import numpy.fft
```

之后你可以使用它：

``` python
np.fft.fft2(...)
```

**为方便起见，** **下面包含**[格式标准](https://numpydoc.readthedocs.io/en/latest/format.html) **和示例**

## numpydoc docstring指南

本文档描述了与[Sphinx](https://www.sphinx-doc.org/)的numpydoc扩展一起使用的文档字符串的语法和最佳实践。

::: tip 注意

有关附带的示例，请参见[example.py](#example)。

本文档中描述的某些功能需要最新版本的  ``numpydoc``。例如，**Yields**部分以``numpydoc``0.6 加入。

:::

### 概述

我们主要遵循这里描述的标准Python样式约定：

- [C代码风格指南](https://python.org/dev/peps/pep-0007/)
- [Python代码的样式指南](https://python.org/dev/peps/pep-0008/)
- [Docstring约定](https://python.org/dev/peps/pep-0257/)

关于代码文档的其他PEP：

- [文档字符串处理框架](https://python.org/dev/peps/pep-0256/)
- [Docutils设计规范](https://python.org/dev/peps/pep-0258/)

使用代码检查器：

- [pylint的](https://www.logilab.org/857)
- [pyflakes](https://pypi.python.org/pypi/pyflakes)
- [pep8.py](http://svn.browsershots.org/trunk/devtools/pep8/pep8.py)
- [flake8](https://pypi.python.org/pypi/flake8)
- [vim-flake8](https://github.com/nvie/vim-flake8)插件，用于使用flake8自动检查语法和样式

### 导入约定

在整个NumPy源和文档中使用以下导入约定：

``` python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
```

不要缩写``scipy``。没有激励用例在现实世界中缩写它，所以我们在文档中避免它以避免混淆。

### 文档串标准

文档字符串（docstring）是一个描述模块，函数，类或方法定义的字符串。docstring是object（``object.__doc__``）的一个特殊属性，为了保持一致性，它被三重双引号括起来，即：

``` python
"""This is the form of a docstring.

It can be spread over several lines.

"""
```

NumPy，[SciPy](https://www.scipy.org)和scikits遵循文档字符串的通用约定，提供一致性，同时还允许我们的工具链生成格式良好的参考指南。本文档描述了当前社区对此类标准的共识。如果您有改进建议，请将它们发布在[numpy-discussion列表中](https://www.scipy.org/scipylib/mailing-lists.html)。

我们的docstring标准使用[重新结构化文本（reST）](http://docutils.sourceforge.net/rst.html)语法，并使用[Sphinx](http://sphinx.pocoo.org)（一种了解我们正在使用的特定文档样式的预处理器[）](http://docutils.sourceforge.net/rst.html)呈现。虽然可以使用丰富的标记集，但我们将自己限制在一个非常基本的子集中，以便提供在纯文本终端上易于阅读的文档字符串。

一个指导原则是文本的人类读者优先于扭曲文档字符串，因此我们的工具产生了良好的输出。我们编写了预处理器来帮助[Sphinx](http://sphinx.pocoo.org)完成任务，而不是牺牲文档字符串的可读性。

文档字符串行的长度应保持为75个字符，以便于阅读文本终端中的文档字符串。

### 章节

docstring由标题分隔的许多部分组成（弃用警告除外）。每个标题应以连字符加下划线，并且章节顺序应与下面的描述一致。

函数文档字符串的各个部分是：


1. Short summary

    A one-line summary that does not use variable names or the function name, e.g.

    ``` python
    def add(a, b):
      """The sum of two numbers.

      """
    ```

    The function signature is normally found by introspection and displayed by the help function. For some functions (notably those written in C) the signature is not available, so we have to specify it as the first line of the docstring:

    ``` python
    """
    add(a, b)

    The sum of two numbers.

    """
    ```
1. Deprecation warning

    A section (use if applicable) to warn users that the object is deprecated. Section contents should include:

      - In what NumPy version the object was deprecated, and when it will be removed.
      - Reason for deprecation if this is useful information (e.g., object is superseded, duplicates functionality found elsewhere, etc.).
      - New recommended way of obtaining the same functionality.

    This section should use the ``deprecated`` Sphinx directive instead of an underlined section header.

    ```
    .. deprecated:: 1.6.0
              `ndobj_old` will be removed in NumPy 2.0.0, it is replaced by
              `ndobj_new` because the latter works also with array subclasses.
    ```

1. Extended Summary

    A few sentences giving an extended description. This section should be used to clarify functionality, not to discuss implementation detail or background theory, which should rather be explored in the Notes section below. You may refer to the parameters and the function name, but parameter descriptions still belong in the Parameters section.

1. Parameters

    Description of the function arguments, keywords and their respective types.

    ```
    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y
        Description of parameter `y` (with type not specified)
    ```

    Enclose variables in single backticks. The colon must be preceded by a space, or omitted if the type is absent.

    For the parameter types, be as precise as possible. Below are a few examples of parameters and their types.

    ```
    Parameters
    ----------
    filename : str
    copy : bool
    dtype : data-type
    iterable : iterable object
    shape : int or tuple of int
    files : list of str
    ```

    If it is not necessary to specify a keyword argument, use optional:

    ```
    x : int, optional
    ```

    Optional keyword parameters have default values, which are displayed as part of the function signature. They can also be detailed in the description:

    ```
    Description of parameter `x` (the default is -1, which implies summation
    over all axes).
    ```

    When a parameter can only assume one of a fixed set of values, those values can be listed in braces, with the default appearing first:

    ```
    order : {'C', 'F', 'A'}
        Description of `order`.
    ```

    When two or more input parameters have exactly the same type, shape and description, they can be combined:

    ```
    x1, x2 : array_like
        Input arrays, description of `x1`, `x2`.
    ```

1. Returns

    Explanation of the returned values and their types. Similar to the **Parameters** section, except the name of each return value is optional. The type of each return value is always required:

    ```
    Returns
    -------
    int
        Description of anonymous integer return value.
    ```

    If both the name and type are specified, the Returns section takes the same form as the **Parameters** section:

    ```
    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    Yields
    ```

    Explanation of the yielded values and their types. This is relevant to generators only. Similar to the **Returns** section in that the name of each value is optional, but the type of each value is always required:

    ```
    Yields
    ------
    int
        Description of the anonymous integer return value.
    ```

    If both the name and type are specified, the Yields section takes the same form as the **Returns** section:

    ```
    Yields
    ------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    ```

    Support for the **Yields** section was added in [numpydoc](https://github.com/numpy/numpydoc) version 0.6.

1. Receives

    Explanation of parameters passed to a generator’s ``.send()`` method, formatted as for Parameters, above. Since, like for Yields and Returns, a single object is always passed to the method, this may describe either the single parameter, or positional arguments passed as a tuple. If a docstring includes Receives it must also include Yields.

1. Other Parameters

    An optional section used to describe infrequently used parameters. It should only be used if a function has a large number of keyword parameters, to prevent cluttering the Parameters section.

1. Raises

    An optional section detailing which errors get raised and under what conditions:

    ```
    Raises
    ------
    LinAlgException
        If the matrix is not numerically invertible.
    ```

    This section should be used judiciously, i.e., only for errors that are non-obvious or have a large chance of getting raised.

1. Warns

    An optional section detailing which warnings get raised and under what conditions, formatted similarly to Raises.
    
1. Warnings

    An optional section with cautions to the user in free text/reST.

1. See Also

    An optional section used to refer to related code. This section can be very useful, but should be used judiciously. The goal is to direct users to other functions they may not be aware of, or have easy means of discovering (by looking at the module docstring, for example). Routines whose docstrings further explain parameters used by this function are good candidates.

    As an example, for ``numpy.mean`` we would have:

    ```
    See Also
    --------
    average : Weighted average
    ```

    When referring to functions in the same sub-module, no prefix is needed, and the tree is searched upwards for a match.

    Prefix functions from other sub-modules appropriately. E.g., whilst documenting the random module, refer to a function in fft by

    ```
    fft.fft2 : 2-D fast discrete Fourier transform
    ```

    When referring to an entirely different module:

    ```
    scipy.random.norm : Random variates, PDFs, etc.
    ```

    Functions may be listed without descriptions, and this is preferable if the functionality is clear from the function name:

    ```
    See Also
    --------
    func_a : Function a with its description.
    func_b, func_c_, func_d
    func_e
    ```

1. Notes

    An optional section that provides additional information about the code, possibly including a discussion of the algorithm. This section may include mathematical equations, written in [LaTeX](https://www.latex-project.org/) format:

    ```
    The FFT is a fast implementation of the discrete Fourier transform:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}
    ```

    Equations can also be typeset underneath the math directive:

    ```
    The discrete-time Fourier time-convolution property states that

    .. math::

        x(n) * y(n) \Leftrightarrow X(e^{j\omega } )Y(e^{j\omega } )\\
        another equation here
    ```

    Math can furthermore be used inline, i.e.

    ```
    The value of :math:`\omega` is larger than 5.
    ```

    Variable names are displayed in typewriter font, obtained by using ``\mathtt{var}``:

    ```
    We square the input parameter `alpha` to obtain
    :math:`\mathtt{alpha}^2`.
    ```

    Note that LaTeX is not particularly easy to read, so use equations sparingly.

    Images are allowed, but should not be central to the explanation; users viewing the docstring as text must be able to comprehend its meaning without resorting to an image viewer. These additional illustrations are included using:

    ```
    .. image:: filename
    ```

    where filename is a path relative to the reference guide source directory.

1. References

    References cited in the notes section may be listed here, e.g. if you cited the article below using the text ``[1]_``, include it as in the list as follows:

    ```
    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
      expert systems and adaptive co-kriging for environmental habitat
      modelling of the Highland Haggis using object-oriented, fuzzy-logic
      and neural-network techniques," Computers & Geosciences, vol. 22,
      pp. 585-588, 1996.
    ```

    which renders as [[1]](#id2):

    O. McNoleg, “The integration of GIS, remote sensing, expert systems and adaptive co-kriging for environmental habitat modelling of the Highland Haggis using object-oriented, fuzzy-logic and neural-network techniques,” Computers & Geosciences, vol. 22, pp. 585-588, 1996.

    Referencing sources of a temporary nature, like web pages, is discouraged. References are meant to augment the docstring, but should not be required to understand it. References are numbered, starting from one, in the order in which they are cited.

    ::: danger Warning

    References will break tables

    Where references like [1] appear in a tables within a numpydoc docstring, the table markup will be broken by numpydoc processing. See [numpydoc issue #130](https://github.com/numpy/numpydoc/issues/130)

    :::

1. Examples

    An optional section for examples, using the [doctest](https://docs.python.org/library/doctest.html) format. This section is meant to illustrate usage, not to provide a testing framework – for that, use the ``tests/`` directory. While optional, this section is very strongly encouraged.

    When multiple examples are provided, they should be separated by blank lines. Comments explaining the examples should have blank lines both above and below them:

    ``` python
    >>> np.add(1, 2)
    3

    Comment explaining the second example

    >>> np.add([1, 2], [3, 4])
    array([4, 6])
    ```

    The example code may be split across multiple lines, with each line after the first starting with ‘… ‘:

    ``` python
    >>> np.add([[1, 2], [3, 4]],
    ...        [[5, 6], [7, 8]])
    array([[ 6,  8],
          [10, 12]])
    ```

    For tests with a result that is random or platform-dependent, mark the output as such:

    ``` python
    >>> import numpy.random
    >>> np.random.rand(2)
    array([ 0.35773152,  0.38568979])  #random
    ```

    You can run examples as doctests using:

    ``` python
    >>> np.test(doctests=True)
    >>> np.linalg.test(doctests=True)  # for a single module
    ```

    In IPython it is also possible to run individual examples simply by copy-pasting them in doctest mode:

    ``` python
    In [1]: %doctest_mode
    Exception reporting mode: Plain
    Doctest mode is: ON
    >>> %paste
    import numpy.random
    np.random.rand(2)
    ## -- End pasted text --
    array([ 0.8519522 ,  0.15492887])
    ```

    It is not necessary to use the doctest markup ``<BLANKLINE>`` to indicate empty lines in the output. Note that the option to run the examples through ``numpy.test`` is provided for checking if the examples work, not for making the examples part of the testing framework.

    The examples may assume that ``import numpy as np`` is executed before the example code in numpy. Additional examples may make use of matplotlib for plotting, but should import it explicitly, e.g., ``import matplotlib.pyplot as plt``. All other imports, including the demonstrated function, must be explicit.

    When matplotlib is imported in the example, the Example code will be wrapped in matplotlib’s Sphinx `plot directive <http://matplotlib.org/sampledoc/extensions.html>`_. When matplotlib is not explicitly imported, *plot::* can be used directly if matplotlib.sphinxext.plot_directive is loaded as a Sphinx extension in ``conf.py``.

### Documenting 类

#### 类文档字符串

使用与上面相同的部分（除了 ``Returns`` 之外的所有部分都适用）。 此处还应记录 constructor(__init__)，docstring 的 **Parameters** 部分详细说明了构造函数参数。

位于 **参数** 部分下方的 **属性** 部分可用于描述类的非方法属性：

```
Attributes
----------
x : float
    The X coordinate.
y : float
    The Y coordinate.
```

属性和具有自己的文档字符串的属性可以按名称简单列出：

```
Attributes
----------
real
imag
x : float
    The X coordinate
y : float
    The Y coordinate
```

通常，没有必要列出类方法。那些不属于公共API的名称以下划线开头。然而，在某些情况下，一个类可能有很多方法，其中只有少数是相关的（例如，ndarray的子​​类）。然后，有一个额外的**方法**部分变得有用：

``` python
class Photo(ndarray):
    """
    Array with associated photographic information.

    ...

    Attributes
    ----------
    exposure : float
        Exposure in seconds.

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    gamma(n=1.0)
        Change the photo's gamma exposure.

    """
```

如果有必要解释私有方法（谨慎使用！），可以在**扩展摘要**或**Notes**部分中引用它。不要在**方法**部分列出私有方法。

需要注意的是 *自我* 的 *不* 列为方法的第一个参数。

#### 方法文档字符串

像任何其他功能一样记录这些。不要包含
 ``self``在参数列表中。如果方法具有等效函数（例如，许多ndarray方法就是这种情况），则函数docstring应该包含详细的文档，而docstring方法应该引用它。仅提供简要摘要，**另请参阅**方法docstring中的部分。该方法应根据需要使用“ **退货”**或“ **收益”**部分。

### 记录类实例

属于NumPy API的类的实例（例如 *np.r_*  
 *np，c_* ， *np.index_exp* 等）可能需要一些小心。要为这些实例提供有用的文档字符串，我们执行以下操作：

- 单实例：如果只公开一个类的实例，请记录该类。示例可以使用实例名称。
- 多个实例：如果公开了多个实例，则会``__doc__``在运行时写入每个实例的文档字符串并将其分配给实例的
 属性。该类按常规进行记录，并且可以在**Notes**和**See Also** 
部分中提及公开的实例。

### 记录生成器

应记录发电机，就像记录功能一样。唯一的区别是，应该使用**Yields**部分而不是**Returns**部分。[numpydoc](https://github.com/numpy/numpydoc)版本0.6中添加了对**Yields**部分的
 支持。[](https://github.com/numpy/numpydoc)

### 记录常量

使用与适用的功能概述相同的部分：

```
1. summary
2. extended summary (optional)
3. see also (optional)
4. references (optional)
5. examples (optional)
```

常量的文档字符串在文本终端中不可见（常量是不可变类型，因此文档字符串不能像为类实例一样分配给它们），但会出现在使用Sphinx构建的文档中。

### 记录模块

每个模块都应该有一个至少包含摘要行的文档字符串。其他部分是可选的，并且应该在适当时使用与记录函数相同的顺序：

```
1. summary
2. extended summary
3. routine listings
4. see also
5. notes
6. references
7. examples
```

鼓励使用常规列表，特别是对于大型模块，很难通过查看源文件或``__all__``dict 来很好地概述所有功能。

请注意，许可证和作者信息虽然通常包含在源文件中，但不属于文档字符串。

### 其他要记住的要点

- 等式：如上面的**注释**部分所述，LaTeX格式应保持最小。通常可以将方程式显示为Python代码或伪代码，这在终端中更易读。对于内联显示器，请使用双反键（如）。要在上方和下方显示空行，请使用双冒号并缩进代码，例如：``y = np.sin(x)``

```
end of previous sentence::

    y = np.sin(x)
```
- 注释和警告：如果文档字符串中的某些点值得特别强调，则可以在警告的上下文附近（部分内部）使用注释或警告的reST指令。句法：

```
.. warning:: Warning text.

.. note:: Note text.
```

谨慎使用这些，因为它们在文本终端中看起来不太好并且通常不是必需的。警告可能有用的一种情况是标记尚未修复的已知错误。
- array_like：对于带有不仅可以有 *ndarray* 类型的参数的函数，还有可以转换为ndarray的类型（即标量类型，序列类型），可以使用 *array_like* 类型记录这些参数。
- 链接：如果您需要在docstring中包含超链接，请注意某些文档字符串部分未被解析为标准reST，并且在这些部分中，numpydoc可能会被超链接目标混淆，例如：

```
.. _Example: http://www.example.com
```

如果Sphinx构建发出表单警告
 ，那就是正在发生的事情。要避免此问题，请使用内联超链接表单：``WARNING: Unknown target name: "example"``

```
`Example <http://www.example.com>`_
```

### 常见的reST概念

对于段落，缩进很重要并且表示输出中的缩进。新段落标有空行。

使用``*italics*``，``**bold**``并且``monospace``如果需要任何的解释（但不包括变量名和文档测试代码或者多行代码）。变量，模块，函数和类名应该写在单个back-ticks（```numpy```）之间。

在[此示例文档中](http://docutils.sourceforge.net/docs/user/rst/demo.txt)可以找到更广泛的reST标记[示例](http://docutils.sourceforge.net/docs/user/rst/demo.txt) ; 该[快速参考](http://docutils.sourceforge.net/docs/user/rst/quickref.html)是编辑时非常有用。

线间距和压痕很重要，应该仔细遵循。

### 结论

该文档本身是用ReStructuredText编写的。这个[例子](#example)这里显示的格式是可用的。
