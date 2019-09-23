---
meta:
  - name: keywords
    content: 构建NumPy API和参考文档
  - name: description
    content: 我们目前使用Sphinx为NumPy生成API和参考文档。您将需要Sphinx 1.8.3或更高版本。
---

# 构建NumPy API和参考文档

我们目前使用[Sphinx](https://www.sphinx-doc.org/)为NumPy生成API和参考文档。您将需要Sphinx 1.8.3或更高版本。

如果您只想获取文档，请注意可以在以下位置找到预构建的版本

  - [https://docs.scipy.org/](https://docs.scipy.org/)

有几种不同的格式。

## 说明

如果您通过git获得了NumPy，那么还要获取包含构建文档所需的其他部分的git子模块：

``` python
git submodule init
git submodule update
```

此外，构建文档需要Sphinx扩展
  *plot_directive* ，它随[Matplotlib](https://matplotlib.org/)一起提供。可以通过安装Matplotlib来安装此Sphinx扩展。你还需要python3.6。

由于主要文档的大部分是从numpy via获取
 并检查文档字符串，因此您需要首先构建NumPy，然后安装它以便导入正确的版本。``import numpy``

请注意，你可以例如。将NumPy安装到临时位置并适当地设置PYTHONPATH环境变量。

安装NumPy后，安装SciPy，因为随机模块中的某些图需要[``scipy.special``](https://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special)正确显示。现在您已准备好生成文档，所以写道：

``` python
make html
```

在 ``doc/`` 目录中。如果一切顺利，这将生成包含构建文档的 ``build/html`` 子目录。
如果您收到有关 ``已安装的numpy != current repo git版本`` 的消息，则必须通过设置 ``GITVER`` 或重新安装 NumPy 来覆盖检查。

请注意，目前尚未主动在Windows上构建文档，尽管应该可以。（有关更多信息，请参阅[Sphinx](https://www.sphinx-doc.org/)文档。）

要构建PDF文档，请改为：

``` python
make latex
make -C build/latex all-pdf
```

您需要为此安装Latex。

除了上述内容，您还可以：

``` python
make dist
```

这将重建NumPy，将其安装到临时位置，并以所有格式构建文档。这很可能只会在Unix平台上运行。

以html和pdf格式在[https://docs.scipy.org](https://docs.scipy.org)上发布的NumPy文档也是用。有关如何更新[https://docs.scipy.org的](https://docs.scipy.org)详细信息，请参阅[HOWTO RELEASE](https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt)。``make dist``[](https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt)[](https://docs.scipy.org)

## Sphinx 扩展

NumPy的文档使用了Sphinx的几个自定义扩展。它们在 ``sphinxext/`` 目录中提供（作为git子模块，如上所述），并在构建NumPy文档时自动启用。

如果你想使用第三方项目这些扩展，它们可在 [PyPI](https://pypi.org/) 中作为 [numpydoc](https://python.org/pypi/numpydoc) 包。
