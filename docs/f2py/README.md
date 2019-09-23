---
meta:
  - name: keywords
    content: NumPy F2PY用户指南和参考手册
  - name: description
    content: 该的目的 F2PY —— Fortran 语言Python接口生成器 —— 是提供的Python和Fortran语言之间的连接。 F2PY是 NumPy（numpy.f2py）的一部分...
---

# F2PY用户指南和参考手册

该的目的 ``F2PY`` —— *Fortran 语言Python接口生成器* —— 是提供的Python和Fortran语言之间的连接。
F2PY是 [NumPy](https://www.numpy.org/)（``numpy.f2py``）的一部分，并且f2py在numpy安装时也可作为独立的命令行工具使用，
这有助于创建/构建可实现的Python C / API扩展模块

- 调用 Fortran 77/90/95 外部子程序和 Fortran 90/95 模块子程序以及C函数;
- 访问 Fortran 77 ``COMMON`` 块和 Fortran 90/95 模块数据，包括可分配数组。

来自Python。

- [打包的三种方法 - 入门](getting-started.html)
  - [快捷的方式](getting-started.html#快捷的方式)
  - [聪明的方式](getting-started.html#聪明的方式)
  - [快捷而聪明的方式](getting-started.html#快捷而聪明的方式)
- [签名文件](signature-file.html)
  - [Python模块块](signature-file.html#Python模块块)
  - [Fortran / C 的例程签名](signature-file.html#fortran-c-的例程签名)
  - [拓展](signature-file.html#拓展)
- [在Python中使用F2PY构建](python-usage.html)
  - [标量参数](python-usage.html#标量参数)
  - [字符串参数](python-usage.html#字符串参数)
  - [数组参数](python-usage.html#数组参数)
  - [回调参数](python-usage.html#回调参数)
  - [常见的块](python-usage.html#常见的块)
  - [Fortran 90模块数据](python-usage.html#Fortran-90模块数据)
- [使用 F2PY](usage.html)
  - [f2py 命令](usage.html#f2py-命令)
  - [Python 模块 numpy.f2py](usage.html#python-模块-numpy-f2py)
- [使用 numpy.distutils 模块](distutils.html)
- [高级 f2py 用法](advanced.html)
  - [将自编写函数添加到F2PY生成的模块](advanced.html#将自编写函数添加到F2PY生成的模块)
  - [修改F2PY生成模块的字典](advanced.html#修改F2PY生成模块的字典)
