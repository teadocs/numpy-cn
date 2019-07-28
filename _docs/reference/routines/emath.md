# 自动域数学函数

> **注意**
> ``numpy.emath``是``numpy.lib.scimath``的首选别名，可在导入``numpy``后使用。

包装器用于更加用户友好地调用某些数学函数，这些函数的输出数据类型与输入的某些域中的输入数据类型不同。

例如，对于带有分支切割的``log``这样的函数，此模块中的版本在复杂平面中提供数学上有效的答案：

```python
>>> import math
>>> from numpy.lib import scimath
>>> scimath.log(-math.exp(1)) == (1+1j*math.pi)
True
```

类似地，正确处理``sqrt``和其他基本对数``power``和trig函数。有关具体示例，请参阅各自的文档。