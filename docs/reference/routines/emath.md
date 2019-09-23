# 具有自动域的数学函数（ ``numpy.emath``）

::: tip 注意

``numpy.emath`` 是 [``numpy.lib.scimath``](#module-numpy.lib.scimath) 的首选别名，
在导入 [``numpy``](index.html#module-numpy) 后可用。

:::

包装器函数对某些数学函数的调用更加用户友好，这些数学函数的输出数据类型与输入的某些域中的输入数据类型不同。

例如，对于带有分支切割的 ``log`` 之类的功能，此模块中的版本在复杂平面中提供数学上有效的答案：

``` python
>>> import math
>>> from numpy.lib import scimath
>>> scimath.log(-math.exp(1)) == (1+1j*math.pi)
True
```

同样，``sqrt``，其他基本对数，``幂``和触发函数也可以正确处理。有关特定示例，请参见其各自的文档。
