# numpy.ma模块的常量

除了``MaskedArray``类之外，``numpy.ma``模块还定义了几个常量。

## ``numpy.ma.masked``

``masked``常量是``MaskedArray``的特例，具有float数据类型和null形状。 它用于测试是否屏蔽了掩码数组的特定条目，或者屏蔽掩码数组的一个或多个条目：

```python
>>> x = ma.array([1, 2, 3], mask=[0, 1, 0])
>>> x[1] is ma.masked
True
>>> x[-1] = ma.masked
>>> x
masked_array(data = [1 -- --],
             mask = [False  True  True],
       fill_value = 999999)
```

## ``numpy.ma.nomask``

表示被屏蔽数组没有无效条目的值。 内部使用``nomask``来加速不需要掩码时的计算。

## ``numpy.ma.masked_print_options``

打印掩码数组时使用的字符串代替缺少的数据。 默认情况下，这个字符串是 ``'--'``。