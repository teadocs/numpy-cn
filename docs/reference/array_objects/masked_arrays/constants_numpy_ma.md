# numpy.ma模块的常量

In addition to the ``MaskedArray`` class, the ``numpy.ma`` module defines several constants.

## ``numpy.ma.masked``

The ``masked`` constant is a special case of ``MaskedArray``, with a float datatype and a null shape. It is used to test whether a specific entry of a masked array is masked, or to mask one or several entries of a masked array:

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

Value indicating that a masked array has no invalid entry. ``nomask`` is used internally to speed up computations when the mask is not needed.

## ``numpy.ma.masked_print_options``

String used in lieu of missing data when a masked array is printed. By default, this string is ``'--'``.