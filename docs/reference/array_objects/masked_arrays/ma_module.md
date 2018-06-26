# numpy.ma模块

## 解释

Masked arrays are arrays that may have missing or invalid entries. The ``numpy.ma`` module provides a nearly work-alike replacement for numpy that supports data arrays with masks.

## What is a masked array?

In many circumstances, datasets can be incomplete or tainted by the presence of invalid data. For example, a sensor may have failed to record a data, or recorded an invalid value. The ``numpy.ma`` module provides a convenient way to address this issue, by introducing masked arrays.

A masked array is the combination of a standard ``numpy.ndarray`` and a mask. A mask is either ``nomask``, indicating that no value of the associated array is invalid, or an array of booleans that determines for each element of the associated array whether the value is valid or not. When an element of the mask is ``False``, the corresponding element of the associated array is valid and is said to be unmasked. When an element of the mask is ``True``, the corresponding element of the associated array is said to be masked (invalid).

The package ensures that masked entries are not used in computations.

As an illustration, let’s consider the following dataset:

```python
>>> import numpy as np
>>> import numpy.ma as ma
>>> x = np.array([1, 2, 3, -1, 5])
```

We wish to mark the fourth entry as invalid. The easiest is to create a masked array:

```python
>>> mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0])
```

We can now compute the mean of the dataset, without taking the invalid data into account:

```python
>>> mx.mean()
2.75
```

## The ``numpy.ma`` module

The main feature of the ``numpy.ma`` module is the ``MaskedArray`` class, which is a subclass of ``numpy.ndarray``. The class, its attributes and methods are described in more details in the MaskedArray class section.

The ``numpy.ma`` module can be used as an addition to ``numpy``:

```python
>>> import numpy as np
>>> import numpy.ma as ma
```

To create an array with the second element invalid, we would do:

```python
>>> y = ma.array([1, 2, 3], mask = [0, 1, 0])
```

To create a masked array where all values close to 1.e20 are invalid, we would do:

```python
>>> z = masked_values([1.0, 1.e20, 3.0, 4.0], 1.e20)
```

For a complete discussion of creation methods for masked arrays please see section Constructing masked arrays.