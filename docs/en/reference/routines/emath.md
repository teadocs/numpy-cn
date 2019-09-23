# Mathematical functions with automatic domain (``numpy.emath``)

::: tip Note

``numpy.emath`` is a preferred alias for [``numpy.lib.scimath``](#module-numpy.lib.scimath),
available after [``numpy``](index.html#module-numpy) is imported.

:::

Wrapper functions to more user-friendly calling of certain math functions
whose output data-type is different than the input data-type in certain
domains of the input.

For example, for functions like ``log`` with branch cuts, the versions in this
module provide the mathematically valid answers in the complex plane:

``` python
>>> import math
>>> from numpy.lib import scimath
>>> scimath.log(-math.exp(1)) == (1+1j*math.pi)
True
```

Similarly, ``sqrt``, other base logarithms, ``power`` and trig functions are
correctly handled.  See their respective docstrings for specific examples.
