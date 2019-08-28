# Constants

NumPy includes several constants:

- numpy.``Inf``

  IEEE 754 floating point representation of (positive) infinity.

  Use [``inf``](#numpy.inf) because [``Inf``](#numpy.Inf), [``Infinity``](#numpy.Infinity), [``PINF``](#numpy.PINF) and [``infty``](#numpy.infty) are aliases for
  [``inf``](#numpy.inf). For more details, see [``inf``](#numpy.inf).

  ::: tip See Also

  inf

  :::

- numpy.``Infinity``

  IEEE 754 floating point representation of (positive) infinity.

  Use [``inf``](#numpy.inf) because [``Inf``](#numpy.Inf), [``Infinity``](#numpy.Infinity), [``PINF``](#numpy.PINF) and [``infty``](#numpy.infty) are aliases for
  [``inf``](#numpy.inf). For more details, see [``inf``](#numpy.inf).

  ::: tip See Also

  inf

  :::

- numpy.``NAN``

  IEEE 754 floating point representation of Not a Number (NaN).

  [``NaN``](#numpy.NaN) and [``NAN``](#numpy.NAN) are equivalent definitions of [``nan``](#numpy.nan). Please use
  [``nan``](#numpy.nan) instead of [``NAN``](#numpy.NAN).

  ::: tip See Also

  nan

  :::

- numpy.``NINF``

  IEEE 754 floating point representation of negative infinity.

  **Returns** 

  y : *float* (A floating point representation of negative infinity.)

  ::: tip See Also

  isinf : Shows which elements are positive or negative infinity

  isposinf : Shows which elements are positive infinity

  isneginf : Shows which elements are negative infinity

  isnan : Shows which elements are Not a Number

  isfinite : Shows which elements are finite (not one of Not a Number,
  positive infinity and negative infinity)

  :::

  ::: tip Notes

  NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
  (IEEE 754). This means that Not a Number is not equivalent to infinity.
  Also that positive infinity is not equivalent to negative infinity. But
  infinity is equivalent to positive infinity.

  :::

  Examples

  ``` python
  >>> np.NINF
  -inf
  >>> np.log(0)
  -inf
  ```

- numpy.``NZERO``

  IEEE 754 floating point representation of negative zero.

  **Returns**

  y : *float* A (floating point representation of negative zero.)

  ::: tip See Also

  PZERO : Defines positive zero.

  isinf : Shows which elements are positive or negative infinity.

  isposinf : Shows which elements are positive infinity.

  isneginf : Shows which elements are negative infinity.

  isnan : Shows which elements are Not a Number.

  isfinite : *Shows which elements are finite - not one of* (Not a Number, positive infinity and negative infinity.)
  
  :::

  ::: tip Notes

  NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
  (IEEE 754). Negative zero is considered to be a finite number.

  :::

  Examples

  ``` python
  >>> np.NZERO
  -0.0
  >>> np.PZERO
  0.0
  ```

  ``` python
  >>> np.isfinite([np.NZERO])
  array([ True])
  >>> np.isnan([np.NZERO])
  array([False])
  >>> np.isinf([np.NZERO])
  array([False])
  ```

- numpy.``NaN``

  IEEE 754 floating point representation of Not a Number (NaN).

  [``NaN``](#numpy.NaN) and [``NAN``](#numpy.NAN) are equivalent definitions of [``nan``](#numpy.nan). Please use
  [``nan``](#numpy.nan) instead of [``NaN``](#numpy.NaN).

  ::: tip See Also

  nan

  :::

- numpy.``PINF``

  IEEE 754 floating point representation of (positive) infinity.

  Use [``inf``](#numpy.inf) because [``Inf``](#numpy.Inf), [``Infinity``](#numpy.Infinity), [``PINF``](#numpy.PINF) and [``infty``](#numpy.infty) are aliases for
  [``inf``](#numpy.inf). For more details, see [``inf``](#numpy.inf).

  ::: tip See Also

  inf

  :::

- numpy.``PZERO``

  IEEE 754 floating point representation of positive zero.

  **Returns**

  y : *float* (A floating point representation of positive zero.)

  ::: tip See Also

  NZERO : Defines negative zero.

  isinf : Shows which elements are positive or negative infinity.

  isposinf : Shows which elements are positive infinity.

  isneginf : Shows which elements are negative infinity.

  isnan : Shows which elements are Not a Number.

  isfinite : *Shows which elements are finite - not one of* (Not a Number, positive infinity and negative infinity.)
  
  :::

  ::: tip Notes

  NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
  (IEEE 754). Positive zero is considered to be a finite number.

  :::

  Examples

  ``` python
  >>> np.PZERO
  0.0
  >>> np.NZERO
  -0.0
  ```

  ``` python
  >>> np.isfinite([np.PZERO])
  array([ True])
  >>> np.isnan([np.PZERO])
  array([False])
  >>> np.isinf([np.PZERO])
  array([False])
  ```

- numpy.``e``

  Euler’s constant, base of natural logarithms, Napier’s constant.

  ``e = 2.71828182845904523536028747135266249775724709369995...``

  ::: tip See Also

  exp : Exponential function log : Natural logarithm

  :::

  References

  [https://en.wikipedia.org/wiki/E_%28mathematical_constant%29](https://en.wikipedia.org/wiki/E_%28mathematical_constant%29)

- numpy.``euler_gamma``

  ``γ = 0.5772156649015328606065120900824024310421...``

  References

  [https://en.wikipedia.org/wiki/Euler-Mascheroni_constant](https://en.wikipedia.org/wiki/Euler-Mascheroni_constant)

- numpy.``inf``

  IEEE 754 floating point representation of (positive) infinity.

  Returns y : *float* (A floating point representation of positive infinity.)

  ::: tip See Also

  isinf : Shows which elements are positive or negative infinity

  isposinf : Shows which elements are positive infinity

  isneginf : Shows which elements are negative infinity

  isnan : Shows which elements are Not a Number

  isfinite : Shows which elements are finite (not one of Not a Number, positive infinity and negative infinity)

  :::

  ::: tip Notes

  NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
  (IEEE 754). This means that Not a Number is not equivalent to infinity.
  Also that positive infinity is not equivalent to negative infinity. But
  infinity is equivalent to positive infinity.

  [``Inf``](#numpy.Inf), [``Infinity``](#numpy.Infinity), [``PINF``](#numpy.PINF) and [``infty``](#numpy.infty) are aliases for [``inf``](#numpy.inf).

  Examples

  ``` python
  >>> np.inf
  inf
  >>> np.array([1]) / 0.
  array([ Inf])
  ```

- numpy.``infty``

  IEEE 754 floating point representation of (positive) infinity.

  Use [``inf``](#numpy.inf) because [``Inf``](#numpy.Inf), [``Infinity``](#numpy.Infinity), [``PINF``](#numpy.PINF) and [``infty``](#numpy.infty) are aliases for
  [``inf``](#numpy.inf). For more details, see [``inf``](#numpy.inf).

  ::: tip See Also

  inf

  :::  

- numpy.``nan``

  IEEE 754 floating point representation of Not a Number (NaN).

  Returns y : A floating point representation of Not a Number.

  ::: tip See Also

  isnan : Shows which elements are Not a Number.

  isfinite : Shows which elements are finite (not one of
  Not a Number, positive infinity and negative infinity)

  :::

  ::: tip Notes

  NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
  (IEEE 754). This means that Not a Number is not equivalent to infinity.

  [``NaN``](#numpy.NaN) and [``NAN``](#numpy.NAN) are aliases of [``nan``](#numpy.nan).

  :::

  Examples

  ``` python
  >>> np.nan
  nan
  >>> np.log(-1)
  nan
  >>> np.log([-1, 1, 2])
  array([        NaN,  0.        ,  0.69314718])
  ```

- numpy.``newaxis``

  A convenient alias for None, useful for indexing arrays.

  ::: tip See Also

  [``numpy.doc.indexing``](https://numpy.org/devdocs/user/basics.indexing.html#module-numpy.doc.indexing)

  :::
  
  Examples

  ``` python
  >>> newaxis is None
  True
  >>> x = np.arange(3)
  >>> x
  array([0, 1, 2])
  >>> x[:, newaxis]
  array([[0],
  [1],
  [2]])
  >>> x[:, newaxis, newaxis]
  array([[[0]],
  [[1]],
  [[2]]])
  >>> x[:, newaxis] * x
  array([[0, 0, 0],
  [0, 1, 2],
  [0, 2, 4]])
  ```

  Outer product, same as ``outer(x, y)``:

  ``` python
  >>> y = np.arange(3, 6)
  >>> x[:, newaxis] * y
  array([[ 0,  0,  0],
  [ 3,  4,  5],
  [ 6,  8, 10]])
  ```

  ``x[newaxis, :]`` is equivalent to ``x[newaxis]`` and ``x[None]``:

  ``` python
  >>> x[newaxis, :].shape
  (1, 3)
  >>> x[newaxis].shape
  (1, 3)
  >>> x[None].shape
  (1, 3)
  >>> x[:, newaxis].shape
  (3, 1)
  ```

- numpy.``pi``

  ``pi = 3.1415926535897932384626433...``

  References

  [https://en.wikipedia.org/wiki/Pi](https://en.wikipedia.org/wiki/Pi)
