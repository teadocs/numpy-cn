---
meta:
  - name: keywords
    content: NumPy 常量
  - name: description
    content: NumPy包括几个常量：
---

# 常量

NumPy包括几个常量：

- numpy.``Inf``

  IEEE 754 浮点表示（正）无穷大。

  使用 [``inf``](#numpy.inf) 是因为 [``Inf``](#numpy.Inf)，[``Infinity``](#numpy.Infinity)，[``PINF``](#numpy.PINF) 和 [``infty``](#numpy.infty) 是 [``inf``](#numpy.inf) 的别名。有关更多详细信息，请参阅 [``inf``](#numpy.inf) 。

  ::: tip 另见

  inf

  :::

- numpy.``Infinity``

  IEEE 754 浮点表示（正）无穷大。

  使用 [``inf``](#numpy.inf) 是因为 [``Inf``](#numpy.Inf)，[``Infinity``](#numpy.Infinity)，[``PINF``](#numpy.PINF) 和 [``infty``](#numpy.infty) 是 [``inf``](#numpy.inf) 的别名。有关更多详细信息，请参阅 [``inf``](#numpy.inf)。

  ::: tip 另见

  inf

  :::

- numpy.``NAN``
  
  IEEE 754 浮点表示非数字（NaN）。

  [``NaN``](#numpy.NaN)  和 [``NAN``](#numpy.NAN) 是 [``nan``](#numpy.nan) 的等价定义。请使用 [``nan``](#numpy.nan) 而不是 [``NAN``](#numpy.NAN)。

  ::: tip 另见

  nan

  :::

- numpy.``NINF``

  IEEE 754 浮点表示负无穷大。

  **返回** 

  y : *float* (负无穷大的浮点表示)

  ::: tip 另见

  isinf : 显示哪些元素为正或负无穷大

  isposinf : 显示哪些元素是正无穷大

  isneginf : 显示哪些元素为负无穷大

  isnan : 显示哪些元素不是数字

  isfinite : 显示哪些元素是有限的（不是非数字，正无穷大和负无穷大中的一个）

  :::

  ::: tip 注意

  NumPy使用IEEE二进制浮点算法标准（IEEE 754）。
  这意味着Not a Number不等于无穷大。
  此外，正无穷大不等于负无穷大。
  但无穷大相当于正无穷大。

  :::

  **例子：**

  ``` python
  >>> np.NINF
  -inf
  >>> np.log(0)
  -inf
  ```

- numpy.``NZERO``

  IEEE 754 浮点表示负零。

  **返回**

  y : *float* A (负零点的浮点表示)

  ::: tip 另见

  PZERO : 定义正零。

  isinf : 显示哪些元素为正或负无穷大。

  isposinf : 显示哪些元素是正无穷大。

  isneginf : 显示哪些元素为负无穷大。

  isnan : 显示哪些元素不是数字。

  isfinite : *显示哪些元素是有限的* - 不是（非数字，正无穷大和负无穷大）之一。
  
  :::

  ::: tip 注意

  NumPy使用IEEE二进制浮点算法标准（IEEE 754）。 负零被认为是有限数。

  :::

  **例子：**

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

  IEEE 754浮点表示非数字（NaN）。

  [``NaN``](#numpy.NaN) 和 [``NAN``](#numpy.NaN)是 [``nan``](#numpy.nan) 的等价定义。 请使用 [``nan``](#numpy.nan) 而不是 [``NaN``](#numpy.NaN)。

  ::: tip 另见

  nan

  :::

- numpy.``PINF``

  IEEE 754 浮点表示（正）无穷大。

  使用 [``inf``](#numpy.inf) 是因为 [``Inf``](#numpy.Inf)，[``Infinity``](#numpy.Infinity)，[``PINF``](#numpy.PINF) 和 [``infty``](#numpy.infty) 是 [``inf``](#numpy.inf) 的别名。有关更多详细信息，请参阅 [``inf``](#numpy.inf) 。

  ::: tip 另见

  inf

  :::

- numpy.``PZERO``

  IEEE 754浮点表示正零。

  **返回**

  y : *float* （正零的浮点表示。）

  ::: tip 另见

  NZERO : 定义负零。

  isinf : 显示哪些元素为正或负无穷大。

  isposinf : 显示哪些元素是正无穷大。

  isneginf : 显示哪些元素为负无穷大。

  isnan : 显示哪些元素不是数字。

  isfinite : *显示哪些元素是有限的* - 不是（非数字，正无穷大和负无穷大）之一。
  
  :::

  ::: tip 注意

  NumPy使用IEEE二进制浮点算法标准（IEEE 754）。正零被认为是有限数。

  :::

  **例子：**

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

  欧拉的常数，自然对数的基础，纳皮尔的常数。

  ``e = 2.71828182845904523536028747135266249775724709369995...``

  ::: tip 另见

  exp : 指数函数日志：自然对数

  :::

  参考

  [https://en.wikipedia.org/wiki/E_%28mathematical_constant%29](https://en.wikipedia.org/wiki/E_%28mathematical_constant%29)

- numpy.``euler_gamma``

  ``γ = 0.5772156649015328606065120900824024310421...``

  参考

  [https://en.wikipedia.org/wiki/Euler-Mascheroni_constant](https://en.wikipedia.org/wiki/Euler-Mascheroni_constant)

- numpy.``inf``

  IEEE 754浮点表示（正）无穷大。

  返回 y : *float* （正无穷大的浮点表示。）

  ::: tip 另见

  isinf : 显示哪些元素为正或负无穷大。

  isposinf : 显示哪些元素是正无穷大。

  isneginf : 显示哪些元素为负无穷大。

  isnan : 显示哪些元素不是数字。

  isfinite : 显示哪些元素是有限的（不是非数字，正无穷大和负无穷大中的一个）

  :::

  ::: tip 注意

  NumPy使用IEEE二进制浮点算法标准（IEEE 754）。
  这意味着Not a Number不等于无穷大。 此外，正无穷大不等于负无穷大。
  但无穷大相当于正无穷大。

  [``Inf``](#numpy.Inf), [``Infinity``](#numpy.Infinity), [``PINF``](#numpy.PINF) 和 [``infty``](#numpy.infty) 是 [``inf``](#numpy.inf) 的别名。

  **例子：**

  ``` python
  >>> np.inf
  inf
  >>> np.array([1]) / 0.
  array([ Inf])
  ```

- numpy.``infty``

  IEEE 754浮点表示（正）无穷大。

  使用 [``inf``](#numpy.inf) 是因为 [``Inf``](#numpy.Inf) ，[``Infinity``](#numpy.Infinity)，[``PINF``](#numpy.PINF) 和 [``infty``](#numpy.infty)  是 [``inf``](#numpy.inf) 的别名。有关更多详细信息，请参阅 [``inf``](#numpy.inf)。

  ::: tip 另见

  inf

  :::  

- numpy.``nan``

  IEEE 754浮点表示非数字（NaN）。

  返回 y : 非数字的浮点表示。

  ::: tip 另见

  isnan : 显示哪些元素不是数字。

  isfinite : 显示哪些元素是有限的（不是非数字，正无穷大和负无穷大中的一个）

  :::

  ::: tip 注意

  NumPy使用IEEE二进制浮点算法标准（IEEE 754）。 这意味着Not a Number不等于无穷大。

  [``NaN``](#numpy.NaN) 和 [``NAN``](#numpy.NAN) 是 [``nan``](#numpy.nan) 的别名。

  :::

  **例子：**

  ``` python
  >>> np.nan
  nan
  >>> np.log(-1)
  nan
  >>> np.log([-1, 1, 2])
  array([        NaN,  0.        ,  0.69314718])
  ```

- numpy.``newaxis``

  None的便捷别名，对索引数组很有用。

  ::: tip 另见

  [``numpy.doc.indexing``](https://numpy.org/devdocs/user/basics.indexing.html#module-numpy.doc.indexing)

  :::
  
  **例子：**

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

  外积，与 ``outer(x, y)`` 相同：

  ``` python
  >>> y = np.arange(3, 6)
  >>> x[:, newaxis] * y
  array([[ 0,  0,  0],
  [ 3,  4,  5],
  [ 6,  8, 10]])
  ```

  ``x[newaxis, :]`` 相当于 ``x[newaxis]`` 和 ``x[None]``：

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

  参考

  [https://en.wikipedia.org/wiki/Pi](https://en.wikipedia.org/wiki/Pi)
