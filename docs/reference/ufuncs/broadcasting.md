# 广播

Each universal function takes array inputs and produces array outputs by performing the core function element-wise on the inputs (where an element is generally a scalar, but can be a vector or higher-order sub-array for generalized ufuncs). Standard broadcasting rules are applied so that inputs not sharing exactly the same shapes can still be usefully operated on. Broadcasting can be understood by four rules:

1. All input arrays with ndim smaller than the input array of largest ndim, have 1’s prepended to their shapes.
1. The size in each dimension of the output shape is the maximum of all the input sizes in that dimension.
1. An input can be used in the calculation if its size in a particular dimension either matches the output size in that dimension, or has value exactly 1.
1. If an input has a dimension size of 1 in its shape, the first data entry in that dimension will be used for all calculations along that dimension. In other words, the stepping machinery of the ufunc will simply not step along that dimension (the stride will be 0 for that dimension).

Broadcasting is used throughout NumPy to decide how to handle disparately shaped arrays; for example, all arithmetic operations (+, -, *, …) between ndarrays broadcast the arrays before operation.

A set of arrays is called “broadcastable” to the same shape if the above rules produce a valid result, i.e., one of the following is true:

1. The arrays all have exactly the same shape.
1. The arrays all have the same number of dimensions and the length of each dimensions is either a common length or 1.
1. The arrays that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2.

**Example**

If ``a.shape`` is (5,1), ``b.shape`` is (1,6), ``c.shape`` is (6,) and ``d.shape`` is () so that d is a scalar, then a, b, c, and d are all broadcastable to dimension (5,6); and

- a acts like a (5,6) array where ``a[:,0]`` is broadcast to the other columns,
- b acts like a (5,6) array where ``b[0,:]`` is broadcast to the other rows,
- c acts like a (1,6) array and therefore like a (5,6) array where ``c[:]`` is broadcast to every row, and finally,
- d acts like a (5,6) array where the single value is repeated.