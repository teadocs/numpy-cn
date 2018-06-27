# 例子 

## Data with a given value representing missing data

Let’s consider a list of elements, ``x``, where values of -9999. represent missing data. We wish to compute the average value of the data and the vector of anomalies (deviations from the average):

```python
>>> import numpy.ma as ma
>>> x = [0.,1.,-9999.,3.,4.]
>>> mx = ma.masked_values (x, -9999.)
>>> print mx.mean()
2.0
>>> print mx - mx.mean()
[-2.0 -1.0 -- 1.0 2.0]
>>> print mx.anom()
[-2.0 -1.0 -- 1.0 2.0]
```

## Filling in the missing data
Suppose now that we wish to print that same data, but with the missing values replaced by the average value.

```python
>>> print mx.filled(mx.mean())
[ 0.  1.  2.  3.  4.]
```

## Numerical operations

Numerical operations can be easily performed without worrying about missing values, dividing by zero, square roots of negative numbers, etc.:

```python
>>> import numpy as np, numpy.ma as ma
>>> x = ma.array([1., -1., 3., 4., 5., 6.], mask=[0,0,0,0,1,0])
>>> y = ma.array([1., 2., 0., 4., 5., 6.], mask=[0,0,0,0,0,1])
>>> print np.sqrt(x/y)
[1.0 -- -- 1.0 -- --]
```

Four values of the output are invalid: the first one comes from taking the square root of a negative number, the second from the division by zero, and the last two where the inputs were masked.

## Ignoring extreme values

Let’s consider an array ``d`` of random floats between 0 and 1. We wish to compute the average of the values of d while ignoring any data outside the range ``[0.1, 0.9]``:

```python
>>> print ma.masked_outside(d, 0.1, 0.9).mean()
```
