# 使用NumPy进行数组编程

## 目录

- Getting into Shape: Intro to NumPy Arrays
- What is Vectorization?
    - Counting: Easy as 1, 2, 3…
    - Buy Low, Sell High
- Intermezzo: Understanding Axes Notation
- Broadcasting
- Array Programming in Action: Examples
    - Clustering Algorithms
    - Amortization Tables
    - Image Feature Extraction
- A Parting Thought: Don’t Over-Optimize
- More Resources

## 前言

It is sometimes said that Python, compared to low-level languages such as C++, improves development time at the expense of runtime. Fortunately, there are a handful of ways to speed up operation runtime in Python without sacrificing ease of use. One option suited for fast numerical operations is NumPy, which deservedly bills itself as the fundamental package for scientific computing with Python.

Granted, few people would categorize something that takes 50 microseconds (fifty millionths of a second) as “slow.” However, computers might beg to differ. The runtime of an operation taking 50 microseconds (50 μs) falls under the realm of microperformance, which can loosely be defined as operations with a runtime between 1 microsecond and 1 millisecond.

Why does speed matter? The reason that microperformance is worth monitoring is that small differences in runtime become amplified with repeated function calls: an incremental 50 μs of overhead, repeated over 1 million function calls, translates to 50 seconds of incremental runtime.

When it comes to computation, there are really three concepts that lend NumPy its power:

- Vectorization
- Broadcasting
- Indexing

In this tutorial, you’ll see step by step **how to take advantage of vectorization and broadcasting**, so that you can use NumPy to its full capacity. While you will use some indexing in practice here, NumPy’s complete indexing schematics, which extend Python’s [slicing syntax](https://docs.python.org/3/reference/expressions.html?highlight=slice#slicings), are their own beast. If you’re looking to read more on NumPy [indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html), grab some coffee and head to the Indexing section in the NumPy docs.

## Getting into Shape: Intro to NumPy Arrays

The fundamental object of NumPy is its ndarray (or numpy.array), an n-dimensional array that is also present in some form in array-oriented languages such as Fortran 90, R, and MATLAB, as well as predecessors APL and J.

Let’s start things off by forming a 3-dimensional array with 36 elements:

```python
>>> import numpy as np

>>> arr = np.arange(36).reshape(3, 4, 3)
>>> arr
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]],

       [[12, 13, 14],
        [15, 16, 17],
        [18, 19, 20],
        [21, 22, 23]],

       [[24, 25, 26],
        [27, 28, 29],
        [30, 31, 32],
        [33, 34, 35]]])
```

Picturing high-dimensional arrays in two dimensions can be difficult. One intuitive way to think about an array’s shape is to simply “read it from left to right.” arr is a 3 by 4 by 3 array:

```python
>>> arr.shape
(3, 4, 3)
```

Visually, arr could be thought of as a container of three 4x3 grids (or a rectangular prism) and would look like this:

Higher dimensional arrays can be tougher to picture, but they will still follow this “arrays within an array” pattern.

Where might you see data with greater than two dimensions?

- [Panel data](https://en.wikipedia.org/wiki/Panel_data) can be represented in three dimensions. Data that tracks attributes of a cohort (group) of individuals over time could be structured as (respondents, dates, attributes). The 1979 [National Longitudinal Survey of Youth](https://www.nlsinfo.org/content/cohorts/nlsy79) follows 12,686 respondents over 27 years. Assuming that you have ~500 directly asked or derived data points per individual, per year, this data would have shape (12686, 27, 500) for a total of 177,604,000 data points.

- Color-image data for multiple images is typically stored in four dimensions. Each image is a three-dimensional array of (height, width, channels), where the channels are usually red, green, and blue (RGB) values. A collection of images is then just (image_number, height, width, channels). One thousand 256x256 RGB images would have shape (1000, 256, 256, 3). (An extended representation is RGBA, where the A–alpha–denotes the level of opacity.)
For more detail on real-world examples of high-dimensional data, see Chapter 2 of François Chollet’s Deep Learning with Python.

For more detail on real-world examples of high-dimensional data, see Chapter 2 of François Chollet’s [Deep Learning with Python](https://realpython.com/asins/1617294438/).

## What is Vectorization?

Vectorization is a powerful ability within NumPy to express operations as occurring on entire arrays rather than their individual elements. Here’s a concise definition from Wes McKinney:

> This practice of replacing explicit loops with array expressions is commonly referred to as vectorization. In general, vectorized array operations will often be one or two (or more) orders of magnitude faster than their pure Python equivalents, with the biggest impact [seen] in any kind of numerical computations. [source](https://www.safaribooksonline.com/library/view/python-for-data/9781449323592/ch04.html)

When looping over an array or any data structure in Python, there’s a lot of overhead involved. Vectorized operations in NumPy delegate the looping internally to highly optimized C and Fortran functions, making for cleaner and faster Python code.

### Counting: Easy as 1, 2, 3…

As an illustration, consider a 1-dimensional vector of True and False for which you want to count the number of “False to True” transitions in the sequence:

```python
>>> np.random.seed(444)

>>> x = np.random.choice([False, True], size=100000)
>>> x
array([ True, False,  True, ...,  True, False,  True])
```

With a Python for-loop, one way to do this would be to evaluate, in pairs, the [truth value](https://docs.python.org/3/library/stdtypes.html#truth-value-testing) of each element in the sequence along with the element that comes right after it:

```python
>>> def count_transitions(x) -> int:
...     count = 0
...     for i, j in zip(x[:-1], x[1:]):
...         if j and not i:
...             count += 1
...     return count
...
>>> count_transitions(x)
24984
```

In vectorized form, there’s no explicit for-loop or direct reference to the individual elements:

```python
>>> np.count_nonzero(x[:-1] < x[1:])
24984
```

How do these two equivalent functions compare in terms of performance? In this particular case, the vectorized NumPy call wins out by a factor of about 70 times:

```python
>>> from timeit import timeit
>>> setup = 'from __main__ import count_transitions, x; import numpy as np'
>>> num = 1000
>>> t1 = timeit('count_transitions(x)', setup=setup, number=num)
>>> t2 = timeit('np.count_nonzero(x[:-1] < x[1:])', setup=setup, number=num)
>>> print('Speed difference: {:0.1f}x'.format(t1 / t2))
Speed difference: 71.0x
```

Technical Detail: Another term is [vector processor](https://blogs.msdn.microsoft.com/nativeconcurrency/2012/04/12/what-is-vectorization/), which is related to a computer’s hardware. When I speak about vectorization here, I’m referring to concept of replacing explicit for-loops with array expressions, which in this case can then be computed internally with a low-level language.

### Buy Low, Sell High

Here’s another example to whet your appetite. Consider the following classic technical interview problem:

> Given a stock’s price history as a sequence, and assuming that you are only allowed to make one purchase and one sale, what is the maximum profit that can be obtained? For example, given prices = (20, 18, 14, 17, 20, 21, 15), the max profit would be 7, from buying at 14 and selling at 21.

(To all of you finance people: no, short-selling is not allowed.)

There is a solution with n-squared [time complexity](https://en.wikipedia.org/wiki/Time_complexity) that consists of taking every combination of two prices where the second price “comes after” the first and determining the maximum difference.

However, there is also an O(n) solution that consists of iterating through the sequence just once and finding the difference between each price and a running minimum. It goes something like this:

```python
>>> def profit(prices):
...     max_px = 0
...     min_px = prices[0]
...     for px in prices[1:]:
...         min_px = min(min_px, px)
...         max_px = max(px - min_px, max_px)
...     return max_px

>>> prices = (20, 18, 14, 17, 20, 21, 15)
>>> profit(prices)
7
```

Can this be done in NumPy? You bet. But first, let’s build a quasi-realistic example:

```python
# Create mostly NaN array with a few 'turning points' (local min/max).
>>> prices = np.full(100, fill_value=np.nan)
>>> prices[[0, 25, 60, -1]] = [80., 30., 75., 50.]

# Linearly interpolate the missing values and add some noise.
>>> x = np.arange(len(prices))
>>> is_valid = ~np.isnan(prices)
>>> prices = np.interp(x=x, xp=x[is_valid], fp=prices[is_valid])
>>> prices += np.random.randn(len(prices)) * 2
```

Here’s what this looks like with [matplotlib](https://realpython.com/python-matplotlib-guide/). The adage is to buy low (green) and sell high (red):

```python
>>> import matplotlib.pyplot as plt

# Warning! This isn't a fully correct solution, but it works for now.
# If the absolute min came after the absolute max, you'd have trouble.
>>> mn = np.argmin(prices)
>>> mx = mn + np.argmax(prices[mn:])
>>> kwargs = {'markersize': 12, 'linestyle': ''}

>>> fig, ax = plt.subplots()
>>> ax.plot(prices)
>>> ax.set_title('Price History')
>>> ax.set_xlabel('Time')
>>> ax.set_ylabel('Price')
>>> ax.plot(mn, prices[mn], color='green', **kwargs)
>>> ax.plot(mx, prices[mx], color='red', **kwargs)
```

## 文章出处

由NumPy中文文档翻译，原作者为 [Brad Solomon](https://realpython.com/team/bsolomon/)，翻译至：[https://realpython.com/numpy-array-programming/](https://realpython.com/numpy-array-programming/) 