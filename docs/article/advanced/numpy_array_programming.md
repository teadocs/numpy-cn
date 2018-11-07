# 使用NumPy进行数组编程

## 目录

- [Getting into Shape: Intro to NumPy Arrays](#Getting-into-Shape:-Intro-to-NumPy-Arrays)
- [What is Vectorization?](#What-is-Vectorization?)
    - [Counting: Easy as 1, 2, 3…](#Counting:-Easy-as-1,-2,-3…)
    - [Buy Low, Sell High](#Buy-Low,-Sell-High)
- [Intermezzo: Understanding Axes Notation](#Intermezzo:-Understanding-Axes-Notation)
- [Broadcasting](#Broadcasting)
- [Array Programming in Action: Examples](#Array-Programming-in-Action:-Examples)
    - [Clustering Algorithms](#Clustering-Algorithms)
    - [Amortization Tables](#Amortization-Tables)
    - [Image Feature Extraction](#Image-Feature-Extraction)
- [A Parting Thought: Don’t Over-Optimize](#A-Parting-Thought:-Don’t-Over-Optimize)
- [More Resources](#More-Resources)

## 前言

人们有时会说，与C++这种低级语言相比，Python以运行速度为代价改善了开发时间和效率。幸运的是，有一些方法可以在不牺牲易用性的情况下加速Python中的操作运行时。适用于快速数值运算的一个选项是NumPy，它当之无愧地将自己称为使用Python进行科学计算的基本软件包。

当然，很少有人将50微秒（百万分之五十秒）的东西归类为“慢”。然而，计算机可能会有所不同。运行50微秒（50微秒）的运行时属于微执行领域，可以松散地定义为运行时间在1微秒和1毫秒之间的运算。

为什么速度很重要？微观性能值得监控的原因是运行时的小差异会随着重复的函数调用而放大：增量50μs的开销，重复超过100万次函数调用，转换为50秒的增量运行时间。

在计算方面，实际上有三个概念为NumPy提供了强大的功能：

- 矢量化
- 广播
- 索引

在本教程中，你将逐步了解**如何利用矢量化和广播**，以便你可以充分使用NumPy。虽然你在这里将使用一些索引，但NumPy的完整索引原理图(它扩展了Python的[切片语法]((https://docs.python.org/3/reference/expressions.html?highlight=slice#slicings)))是它们自己的工具。如果你想了解有关[NumPy索引](/reference/array_objects/indexing.html)的更多信息，请喝点咖啡，然后前往NumPy文档中的索引部分。

## 进入状态：介绍NumPy数组

NumPy的基本对象是它的ndarray（或numpy.array），这是一个n维数组，它也以某种形式出现在面向数组的语言中，如Fortran 90、R和MATLAB，以及以前的 APL 和 J。

让我们从形成一个包含36个元素的三维数组开始：

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

在二维中描绘高维阵列可能会比较困难。考虑数组形状的一种直观方法是简单地“从左到右读取它”。arr 是一个3乘4乘3的数组：

```python
>>> arr.shape
(3, 4, 3)
```

在视觉上，arr可以被认为是三个4x3网格（或矩形棱镜）的容器，看起来像这样：

![NumPy三维数组](/static/images/article/arr3d.7442cd4e11c6.jpg)

更高维度的数组可能更难以用图像表达出来，但它们仍将遵循这种“阵列内的阵列”模式。

你在哪里可以看到超过两个维度的数据？

- [面板数据](https://en.wikipedia.org/wiki/Panel_data)可以用三维表示。跟踪个体群组（群体）随时间变化的数据可以被构造为（受访者，日期，属性）。 1979年[全国青年纵向调查（iq调查）](https://www.nlsinfo.org/content/cohorts/nlsy79)对27岁以上的12,686名受访者进行了调查。假设每个人每年有大约500个直接询问或派生的数据点，这些数据将具有形状（12686,27,500），总共177,604,000个数据点。
- 用于多幅图像的彩色图像数据通常存储在四个维度中。每个图像是一个三维数组(高度、宽度、通道)，通道通常是红色、绿色和蓝色(RGB)值。然后，图像的集合就是(图像数、高度、宽度、通道)。1，000张256x256 RGB图像将具有形状(1000，256，256，3)。(扩展的表示是RGBA，其中A-alpha-表示不透明的级别。)。

有关高维数据的真实示例的更多详细信息，请参阅FrançoisChollet[使用Python进行深度学习](https://realpython.com/asins/1617294438/)的第2章。

## 什么是矢量化？

矢量化是NumPy中的一种强大功能，可以将操作表达为在整个阵列上而不是在各个元素上发生。以下是Wes McKinney的简明定义：

> 这种用数组表达式替换显式循环的做法通常称为向量化。通常，矢量化数组操作通常比其纯Python等价物快一个或两个（或更多）数量级，在任何类型的数值计算中都具有最大的影响。[查看源码](https://www.safaribooksonline.com/library/view/python-for-data/9781449323592/ch04.html)

在Python中循环数组或任何数据结构时，会涉及很多开销。 NumPy中的向量化操作将内部循环委托给高度优化的C和Fortran函数，从而实现更清晰，更快速的Python代码。

### 计数: 简单的如：1, 2, 3…

作为示例，考虑一个True和False的一维向量，你要为其计算序列中“False to True”转换的数量：

```python
>>> np.random.seed(444)

>>> x = np.random.choice([False, True], size=100000)
>>> x
array([ True, False,  True, ...,  True, False,  True])
```

使用Python for循环，一种方法是成对地评估序列中每个元素的[真值](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)以及紧随其后的元素：

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

在矢量化形式中，没有明确的for循环或直接引用各个元素：

```python
>>> np.count_nonzero(x[:-1] < x[1:])
24984
```

这两个等效函数在性能方面有何比较？ 在这种特殊情况下，向量化的NumPy调用胜出约70倍：

```python
>>> from timeit import timeit
>>> setup = 'from __main__ import count_transitions, x; import numpy as np'
>>> num = 1000
>>> t1 = timeit('count_transitions(x)', setup=setup, number=num)
>>> t2 = timeit('np.count_nonzero(x[:-1] < x[1:])', setup=setup, number=num)
>>> print('Speed difference: {:0.1f}x'.format(t1 / t2))
Speed difference: 71.0x
```

**技术细节**: 另一个术语是[矢量处理器](https://blogs.msdn.microsoft.com/nativeconcurrency/2012/04/12/what-is-vectorization/)，它与计算机的硬件有关。 当我在这里谈论矢量化时，我指的是用数组表达式替换显式for循环的概念，在这种情况下，可以使用低级语言在内部计算。

### 买低，卖高

这是另一个激发你胃口的例子。考虑以下经典技术面试问题：

> 假定一只股票的历史价格是一个序列，假设你只允许进行一次购买和一次出售，那么可以获得的最大利润是多少？例如，假设价格=(20，18，14，17，20，21，15)，最大利润将是7，从14买到21卖。

(对所有金融界人士说：不，卖空是不允许的。)

存在具有n平方[时间复杂度]((https://en.wikipedia.org/wiki/Time_complexity))的解决方案，其包括采用两个价格的每个组合，其中第二价格“在第一个之后”并且确定最大差异。

然而，还有一个O(n)解决方案，它包括迭代序列一次，找出每个价格和运行最小值之间的差异。 它是这样的：

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

这可以用NumPy实现吗？行!没问题。但首先，让我们构建一个准现实的例子：

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

下面是[matplotlib](https://realpython.com/python-matplotlib-guide/)的示例。俗话说：买低(绿)，卖高(红)：

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

![以序列形式显示股票价格历史的图解](/static/images/article/prices.664958f44799.png)

NumPy实现是什么样的？ 虽然没有np.cummin() “直接”，但NumPy的[通用函数（ufuncs）](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs)都有一个accumulate()方法，它的名字暗示了：

```python
>>> cummin = np.minimum.accumulate
```

从纯Python示例扩展逻辑，你可以找到每个价格和运行最小值（元素方面）之间的差异，然后获取此序列的最大值：

```python
>>> def profit_with_numpy(prices):
...     """Price minus cumulative minimum price, element-wise."""
...     prices = np.asarray(prices)
...     return np.max(prices - cummin(prices))

>>> profit_with_numpy(prices)
44.2487532293278
>>> np.allclose(profit_with_numpy(prices), profit(prices))
True
```

这两个具有相同理论时间复杂度的操作如何在实际运行时进行比较？ 首先，让我们采取更长的序列。（此时不一定需要是股票价格的时间序列。）

```python
>>> seq = np.random.randint(0, 100, size=100000)
>>> seq
array([ 3, 23,  8, 67, 52, 12, 54, 72, 41, 10, ..., 46,  8, 90, 95, 93,
       28, 24, 88, 24, 49])
```

现在，对于一个有点不公平的比较：

```python
>>> setup = ('from __main__ import profit_with_numpy, profit, seq;'
...          ' import numpy as np')
>>> num = 250
>>> pytime = timeit('profit(seq)', setup=setup, number=num)
>>> nptime = timeit('profit_with_numpy(seq)', setup=setup, number=num)
>>> print('Speed difference: {:0.1f}x'.format(pytime / nptime))
Speed difference: 76.0x
```

在上面，将profit_with_numpy() 视为伪代码（不考虑NumPy的底层机制），实际上有三个遍历序列：

- cummin(prices) 具有O(n)时间复杂度
- prices - cummin(prices) 是 O(n)的时间复杂度
- max(...) 是O(n)的时间复杂度

这就减少到O(n)，因为O(3n)只剩下O(n)-当n接近无穷大时，n “占主导地位”。

因此，这两个函数具有等价的最坏情况时间复杂度。(不过，顺便提一下，NumPy函数的空间复杂度要高得多。)。但这可能是最不重要的内容。这里我们有一个教训是：虽然理论上的时间复杂性是一个重要的考虑因素，运行时机制也可以发挥很大的作用。NumPy不仅可以委托给C，而且通过一些元素操作和线性代数，它还可以利用多线程中的计算。但是这里有很多因素在起作用，包括所使用的底层库(BLAS/LAPACK/Atlas)，而这些细节完全是另一篇文章的全部内容。

## Intermezzo：理解轴符号

在NumPy中，轴指向多维数组的单个维度：

```python
>>> arr = np.array([[1, 2, 3],
...                 [10, 20, 30]])
>>> arr.sum(axis=0)
array([11, 22, 33])
>>> arr.sum(axis=1)
array([ 6, 60])
```

围绕轴的术语和描述它们的方式可能有点不直观。在Pandas(在NumPy之上构建的库)的文档中，你可能经常看到如下内容：

```
axis : {'index' (0), 'columns' (1)}
```

根据这一描述，你可以争辩说，上面的结果应该是“反向的”。但是，关键是轴指向调用函数的轴。杰克·范德普拉斯很好地阐述了这一点：

> 此处指定轴的方式可能会让来自其他语言的用户感到困惑。AXIS关键字指定将折叠的数组的维度，而不是将要返回的维度。因此，指定Axis=0意味着第一个轴将折叠：对于二维数组，这意味着每列中的值将被聚合。 [查看源码](https://realpython.com/asins/1491912057/)

换句话说，如果将AXIS=0的数组相加，则会使用按列计算的方式折叠数组的行。

考虑到这一区别，让我们继续探讨广播的概念。

## 广播

广播是另一个重要的NumPy抽象。你已经看到了两个NumPy数组(大小相等)之间的操作是按元素操作的：

```python
>>> a = np.array([1.5, 2.5, 3.5])
>>> b = np.array([10., 5., 1.])
>>> a / b
array([0.15, 0.5 , 3.5 ])
```

但是，大小不相等的数组呢？这就是广播的意义所在：

> 术语广播描述了在算术运算期间NumPy如何处理具有不同形状的数组。受某些约束条件的限制，较小的数组会在较大的数组中“广播”，以便它们具有兼容的形状。广播提供了一种向量化数组操作的方法，因此循环是在C而不是Python中进行的。[查看源码](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

当使用两个以上的数组时，广播的实现方式可能会变得乏味。但是，如果只有两个数组，那么可以用两条简短的规则来描述它们的广播能力：

> 在对两个数组进行操作时，NumPy按元素对它们的形状进行比较。它从尾随维度开始，然后继续前进。两个维度在下列情况下是兼容的：
> - 他们是平等的，或者
> - 其中一个是1

非那样做不行。

让我们以一个例子为例，我们想要减去数组的每个列的平均值，元素的平均值：

```python
>>> sample = np.random.normal(loc=[2., 20.], scale=[1., 3.5],
...                           size=(3, 2))
>>> sample
array([[ 1.816 , 23.703 ],
       [ 2.8395, 12.2607],
       [ 3.5901, 24.2115]])
```

在统计术语中，样本由两个独立于两个总体的样本(列)组成，平均值分别为2和20。按列分列的方法应该近似于总体方法(尽管是粗略的，因为样本很小)：

```python
>>> mu = sample.mean(axis=0)
>>> mu
array([ 2.7486, 20.0584])
```

现在，减去列意义是很简单的，因为广播规则检查出来了：

```python
>>> print('sample:', sample.shape, '| means:', mu.shape)
sample: (3, 2) | means: (2,)

>>> sample - mu
array([[-0.9325,  3.6446],
       [ 0.091 , -7.7977],
       [ 0.8416,  4.1531]])
```

下面是一个减去列意义的示例，其中较小的数组被“拉伸”，以便从较大的数组的每一行中减去它：

![NumPy数组广播](/static/images/article/broadcasting.084a0e28dea8.jpg)

**技术细节**：较小的数组或标量不是按字面意义上在内存中展开的：重复的是计算本身。

This extends to [standardizing](https://en.wikipedia.org/wiki/Standard_score) each column as well, making each cell a z-score relative to its respective column:

```python
>>> (sample - sample.mean(axis=0)) / sample.std(axis=0)
array([[-1.2825,  0.6605],
       [ 0.1251, -1.4132],
       [ 1.1574,  0.7527]])
```

However, what if you want to subtract out, for some reason, the row-wise minimums? You’ll run into a bit of trouble:

```python
>>> sample - sample.min(axis=1)
ValueError: operands could not be broadcast together with shapes (3,2) (3,)
```

The problem here is that the smaller array, in its current form, cannot be “stretched” to be shape-compatible with sample. You actually need to expand its dimensionality to meet the broadcasting rules above:

```python
>>> sample.min(axis=1)[:, None]  # 3 minimums across 3 rows
array([[1.816 ],
       [2.8395],
       [3.5901]])

>>> sample - sample.min(axis=1)[:, None]
array([[ 0.    , 21.887 ],
       [ 0.    ,  9.4212],
       [ 0.    , 20.6214]])
```

**Note**: [:, None] is a means by which to expand the dimensionality of an array, to create an axis of length one. [np.newaxis](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#numpy.newaxis) is an alias for None.

There are some significantly more complex cases, too. Here’s a more rigorous definition of when any arbitrary number of arrays of any shape can be broadcast together:

> A set of arrays is called “broadcastable” to the same shape if the following rules produce a valid result, meaning **one of the following is true**:
> 1. The arrays all have exactly the same shape.
> 1. The arrays all have the same number of dimensions, and the length of each dimension is either a common length or 1.
> 1. The arrays that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property #2.
> [source](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting)

This is easier to walk through step by step. Let’s say you have the following four arrays:

```python
>>> a = np.sin(np.arange(10)[:, None])
>>> b = np.random.randn(1, 10)
>>> c = np.full_like(a, 10)
>>> d = 8
```

Before checking shapes, NumPy first converts scalars to arrays with one element:

```python
>>> arrays = [np.atleast_1d(arr) for arr in (a, b, c, d)]
>>> for arr in arrays:
...     print(arr.shape)
...
(10, 1)
(1, 10)
(10, 1)
(1,)
```

Now we can check criterion #1. If all of the arrays have the same shape, a set of their shapes will condense down to one element, because the set() constructor effectively drops duplicate items from its input. This criterion is clearly not met:

```python
>>> len(set(arr.shape for arr in arrays)) == 1
False
```

The first part of criterion #2 also fails, meaning the entire criterion fails:

```python
>>> len(set((arr.ndim) for arr in arrays)) == 1
False
```

The final criterion is a bit more involved:

> The arrays that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property #2.

To codify this, you can first determine the dimensionality of the highest-dimension array and then prepend ones to each shape tuple until all are of equal dimension:

```python
>>> maxdim = max(arr.ndim for arr in arrays)  # Maximum dimensionality
>>> shapes = np.array([(1,) * (maxdim - arr.ndim) + arr.shape
...                    for arr in arrays])
>>> shapes
array([[10,  1],
       [ 1, 10],
       [10,  1],
       [ 1,  1]])
```

Finally, you need to test that the length of each dimension is either (drawn from) a common length, or 1. A trick for doing this is to first mask the array of “shape-tuples” in places where it equals one. Then, you can check if the peak-to-peak (np.ptp()) column-wise differences are all zero:

```python
>>> masked = np.ma.masked_where(shapes == 1, shapes)
>>> np.all(masked.ptp(axis=0) == 0)  # ptp: max - min
True
```

Encapsulated in a single function, this logic looks like this:

```python
>>> def can_broadcast(*arrays) -> bool:
...     arrays = [np.atleast_1d(arr) for arr in arrays]
...     if len(set(arr.shape for arr in arrays)) == 1:
...         return True
...     if len(set((arr.ndim) for arr in arrays)) == 1:
...         return True
...     maxdim = max(arr.ndim for arr in arrays)
...     shapes = np.array([(1,) * (maxdim - arr.ndim) + arr.shape
...                        for arr in arrays])
...     masked = np.ma.masked_where(shapes == 1, shapes)
...     return np.all(masked.ptp(axis=0) == 0)
...
>>> can_broadcast(a, b, c, d)
True
```

Luckily, you can take a shortcut and use np.broadcast() for this sanity-check, although it’s not explicitly designed for this purpose:

```python
>>> def can_broadcast(*arrays) -> bool:
...     try:
...         np.broadcast(*arrays)
...         return True
...     except ValueError:
...         return False
...
>>> can_broadcast(a, b, c, d)
True
```

For those interested in digging a little deeper, [PyArray_Broadcast](https://github.com/numpy/numpy/blob/7dcee7a469ad1bbfef1cd8980dc18bf5869c5391/numpy/core/src/multiarray/iterators.c#L1274) is the underlying C function that encapsulates broadcasting rules.

## Array Programming in Action: Examples

In the following 3 examples, you’ll put vectorization and broadcasting to work with some real-world applications.

### Clustering Algorithms

Machine learning is one domain that can frequently take advantage of vectorization and broadcasting. Let’s say that you have the vertices of a triangle (each row is an x, y coordinate):

```python
>>> tri = np.array([[1, 1],
...                 [3, 1],
...                 [2, 3]])
```

The [centroid](https://en.wikipedia.org/wiki/Centroid) of this “cluster” is an (x, y) coordinate that is the arithmetic mean of each column:

```python
>>> centroid = tri.mean(axis=0)
>>> centroid
array([2.    , 1.6667])
```

It’s helpful to visualize this:

```python
>>> trishape = plt.Polygon(tri, edgecolor='r', alpha=0.2, lw=5)
>>> _, ax = plt.subplots(figsize=(4, 4))
>>> ax.add_patch(trishape)
>>> ax.set_ylim([.5, 3.5])
>>> ax.set_xlim([.5, 3.5])
>>> ax.scatter(*centroid, color='g', marker='D', s=70)
>>> ax.scatter(*tri.T, color='b',  s=70)
```

![三角形的图像](/static/images/article/tri.521228ffdca0.png)

Many [clustering algorithms](http://scikit-learn.org/stable/modules/clustering.html) make use of Euclidean distances of a collection of points, either to the origin or relative to their centroids.

In Cartesian coordinates, the Euclidean distance between points p and q is:

![点之间欧氏距离的计算公式](/static/images/article/euclid.ffdfd280d315.png)

[[source: Wikipedia](https://en.wikipedia.org/wiki/Euclidean_distance#Definition)]

So for the set of coordinates in tri from above, the Euclidean distance of each point from the origin (0, 0) would be:

```pyhon
>>> np.sum(tri**2, axis=1) ** 0.5  # Or: np.sqrt(np.sum(np.square(tri), 1))
array([1.4142, 3.1623, 3.6056])
```

You may recognize that we are really just finding Euclidean norms:

```python
>>> np.linalg.norm(tri, axis=1)
array([1.4142, 3.1623, 3.6056])
```

Instead of referencing the origin, you could also find the norm of each point relative to the triangle’s centroid:

```python
>>> np.linalg.norm(tri - centroid, axis=1)
array([1.2019, 1.2019, 1.3333])
```

Finally, let’s take this one step further: let’s say that you have a 2d array X and a 2d array of multiple (x, y) “proposed” centroids. Algorithms such as [K-Means clustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) work by randomly assigning initial “proposed” centroids, then reassigning each data point to its closest centroid. From there, new centroids are computed, with the algorithm converging on a solution once the re-generated labels (an encoding of the centroids) are unchanged between iterations. A part of this iterative process requires computing the Euclidean distance of each point from each centroid:

```python
>>> X = np.repeat([[5, 5], [10, 10]], [5, 5], axis=0)
>>> X = X + np.random.randn(*X.shape)  # 2 distinct "blobs"
>>> centroids = np.array([[5, 5], [10, 10]])

>>> X
array([[ 3.3955,  3.682 ],
       [ 5.9224,  5.785 ],
       [ 5.9087,  4.5986],
       [ 6.5796,  3.8713],
       [ 3.8488,  6.7029],
       [10.1698,  9.2887],
       [10.1789,  9.8801],
       [ 7.8885,  8.7014],
       [ 8.6206,  8.2016],
       [ 8.851 , 10.0091]])

>>> centroids
array([[ 5,  5],
       [10, 10]])
```

In other words, we want to answer the question, to which centroid does each point within X belong? We need to do some reshaping to enable broadcasting here, in order to calculate the Euclidean distance between each point in X and each point in centroids:

```python
>>> centroids[:, None]
array([[[ 5,  5]],

       [[10, 10]]])

>>> centroids[:, None].shape
(2, 1, 2)
```

This enables us to cleanly subtract one array from another using a **combinatoric product of their rows:**

```python
>>> np.linalg.norm(X - centroids[:, None], axis=2).round(2)
array([[2.08, 1.21, 0.99, 1.94, 2.06, 6.72, 7.12, 4.7 , 4.83, 6.32],
       [9.14, 5.86, 6.78, 7.02, 6.98, 0.73, 0.22, 2.48, 2.27, 1.15]])
```

In other words, the shape of X - centroids[:, None] is (2, 10, 2), essentially representing two stacked arrays that are each the size of X. Next, we want the label (index number) of each closest centroid, finding the minimum distance on the 0th axis from the array above:

```python
>>> np.argmin(np.linalg.norm(X - centroids[:, None], axis=2), axis=0)
array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
```

You can put all this together in functional form:

```python
>>> def get_labels(X, centroids) -> np.ndarray:
...     return np.argmin(np.linalg.norm(X - centroids[:, None], axis=2),
...                      axis=0)
>>> labels = get_labels(X, centroids)
```

Let’s inspect this visually, plotting both the two clusters and their assigned labels with a color-mapping:

```python
>>> c1, c2 = ['#bc13fe', '#be0119']  # https://xkcd.com/color/rgb/
>>> llim, ulim  = np.trunc([X.min() * 0.9, X.max() * 1.1])

>>> _, ax = plt.subplots(figsize=(5, 5))
>>> ax.scatter(*X.T, c=np.where(labels, c2, c1), alpha=0.4, s=80)
>>> ax.scatter(*centroids.T, c=[c1, c2], marker='s', s=95,
...            edgecolor='yellow')
>>> ax.set_ylim([llim, ulim])
>>> ax.set_xlim([llim, ulim])
>>> ax.set_title('One K-Means Iteration: Predicted Classes')
```

![预测类颜色映射](/static/images/article/classes.cdaa3e38d62f.png)

### Amortization Tables

Vectorization has applications in finance as well.

Given an annualized interest rate, payment frequency (times per year), initial loan balance, and loan term, you can create an amortization table with monthly loan balances and payments, in a vectorized fashion. Let’s set some scalar constants first:

```python
>>> freq = 12     # 12 months per year
>>> rate = .0675  # 6.75% annualized
>>> nper = 30     # 30 years
>>> pv = 200000   # Loan face value

>>> rate /= freq  # Monthly basis
>>> nper *= freq  # 360 months
```

NumPy comes preloaded with a handful of [financial functions](https://docs.scipy.org/doc/numpy/reference/routines.financial.html) that, unlike their [Excel cousins](http://www.tvmcalcs.com/index.php/calculators/apps/excel_loan_amortization), are capable of producing vector outputs.

The debtor (or lessee) pays a constant monthly amount that is composed of a principal and interest component. As the outstanding loan balance declines, the interest portion of the total payment declines with it.

```python
>>> periods = np.arange(1, nper + 1, dtype=int)
>>> principal = np.ppmt(rate, periods, nper, pv)
>>> interest = np.ipmt(rate, periods, nper, pv)
>>> pmt = principal + interest  # Or: pmt = np.pmt(rate, nper, pv)
```

Next, you’ll need to calculate a monthly balance, both before and after that month’s payment, which can be defined as the [future value of the original balance minus the future value of an annuity](http://financeformulas.net/Remaining_Balance_Formula.html) (a stream of payments), using a discount factor d:

![原始余额未来价值计算的财务公式图](/static/images/article/fv.7346eb669ac7.png)

Functionally, this looks like:

```python
>>> def balance(pv, rate, nper, pmt) -> np.ndarray:
...     d = (1 + rate) ** nper  # Discount factor
...     return pv * d - pmt * (d - 1) / rate
```

Finally, you can drop this into a tabular format with a Pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html). Be careful with signs here. PMT is an outflow from the perspective of the debtor.

```python
>>> import pandas as pd

>>> cols = ['beg_bal', 'prin', 'interest', 'end_bal']
>>> data = [balance(pv, rate, periods - 1, -pmt),
...         principal,
...         interest,
...         balance(pv, rate, periods, -pmt)]

>>> table = pd.DataFrame(data, columns=periods, index=cols).T
>>> table.index.name = 'month'

>>> with pd.option_context('display.max_rows', 6):
...     # Note: Using floats for $$ in production-level code = bad
...     print(table.round(2))
...
         beg_bal     prin  interest    end_bal
month
1      200000.00  -172.20  -1125.00  199827.80
2      199827.80  -173.16  -1124.03  199654.64
3      199654.64  -174.14  -1123.06  199480.50
...          ...      ...       ...        ...
358      3848.22 -1275.55    -21.65    2572.67
359      2572.67 -1282.72    -14.47    1289.94
360      1289.94 -1289.94     -7.26      -0.00
```

At the end of year 30, the loan is paid off:

```python
>>> final_month = periods[-1]
>>> np.allclose(table.loc[final_month, 'end_bal'], 0)
True
```

**Note**: While using floats to represent money can be useful for concept illustration in a scripting environment, using Python floats for financial calculations in a production environment might cause your calculation to be a penny or two off in some cases.

### Image Feature Extraction

In one final example, we’ll work with an October 1941 [image](https://www.history.navy.mil/our-collections/photography/numerical-list-of-images/nara-series/80-g/80-G-410000/80-G-416362.html) of the USS Lexington (CV-2), the wreck of which was discovered off the coast of Australia in March 2018. First, we can map the image into a NumPy array of its pixel values:

```python
>>> from skimage import io

>>> url = ('https://www.history.navy.mil/bin/imageDownload?image=/'
...        'content/dam/nhhc/our-collections/photography/images/'
...        '80-G-410000/80-G-416362&rendition=cq5dam.thumbnail.319.319.png')
>>> img = io.imread(url, as_grey=True)

>>> fig, ax = plt.subplots()
>>> ax.imshow(img, cmap='gray')
>>> ax.grid(False)
```

![列克星敦号航空母舰的图像](/static/images/article/lex.77b7efabdb0c.png)

For simplicity’s sake, the image is loaded in grayscale, resulting in a 2d array of 64-bit floats rather than a 3-dimensional MxNx4 RGBA array, with lower values denoting darker spots:

```python
>>> img.shape
(254, 319)

>>> img.min(), img.max()
(0.027450980392156862, 1.0)

>>> img[0, :10]  # First ten cells of the first row
array([0.8078, 0.7961, 0.7804, 0.7882, 0.7961, 0.8078, 0.8039, 0.7922,
       0.7961, 0.7961])
>>> img[-1, -10:]  # Last ten cells of the last row
array([0.0784, 0.0784, 0.0706, 0.0706, 0.0745, 0.0706, 0.0745, 0.0784,
       0.0784, 0.0824])
```

One technique commonly employed as an intermediary step in image analysis is patch extraction. As the name implies, this consists of extracting smaller overlapping sub-arrays from a larger array and can be used in cases where it is advantageous to “denoise” or blur an image.

This concept extends to other fields, too. For example, you’d be doing something similar by taking “rolling” windows of a time series with multiple features (variables). It’s even useful for building [Conway’s Game of Life](https://bitstorm.org/gameoflife/). (Although, [convolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html) with a 3x3 kernel is a more direct approach.)

Here, we will find the mean of each overlapping 10x10 patch within img. Taking a miniature example, the first 3x3 patch array in the top-left corner of img would be:

```python
>>> img[:3, :3]
array([[0.8078, 0.7961, 0.7804],
       [0.8039, 0.8157, 0.8078],
       [0.7882, 0.8   , 0.7961]])

>>> img[:3, :3].mean()
0.7995642701525054
```

The pure-Python approach to creating sliding patches would involve a nested for-loop. You’d need to consider that the starting index of the right-most patches will be at index n - 3 + 1, where n is the width of the array. In other words, if you were extracting 3x3 patches from a 10x10 array called arr, the last patch taken would be from arr[7:10, 7:10]. Also keep in mind that Python’s range() does not include its stop parameter:

```python
>>> size = 10
>>> m, n = img.shape
>>> mm, nn = m - size + 1, n - size + 1
>>>
>>> patch_means = np.empty((mm, nn))
>>> for i in range(mm):
...     for j in range(nn):
...         patch_means[i, j] = img[i: i+size, j: j+size].mean()

>>> fig, ax = plt.subplots()
>>> ax.imshow(patch_means, cmap='gray')
>>> ax.grid(False)
```

![莱克星顿号航空母舰的模糊图像](/static/images/article/lexblur.0f886a01be97.png)

With this loop, you’re performing a lot of Python calls.

An alternative that will be scalable to larger RGB or RGBA images is NumPy’s stride_tricks.

An instructive first step is to visualize, given the patch size and image shape, what a higher-dimensional array of patches would look like. We have a 2d array img with shape (254, 319)and a (10, 10) 2d patch. This means our output shape (before taking the mean of each “inner” *10x10* array) would be:

```python
>>> shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
>>> shape
(245, 310, 10, 10)
```

You also need to specify the **strides** of the new array. An array’s strides is a tuple of bytes to jump in each dimension when moving along the array. Each pixel in img is a 64-bit (8-byte) float, meaning the total image size is *254 x 319 x 8 = 648,208* bytes.

```python
>>> img.dtype
dtype('float64')

>>> img.nbytes
648208
```

Internally, img is kept in memory as one contiguous block of 648,208 bytes. strides is hence a sort of “metadata”-like attribute that tells us how many bytes we need to jump ahead to move to the next position along each axis. We move in blocks of 8 bytes along the rows but need to traverse *8 x 319 = 2,552* bytes to move “down” from one row to another.

```python
>>> img.strides
(2552, 8)
```

In our case, the strides of the resulting patches will just repeat the strides of img twice:

```python
>>> strides = 2 * img.strides
>>> strides
(2552, 8, 2552, 8)
```

Now, let’s put these pieces together with NumPy’s [stride_tricks](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html):

```python
>>> from numpy.lib import stride_tricks

>>> patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
>>> patches.shape
(245, 310, 10, 10)
```

Here’s the first *10x10* patch:

```python
>>> patches[0, 0].round(2)
array([[0.81, 0.8 , 0.78, 0.79, 0.8 , 0.81, 0.8 , 0.79, 0.8 , 0.8 ],
       [0.8 , 0.82, 0.81, 0.79, 0.79, 0.79, 0.78, 0.81, 0.81, 0.8 ],
       [0.79, 0.8 , 0.8 , 0.79, 0.8 , 0.8 , 0.82, 0.83, 0.79, 0.81],
       [0.8 , 0.79, 0.81, 0.81, 0.8 , 0.8 , 0.78, 0.76, 0.8 , 0.79],
       [0.78, 0.8 , 0.8 , 0.78, 0.8 , 0.79, 0.78, 0.78, 0.79, 0.79],
       [0.8 , 0.8 , 0.78, 0.78, 0.78, 0.8 , 0.8 , 0.8 , 0.81, 0.79],
       [0.78, 0.77, 0.78, 0.76, 0.77, 0.8 , 0.8 , 0.77, 0.8 , 0.8 ],
       [0.79, 0.76, 0.77, 0.78, 0.77, 0.77, 0.79, 0.78, 0.77, 0.76],
       [0.78, 0.75, 0.76, 0.76, 0.73, 0.75, 0.78, 0.76, 0.77, 0.77],
       [0.78, 0.79, 0.78, 0.78, 0.78, 0.78, 0.77, 0.76, 0.77, 0.77]])
```

The last step is tricky. To get a vectorized mean of each inner 10x10 array, we need to think carefully about the dimensionality of what we have now. The result should collapse the last two dimensions so that we’re left with a single 245x310 array.

One (suboptimal) way would be to reshape patches first, flattening the inner 2d arrays to length-100 vectors, and then computing the mean on the final axis:

```python
>>> veclen = size ** 2
>>> patches.reshape(*patches.shape[:2], veclen).mean(axis=-1).shape
(245, 310)
```

However, you can also specify axis as a tuple, computing a mean over the last two axes, which should be more efficient than reshaping:

```python
>>> patches.mean(axis=(-1, -2)).shape
(245, 310)
```

Let’s make sure this checks out by comparing equality to our looped version. It does:

```python
>>> strided_means = patches.mean(axis=(-1, -2))
>>> np.allclose(patch_means, strided_means)
True
```

If the concept of strides has you drooling, don’t worry: Scikit-Learn has already [embedded this entire process](http://scikit-learn.org/stable/modules/feature_extraction.html#image-feature-extraction) nicely within its feature_extraction module.

## A Parting Thought: Don’t Over-Optimize

In this article, we discussed optimizing runtime by taking advantage of array programming in NumPy. When you are working with large datasets, it’s important to be mindful of microperformance.

However, there is a subset of cases where avoiding a native Python for-loop isn’t possible. As Donald Knuth [advised](http://web.archive.org/web/20130731202547/http://pplab.snu.ac.kr/courses/adv_pl05/papers/p261-knuth.pdf), “Premature optimization is the root of all evil.” Programmers may incorrectly predict where in their code a bottleneck will appear, spending hours trying to fully vectorize an operation that would result in a relatively insignificant improvement in runtime.

There’s nothing wrong with for-loops sprinkled here and there. Often, it can be more productive to think instead about optimizing the flow and structure of the entire script at a higher level of abstraction.

## More Resources

Free Bonus: [Click here to get access to a free NumPy Resources Guide](https://realpython.com/numpy-array-programming/#) that points you to the best tutorials, videos, and books for improving your NumPy skills.

NumPy Documentation:

- [What is NumPy?](https://docs.scipy.org/doc/numpy/user/whatisnumpy.html)
- [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
- [Universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html)
- [NumPy for MATLAB Users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html)
- The complete [NumPy Reference](https://docs.scipy.org/doc/numpy/reference/index.html) index

Books:

- Travis Oliphant’s [Guide to NumPy, 2nd ed](https://realpython.com/asins/151730007X/). (Travis is the primary creator of NumPy)
- Chapter 2 (“Introduction to NumPy”) of Jake VanderPlas’ [Python Data Science Handbook](https://realpython.com/asins/1491912057/)
- Chapter 4 (“NumPy Basics”) and Chapter 12 (“Advanced NumPy”) of Wes McKinney’s [Python for Data Analysis 2nd ed](https://realpython.com/asins/B075X4LT6K/).
- Chapter 2 (“The Mathematical Building Blocks of Neural Networks”) from François Chollet’s [Deep Learning with Python](https://realpython.com/asins/1617294438/)
- Robert Johansson’s [Numerical Python](https://realpython.com/asins/1484205545/)
- Ivan Idris: [Numpy Beginner’s Guide, 3rd ed](https://realpython.com/asins/1785281968/).

Other Resources:

- Wikipedia: [Array Programming](https://en.wikipedia.org/wiki/Array_programming)
- SciPy Lecture Notes: [Basic]() and [Advanced]() NumPy
- EricsBroadcastingDoc: [Array Broadcasting in NumPy](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc)
- SciPy Cookbook: [Views versus copies in NumPy](http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html)
- Nicolas Rougier: [From Python to Numpy](http://www.labri.fr/perso/nrougier/from-python-to-numpy/) and [100 NumPy Exercises](http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html)
- TensorFlow docs: [Broadcasting Semantics](https://www.tensorflow.org/performance/xla/broadcasting)
- Theano docs: [Broadcasting](http://deeplearning.net/software/theano/tutorial/broadcasting.html)
- Eli Bendersky: [Broadcasting Arrays in Numpy](https://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/)

## 文章出处

由NumPy中文文档翻译，原作者为 [Brad Solomon](https://realpython.com/team/bsolomon/)，翻译至：[https://realpython.com/numpy-array-programming/](https://realpython.com/numpy-array-programming/) 