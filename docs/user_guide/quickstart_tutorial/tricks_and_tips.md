# 技巧和提示

在这里，我们列出一些简短而有用的提示。

## “自动”重定义数组形状

要更改数组的大小，你可以省略其中一个size，它将被自动推导出来：

```
>>> a = np.arange(30)
>>> a.shape = 2,-1,3  # -1 means "whatever is needed"
>>> a.shape
(2, 5, 3)
>>> a
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [12, 13, 14]],
       [[15, 16, 17],
        [18, 19, 20],
        [21, 22, 23],
        [24, 25, 26],
        [27, 28, 29]]])
```

## 向量堆叠

我们如何从一个相同大小的行向量列表构造一个二维数组？在MATLAB中，这很容易：如果x和y是两个长度相同的向量，那么只需要 ``m=[x;y]`` 。在NumPy中，这通过函数 ``column_stack`` ，``dstack`` ，``hstack`` 和 ``vstack`` 工作，具体取决于要做什么堆叠。例如：

```
x = np.arange(0,10,2)                     # x=([0,2,4,6,8])
y = np.arange(5)                          # y=([0,1,2,3,4])
m = np.vstack([x,y])                      # m=([[0,2,4,6,8],
                                          #     [0,1,2,3,4]])
xy = np.hstack([x,y])                     # xy =([0,2,4,6,8,0,1,2,3,4])
```

这些功能背后的逻辑可能很奇怪。

另见：

> NumPy for Matlab users

## 直方图

NumPy的 ``histogram`` 函数应用于一个数组，并返回一对向量：数组的histogram和向量的bin。注意： ``matplotlib`` 也具有构建histograms的函数（在Matlab中称为 ``hist`` ），它与NumPy中的不同。主要区别是 ``pylab.hist`` 自动绘制histogram，而 ``numpy.histogram`` 仅生成数据。

```
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> # Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
>>> mu, sigma = 2, 0.5
>>> v = np.random.normal(mu,sigma,10000)
>>> # Plot a normalized histogram with 50 bins
>>> plt.hist(v, bins=50, normed=1)       # matplotlib version (plot)
>>> plt.show()
```

![quickstart-2_00_00](/static/images/quickstart-2_00_00.png)

```
>>> # Compute the histogram with numpy and then plot it
>>> (n, bins) = np.histogram(v, bins=50, normed=True)  # NumPy version (no plot)
>>> plt.plot(.5*(bins[1:]+bins[:-1]), n)
>>> plt.show()
```

![quickstart-2_01_00](/static/images/quickstart-2_01_00.png)
