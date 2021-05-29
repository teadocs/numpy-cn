---
home: true
meta:
  - name: keywords
    content: numpy中文文档,numpy中文api,numpy中文手册,numpy教程,numpy下载安装,numpy
  - name: description
    content: 这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包。
heroImage: https://static.numpy.org.cn/site/logo.png
actionText: 快速了解 →
actionLink: /user/
action2Text: 开始深度学习
action2Link: https://analytics.numpy.org.cn/course.html
footer: 署名-非商业性使用-相同方式共享 3.0 中国大陆 (CC BY-NC-SA 3.0 CN) | Copyright © 2019-present Zhi Bing
---

<div class="features">
  <div class="feature">
    <h2>NumPy 是什么？</h2>
    <p>
      NumPy是使用Python进行科学计算的基础软件包。除其他外，它包括：
    </p>
    <ul>
      <li>
        功能强大的N维数组对象。
      </li>
      <li>
        精密广播功能函数。
      </li>
      <li>
        集成 C/C+和Fortran 代码的工具。
      </li>
      <li>
        强大的线性代数、傅立叶变换和随机数功能。
      </li>
    </ul>
  </div>
  <div class="feature">
    <h2>利器之一：Ndarray</h2>
    <p>NumPy 最重要的一个特点是其 N 维数组对象 ndarray，它是一系列同类型数据的集合，以 0 下标为开始进行集合中元素的索引。ndarray 对象是用于存放同类型元素的多维数组。ndarray 中的每个元素在内存中都有相同存储大小的区域。</p>
  </div>
  <div class="feature">
    <h2>利器之一：切片和索引</h2>
    <p>ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。ndarray 数组可以基于 0 - n 的下标进行索引，切片对象可以通过内置的 slice 函数，并设置 start, stop 及 step 参数进行，从原数组中切割出一个新数组。</p>
  </div>
</div>


### 网站阅读导航

- 如果使用手机预览，请点击**左上角**的菜单图标展开文档的菜单。
- 假设你是新手同学，推荐阅读基础文章中的：[理解Numpy](/article/basics/understanding_numpy.html)、[NumPy简单入门教程](/article/basics/an_introduction_to_scientific_python_numpy.html)、[创建Numpy数组的不同方式](/article/basics/different_ways_create_numpy_arrays.html)。还有中文文档提供的[精选资源](/awesome/)。
- 想了解**神经网络**或者**强化学习**相关的可以参看 [NumPy 与 神经网络](/article/advanced/numpy_kmeans.html)、[ NumPy实现DNC、RNN和LSTM神经网络算法](/article/advanced/dnc_rnn_lstm.html)。
- 想查找手册？请指教点击左上角的搜索框进行搜索。
- 想系统的学习NumPy？请直接从本文档第一篇一直阅读到最后一篇，你可能不需要为任何教程/内容付费就可以学会。
- 如果有疑问请在右侧**快捷留言板**留言 或者 加入**NumPy 中文社区**的QQ/微信群。
- 另外，**捐赠**可以点击下面**捐赠网站**按钮。🙏

### 就像1、2、3 一样简单

``` bash
# 1、安装包
$ pip install numpy

# 2、进入python的交互式界面
$ python -i

# 3、使用Numpy
>>> from numpy import *
>>> eye(4)

# 4、输出结果
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
```

::: warning 提醒

改变世界从 **Python** 开始。

本网站推荐使用[Python3.x](https://www.python.org/downloads/)及以上版本。

:::

<ahome-article></ahome-article>

<ahome-nav></ahome-nav>

<ahome-footer></ahome-footer>
