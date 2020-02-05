---
home: true
meta:
  - name: keywords
    content: numpy中文文档,numpy中文api,numpy中文手册,numpy教程,numpy下载安装,numpy
  - name: description
    content: 这是 NumPy 官方的中文文档，NumPy 是用 Python 进行科学计算的基础软件包。
heroImage: https://extraimage.net/images/2019/09/23/5ac6e9d90002903efacacdcb8182b8ed.png
actionText: 快速了解 →
actionLink: /user/
action2Text: 学习深度学习 →
action2Link: /deep/
footer: 署名-非商业性使用-相同方式共享 3.0 中国大陆 (CC BY-NC-SA 3.0 CN) | Copyright © 2019-present Zhi Bing
---

<div class="features">
  <div class="feature">
    <h2>NumPy 是什么？</h2>
    <p>
      NumPy 是 Python 科学计算的基础包。它主要包括：
    </p>
    <ul>
      <li>
        功能强大的多维数组对象。
      </li>
      <li>
        先进的广播函数。
      </li>
      <li>
        集成 C/C++ 和 Fortran 代码的工具。
      </li>
      <li>
        实用的线性代数、傅立叶变换和随机数等功能。
      </li>
    </ul>
  </div>
  <div class="feature">
    <h2>利器之一：ndarray</h2>
    <p>NumPy 最重要的特性之一是其多维数组对象 ndarray。它用于存放同类型元素，其中每个元素在内存中都有相同大小的空间。ndarray 以从 0 开始的下标索引其中的元素。</p>
  </div>
  <div class="feature">
    <h2>利器之二：索引和切片</h2>
    <p>可通过索引和切片来访问、修改 ndarray 对象，与 Python 中的 list 一样。ndarray 数组可以使用 0~n 的下标进行索引，也可以通过内置的 slice 函数切片，即通过设置 start, stop 和 step 参数从原数组中提取出一个新数组。</p>
  </div>
</div>


### 阅读导航

- 如果您在使用手机预览，请点击**左上角**的菜单图标展开文档的菜单。
- 如果您是新手，我们推荐阅读：[理解 Numpy](/article/basics/understanding_numpy.html)、[NumPy 简单入门教程](/article/basics/an_introduction_to_scientific_python_numpy.html)、[创建 Numpy 数组的不同方式](/article/basics/different_ways_create_numpy_arrays.html)。还有中文文档提供的[精选资源](/awesome/)。
- 了解**神经网络**或者**强化学习**的相关知识，参看 [NumPy 与 神经网络](/article/advanced/numpy_kmeans.html)、[NumPy 实现 DNC、RNN 和 LSTM神经网络算法](/article/advanced/dnc_rnn_lstm.html)。
- 查找资料？请在页面中左上角的搜索框进行搜索。
- 系统地学习 NumPy？请从头到尾阅读此文档，你可能不需要为任何教程/内容付费。
- 如果有疑问请在右侧**快捷留言板**留言，或者加入**NumPy 中文社区**的 QQ / 微信群。
- 另外，点击下面的**捐赠网站**以向我们捐赠。🙏

### 快速开始

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

改变世界，从 **Python** 开始。

本网站推荐使用 [Python3.x](https://www.python.org/downloads/) 或更高版本。

:::

<ahome-wxpub></ahome-wxpub>

<ahome-nav></ahome-nav>

<ahome-footer></ahome-footer>
