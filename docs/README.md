---
home: true
meta:
  - name: keywords
    content: numpy中文文档,numpy中文api,numpy中文手册,numpy教程,numpy下载安装,numpy
  - name: description
    content: 这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包。
heroImage: /logo.svg
actionText: 快速开始 →
actionLink: /docs/
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

本网站推荐使用[Python3.x](https://www.python.org/downloads/)及以上版本。

:::

<ahome-wxpub></ahome-wxpub>

<ahome-nav></ahome-nav>

<ahome-footer></ahome-footer>
