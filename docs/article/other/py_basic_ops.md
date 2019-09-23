---
meta:
  - name: keywords
    content: OpenCV中的图像的基本操作
  - name: description
    content: 本节中的几乎所有操作都主要与 Numpy 有关，而不是OpenCV。要使用OpenCV编写更好的优...
---

# OpenCV中的图像的基本操作

## 目标

学到如下项目:

- 访问像素值并修改它们
- 存取图像特性
- 图像设置区域(ROI)
- 分割和合并图像

本节中的几乎所有操作都主要与``Numpy``有关，而不是OpenCV。要使用OpenCV编写更好的优化代码，需要对Numpy有很好的了解。

*(示例将在Python终端中显示，因为大多数示例都是单行代码)*

## 访问和修改像素值

让我们先加载一个彩色图像：

```python
>>> import cv2
>>> import numpy as np

>>> img = cv2.imread('messi5.jpg')
```

你可以根据像素值的行和列坐标来访问它。对于BGR图像，它返回一个蓝色、绿色、红色值数组。为。
灰度图像，只返回相应的亮度。

```python
>>> px = img[100,100]
>>> print px
[157 166 200]

# accessing only blue pixel
>>> blue = img[100,100,0]
>>> print blue
157
```

你可以同样的方式修改像素值。

```python
>>> img[100,100] = [255,255,255]
>>> print img[100,100]
[255 255 255]
```

<div class="warning-warp">
<b>警告</b>

<p>Numpy是一个用于快速数组计算的优化库。因此，使用原生python的数组简单地访问每个像素值，并修改它将非常缓慢，我们并不推荐这种方法。</p>
</div>

> **注意：** 上述方法通常用于选择数组区域，例如前5行和最后3列。对于单个像素访问，但用Numpy数组的方法、array.tem()和array.itemset() 会更适合。因为它总是返回一个标量。因此，如果你想访问所有的B，G，R值，你需要为所有人分别调用array.tem()。

更好的像素访问和编辑方法：

```python
# accessing RED value
>>> img.item(10,10,2)
59

# modifying RED value
>>> img.itemset((10,10,2),100)
>>> img.item(10,10,2)
100
```

## 访问图像属性

图像属性包括行数，列数和通道数，图像数据类型，像素数等。

img.shape可以访问图像的形状。 它返回一组行，列和通道的元组（如果图像是彩色的）：

```python
>>> print img.shape
(342, 548, 3)
```

> **注意：** 如果图像是灰度，则返回的元组仅包含行数和列数。因此，检查加载的图像是灰度还是彩色图像是一种很好的方法。

img.size访问的像素总数：

```python
>>> print img.size
562248
```

图像数据类型由img.dtype获得：

```python
>>> print img.dtype
uint8
```

> **注意：** img.dtype在调试时非常重要，因为OpenCV-Python代码中的大量错误是由无效的数据类型引起的。

## 图像 ROI

有时，你必须使用某些图像区域。对于图像中的眼睛检测，在整个图像上进行第一次面部检测，并且当获得面部时，我们单独选择面部区域并搜索其内部的眼睛而不是搜索整个图像。它提高了准确性（因为眼睛总是在脸上：D）和表现（因为我们搜索的是一小块区域）

使用Numpy索引再次获得ROI。在这里，我选择球并将其复制到图像中的另一个区域：

```python
>>> ball = img[280:340, 330:390]
>>> img[273:333, 100:160] = ball
```

检查以下结果：

![roi](/static/images/roi.jpg)

## 拆分和合并图像通道

有时你需要在B,G,R通道图像上单独工作。然后，你需要将BGR图像分割为单个平面。或者，你可能需要将这些单独的通道连接到BGR图像。你可以通过以下方式完成：

```python
>>> b,g,r = cv2.split(img)
>>> img = cv2.merge((b,g,r))
```

或者这样写：

```python
>>> b = img[:,:,0]
```

假设，你想要将所有红色像素设为零，你不需要像这样分割并将其等于零。 你可以简单地使用Numpy索引，这样更快。

```python
>>> img[:,:,2] = 0
```

<div class="warning-warp">
<b>警告</b>

<p>cv2.split() 是一项代价昂贵的的操作（就运算时间而言）。所以只有在你需要时才使用这个方法。否则请使用Numpy索引。</p>
</div>

## 制作图像边框（填充）

如果要在图像周围创建边框，比如相框，可以使用cv2.copyMakeBorder() 函数。但它有更多卷积运算，零填充等应用。该函数采用以下参数：

- src - 输入图像
- top, bottom, left, right - 相应方向上的像素数的边界宽度
- borderType - 标志定义要添加的边框类型。它可以是以下类型：
    - cv2.BORDER_CONSTANT - 添加恒定的彩色边框。 该值应作为下一个参数给出。
    - cv2.BORDER_REFLECT - 边框将镜像反射边界元素，如：fedcba | abcdefgh | hgfedcb
    - cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT - 边框将镜像反射边界元素，如：fedcba | abcdefgh | hgfedcb
    - cv2.BORDER_REPLICATE - 最后一个元素被复制，如下所示：aaaaaa | abcdefgh | hhhhhhh
    - cv2.BORDER_WRAP - 无法解释，它看起来像这样：cdefgh | abcdefgh | abcdefg
- value - 如果边框类型为cv2.BORDER_CONSTANT，则为边框颜色

下面是一个示例代码，演示了所有这些边框类型，以便更好地理解：

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255,0,0]

img1 = cv2.imread('opencv_logo.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()
```

请参阅下面的结果。（图像与matplotlib一起显示。因此RED和BLUE平面将互换）：

![border](/static/images/border.jpg)

## 帮助和反馈

你找不到你想要的东西？
- 在[问答论坛](http://answers.opencv.org/)上提问。
- 如果你认为文档中缺少某些内容或错误，请提交[错误报告](http://code.opencv.org/)。

## 文章出处

由NumPy中文文档翻译，原作者为OpenCV官方英文文档，翻译至：[https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html)