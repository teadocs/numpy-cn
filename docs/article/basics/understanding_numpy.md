---
meta:
  - name: keywords
    content: 理解 NumPy
  - name: description
    content: 在这篇文章中，我们将介绍使用NumPy的基础知识，NumPy是一个功能强大的Python库，允许更高级的数据操作和数学计算。
---

# 理解 NumPy

在这篇文章中，我们将介绍使用NumPy的基础知识，NumPy是一个功能强大的Python库，允许更高级的数据操作和数学计算。

## 什么是 NumPy?

NumPy是一个功能强大的Python库，主要用于对多维数组执行计算。NumPy这个词来源于两个单词-- ``Numerical``和``Python``。NumPy提供了大量的库函数和操作，可以帮助程序员轻松地进行数值计算。这类数值计算广泛用于以下任务：

- **机器学习模型**：在编写机器学习算法时，需要对矩阵进行各种数值计算。例如矩阵乘法、换位、加法等。NumPy提供了一个非常好的库，用于简单(在编写代码方面)和快速(在速度方面)计算。NumPy数组用于存储训练数据和机器学习模型的参数。

- **图像处理和计算机图形学**：计算机中的图像表示为多维数字数组。NumPy成为同样情况下最自然的选择。实际上，NumPy提供了一些优秀的库函数来快速处理图像。例如，镜像图像、按特定角度旋转图像等。

- **数学任务**：NumPy对于执行各种数学任务非常有用，如数值积分、微分、内插、外推等。因此，当涉及到数学任务时，它形成了一种基于Python的MATLAB的快速替代。

## NumPy 的安装

在你的计算机上安装NumPy的最快也是最简单的方法是在shell上使用以下命令：``pip install numpy``。

这将在你的计算机上安装最新/最稳定的NumPy版本。通过PIP安装是安装任何Python软件包的最简单方法。现在让我们来谈谈NumPy中最重要的概念，NumPy数组。

## NumPy 中的数组

NumPy提供的最重要的数据结构是一个称为NumPy数组的强大对象。NumPy数组是通常的Python数组的扩展。NumPy数组配备了大量的函数和运算符，可以帮助我们快速编写上面讨论过的各种类型计算的高性能代码。让我们看看如何快速定义一维NumPy数组：

```python
import numpy as np 
my_array = np.array([1, 2, 3, 4, 5]) 
print my_array
```

在上面的简单示例中，我们首先使用import numpy作为np导入NumPy库。然后，我们创建了一个包含5个整数的简单NumPy数组，然后我们将其打印出来。继续在自己的机器上试一试。在看 “NumPy安装” 部分下面的步骤的时候，请确保已在计算机中安装了NumPy。

现在让我们看看我们可以用这个特定的NumPy数组能做些什么。

```python
print my_array.shape
```

它会打印我们创建的数组的形状：``(5, )``。意思就是 my_array 是一个包含5个元素的数组。

我们也可以打印各个元素。就像普通的Python数组一样，NumPy数组的起始索引编号为0。

```python
print my_array[0]
print my_array[1]
```

上述命令将分别在终端上打印1和2。我们还可以修改NumPy数组的元素。例如，假设我们编写以下2个命令：

```python
my_array[0] = -1
print my_array
```

我们将在屏幕上看到：``[-1,2,3,4,5]``。

现在假设，我们要创建一个长度为5的NumPy数组，但所有元素都为0，我们可以这样做吗？是的。NumPy提供了一种简单的方法来做同样的事情。

```python
my_new_array = np.zeros((5)) 
print my_new_array
```

我们将看到输出了 ``[0., 0., 0., 0., 0.] ``。与 ``np.zeros`` 类似，我们也有 ``np.ones``。 如果我们想创建一个随机值数组怎么办？

```python
my_random_array = np.random.random((5))
print my_random_array
```

我们得到的输出看起来像 [0.22051844 0.35278286 0.11342404 0.79671772 0.62263151] 这样的数据。你获得的输出可能会有所不同，因为我们使用的是随机函数，它为每个元素分配0到1之间的随机值。

现在让我们看看如何使用NumPy创建二维数组。

```python
my_2d_array = np.zeros((2, 3)) print my_2d_array
```

这将在屏幕上打印以下内容：

```
[[0. 0. 0.]

[0. 0. 0.]]
```

猜猜以下代码的输出结果如何：

```python
my_2d_array_new = np.ones((2, 4)) print my_2d_array_new
```

这里是：

```
[[1. 1. 1. 1.]

[1. 1. 1. 1.]]
```

基本上，当你使用函数``np.zeros()``或``np.ones()``时，你可以指定讨论数组大小的元组。在上面的两个例子中，我们使用以下元组，(2, 3) 和(2, 4) 分别表示2行，3列和4列。像上面那样的多维数组可以用 ``my_array[i][j]`` 符号来索引，其中i表示行号，j表示列号。i和j都从0开始。

```python
my_array = np.array([[4, 5], [6, 1]])
print my_array[0][1]
```

上面的代码片段的输出是5，因为它是索引0行和索引1列中的元素。

你还可以按如下方式打印my_array的形状：

```python
print my_array.shape
```

输出为(2, 2)，表示数组中有2行2列。

NumPy提供了一种提取多维数组的行/列的强大方法。例如，考虑我们上面定义的``my_array``的例子。

```python
[[4 5] [6 1]]
```

假设，我们想从中提取第二列（索引1）的所有元素。在这里，我们肉眼可以看出，第二列由两个元素组成：``5`` 和 ``1``。为此，我们可以执行以下操作：

```python
my_array_column_2 = my_array[:, 1] 
print my_array_column_2
```

注意，我们使用了冒号(``:``)而不是行号，而对于列号，我们使用了值``1``，最终输出是：``[5, 1]``。

我们可以类似地从多维NumPy数组中提取一行。现在，让我们看看NumPy在多个数组上执行计算时提供的强大功能。

## NumPy中的数组操作

使用NumPy，你可以轻松地在数组上执行数学运算。例如，你可以添加NumPy数组，你可以减去它们，你可以将它们相乘，甚至可以将它们分开。 以下是一些例子：

```python
import numpy as np 
a = np.array([[1.0, 2.0], [3.0, 4.0]]) 
b = np.array([[5.0, 6.0], [7.0, 8.0]]) 
sum = a + b 
difference = a - b 
product = a * b 
quotient = a / b 
print "Sum = \n", sum 
print "Difference = \n", difference 
print "Product = \n", product 
print "Quotient = \n", quotient 

# The output will be as follows: 

Sum = [[ 6. 8.] [10. 12.]]
Difference = [[-4. -4.] [-4. -4.]]
Product = [[ 5. 12.] [21. 32.]]
Quotient = [[0.2 0.33333333] [0.42857143 0.5 ]]
```

如你所见，乘法运算符执行逐元素乘法而不是矩阵乘法。 要执行矩阵乘法，你可以执行以下操作：

```python
matrix_product = a.dot(b) 
print "Matrix Product = ", matrix_product
```

输出将是：

```
[[19. 22.]

[43. 50.]]
```

## 总结

如你所见，NumPy在其提供的库函数方面非常强大。你可以使用NumPy公开的优秀的API在单行代码中执行大型计算。这使它成为各种数值计算的优雅工具。如果你希望自己成为一名数学家或数据科学家，你一定要考虑掌握它。在熟练掌握NumPy之前，你需要了解Python。

你可以在 Hackr.io 上找到编程社区推荐的最佳[Python 教程](https://hackr.io/tutorials/learn-python)，愿上帝保佑你！

## 文章出处

由NumPy中文文档翻译，原作者为 [Vijay Singh](https://dzone.com/users/3404598/vijayhackr.html)，翻译至：[https://dzone.com/article/understanding-numpy](https://dzone.com/article/understanding-numpy)。