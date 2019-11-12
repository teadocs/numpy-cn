---
meta:
  - name: keywords
    content: NumPy 数据分析练习
  - name: description
    content: Numpy练习的目标仅作为学习numpy的参考，并让你脱离基础性的NumPy使用。这些问题有4个级别的难度，其中L1是最容易的...
---

# NumPy 数据分析练习

Numpy练习的目标仅作为学习numpy的参考，并让你脱离基础性的NumPy使用。这些问题有4个级别的难度，其中L1是最容易的，L4是最难的。

![Numpy教程第2部分：数据分析的重要函数。图片由安娜贾斯汀卢布克拍摄。](/static/images/101-numpy-exercises-1024x683.jpg)

如果你想快速进阶你的numpy知识，那么[numpy基础知识](https://www.machinelearningplus.com/numpy-tutorial-part1-array-python-examples)和[高级numpy教程](https://www.machinelearningplus.com/numpy-tutorial-python-part2)可能就是你要寻找的内容。

**更新：**现在有一套类似的关于[pandas](https://www.machinelearningplus.com/python/101-pandas-exercises-python/)的练习。 

## NumPy数据分析问答

### 1、导入numpy作为np，并查看版本

**难度等级：**L1
**问题：**将numpy导入为 ``np`` 并打印版本号。
**答案：**

```python
import numpy as np
print(np.__version__)
# > 1.13.3
```

你必须将numpy导入np，才能使本练习中的其余代码正常工作。

要安装numpy，建议安装anaconda，里面已经包含了numpy。

### 2、如何创建一维数组？

**难度等级：**L1
**问题：**创建从0到9的一维数字数组

**期望输出：**
```python
# > array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

**答案：**
```python
arr = np.arange(10)
arr
# > array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### 3. 如何创建一个布尔数组？

**难度等级：**L1

**问题：**创建一个numpy数组元素值全为True（真）的数组

**答案：**

```python
np.full((3, 3), True, dtype=bool)
# > array([[ True,  True,  True],
# >        [ True,  True,  True],
# >        [ True,  True,  True]], dtype=bool)

# Alternate method:
np.ones((3,3), dtype=bool)
```

### 4. 如何从一维数组中提取满足指定条件的元素？
**难度等级：**L1

**问题：**从 arr 中提取所有的奇数

**给定：**

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

**期望的输出：**

```python
# > array([1, 3, 5, 7, 9])
```

**答案：**

```python
# Input
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Solution
arr[arr % 2 == 1]
# > array([1, 3, 5, 7, 9])
```

### 5. 如何用numpy数组中的另一个值替换满足条件的元素项？
**难度等级：**L1

**问题：**将arr中的所有奇数替换为-1。

**给定：**

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

**期望的输出：**
```python
# >  array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
```

**答案：**
```python
arr[arr % 2 == 1] = -1
arr
# > array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
```

### 6. 如何在不影响原始数组的情况下替换满足条件的元素项？
**难度等级：**L2

**问题：**将arr中的所有奇数替换为-1，而不改变arr。

**给定：**
```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

**期望的输出：**

```python
out
# >  array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
arr
# >  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

**答案：**

```python
arr = np.arange(10)
out = np.where(arr % 2 == 1, -1, arr)
print(arr)
out
# > [0 1 2 3 4 5 6 7 8 9]
array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
```

### 7. 如何改变数组的形状？
**难度等级：**L1

**问题：**将一维数组转换为2行的2维数组

**给定：**

```python
np.arange(10)

# > array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

**期望的输出：**

```
# > array([[0, 1, 2, 3, 4],
# >        [5, 6, 7, 8, 9]])
```

**答案：**
```python
arr = np.arange(10)
arr.reshape(2, -1)  # Setting to -1 automatically decides the number of cols
# > array([[0, 1, 2, 3, 4],
# >        [5, 6, 7, 8, 9]])
```

### 8. 如何垂直叠加两个数组？

**难度等级：**L2

**问题：**垂直堆叠数组a和数组b

**给定：**

```python
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
```

**期望的输出：**

```python
# > array([[0, 1, 2, 3, 4],
# >        [5, 6, 7, 8, 9],
# >        [1, 1, 1, 1, 1],
# >        [1, 1, 1, 1, 1]])
```

**答案：**

```python
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)

# Answers
# Method 1:
np.concatenate([a, b], axis=0)

# Method 2:
np.vstack([a, b])

# Method 3:
np.r_[a, b]
# > array([[0, 1, 2, 3, 4],
# >        [5, 6, 7, 8, 9],
# >        [1, 1, 1, 1, 1],
# >        [1, 1, 1, 1, 1]])
```
### 9. 如何水平叠加两个数组？
**难度等级：**L2

**问题：**将数组a和数组b水平堆叠。

**给定：**

```python
a = np.arange(10).reshape(2,-1)

b = np.repeat(1, 10).reshape(2,-1)
```

**期望的输出：**

```python
# > array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
# >        [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
```

**答案：**

```python
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)

# Answers
# Method 1:
np.concatenate([a, b], axis=1)

# Method 2:
np.hstack([a, b])

# Method 3:
np.c_[a, b]
# > array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
# >        [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
```

### 10. 如何在无硬编码的情况下生成numpy中的自定义序列？
**难度等级：**L2

**问题：**创建以下模式而不使用硬编码。只使用numpy函数和下面的输入数组a。

**给定：**

```python
a = np.array([1,2,3])`
```

**期望的输出：**

```python
# > array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
```

**答案：**

```python
np.r_[np.repeat(a, 3), np.tile(a, 3)]
# > array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
```

### 11. 如何获取两个numpy数组之间的公共项？
**难度等级：**L2

**问题：**获取数组a和数组b之间的公共项。

**给定：**

```python
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
```

**期望的输出：**

```python
array([2, 4])
```

**答案：**

```python
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)
# > array([2, 4])
```

### 12. 如何从一个数组中删除存在于另一个数组中的项？
**难度等级：**L2

**问题：**从数组a中删除数组b中的所有项。

**给定：**

```python
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
```

**期望的输出：**

```python
array([1,2,3,4])
```

**答案：**

```python
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])

# From 'a' remove all of 'b'
np.setdiff1d(a,b)
# > array([1, 2, 3, 4])
```

### 13. 如何得到两个数组元素匹配的位置？
**难度等级：**L2

**问题：**获取a和b元素匹配的位置。

**给定：**

```python
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
```

**期望的输出：**

```python
# > (array([1, 3, 5, 7]),)
```

**答案：**
```python
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])

np.where(a == b)
# > (array([1, 3, 5, 7]),)
```

### 14. 如何从numpy数组中提取给定范围内的所有数字？
**难度等级：**L2

**问题：**获取5到10之间的所有项目。

**给定：**

```python
a = np.array([2, 6, 1, 9, 10, 3, 27])
```

**期望的输出：**

```python
(array([6, 9, 10]),)
```

**答案：**

```python
a = np.arange(15)

# Method 1
index = np.where((a >= 5) & (a <= 10))
a[index]

# Method 2:
index = np.where(np.logical_and(a>=5, a<=10))
a[index]
# > (array([6, 9, 10]),)

# Method 3: (thanks loganzk!)
a[(a >= 5) & (a <= 10)]
```

### 15. 如何创建一个python函数来处理scalars并在numpy数组上工作？
**难度等级：**L2

**问题：**转换适用于两个标量的函数maxx，以处理两个数组。

**给定：**

```python
def maxx(x, y):
    """Get the maximum of two items"""
    if x >= y:
        return x
    else:
        return y

maxx(1, 5)
# > 5
```

**期望的输出：**

```python
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
pair_max(a, b)
# > array([ 6.,  7.,  9.,  8.,  9.,  7.,  5.])
```

**答案：**

```python
def maxx(x, y):
    """Get the maximum of two items"""
    if x >= y:
        return x
    else:
        return y

pair_max = np.vectorize(maxx, otypes=[float])

a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])

pair_max(a, b)
# > array([ 6.,  7.,  9.,  8.,  9.,  7.,  5.])
```

### 16. 如何交换二维numpy数组中的两列？

**难度等级：**L2

**问题：**在数组arr中交换列1和2。

**给定：**

```python
arr = np.arange(9).reshape(3,3)
arr
```

**答案：**

```python
# Input
arr = np.arange(9).reshape(3,3)
arr

# Solution
arr[:, [1,0,2]]
# > array([[1, 0, 2],
# >        [4, 3, 5],
# >        [7, 6, 8]])
```

### 17. 如何交换二维numpy数组中的两行？
**难度等级：**L2

**问题：**交换数组arr中的第1和第2行：

**给定：**

```python
arr = np.arange(9).reshape(3,3)
arr
```

**答案：**

```python
# Input
arr = np.arange(9).reshape(3,3)

# Solution
arr[[1,0,2], :]
# > array([[3, 4, 5],
# >        [0, 1, 2],
# >        [6, 7, 8]])
```

### 18. 如何反转二维数组的行？
**难度等级：**L2

**问题：**反转二维数组arr的行。

**给定：**

```python
# Input
arr = np.arange(9).reshape(3,3)
```

**答案：**

```python
# Input
arr = np.arange(9).reshape(3,3)
```

```python
# Solution
arr[::-1]
array([[6, 7, 8],
       [3, 4, 5],
       [0, 1, 2]])
```

### 19. 如何反转二维数组的列？
**难度等级：**L2

**问题：**反转二维数组arr的列。

**给定：**

```python
# Input
arr = np.arange(9).reshape(3,3)
```

**答案：**

```python
# Input
arr = np.arange(9).reshape(3,3)

# Solution
arr[:, ::-1]
# > array([[2, 1, 0],
# >        [5, 4, 3],
# >        [8, 7, 6]])
```

### 20. 如何创建包含5到10之间随机浮动的二维数组？
**难度等级：**L2

**问题：**创建一个形状为5x3的二维数组，以包含5到10之间的随机十进制数。

**答案：**

```python
# Input
arr = np.arange(9).reshape(3,3)

# Solution Method 1:
rand_arr = np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))
# print(rand_arr)

# Solution Method 2:
rand_arr = np.random.uniform(5,10, size=(5,3))
print(rand_arr)
# > [[ 8.50061025  9.10531502  6.85867783]
# >  [ 9.76262069  9.87717411  7.13466701]
# >  [ 7.48966403  8.33409158  6.16808631]
# >  [ 7.75010551  9.94535696  5.27373226]
# >  [ 8.0850361   5.56165518  7.31244004]]
```

### 21. 如何在numpy数组中只打印小数点后三位？
**难度等级：**L1

**问题：**只打印或显示numpy数组rand_arr的小数点后3位。

**给定：**

```python
rand_arr = np.random.random((5,3))
```

**答案：**
```python
# Input
rand_arr = np.random.random((5,3))

# Create the random array
rand_arr = np.random.random([5,3])

# Limit to 3 decimal places
np.set_printoptions(precision=3)
rand_arr[:4]
# > array([[ 0.443,  0.109,  0.97 ],
# >        [ 0.388,  0.447,  0.191],
# >        [ 0.891,  0.474,  0.212],
# >        [ 0.609,  0.518,  0.403]])
```

### 22. 如何通过e式科学记数法（如1e10）来打印一个numpy数组？
**难度等级：**L1

**问题：**通过e式科学记数法来打印rand_arr（如1e10）

**给定：**

```python
# Create the random array
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
rand_arr

# > array([[  5.434049e-04,   2.783694e-04,   4.245176e-04],
# >        [  8.447761e-04,   4.718856e-06,   1.215691e-04],
# >        [  6.707491e-04,   8.258528e-04,   1.367066e-04]])
```

**期望的输出：**

```python
# > array([[ 0.000543,  0.000278,  0.000425],
# >        [ 0.000845,  0.000005,  0.000122],
# >        [ 0.000671,  0.000826,  0.000137]])
```

**答案：**

```python
# Reset printoptions to default
np.set_printoptions(suppress=False)

# Create the random array
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
rand_arr
# > array([[  5.434049e-04,   2.783694e-04,   4.245176e-04],
# >        [  8.447761e-04,   4.718856e-06,   1.215691e-04],
# >        [  6.707491e-04,   8.258528e-04,   1.367066e-04]])
```

```python
np.set_printoptions(suppress=True, precision=6)  # precision is optional
rand_arr
# > array([[ 0.000543,  0.000278,  0.000425],
# >        [ 0.000845,  0.000005,  0.000122],
# >        [ 0.000671,  0.000826,  0.000137]])
```

### 23. 如何限制numpy数组输出中打印的项目数？
**难度等级：**L1

**问题：**将numpy数组a中打印的项数限制为最多6个元素。

**给定：**

```python
a = np.arange(15)
# > array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
```

**期望的输出：**

```python
# > array([ 0,  1,  2, ..., 12, 13, 14])
```

**答案：**

```python
np.set_printoptions(threshold=6)
a = np.arange(15)
a
# > array([ 0,  1,  2, ..., 12, 13, 14])
```

### 24. 如何打印完整的numpy数组而不截断
**难度等级：**L1

**问题：**打印完整的numpy数组a而不截断。

**给定：**

```python
np.set_printoptions(threshold=6)
a = np.arange(15)
a
# > array([ 0,  1,  2, ..., 12, 13, 14])
```

**期望的输出：**

```python
a
# > array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
```

**答案：**

```python
# Input
np.set_printoptions(threshold=6)
a = np.arange(15)

# Solution
np.set_printoptions(threshold=np.nan)
a
# > array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
```

### 25. 如何导入数字和文本的数据集保持文本在numpy数组中完好无损？
**难度等级：**L2

**问题：**导入鸢尾属植物数据集，保持文本不变。

**答案：**

```python
# Solution
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Print the first 3 rows
iris[:3]
# > array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa'],
# >        [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa'],
# >        [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa']], dtype=object)
```

### 26. 如何从1维元组数组中提取特定列？
**难度等级：**L2

**问题：**从前面问题中导入的一维鸢尾属植物数据集中提取文本列的物种。

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
```

**答案：**

```python
# **给定：**
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
print(iris_1d.shape)

# Solution:
species = np.array([row[4] for row in iris_1d])
species[:5]
# > (150,)
# > array([b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
# >        b'Iris-setosa'],
# >       dtype='|S18')
```

### 27. 如何将1维元组数组转换为2维numpy数组？
**难度等级：**L2

**问题：**通过省略鸢尾属植物数据集种类的文本字段，将一维鸢尾属植物数据集转换为二维数组iris_2d。

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)
```

**答案：**

```python
# **给定：**
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)

# Solution:
# Method 1: Convert each row to a list and get the first 4 items
iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
iris_2d[:4]

# Alt Method 2: Import only the first 4 columns from source url
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[:4]
# > array([[ 5.1,  3.5,  1.4,  0.2],
# >        [ 4.9,  3. ,  1.4,  0.2],
# >        [ 4.7,  3.2,  1.3,  0.2],
# >        [ 4.6,  3.1,  1.5,  0.2]])
```

### 28. 如何计算numpy数组的均值，中位数，标准差？
**难度等级：**L1

**问题：**求出鸢尾属植物萼片长度的平均值、中位数和标准差(第1列)

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

# Solution
mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mu, med, sd)
# > 5.84333333333 5.8 0.825301291785
```

### 29. 如何规范化数组，使数组的值正好介于0和1之间？
**难度等级：**L2

**问题：**创建一种标准化形式的鸢尾属植物间隔长度，其值正好介于0和1之间，这样最小值为0，最大值为1。

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

# Solution
Smax, Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin)/(Smax - Smin)
# or 
S = (sepallength - Smin)/sepallength.ptp()  # Thanks, David Ojeda!
print(S)
# > [ 0.222  0.167  0.111  0.083  0.194  0.306  0.083  0.194  0.028  0.167
# >   0.306  0.139  0.139  0.     0.417  0.389  0.306  0.222  0.389  0.222
# >   0.306  0.222  0.083  0.222  0.139  0.194  0.194  0.25   0.25   0.111
# >   0.139  0.306  0.25   0.333  0.167  0.194  0.333  0.167  0.028  0.222
# >   0.194  0.056  0.028  0.194  0.222  0.139  0.222  0.083  0.278  0.194
# >   0.75   0.583  0.722  0.333  0.611  0.389  0.556  0.167  0.639  0.25
# >   0.194  0.444  0.472  0.5    0.361  0.667  0.361  0.417  0.528  0.361
# >   0.444  0.5    0.556  0.5    0.583  0.639  0.694  0.667  0.472  0.389
# >   0.333  0.333  0.417  0.472  0.306  0.472  0.667  0.556  0.361  0.333
# >   0.333  0.5    0.417  0.194  0.361  0.389  0.389  0.528  0.222  0.389
# >   0.556  0.417  0.778  0.556  0.611  0.917  0.167  0.833  0.667  0.806
# >   0.611  0.583  0.694  0.389  0.417  0.583  0.611  0.944  0.944  0.472
# >   0.722  0.361  0.944  0.556  0.667  0.806  0.528  0.5    0.583  0.806
# >   0.861  1.     0.583  0.556  0.5    0.944  0.556  0.583  0.472  0.722
# >   0.667  0.722  0.417  0.694  0.667  0.667  0.556  0.611  0.528  0.444]
```

### 30. 如何计算Softmax得分？
**难度等级：**L3

**问题：**计算sepallength的softmax分数。

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.array([float(row[0]) for row in iris])

# Solution
def softmax(x):
    """Compute softmax values for each sets of scores in x.
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

print(softmax(sepallength))
# > [ 0.002  0.002  0.001  0.001  0.002  0.003  0.001  0.002  0.001  0.002
# >   0.003  0.002  0.002  0.001  0.004  0.004  0.003  0.002  0.004  0.002
# >   0.003  0.002  0.001  0.002  0.002  0.002  0.002  0.002  0.002  0.001
# >   0.002  0.003  0.002  0.003  0.002  0.002  0.003  0.002  0.001  0.002
# >   0.002  0.001  0.001  0.002  0.002  0.002  0.002  0.001  0.003  0.002
# >   0.015  0.008  0.013  0.003  0.009  0.004  0.007  0.002  0.01   0.002
# >   0.002  0.005  0.005  0.006  0.004  0.011  0.004  0.004  0.007  0.004
# >   0.005  0.006  0.007  0.006  0.008  0.01   0.012  0.011  0.005  0.004
# >   0.003  0.003  0.004  0.005  0.003  0.005  0.011  0.007  0.004  0.003
# >   0.003  0.006  0.004  0.002  0.004  0.004  0.004  0.007  0.002  0.004
# >   0.007  0.004  0.016  0.007  0.009  0.027  0.002  0.02   0.011  0.018
# >   0.009  0.008  0.012  0.004  0.004  0.008  0.009  0.03   0.03   0.005
# >   0.013  0.004  0.03   0.007  0.011  0.018  0.007  0.006  0.008  0.018
# >   0.022  0.037  0.008  0.007  0.006  0.03   0.007  0.008  0.005  0.013
# >   0.011  0.013  0.004  0.012  0.011  0.011  0.007  0.009  0.007  0.005]
```

### 31. 如何找到numpy数组的百分位数？
**难度等级：**L1

**问题：**找到鸢尾属植物数据集的第5和第95百分位数

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

# Solution
np.percentile(sepallength, q=[5, 95])
# > array([ 4.6  ,  7.255])
```

### 32. 如何在数组中的随机位置插入值？
**难度等级：**L2

**问题：**在iris_2d数据集中的20个随机位置插入np.nan值

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

# Method 1
i, j = np.where(iris_2d)

# i, j contain the row numbers and column numbers of 600 elements of iris_x
np.random.seed(100)
iris_2d[np.random.choice((i), 20), np.random.choice((j), 20)] = np.nan

# Method 2
np.random.seed(100)
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Print first 10 rows
print(iris_2d[:10])
# > [[b'5.1' b'3.5' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.9' b'3.0' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
# >  [b'4.6' b'3.1' b'1.5' b'0.2' b'Iris-setosa']
# >  [b'5.0' b'3.6' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'5.4' b'3.9' b'1.7' b'0.4' b'Iris-setosa']
# >  [b'4.6' b'3.4' b'1.4' b'0.3' b'Iris-setosa']
# >  [b'5.0' b'3.4' b'1.5' b'0.2' b'Iris-setosa']
# >  [b'4.4' nan b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']]
```

### 33. 如何在numpy数组中找到缺失值的位置？
**难度等级：**L2

**问题：**在iris_2d的sepallength中查找缺失值的数量和位置（第1列）

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution
print("Number of missing values: \n", np.isnan(iris_2d[:, 0]).sum())
print("Position of missing values: \n", np.where(np.isnan(iris_2d[:, 0])))
# > Number of missing values: 
# >  5
# > Position of missing values: 
# >  (array([ 39,  88,  99, 130, 147]),)
```

### 34. 如何根据两个或多个条件过滤numpy数组？
**难度等级：**L3

**问题：**过滤具有petallength（第3列）> 1.5 和 sepallength（第1列）< 5.0 的iris_2d行

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# Solution
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
iris_2d[condition]
# > array([[ 4.8,  3.4,  1.6,  0.2],
# >        [ 4.8,  3.4,  1.9,  0.2],
# >        [ 4.7,  3.2,  1.6,  0.2],
# >        [ 4.8,  3.1,  1.6,  0.2],
# >        [ 4.9,  2.4,  3.3,  1. ],
# >        [ 4.9,  2.5,  4.5,  1.7]])
```

### 35. 如何从numpy数组中删除包含缺失值的行？
**难度等级：**L3:

**问题：**选择没有任何nan值的iris_2d行。

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution
# No direct numpy function for this.
# Method 1:
any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
iris_2d[any_nan_in_row][:5]

# Method 2: (By Rong)
iris_2d[np.sum(np.isnan(iris_2d), axis = 1) == 0][:5]
# > array([[ 4.9,  3. ,  1.4,  0.2],
# >        [ 4.7,  3.2,  1.3,  0.2],
# >        [ 4.6,  3.1,  1.5,  0.2],
# >        [ 5. ,  3.6,  1.4,  0.2],
# >        [ 5.4,  3.9,  1.7,  0.4]])
```

### 36. 如何找到numpy数组的两列之间的相关性？
**难度等级：**L2

**问题：**在iris_2d中找出SepalLength（第1列）和PetalLength（第3列）之间的相关性

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# Solution 1
np.corrcoef(iris[:, 0], iris[:, 2])[0, 1]

# Solution 2
from scipy.stats.stats import pearsonr  
corr, p_value = pearsonr(iris[:, 0], iris[:, 2])
print(corr)

# Correlation coef indicates the degree of linear relationship between two numeric variables.
# It can range between -1 to +1.

# The p-value roughly indicates the probability of an uncorrelated system producing 
# datasets that have a correlation at least as extreme as the one computed.
# The lower the p-value (<0.01), stronger is the significance of the relationship.
# It is not an indicator of the strength.
# > 0.871754157305
```

### 37. 如何查找给定数组是否具有任何空值？
**难度等级：**L2

**问题：**找出iris_2d是否有任何缺失值。

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

np.isnan(iris_2d).any()
# > False
```

### 38. 如何在numpy数组中用0替换所有缺失值？
**难度等级：**L2

**问题：**在numpy数组中将所有出现的nan替换为0

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# Solution
iris_2d[np.isnan(iris_2d)] = 0
iris_2d[:4]
# > array([[ 5.1,  3.5,  1.4,  0. ],
# >        [ 4.9,  3. ,  1.4,  0.2],
# >        [ 4.7,  3.2,  1.3,  0.2],
# >        [ 4.6,  3.1,  1.5,  0.2]])
```

### 39. 如何在numpy数组中查找唯一值的计数？
**难度等级：**L2

**问题：**找出鸢尾属植物物种中的独特值和独特值的数量

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
```

**答案：**

```python
# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Solution
# Extract the species column as an array
species = np.array([row.tolist()[4] for row in iris])

# Get the unique values and the counts
np.unique(species, return_counts=True)
# > (array([b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica'],
# >        dtype='|S15'), array([50, 50, 50]))
```

### 40. 如何将数字转换为分类（文本）数组？
**难度等级：**L2

**问题：**将iris_2d的花瓣长度（第3列）加入以形成文本数组，这样如果花瓣长度为：

- Less than 3 --> 'small'
- 3-5 --> 'medium'
- '>=5 --> 'large'

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# Bin petallength 
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])

# Map it to respective category
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]

# View
petal_length_cat[:4]
<# > ['small', 'small', 'small', 'small']
```

### 41. 如何从numpy数组的现有列创建新列？
**难度等级：**L2

**问题：**在iris_2d中为卷创建一个新列，其中volume是``（pi x petallength x sepal_length ^ 2）/ 3``

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution
# Compute volume
sepallength = iris_2d[:, 0].astype('float')
petallength = iris_2d[:, 2].astype('float')
volume = (np.pi * petallength * (sepallength**2))/3

# Introduce new dimension to match iris_2d's
volume = volume[:, np.newaxis]

# Add the new column
out = np.hstack([iris_2d, volume])

# View
out[:4]
# > array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa', 38.13265162927291],
# >        [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa', 35.200498485922445],
# >        [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa', 30.0723720777127],
# >        [b'4.6', b'3.1', b'1.5', b'0.2', b'Iris-setosa', 33.238050274980004]], dtype=object)
```

### 42. 如何在numpy中进行概率抽样？
**难度等级：**L3

**问题：**随机抽鸢尾属植物的种类，使得刚毛的数量是云芝和维吉尼亚的两倍

**给定：**

```python
# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
```

**答案：**

```python
# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution
# Get the species column
species = iris[:, 4]

# Approach 1: Generate Probablistically
np.random.seed(100)
a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])

# Approach 2: Probablistic Sampling (preferred)
np.random.seed(100)
probs = np.r_[np.linspace(0, 0.500, num=50), np.linspace(0.501, .750, num=50), np.linspace(.751, 1.0, num=50)]
index = np.searchsorted(probs, np.random.random(150))
species_out = species[index]
print(np.unique(species_out, return_counts=True))

# > (array([b'Iris-setosa', b'Iris-versicolor', b'Iris-virginica'], dtype=object), array([77, 37, 36]))
```

方法2是首选方法，因为它创建了一个索引变量，该变量可用于取样2维表格数据。

### 43. 如何在按另一个数组分组时获取数组的第二大值？
**难度等级：**L2

**问题：**第二长的物种setosa的价值是多少

**给定：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
```

**答案：**

```python

# Import iris keeping the text column intact
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution
# Get the species and petal length columns
petal_len_setosa = iris[iris[:, 4] == b'Iris-setosa', [2]].astype('float')

# Get the second last value
np.unique(np.sort(petal_len_setosa))[-2]
# > 1.7
```

### 44. 如何按列对2D数组进行排序
**难度等级：**L2

**问题：**根据sepallength列对虹膜数据集进行排序。

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
```

**答案：**

```python
# Sort by column position 0: SepalLength
print(iris[iris[:,0].argsort()][:20])
# > [[b'4.3' b'3.0' b'1.1' b'0.1' b'Iris-setosa']
# >  [b'4.4' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
# >  [b'4.4' b'3.0' b'1.3' b'0.2' b'Iris-setosa']
# >  [b'4.4' b'2.9' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.5' b'2.3' b'1.3' b'0.3' b'Iris-setosa']
# >  [b'4.6' b'3.6' b'1.0' b'0.2' b'Iris-setosa']
# >  [b'4.6' b'3.1' b'1.5' b'0.2' b'Iris-setosa']
# >  [b'4.6' b'3.4' b'1.4' b'0.3' b'Iris-setosa']
# >  [b'4.6' b'3.2' b'1.4' b'0.2' b'Iris-setosa']
# >  [b'4.7' b'3.2' b'1.3' b'0.2' b'Iris-setosa']
# >  [b'4.7' b'3.2' b'1.6' b'0.2' b'Iris-setosa']
# >  [b'4.8' b'3.0' b'1.4' b'0.1' b'Iris-setosa']
# >  [b'4.8' b'3.0' b'1.4' b'0.3' b'Iris-setosa']
# >  [b'4.8' b'3.4' b'1.9' b'0.2' b'Iris-setosa']
# >  [b'4.8' b'3.4' b'1.6' b'0.2' b'Iris-setosa']
# >  [b'4.8' b'3.1' b'1.6' b'0.2' b'Iris-setosa']
# >  [b'4.9' b'2.4' b'3.3' b'1.0' b'Iris-versicolor']
# >  [b'4.9' b'2.5' b'4.5' b'1.7' b'Iris-virginica']
# >  [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']
# >  [b'4.9' b'3.1' b'1.5' b'0.1' b'Iris-setosa']]
```

### 45. 如何在numpy数组中找到最常见的值？
**难度等级：**L1

**问题：**在鸢尾属植物数据集中找到最常见的花瓣长度值（第3列）。

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
```

**答案：**

```python
# **给定：**
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution:
vals, counts = np.unique(iris[:, 2], return_counts=True)
print(vals[np.argmax(counts)])
# > b'1.5'
```

### 46. 如何找到第一次出现的值大于给定值的位置？
**难度等级：**L2

**问题：**在虹膜数据集的petalwidth第4列中查找第一次出现的值大于1.0的位置。

```python
# **给定：**
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
```

**答案：**

```python
# **给定：**
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')

# Solution: (edit: changed argmax to argwhere. Thanks Rong!)
np.argwhere(iris[:, 3].astype(float) > 1.0)[0]
# > 50
```

### 47. 如何将大于给定值的所有值替换为给定的截止值？
**难度等级：**L2

**问题：**从数组a中，替换所有大于30到30和小于10到10的值。

**给定：**
```python
np.random.seed(100)
a = np.random.uniform(1,50, 20)
```

**答案：**

```python
# Input
np.set_printoptions(precision=2)
np.random.seed(100)
a = np.random.uniform(1,50, 20)

# Solution 1: Using np.clip
np.clip(a, a_min=10, a_max=30)

# Solution 2: Using np.where
print(np.where(a < 10, 10, np.where(a > 30, 30, a)))
# > [ 27.63  14.64  21.8   30.    10.    10.    30.    30.    10.    29.18  30.
# >   11.25  10.08  10.    11.77  30.    30.    10.    30.    14.43]
```

### 48. 如何从numpy数组中获取最大n值的位置？
**难度等级：**L2

**问题：**获取给定数组a中前5个最大值的位置。

```python
np.random.seed(100)
a = np.random.uniform(1,50, 20)
```

**答案：**

```python
# Input
np.random.seed(100)
a = np.random.uniform(1,50, 20)

# Solution:
print(a.argsort())
# > [18 7 3 10 15]

# Solution 2:
np.argpartition(-a, 5)[:5]
# > [15 10  3  7 18]

# Below methods will get you the values.
# Method 1:
a[a.argsort()][-5:]

# Method 2:
np.sort(a)[-5:]

# Method 3:
np.partition(a, kth=-5)[-5:]

# Method 4:
a[np.argpartition(-a, 5)][:5]
```

### 49. 如何计算数组中所有可能值的行数？
**难度等级：**L4

**问题：**按行计算唯一值的计数。

**给定：**

```python
np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))
arr
> array([[ 9,  9,  4,  8,  8,  1,  5,  3,  6,  3],
>        [ 3,  3,  2,  1,  9,  5,  1, 10,  7,  3],
>        [ 5,  2,  6,  4,  5,  5,  4,  8,  2,  2],
>        [ 8,  8,  1,  3, 10, 10,  4,  3,  6,  9],
>        [ 2,  1,  8,  7,  3,  1,  9,  3,  6,  2],
>        [ 9,  2,  6,  5,  3,  9,  4,  6,  1, 10]])
```

**期望的输出：**

```python
> [[1, 0, 2, 1, 1, 1, 0, 2, 2, 0],
>  [2, 1, 3, 0, 1, 0, 1, 0, 1, 1],
>  [0, 3, 0, 2, 3, 1, 0, 1, 0, 0],
>  [1, 0, 2, 1, 0, 1, 0, 2, 1, 2],
>  [2, 2, 2, 0, 0, 1, 1, 1, 1, 0],
>  [1, 1, 1, 1, 1, 2, 0, 0, 2, 1]]
```

输出包含10列，表示从1到10的数字。这些值是各行中数字的计数。
例如，cell(0，2)的值为2，这意味着数字3在第一行中恰好出现了2次。

**答案：**

```python
# **给定：**
np.random.seed(100)
arr = np.random.randint(1,11,size=(6, 10))
arr
# > array([[ 9,  9,  4,  8,  8,  1,  5,  3,  6,  3],
# >        [ 3,  3,  2,  1,  9,  5,  1, 10,  7,  3],
# >        [ 5,  2,  6,  4,  5,  5,  4,  8,  2,  2],
# >        [ 8,  8,  1,  3, 10, 10,  4,  3,  6,  9],
# >        [ 2,  1,  8,  7,  3,  1,  9,  3,  6,  2],
# >        [ 9,  2,  6,  5,  3,  9,  4,  6,  1, 10]])
```

```python
# Solution
def counts_of_all_values_rowwise(arr2d):
    # Unique values and its counts row wise
    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]

    # Counts of all values row wise
    return([[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array])

# Print
print(np.arange(1,11))
counts_of_all_values_rowwise(arr)
# > [ 1  2  3  4  5  6  7  8  9 10]

# > [[1, 0, 2, 1, 1, 1, 0, 2, 2, 0],
# >  [2, 1, 3, 0, 1, 0, 1, 0, 1, 1],
# >  [0, 3, 0, 2, 3, 1, 0, 1, 0, 0],
# >  [1, 0, 2, 1, 0, 1, 0, 2, 1, 2],
# >  [2, 2, 2, 0, 0, 1, 1, 1, 1, 0],
# >  [1, 1, 1, 1, 1, 2, 0, 0, 2, 1]]
```

```python
# Example 2:
arr = np.array([np.array(list('bill clinton')), np.array(list('narendramodi')), np.array(list('jjayalalitha'))])
print(np.unique(arr))
counts_of_all_values_rowwise(arr)
# > [' ' 'a' 'b' 'c' 'd' 'e' 'h' 'i' 'j' 'l' 'm' 'n' 'o' 'r' 't' 'y']

# > [[1, 0, 1, 1, 0, 0, 0, 2, 0, 3, 0, 2, 1, 0, 1, 0],
# >  [0, 2, 0, 0, 2, 1, 0, 1, 0, 0, 1, 2, 1, 2, 0, 0],
# >  [0, 4, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1]]
```

### 50. 如何将数组转换为平面一维数组？
**难度等级：**L2

**问题：**将array_of_arrays转换为扁平线性1d数组。

**给定：**

```python
# **给定：**
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)

array_of_arrays = np.array([arr1, arr2, arr3])
array_of_arrays
# > array([array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9])], dtype=object)
```

**期望的输出：**

```python
# > array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

**答案：**

```python
 # **给定：**
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)

array_of_arrays = np.array([arr1, arr2, arr3])
print('array_of_arrays: ', array_of_arrays)

# Solution 1
arr_2d = np.array([a for arr in array_of_arrays for a in arr])

# Solution 2:
arr_2d = np.concatenate(array_of_arrays)
print(arr_2d)
# > array_of_arrays:  [array([0, 1, 2]) array([3, 4, 5, 6]) array([7, 8, 9])]
# > [0 1 2 3 4 5 6 7 8 9]
```

### 51. 如何在numpy中为数组生成单热编码？
**难度等级：**L4

**问题：**计算一次性编码(数组中每个唯一值的虚拟二进制变量)

**给定：**

```python
np.random.seed(101) 
arr = np.random.randint(1,4, size=6)
arr
# > array([2, 3, 2, 2, 2, 1])
```

**期望输出：**

```python
# > array([[ 0.,  1.,  0.],
# >        [ 0.,  0.,  1.],
# >        [ 0.,  1.,  0.],
# >        [ 0.,  1.,  0.],
# >        [ 0.,  1.,  0.],
# >        [ 1.,  0.,  0.]])
```

**答案：**

```python
# **给定：**
np.random.seed(101) 
arr = np.random.randint(1,4, size=6)
arr
# > array([2, 3, 2, 2, 2, 1])

# Solution:
def one_hot_encodings(arr):
    uniqs = np.unique(arr)
    out = np.zeros((arr.shape[0], uniqs.shape[0]))
    for i, k in enumerate(arr):
        out[i, k-1] = 1
    return out

one_hot_encodings(arr)
# > array([[ 0.,  1.,  0.],
# >        [ 0.,  0.,  1.],
# >        [ 0.,  1.,  0.],
# >        [ 0.,  1.,  0.],
# >        [ 0.,  1.,  0.],
# >        [ 1.,  0.,  0.]])

# Method 2:
(arr[:, None] == np.unique(arr)).view(np.int8)
```

### 52. 如何创建按分类变量分组的行号？
**难度等级：**L3

**问题：**创建按分类变量分组的行号。使用以下来自鸢尾属植物物种的样本作为输入。

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))
species_small
# > array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
# >        'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
# >       dtype='<U15')
```

**期望的输出：**

```python
# > [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7]
```

**答案：**

```python
# **给定：**
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
np.random.seed(100)
species_small = np.sort(np.random.choice(species, size=20))
species_small
# > array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
# >        'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica'],
# >       dtype='<U15')
```

```python
print([i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])])
```

```python
[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5]
```

### 53. 如何根据给定的分类变量创建组ID？
**难度等级：**L4

**问题：**根据给定的分类变量创建组ID。使用以下来自鸢尾属植物物种的样本作为输入。

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
species_small = np.sort(np.random.choice(species, size=20))
species_small
# > array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
# >        'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
# >       dtype='<U15')
```

**期望的输出：**

```python
# > [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
```

**答案：**

```python
# **给定：**
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)
np.random.seed(100)
species_small = np.sort(np.random.choice(species, size=20))
species_small
# > array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
# >        'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
# >        'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
# >        'Iris-virginica'],
# >       dtype='<U15')
```

```python
# Solution:
output = [np.argwhere(np.unique(species_small) == s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]]

# Solution: For Loop version
output = []
uniqs = np.unique(species_small)

for val in uniqs:  # uniq values in group
    for s in species_small[species_small==val]:  # each element in group
        groupid = np.argwhere(uniqs == s).tolist()[0][0]  # groupid
        output.append(groupid)

print(output)
# > [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
```

### 54. 如何使用numpy对数组中的项进行排名？
**难度等级：**L2

**问题：**为给定的数字数组a创建排名。

**给定：**

```python
np.random.seed(10)
a = np.random.randint(20, size=10)
print(a)
# > [ 9  4 15  0 17 16 17  8  9  0]
```

**期望输出：**

```python
[4 2 6 0 8 7 9 3 5 1]
```

**答案：**

```python
np.random.seed(10)
a = np.random.randint(20, size=10)
print('Array: ', a)

# Solution
print(a.argsort().argsort())
print('Array: ', a)
# > Array:  [ 9  4 15  0 17 16 17  8  9  0]
# > [4 2 6 0 8 7 9 3 5 1]
# > Array:  [ 9  4 15  0 17 16 17  8  9  0]
```

### 55. 如何使用numpy对多维数组中的项进行排名？
**难度等级：**L3

**问题：**创建与给定数字数组a相同形状的排名数组。

**给定：**

```python
np.random.seed(10)
a = np.random.randint(20, size=[2,5])
print(a)
# > [[ 9  4 15  0 17]
# >  [16 17  8  9  0]]
```

**期望输出：**

```python
# > [[4 2 6 0 8]
# >  [7 9 3 5 1]]
```

**答案：**

```python
# **给定：**
np.random.seed(10)
a = np.random.randint(20, size=[2,5])
print(a)

# Solution
print(a.ravel().argsort().argsort().reshape(a.shape))
# > [[ 9  4 15  0 17]
# >  [16 17  8  9  0]]
# > [[4 2 6 0 8]
# >  [7 9 3 5 1]]
```

### 56. 如何在二维numpy数组的每一行中找到最大值？
**难度等级：**L2

**问题：**计算给定数组中每行的最大值。

**给定：**

```python
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a
# > array([[9, 9, 4],
# >        [8, 8, 1],
# >        [5, 3, 6],
# >        [3, 3, 3],
# >        [2, 1, 9]])
```

**答案：**

```python
# Input
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a

# Solution 1
np.amax(a, axis=1)

# Solution 2
np.apply_along_axis(np.max, arr=a, axis=1)
# > array([9, 8, 6, 3, 9])
```

### 57. 如何计算二维numpy数组每行的最小值？
**难度等级：**L3

**问题：**为给定的二维numpy数组计算每行的最小值。

**给定：**

```python
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a
# > array([[9, 9, 4],
# >        [8, 8, 1],
# >        [5, 3, 6],
# >        [3, 3, 3],
# >        [2, 1, 9]])
```

**答案：**

```python
# Input
np.random.seed(100)
a = np.random.randint(1,10, [5,3])
a

# Solution
np.apply_along_axis(lambda x: np.min(x)/np.max(x), arr=a, axis=1)
# > array([ 0.44444444,  0.125     ,  0.5       ,  1.        ,  0.11111111])
```

### 58. 如何在numpy数组中找到重复的记录？
**难度等级：**L3

**问题：**在给定的numpy数组中找到重复的条目(第二次出现以后)，并将它们标记为True。第一次出现应该是False的。

**给定：**

```python
# Input
np.random.seed(100)
a = np.random.randint(0, 5, 10)
print('Array: ', a)
# > Array: [0 0 3 0 2 4 2 2 2 2]
```

**期望的输出：**

```python
# > [False  True False  True False False  True  True  True  True]
```

**答案：**

```python
# Input
np.random.seed(100)
a = np.random.randint(0, 5, 10)

## Solution
# There is no direct function to do this as of 1.13.3

# Create an all True array
out = np.full(a.shape[0], True)

# Find the index positions of unique elements
unique_positions = np.unique(a, return_index=True)[1]

# Mark those positions as False
out[unique_positions] = False

print(out)
# > [False  True False  True False False  True  True  True  True]
```

### 59. 如何找出数字的分组均值？
**难度等级：**L3

**问题：**在二维数字数组中查找按分类列分组的数值列的平均值

**给定：**

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
```

**理想的输出：**

```python
# > [[b'Iris-setosa', 3.418],
# >  [b'Iris-versicolor', 2.770],
# >  [b'Iris-virginica', 2.974]]
```

**答案：**

```python
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')


# Solution
# No direct way to implement this. Just a version of a workaround.
numeric_column = iris[:, 1].astype('float')  # sepalwidth
grouping_column = iris[:, 4]  # species

# List comprehension version
[[group_val, numeric_column[grouping_column==group_val].mean()] for group_val in np.unique(grouping_column)]

# For Loop version
output = []
for group_val in np.unique(grouping_column):
    output.append([group_val, numeric_column[grouping_column==group_val].mean()])

output
# > [[b'Iris-setosa', 3.418],
# >  [b'Iris-versicolor', 2.770],
# >  [b'Iris-virginica', 2.974]]
```

### 60. 如何将PIL图像转换为numpy数组？
**难度等级：**L3

**问题：**从以下URL导入图像并将其转换为numpy数组。

```
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
```

**答案：**

```python
from io import BytesIO
from PIL import Image
import PIL, requests

# Import image from URL
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
response = requests.get(URL)

# Read it as Image
I = Image.open(BytesIO(response.content))

# Optionally resize
I = I.resize([150,150])

# Convert to numpy array
arr = np.asarray(I)

# Optionaly Convert it back to an image and show
im = PIL.Image.fromarray(np.uint8(arr))
Image.Image.show(im)
```

### 61. 如何删除numpy数组中所有缺少的值？
**难度等级：**L2

**问题：**从一维numpy数组中删除所有NaN值

**给定：**

```python
np.array([1,2,3,np.nan,5,6,7,np.nan])
```

**期望的输出：**

```python
array([ 1.,  2.,  3.,  5.,  6.,  7.])
```

**答案：**

```python
a = np.array([1,2,3,np.nan,5,6,7,np.nan])
a[~np.isnan(a)]
# > array([ 1.,  2.,  3.,  5.,  6.,  7.])
```

### 62. 如何计算两个数组之间的欧氏距离？
**难度等级：**L3

**问题：**计算两个数组a和数组b之间的欧氏距离。

**给定：**
```python
a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])
```

**答案：**

```python
# Input
a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])

# Solution
dist = np.linalg.norm(a-b)
dist
# > 6.7082039324993694
```

### 63. 如何在一维数组中找到所有的局部极大值(或峰值)？
**难度等级：**L4

**问题：**找到一个一维数字数组a中的所有峰值。峰顶是两边被较小数值包围的点。

**给定：**
```python
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
```

**期望的输出：**

```python
# > array([2, 5])
```

其中，2和5是峰值7和6的位置。

**答案：**

```python
a = np.array([1, 3, 7, 1, 2, 6, 0, 1])
doublediff = np.diff(np.sign(np.diff(a)))
peak_locations = np.where(doublediff == -2)[0] + 1
peak_locations
# > array([2, 5])
```

### 64. 如何从二维数组中减去一维数组，其中一维数组的每一项从各自的行中减去？
**难度等级：**L2

**问题：**从2d数组a_2d中减去一维数组b_1D，使得b_1D的每一项从a_2d的相应行中减去。

```python
a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,2,3])
```

**期望的输出：**

```python
# > [[2 2 2]
# >  [2 2 2]
# >  [2 2 2]]
```

**答案：**

```python
# Input
a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,2,3])

# Solution
print(a_2d - b_1d[:,None])
# > [[2 2 2]
# >  [2 2 2]
# >  [2 2 2]]
```

### 65. 如何查找数组中项的第n次重复索引？
**难度等级：**L2

**问题：**找出x中数字1的第5次重复的索引。

```python
x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
```

**答案：**

```python
x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])
n = 5

# Solution 1: List comprehension
[i for i, v in enumerate(x) if v == 1][n-1]

# Solution 2: Numpy version
np.where(x == 1)[0][n-1]
# > 8
```

### 66. 如何将numpy的datetime 64对象转换为datetime的datetime对象？

**难度等级：**L2

**问题：**将numpy的``datetime64``对象转换为datetime的datetime对象

```python
# **给定：** a numpy datetime64 object
dt64 = np.datetime64('2018-02-25 22:10:10')
```

**答案：**

```python
# **给定：** a numpy datetime64 object
dt64 = np.datetime64('2018-02-25 22:10:10')

# Solution
from datetime import datetime
dt64.tolist()

# or

dt64.astype(datetime)
# > datetime.datetime(2018, 2, 25, 22, 10, 10)
```

### 67. 如何计算numpy数组的移动平均值？
**难度等级：**L3

**问题：**对于给定的一维数组，计算窗口大小为3的移动平均值。

**给定：**

```python
np.random.seed(100)
Z = np.random.randint(10, size=10)
```

**答案：**

```python
# Solution
# Source: https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

np.random.seed(100)
Z = np.random.randint(10, size=10)
print('array: ', Z)
# Method 1
moving_average(Z, n=3).round(2)

# Method 2:  # Thanks AlanLRH!
# np.ones(3)/3 gives equal weights. Use np.ones(4)/4 for window size 4.
np.convolve(Z, np.ones(3)/3, mode='valid') . 


# > array:  [8 8 3 7 7 0 4 2 5 2]
# > moving average:  [ 6.33  6.    5.67  4.67  3.67  2.    3.67  3.  ]
```

### 68. 如何在给定起始点、长度和步骤的情况下创建一个numpy数组序列？
**难度等级：**L2

**问题：**创建长度为10的numpy数组，从5开始，在连续的数字之间的步长为3。

**答案：**

```python
length = 10
start = 5
step = 3

def seq(start, length, step):
    end = start + (step*length)
    return np.arange(start, end, step)

seq(start, length, step)
# > array([ 5,  8, 11, 14, 17, 20, 23, 26, 29, 32])
```

### 69. 如何填写不规则系列的numpy日期中的缺失日期？
**难度等级：**L3

**问题：**给定一系列不连续的日期序列。填写缺失的日期，使其成为连续的日期序列。

**给定：**

```python
# Input
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
print(dates)
# > ['2018-02-01' '2018-02-03' '2018-02-05' '2018-02-07' '2018-02-09'
# >  '2018-02-11' '2018-02-13' '2018-02-15' '2018-02-17' '2018-02-19'
# >  '2018-02-21' '2018-02-23']
```

**答案：**

```python
# Input
dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)
print(dates)

# Solution ---------------
filled_in = np.array([np.arange(date, (date+d)) for date, d in zip(dates, np.diff(dates))]).reshape(-1)

# add the last day
output = np.hstack([filled_in, dates[-1]])
output

# For loop version -------
out = []
for date, d in zip(dates, np.diff(dates)):
    out.append(np.arange(date, (date+d)))

filled_in = np.array(out).reshape(-1)

# add the last day
output = np.hstack([filled_in, dates[-1]])
output
# > ['2018-02-01' '2018-02-03' '2018-02-05' '2018-02-07' '2018-02-09'
# >  '2018-02-11' '2018-02-13' '2018-02-15' '2018-02-17' '2018-02-19'
# >  '2018-02-21' '2018-02-23']

# > array(['2018-02-01', '2018-02-02', '2018-02-03', '2018-02-04',
# >        '2018-02-05', '2018-02-06', '2018-02-07', '2018-02-08',
# >        '2018-02-09', '2018-02-10', '2018-02-11', '2018-02-12',
# >        '2018-02-13', '2018-02-14', '2018-02-15', '2018-02-16',
# >        '2018-02-17', '2018-02-18', '2018-02-19', '2018-02-20',
# >        '2018-02-21', '2018-02-22', '2018-02-23'], dtype='datetime64[D]')
``` 

### 70. 如何从给定的一维数组创建步长？
**难度等级：**L4

**问题：**从给定的一维数组arr中，利用步进生成一个二维矩阵，窗口长度为4，步距为2，类似于 ``[[0,1,2,3], [2,3,4,5], [4,5,6,7]..]``

**给定：**

```python
arr = np.arange(15) 
arr
# > array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
```

**期望的输出：**

```python
# > [[ 0  1  2  3]
# >  [ 2  3  4  5]
# >  [ 4  5  6  7]
# >  [ 6  7  8  9]
# >  [ 8  9 10 11]
# >  [10 11 12 13]]
```

**答案：**

```python
def gen_strides(a, stride_len=5, window_len=5):
    n_strides = ((a.size-window_len)//stride_len) + 1
    # return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])
    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])

print(gen_strides(np.arange(15), stride_len=2, window_len=4))
# > [[ 0  1  2  3]
# >  [ 2  3  4  5]
# >  [ 4  5  6  7]
# >  [ 6  7  8  9]
# >  [ 8  9 10 11]
# >  [10 11 12 13]]
```

未完待续...

## 文章出处

由NumPy中文文档翻译，原作者为 machinelearningplus.com，翻译至：[https://www.machinelearningplus.com/python/101-numpy-exercises-python/](https://www.machinelearningplus.com/python/101-numpy-exercises-python/)