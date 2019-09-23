---
meta:
  - name: keywords
    content: Python、Numpy 教程
  - name: description
    content: 我们将在本课程的所有作业中使用Python编程语言。Python本身就是一种伟大的通用编程语言，并且它在一些其...
---

# Python、Numpy 教程

我们将在本课程的所有作业中使用Python编程语言。Python本身就是一种伟大的通用编程语言，并且它在一些其他流行的Python库(numpy、sciy、matplotlib)的帮助下，它成为了一个强大的科学计算环境。

我们希望你们中大部分人会有一点Python和numpy的使用经验；因为对于大部分人来说，本节将作为关于Python编程语言和使用Python进行科学计算的快速速成课程。

你们中的一些人可能以前学过Matlab接触过相关的知识，如果是这样的话，我推荐你们看一下这篇文章：[Numpy对于Matlab用户](/user_guide/numpy_for_matlab_users.html)。

你还可以在 [这里找到](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb) 由 [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) 和 [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335) 为 [CS 228](https://cs.stanford.edu/~ermon/cs228/index.html) 创建的本教程的IPython笔记版本。

**目录**：

*   [Python](#Python)
    *   [基本数据类型](#基本数据类型)
    *   [容器(Containers)](#容器(Containers))
        *   [列表(Lists)](#列表(Lists))
        *   [字典](#字典)
        *   [集合(Sets)](#集合(Sets))
        *   [元组(Tuples)](#元组(Tuples))
    *   [函数(Functions)](#函数(Functions))
    *   [类(Classes)](#类(Classes))
*   [Numpy](#Numpy)
    *   [数组(Arrays)](#数组(Arrays))
    *   [数组索引](#数组索引)
    *   [数据类型](#数据类型)
    *   [数组中的数学](#数组中的数学)
    *   [广播(Broadcasting)](#广播(Broadcasting))
*   [SciPy](#SciPy)
    *   [图像操作](#图像操作)
    *   [MATLAB文件](#MATLAB文件)
    *   [点之间的距离](#点之间的距离)
*   [Matplotlib](#Matplotlib)
    *   [绘制](#绘制)
    *   [子图](#子图)
    *   [图片](#图片)

## Python

Python是一种高级动态类型的多范式编程语言。Python代码通常被称为可运行的伪代码，因为它允许你在非常少的代码行中表达非常强大的想法，同时具有非常可读性。作为示例，这里是Python中经典快速排序算法的实现：

```
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"
```

### Python 的版本

目前有两种不同的受支持版本的Python，分别是2.7和3.5。有点令人困惑的是，Python 3.0引入了许多向后兼容的语言更改，因此为2.7编写的代码可能无法在3.5下运行，反之亦然。所以我们下面所有的示例的代码都使用Python 3.5来编程。

你可以通过运行 ``python -version`` 在命令行中查看Python的版本。

### 基本数据类型

与大多数语言一样，Python有许多基本类型，包括整数，浮点数，布尔值和字符串。这些数据类型的行为方式与其他编程语言相似。

**Numbers(数字类型)**：代表的是整数和浮点数，它原理与其他语言相同：

```python
x = 3
print(type(x)) # Prints "<class 'int'>"
print(x)       # Prints "3"
print(x + 1)   # Addition; prints "4"
print(x - 1)   # Subtraction; prints "2"
print(x * 2)   # Multiplication; prints "6"
print(x ** 2)  # Exponentiation; prints "9"
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"
```

注意，与许多语言不同，Python没有一元增量(``x+``)或递减(``x-``)运算符。

Python还有用于复数的内置类型；你可以在[这篇文档](https://docs.python.org/3.5/library/stdtypes.html#numeric-types-int-float-complex)中找到所有的详细信息。

**Booleans(布尔类型)**: Python实现了所有常用的布尔逻辑运算符，但它使用的是英文单词而不是符号 (``&&``, ``||``, etc.)：

```python
t = True
f = False
print(type(t)) # Prints "<class 'bool'>"
print(t and f) # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)   # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"
```

**Strings(字符串类型)**：Python对字符串有很好的支持：

```python
hello = 'hello'    # String literals can use single quotes
world = "world"    # or double quotes; it does not matter.
print(hello)       # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
```

String对象有许多有用的方法；例如：

```python
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
```

你可以在[这篇文档](https://docs.python.org/3.5/library/stdtypes.html#string-methods)中找到所有String方法的列表。

### 容器(Containers)

Python包含几种内置的容器类型：列表、字典、集合和元组。

#### 列表(Lists)

列表其实就是Python中的数组，但是可以它可以动态的调整大小并且可以包含不同类型的元素：

```python
xs = [3, 1, 2]    # Create a list
print(xs, xs[2])  # Prints "[3, 1, 2] 2"
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'     # Lists can contain elements of different types
print(xs)         # Prints "[3, 1, 'foo']"
xs.append('bar')  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)      # Prints "bar [3, 1, 'foo']"
```

像往常一样，你可以在[这篇文档](https://docs.python.org/3.5/tutorial/datastructures.html#more-on-lists)中找到有关列表的所有详细信息。

**切片(Slicing)**: 除了一次访问一个列表元素之外，Python还提供了访问子列表的简明语法; 这被称为切片：

```python
nums = list(range(5))     # range is a built-in function that creates a list of integers
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"
```

我们将在numpy数组的上下文中再次看到切片。

**(循环)Loops**: 你可以循环遍历列表的元素，如下所示：

```python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.
```

如果要访问循环体内每个元素的索引，请使用内置的 ``enumerate`` 函数：

```python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
```

**列表推导式(List comprehensions)**: 编程时，我们经常想要将一种数据转换为另一种数据。 举个简单的例子，思考以下计算平方数的代码：

```python
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]
```

你可以使用 **列表推导式** 使这段代码更简单:

```python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]
```

列表推导还可以包含条件：

```
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"
```

#### 字典

字典存储（键，值）对，类似于Java中的``Map``或Javascript中的对象。你可以像这样使用它：

```python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'     # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

你可以在[这篇文档](https://docs.python.org/3.5/library/stdtypes.html#dict)中找到有关字典的所有信息。

**(循环)Loops**: 迭代词典中的键很容易：

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

如果要访问键及其对应的值，请使用``items``方法：

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

**字典推导式(Dictionary comprehensions)**: 类似于列表推导式，可以让你轻松构建词典数据类型。例如：

```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"
```

#### 集合(Sets)

集合是不同元素的无序集合。举个简单的例子，请思考下面的代码：

```python
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')       # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')        # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')     # Remove an element from a set
print(len(animals))       # Prints "2"
```

与往常一样，你想知道的关于集合的所有内容都可以在[这篇文档](https://docs.python.org/3.5/library/stdtypes.html#set)中找到。

**循环(Loops)**: 遍历集合的语法与遍历列表的语法相同；但是，由于集合是无序的，因此不能假设访问集合元素的顺序：

```python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"
```

**集合推导式(Set comprehensions)**: 就像列表和字典一样，我们可以很容易地使用集合理解来构造集合：

```python
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"
```

#### 元组(Tuples)

元组是（不可变的）有序值列表。 元组在很多方面类似于列表; 其中一个最重要的区别是元组可以用作字典中的键和集合的元素，而列表则不能。 这是一个简单的例子：

```python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"
```

[这篇文档](https://docs.python.org/3.5/tutorial/datastructures.html#tuples-and-sequences)包含有关元组的更多信息。

### 函数(Functions)

Python函数使用``def``关键字定义。例如：

```python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# Prints "negative", "zero", "positive"
```

我们经常定义函数来获取可选的关键字参数，如下所示：

```python
def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"
```

[这篇文档](https://docs.python.org/3.5/tutorial/controlflow.html#defining-functions)中有更多关于Python函数的信息。

### 类(Classes)

在Python中定义类的语法很简单：

```python
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```

你可以在[这篇文档](https://docs.python.org/3.5/tutorial/classes.html)中阅读更多关于Python类的内容。

## Numpy

[Numpy](http://www.numpy.org/)是Python中科学计算的核心库。它提供了一个高性能的多维数组对象，以及用于处理这些数组的工具。如果你已经熟悉MATLAB，你可能会发现[这篇教程](/user_guide/numpy_for_matlab_users.html)对于你从MATLAB切换到学习Numpy很有帮助。

### 数组(Arrays)

numpy数组是一个值网格，所有类型都相同，并由非负整数元组索引。 维数是数组的排名; 数组的形状是一个整数元组，给出了每个维度的数组大小。

我们可以从嵌套的Python列表初始化numpy数组，并使用方括号访问元素：

```python
import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
```

Numpy还提供了许多创建数组的函数：

```python
import numpy as np

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
```

你可以在[这篇文档](/user_guide/numpy_basics/array_creation.html)中阅读有关其他数组创建方法的信息。

### 数组索引

Numpy提供了几种索引数组的方法。

**切片(Slicing)**: 与Python列表类似，可以对numpy数组进行切片。由于数组可能是多维的，因此必须为数组的每个维指定一个切片：

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"
```

你还可以将整数索引与切片索引混合使用。 但是，这样做会产生比原始数组更低级别的数组。 请注意，这与MATLAB处理数组切片的方式完全不同：

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```

**整数数组索引**: 使用切片索引到numpy数组时，生成的数组视图将始终是原始数组的子数组。 相反，整数数组索引允许你使用另一个数组中的数据构造任意数组。 这是一个例子：

```python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
```

整数数组索引的一个有用技巧是从矩阵的每一行中选择或改变一个元素：

```python
import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
```

**布尔数组索引**: 布尔数组索引允许你选择数组的任意元素。通常，这种类型的索引用于选择满足某些条件的数组元素。下面是一个例子：

```python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"
```

为简洁起见，我们省略了很多关于numpy数组索引的细节; 如果你想了解更多，你应该阅读[这篇文档](/reference/array_objects/indexing.html)。

### 数据类型

每个numpy数组都是相同类型元素的网格。Numpy提供了一组可用于构造数组的大量数值数据类型。Numpy在创建数组时尝试猜测数据类型，但构造数组的函数通常还包含一个可选参数来显式指定数据类型。这是一个例子：

```python
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"
```

你可以在[这篇文档](/user_guide/numpy_basics/data_types.html)中阅读有关numpy数据类型的所有信息。

### 数组中的数学

基本数学函数在数组上以元素方式运行，既可以作为运算符重载，也可以作为numpy模块中的函数：

```python
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```

请注意，与MATLAB不同，``*``是元素乘法，而不是矩阵乘法。 我们使用``dot``函数来计算向量的内积，将向量乘以矩阵，并乘以矩阵。 ``dot``既可以作为numpy模块中的函数，也可以作为数组对象的实例方法：

```python
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
```

Numpy为在数组上执行计算提供了许多有用的函数；其中最有用的函数之一是 ``SUM``：

```python
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```

你可以在[这篇文档](/reference/routines/math.html)中找到numpy提供的数学函数的完整列表。

除了使用数组计算数学函数外，我们经常需要对数组中的数据进行整形或其他操作。这种操作的最简单的例子是转置一个矩阵；要转置一个矩阵，只需使用一个数组对象的``T``属性：

```python
import numpy as np

x = np.array([[1,2], [3,4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"
```

Numpy提供了许多用于操作数组的函数；你可以在[这篇文档](/reference/routines/array_manipulation_routines.html)中看到完整的列表。

### 广播(Broadcasting)

广播是一种强大的机制，它允许numpy在执行算术运算时使用不同形状的数组。通常，我们有一个较小的数组和一个较大的数组，我们希望多次使用较小的数组来对较大的数组执行一些操作。

例如，假设我们要向矩阵的每一行添加一个常数向量。我们可以这样做：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
```

这会凑效; 但是当矩阵 ``x`` 非常大时，在Python中计算显式循环可能会很慢。注意，向矩阵 ``x`` 的每一行添加向量 ``v`` 等同于通过垂直堆叠多个 ``v`` 副本来形成矩阵 ``vv``，然后执行元素的求和``x`` 和 ``vv``。 我们可以像如下这样实现这种方法：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

Numpy广播允许我们在不实际创建``v``的多个副本的情况下执行此计算。考虑这个需求，使用广播如下：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

``y=x+v``行即使``x``具有形状``(4，3)``和``v``具有形状``(3,)``，但由于广播的关系，该行的工作方式就好像``v``实际上具有形状``(4，3)``，其中每一行都是``v``的副本，并且求和是按元素执行的。

将两个数组一起广播遵循以下规则：

1. 如果数组不具有相同的rank，则将较低等级数组的形状添加1，直到两个形状具有相同的长度。
1. 如果两个数组在维度上具有相同的大小，或者如果其中一个数组在该维度中的大小为1，则称这两个数组在维度上是兼容的。
1. 如果数组在所有维度上兼容，则可以一起广播。
1. 广播之后，每个数组的行为就好像它的形状等于两个输入数组的形状的元素最大值。
1. 在一个数组的大小为1且另一个数组的大小大于1的任何维度中，第一个数组的行为就像沿着该维度复制一样

如果对于以上的解释依然没有理解，请尝试阅读[这篇文档](/user_guide/numpy_basics/broadcasting.html)或[这篇解释](http://wiki.scipy.org/EricsBroadcastingDoc)中的说明。

支持广播的功能称为通用功能。你可以在[这篇文档](/reference/ufuncs/available_ufuncs.html)中找到所有通用功能的列表。

以下是广播的一些应用：

```python
import numpy as np

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)
```

广播通常会使你的代码更简洁，效率更高，因此你应该尽可能地使用它。

### Numpy 的文档

这个简短的概述说明了部分numpy相关的重要事项。查看[numpy参考手册](/reference/index.html)以了解有关numpy的更多信息。

## SciPy

Numpy提供了一个高性能的多维数组和基本工具来计算和操作这些数组。 而[SciPy](https://docs.scipy.org/doc/scipy/reference/)以此为基础，提供了大量在numpy数组上运行的函数，可用于不同类型的科学和工程应用程序。

熟悉SciPy的最佳方法是浏览[它的文档](https://docs.scipy.org/doc/scipy/reference/index.html)。我们将重点介绍SciPy有关的对你有价值的部分内容。

### 图像操作

SciPy提供了一些处理图像的基本函数。例如，它具有将映像从磁盘读入numpy数组、将numpy数组作为映像写入磁盘以及调整映像大小的功能。下面是一个演示这些函数的简单示例：

```python
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
```

![猫咪](/static/images/article/cat.jpg) ![猫咪](/static/images/article/cat_tinted.jpg)

左：原始图像。右：着色和调整大小的图像。

### MATLAB 文件

函数 ``scipy.io.loadmat`` 和 ``scipy.io.savemat`` 允许你读取和写入MATLAB文件。你可以在[这篇文档](https://docs.scipy.org/doc/scipy/reference/io.html)中学习相关操作。

### 点之间的距离

SciPy定义了一些用于计算点集之间距离的有用函数。

函数``scipy.spatial.distance.pdist``计算给定集合中所有点对之间的距离：

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)
```

你可以在[这篇文档](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)中阅读有关此功能的所有详细信息。

类似的函数（``scipy.spatial.distance.cdist``）计算两组点之间所有对之间的距离; 你可以在[这篇文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)中阅读它。

## Matplotlib

[Matplotlib](https://matplotlib.org/)是一个绘图库。本节简要介绍 ``matplotlib.pyplot`` 模块，该模块提供了类似于MATLAB的绘图系统。

### 绘制

matplotlib中最重要的功能是``plot``，它允许你绘制2D数据的图像。这是一个简单的例子：

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.
```

运行此代码会生成以下图表：

![sine](/static/images/article/sine.png)

通过一些额外的工作，我们可以轻松地一次绘制多条线，并添加标题，图例和轴标签：


```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```

![sine_cosine](/static/images/article/sine_cosine.png)

你可以在[这篇文档](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)中阅读有关``绘图``功能的更多信息。

### 子图

你可以使用``subplot``函数在同一个图中绘制不同的东西。 这是一个例子：

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

![sine_cosine_subplot](/static/images/article/sine_cosine_subplot.png)

你可以在[这篇文档](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot)中阅读有关``子图``功能的更多信息。

### 图片

你可以使用 ``imshow`` 函数来显示一张图片。 这是一个例子：

```python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
```

![cat_tinted_imshow](/static/images/article/cat_tinted_imshow.png)

## 文章出处 

由NumPy中文文档翻译，原作者为 [Justin Johnson](https://cs.stanford.edu/people/jcjohns/)，翻译至：[http://cs231n.github.io/python-numpy-tutorial/](http://cs231n.github.io/python-numpy-tutorial/)。