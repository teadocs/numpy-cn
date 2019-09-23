# 数组对象

NumPy提供了一个N维数组类型，即[ndarray](ndarray.html)，
它描述了相同类型的“项目”集合。可以使用例如N个整数来[索引](indexing.html)项目。

所有ndarray都是[同质的](https://numpy.org/devdocs/glossary.html#term-homogenous)：每个项目占用相同大小的内存块，
并且所有块都以完全相同的方式解释。
如何解释数组中的每个项目由单独的[数据类型对象](dtypes.html)指定，
其中一个[对象](dtypes.html)与每个数组相关联。除了基本类型（整数，浮点数 *等* ）之外，
数据类型对象还可以表示数据结构。

从数组中提取的项（ *例如* ，通过索引）由Python对象表示，
其类型是在NumPy中构建的[数组标量类型](scalars.html)之一。
数组标量允许容易地操纵更复杂的数据排列。

![threefundamental](/static/images/threefundamental.png)

**图**的概念表现了用于描述数组中数据的三个基本对象之间的关系：1）ndarray本身，2）描述数组中单个固定大小元素布局的数据类型对象，3）访问数组的单个元素时返回的数组标量Python对象。

- [N维数组(ndarray)](ndarray.html)
  - [构造数组](ndarray.html#构造数组)
  - [索引数组](ndarray.html#索引数组)
  - [ndarray的内存布局](ndarray.html#ndarray的内存布局)
  - [数组属性](ndarray.html#数组属性s)
  - [数组方法](ndarray.html#数组方法)
  - [算术、矩阵乘法和比较运算](ndarray.html#算术、矩阵乘法和比较运算)
  - [特殊方法](ndarray.html#特殊方法)
- [标量](scalars.html)
  - [内置标量类型](scalars.html##内置标量类型)
  - [属性](scalars.html#属性)
  - [索引](scalars.html#索引)
  - [方法](scalars.html#方法)
  - [定义新类型](scalars.html#定义新类型)
- [数据类型对象(dtype)](dtypes.html)
  - [指定和构造数据类型](dtypes.html#指定和构造数据类型)
  - [dtype](dtypes.html#dtype)
- [索引](indexing.html)
  - [基本切片和索引](indexing.html#基本切片和索引)
  - [高级索引](indexing.html#高级索引)
  - [详细说明](indexing.html#详细说明)
  - [字段形式访问](indexing.html#字段形式访问)
  - [Flat Iterator索引](indexing.html#flat-iterator索引)
- [迭代数组](nditer.html)
  - [单数组迭代](nditer.html#单数组迭代)
  - [广播数组迭代](nditer.html#广播数组迭代)
  - [将内循环放在Cython中](nditer.html#将内循环放在Cython中)
- [标准数组子类](classes.html)
  - [特殊属性和方法](classes.html#特殊属性和方法)
  - [矩阵对象](classes.html#矩阵对象)
  - [内存映射文件数组](classes.html#内存映射文件数组)
  - [字符数组（numpy.char）](classes.html#字符数组（numpy-char）)
  - [记录数组（numpy.rec）](classes.html#记录数组（numpy.rec）)
  - [掩码数组（numpy.ma）](classes.html#掩码数组（numpy.ma）)
  - [标准容器类](classes.html#标准容器类)
  - [数组迭代器](classes.html#数组迭代器)
- [掩码数组](maskedarray.html)
  - [numpy.ma 模块](maskedarray.html)
  - [使用 numpy.ma 模块](maskedarray.html##使用-numpy-ma-模块)
  - [示例](maskedarray.generic.html#示例)
  - [numpy.ma模块的常量](https://numpy.org/devdocs/reference/maskedarray.baseclass.html)
  - [MaskedArray类](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#the-maskedarray-class)
  - [MaskedArray方法](https://numpy.org/devdocs/reference/maskedarray.baseclass.html#maskedarray-methods)
  - [Masked数组操作API](/reference/routines/ma.html)
- [数组接口](interface.html)
  - [Python 方法](interface.html#python-side)
  - [C-struct 访问](interface.html#c-struct-访问)
  - [类型描述示例](interface.html#类型描述示例)
  - [与数组接口（版本2）的差异](interface.html#与数组接口（版本2）的差异)
- [日期时间和时间增量](datetime.html)
  - [基本日期时间](datetime.html#基本日期时间)
  - [Datetime 和 Timedelta 算法](datetime.html#datetime-和-timedelta-算法)
  - [日期时间单位](datetime.html#日期时间单位)
  - [Datetime 功能](datetime.html#datetime-功能)
  - [NumPy 1.11 的更改](datetime.html#numpy-1-11-的更改)