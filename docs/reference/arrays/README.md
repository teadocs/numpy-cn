# 数组对象

NumPy提供了一个N维数组类型，即[ndarray](arrays.ndarray.html#arrays-ndarray)，
它描述了相同类型的“项目”集合。可以使用例如N个整数来[索引](arrays.indexing.html#arrays-indexing)项目。

所有ndarray都是[同质的](https://numpy.org/devdocs/glossary.html#term-homogenous)：每个项目占用相同大小的内存块，
并且所有块都以完全相同的方式解释。
如何解释数组中的每个项目由单独的[数据类型对象](arrays.dtypes.html#arrays-dtypes)指定，
其中一个[对象](arrays.dtypes.html#arrays-dtypes)与每个数组相关联。除了基本类型（整数，浮点数 *等* ）之外，
数据类型对象还可以表示数据结构。

从数组中提取的项（ *例如* ，通过索引）由Python对象表示，其类型是在NumPy中构建的[数组标量类型](arrays.scalars.html#arrays-scalars)之一。阵列标量允许容易地操纵更复杂的数据排列。

![threefundamental](/static/images/threefundamental.png)

**图**的概念表现了用于描述数组中数据的三个基本对象之间的关系：1）ndarray本身，2）描述数组中单个固定大小元素布局的数据类型对象，3）访问数组的单个元素时返回的数组标量Python对象。

- [N维数组(ndarray)](ndarray.html)
  - [Constructing arrays](ndarray.html#constructing-arrays)
  - [Indexing arrays](ndarray.html#indexing-arrays)
  - [Internal memory layout of an ndarray](ndarray.html#internal-memory-layout-of-an-ndarray)
  - [Array attributes](ndarray.html#array-attributes)
  - [Array methods](ndarray.html#array-methods)
  - [Arithmetic, matrix multiplication, and comparison operations](ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations)
  - [Special methods](ndarray.html#special-methods)
- [标量](scalars.html)
  - [Built-in scalar types](scalars.html#built-in-scalar-types)
  - [Attributes](scalars.html#attributes)
  - [Indexing](scalars.html#indexing)
  - [Methods](scalars.html#methods)
  - [Defining new types](scalars.html#defining-new-types)
- [数据类型对象(dtype)](dtypes.html)
  - [Specifying and constructing data types](dtypes.html#specifying-and-constructing-data-types)
  - [dtype](dtypes.html#dtype)
- [索引](indexing.html)
  - [Basic Slicing and Indexing](indexing.html#basic-slicing-and-indexing)
  - [Advanced Indexing](indexing.html#advanced-indexing)
  - [Detailed notes](indexing.html#detailed-notes)
  - [Field Access](indexing.html#field-access)
  - [Flat Iterator indexing](indexing.html#flat-iterator-indexing)
- [迭代数组](nditer.html)
  - [Single Array Iteration](nditer.html#single-array-iteration)
  - [Broadcasting Array Iteration](nditer.html#broadcasting-array-iteration)
  - [Putting the Inner Loop in Cython](nditer.html#putting-the-inner-loop-in-cython)
- [标准数组子类](classes.html)
  - [Special attributes and methods](classes.html#special-attributes-and-methods)
  - [Matrix objects](classes.html#matrix-objects)
  - [Memory-mapped file arrays](classes.html#memory-mapped-file-arrays)
  - [Character arrays (numpy.char)](classes.html#character-arrays-numpy-char)
  - [Record arrays (numpy.rec)](classes.html#record-arrays-numpy-rec)
  - [Masked arrays (numpy.ma)](classes.html#masked-arrays-numpy-ma)
  - [Standard container class](classes.html#standard-container-class)
  - [Array Iterators](classes.html#array-iterators)
- [掩码数组](maskedarray.html)
  - [The numpy.ma module](maskedarray.generic.html)
  - [Using numpy.ma](maskedarray.generic.html#using-numpy-ma)
  - [Examples](maskedarray.generic.html#examples)
  - [Constants of the numpy.ma module](maskedarray.baseclass.html)
  - [The MaskedArray class](maskedarray.baseclass.html#the-maskedarray-class)
  - [MaskedArray methods](maskedarray.baseclass.html#maskedarray-methods)
  - [Masked array operations](routines.ma.html)
- [数组接口](interface.html)
  - [Python side](interface.html#python-side)
  - [C-struct access](interface.html#c-struct-access)
  - [Type description examples](interface.html#type-description-examples)
  - [Differences with Array interface (Version 2)](interface.html#differences-with-array-interface-version-2)
- [日期时间和时间增量](datetime.html)
  - [Basic Datetimes](datetime.html#basic-datetimes)
  - [Datetime and Timedelta Arithmetic](datetime.html#datetime-and-timedelta-arithmetic)
  - [Datetime Units](datetime.html#datetime-units)
  - [Business Day Functionality](datetime.html#business-day-functionality)
  - [Changes with NumPy 1.11](datetime.html#changes-with-numpy-1-11)
  - [Differences Between 1.6 and 1.7 Datetimes](datetime.html#differences-between-1-6-and-1-7-datetimes)
