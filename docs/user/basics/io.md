# NumPy与输入输出

## 使用genfromtxt导入数据

NumPy提供了几个函数来根据表格数据创建数组。我们将重点放在``genfromtxt``函数上。

In a nutshell, ``genfromtxt`` runs two main loops. 第一个循环以字符串序列转换文件的每一行。第二个循环将每个字符串转换为适当的数据类型。这种机制比单一循环慢，但提供了更多的灵活性。特别的, ``genfromtxt``考虑到缺失值的情况, 其他更简单的方法如``loadtxt``无法做到这点.

注意
举例时，我们将使用以下约定：

```python
>>> import numpy as np
>>> from io import BytesIO
```

### 定义输入

``genfromtxt``的唯一强制参数是数据的来源。它可以是一个字符串，一串字符串或一个生成器。如果提供了单个字符串，则假定它是本地或远程文件的名称，或者带有``read``方法的开放文件类对象，例如文件或``StringIO.StringIO``对象。如果提供了字符串列表或生成器返回字符串，则每个字符串在文件中被视为一行。当传递远程文件的URL时，该文件将自动下载到当前目录并打开。

识别的文件类型是文本文件和档案。目前，该功能可识别``gzip``和``bz2``（bzip2）档案。归档文件的类型由文件的扩展名决定：如果文件名以``'.gz'``结尾，则需要一个``gzip``归档文件；如果它以``'bz2'``结尾，则假定``bzip2``存档。

### 将行拆分为列

#### ``delimiter``参数
一旦文件被定义并打开进行读取，``genfromtxt``会将每个非空行分割为一串字符串。 空的或注释的行只是略过。 ``delimiter``关键字用于定义拆分应该如何进行。

通常，单个字符标记列之间的分隔。例如，逗号分隔文件（CSV）使用逗号（``,``）或分号（``;``）作为分隔符：

```python
>>> data = "1, 2, 3\n4, 5, 6"
>>> np.genfromtxt(BytesIO(data), delimiter=",")
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]])
```

另一个常用的分隔符是"\t"，即制表符。但是，我们不限于单个字符，任何字符串都可以。默认情况下，``genfromtxt``假定``delimiter=None``，这意味着该行沿着空白区域（包括制表符）分割，并且连续的空白区域被视为单个空白区域。

或者，我们可能正在处理一个固定宽度的文件，其中列被定义为给定数量的字符。在这种情况下，我们需要将``delimiter``设置为单个整数（如果所有列的大小相同）或整数序列（如果列的大小可能不同）：

```python
>>> data = "  1  2  3\n  4  5 67\n890123  4"
>>> np.genfromtxt(BytesIO(data), delimiter=3)
array([[   1.,    2.,    3.],
       [   4.,    5.,   67.],
       [ 890.,  123.,    4.]])
>>> data = "123456789\n   4  7 9\n   4567 9"
>>> np.genfromtxt(BytesIO(data), delimiter=(4, 3, 2))
array([[ 1234.,   567.,    89.],
       [    4.,     7.,     9.],
       [    4.,   567.,     9.]])
```

#### ``autostrip``参数

默认情况下，当一行被分解为一系列字符串时，单个条目不会被剥离前导空白或尾随空白。通过将可选参数autostrip设置为值True，可以覆盖此行为：

```python
>>> data = "1, abc , 2\n 3, xxx, 4"
>>> # Without autostrip
>>> np.genfromtxt(BytesIO(data), delimiter=",", dtype="|S5")
array([['1', ' abc ', ' 2'],
       ['3', ' xxx', ' 4']],
      dtype='|S5')
>>> # With autostrip
>>> np.genfromtxt(BytesIO(data), delimiter=",", dtype="|S5", autostrip=True)
array([['1', 'abc', '2'],
       ['3', 'xxx', '4']],
      dtype='|S5')
```

#### ``comments``参数

可选参数``comments``用于定义标记注释开始的字符串。默认情况下，``genfromtxt``假定``comments='#'``。评论标记可能发生在线上的任何地方。评论标记之后的任何字符都会被忽略：

```python
>>> data = """#
... # Skip me !
... # Skip me too !
... 1, 2
... 3, 4
... 5, 6 #This is the third line of the data
... 7, 8
... # And here comes the last line
... 9, 0
... """
>>> np.genfromtxt(BytesIO(data), comments="#", delimiter=",")
[[ 1.  2.]
 [ 3.  4.]
 [ 5.  6.]
 [ 7.  8.]
 [ 9.  0.]]
```

::: tip 注意

这种行为有一个明显的例外：如果可选参数``names=True``，则会检查第一条注释行的名称。

:::

### 跳过直线并选择列

#### ``skip_header``和``skip_footer``参数

文件中存在标题可能会妨碍数据处理。在这种情况下，我们需要使用``skip_header``可选参数。此参数的值必须是一个整数，与执行任何其他操作之前在文件开头跳过的行数相对应。同样，我们可以使用``skip_footer``属性跳过文件的最后一行``n``，并给它一个``n``的值：

```python
>>> data = "\n".join(str(i) for i in range(10))
>>> np.genfromtxt(BytesIO(data),)
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
>>> np.genfromtxt(BytesIO(data),
...               skip_header=3, skip_footer=5)
array([ 3.,  4.])
```

默认情况下，``skip_header=0``和``skip_footer=0``，这意味着不会跳过任何行。

#### ``usecols``参数

在某些情况下，我们对数据的所有列不感兴趣，但只有其中的一小部分。我们可以用``usecols``参数选择要导入的列。该参数接受与要导入的列的索引相对应的单个整数或整数序列。请记住，按照惯例，第一列的索引为0。负整数的行为与常规Python负向索引相同。

例如，如果我们只想导入第一列和最后一列，我们可以使用``usecols =（0， -1）``：

```python
>>> data = "1 2 3\n4 5 6"
>>> np.genfromtxt(BytesIO(data), usecols=(0, -1))
array([[ 1.,  3.],
       [ 4.,  6.]])
```

如果列有名称，我们也可以通过将它们的名称提供给``usecols``参数来选择要导入哪些列，可以将其作为字符串序列或逗号分隔字符串：

```python
>>> data = "1 2 3\n4 5 6"
>>> np.genfromtxt(BytesIO(data),
...               names="a, b, c", usecols=("a", "c"))
array([(1.0, 3.0), (4.0, 6.0)],
      dtype=[('a', '<f8'), ('c', '<f8')])
>>> np.genfromtxt(BytesIO(data),
...               names="a, b, c", usecols=("a, c"))
    array([(1.0, 3.0), (4.0, 6.0)],
          dtype=[('a', '<f8'), ('c', '<f8')])
```

### 选择数据的类型

控制我们从文件中读取的字符串序列如何转换为其他类型的主要方法是设置``dtype``参数。这个参数的可接受值是：

- 单一类型，如``dtype=float``。除非使用``names``参数将名称与每个列关联（见下文），否则输出将是给定dtype的2D格式。请注意，``dtype=float``是``genfromtxt``的默认值。
- 一系列类型，如``dtype =（int， float， float）``。
- 逗号分隔的字符串，例如``dtype="i4,f8,|S3"``。
- 一个包含两个键``'names'``和``'formats'``的字典。
- a sequence of tuples``(name, type)``, such as ``dtype=[('A', int), ('B', float)]``.
- 现有的``numpy.dtype``对象。
- 特殊值``None``。在这种情况下，列的类型将根据数据本身确定（见下文）。

在所有情况下，除了第一种情况，输出将是一个带有结构化dtype的一维数组。这个dtype与序列中的项目一样多。字段名称由``names``关键字定义。

当``dtype=None``时，每列的类型由其数据迭代确定。我们首先检查一个字符串是否可以转换为布尔值（也就是说，如果字符串在小写字母中匹配``true``或``false``）；然后是否可以将其转换为整数，然后转换为浮点数，然后转换为复数并最终转换为字符串。通过修改``StringConverter``类的默认映射器可以更改此行为。

为方便起见，提供了dtype=None选项。但是，它明显比显式设置dtype要慢。

### 设置名称

#### ``names``参数

处理表格数据时的一种自然方法是为每列分配一个名称。如前所述，第一种可能性是使用明确的结构化dtype。

```python
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> np.genfromtxt(data, dtype=[(_, int) for _ in "abc"])
array([(1, 2, 3), (4, 5, 6)],
      dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<i8')])
```

另一种更简单的可能性是将``names``关键字与一系列字符串或逗号分隔的字符串一起使用：

```python
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> np.genfromtxt(data, names="A, B, C")
array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
      dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])
```

在上面的例子中，我们使用了默认情况下``dtype=float``的事实。通过给出一个名称序列，我们强制输出到一个结构化的dtype。

我们有时可能需要从数据本身定义列名。在这种情况下，我们必须使用``names``关键字的值为``True``。这些名字将从第一行（在``skip_header``之后）被读取，即使该行被注释掉：

```python
>>> data = BytesIO("So it goes\n#a b c\n1 2 3\n 4 5 6")
>>> np.genfromtxt(data, skip_header=1, names=True)
array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
      dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
```

``names``的默认值为``None``。如果我们给关键字赋予任何其他值，新名称将覆盖我们可能用dtype定义的字段名称：

```python
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> ndtype=[('a',int), ('b', float), ('c', int)]
>>> names = ["A", "B", "C"]
>>> np.genfromtxt(data, names=names, dtype=ndtype)
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('A', '<i8'), ('B', '<f8'), ('C', '<i8')])
```

#### ``defaultfmt``参数

如果 ``names=None`` 的时候，只是预计会有一个结构化的dtype，它的名称将使用标准的NumPy默认值 ``"f%i"``来定义，会产生例如``f0``，``f1``等名称：

```python
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> np.genfromtxt(data, dtype=(int, float, int))
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', '<i8')])
```

同样，如果我们没有提供足够的名称来匹配dtype的长度，缺少的名称将使用此默认模板进行定义：

```python
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> np.genfromtxt(data, dtype=(int, float, int), names="a")
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('a', '<i8'), ('f0', '<f8'), ('f1', '<i8')])
```

我们可以使用``defaultfmt``参数覆盖此默认值，该参数采用任何格式字符串：

```python
>>> data = BytesIO("1 2 3\n 4 5 6")
>>> np.genfromtxt(data, dtype=(int, float, int), defaultfmt="var_%02i")
array([(1, 2.0, 3), (4, 5.0, 6)],
      dtype=[('var_00', '<i8'), ('var_01', '<f8'), ('var_02', '<i8')])
```

> 注意！  
> 我们需要记住，仅当预期一些名称但未定义时才使用``defaultfmt``。

#### 验证名称

具有结构化dtype的NumPy数组也可以被视为``recarray``，其中可以像访问属性一样访问字段。因此，我们可能需要确保字段名称不包含任何空格或无效字符，或者它不对应于标准属性的名称（如``size``或``shape``），这会混淆解释者。``genfromtxt``接受三个可选参数，这些参数可以更好地控制名称：

- **``deletechars``** - 给出一个字符串，将所有必须从名称中删除的字符组合在一起。默认情况下，无效字符是``~!@#$%^&*()-=+~\|]}[{';: /?.>,<``
- **``excludelist``** - 给出要排除的名称列表，如``return``，``file``，``print`` ...如果其中一个输入名称是该列表的一部分，则会附加一个下划线字符（``'_'``）。
- **``case_sensitive``** - 是否区分大小写（``case_sensitive=True``），转换为大写（``case_sensitive=False``或``case_sensitive='upper'``）或小写（``case_sensitive='lower'``）。

### 调整转换

#### ``converters``参数

通常，定义一个dtype足以定义字符串序列必须如何转换。但是，有时可能需要一些额外的控制。例如，我们可能希望确保格式为``YYYY/MM/DD``的日期转换为``datetime``对象，或者像``xx%``正确转换为0到1之间的浮点数。在这种情况下，我们应该使用``converters``参数定义转换函数。

该参数的值通常是以列索引或列名称作为关键字的字典，并且转换函数作为值。这些转换函数可以是实际函数或lambda函数。无论如何，它们只应接受一个字符串作为输入，并只输出所需类型的单个元素。

在以下示例中，第二列从代表百分比的字符串转换为0和1之间的浮点数：

```python
>>> convertfunc = lambda x: float(x.strip("%"))/100.
>>> data = "1, 2.3%, 45.\n6, 78.9%, 0"
>>> names = ("i", "p", "n")
>>> # General case .....
>>> np.genfromtxt(BytesIO(data), delimiter=",", names=names)
array([(1.0, nan, 45.0), (6.0, nan, 0.0)],
      dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])
```

我们需要记住，默认情况下，``dtype=float``。因此，对于第二列期望浮点数。但是，字符串``'2.3%'``和``'78.9%``无法转换为浮点数，我们最终改为使用``np.nan``。现在让我们使用一个转换器：

```python
>>> # Converted case ...
>>> np.genfromtxt(BytesIO(data), delimiter=",", names=names,
...               converters={1: convertfunc})
array([(1.0, 0.023, 45.0), (6.0, 0.78900000000000003, 0.0)],
      dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])
```

通过使用第二列（``"p"``）作为关键字而不是其索引（1）的名称，可以获得相同的结果：

```python
>>> # Using a name for the converter ...
>>> np.genfromtxt(BytesIO(data), delimiter=",", names=names,
...               converters={"p": convertfunc})
array([(1.0, 0.023, 45.0), (6.0, 0.78900000000000003, 0.0)],
      dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])
```

转换器也可以用来为缺少的条目提供默认值。在以下示例中，如果字符串为空，则转换器``convert``会将已剥离的字符串转换为相应的浮点型或转换为-999。我们需要明确地从空白处去除字符串，因为它并未默认完成：

```python
>>> data = "1, , 3\n 4, 5, 6"
>>> convert = lambda x: float(x.strip() or -999)
>>> np.genfromtxt(BytesIO(data), delimiter=",",
...               converters={1: convert})
array([[   1., -999.,    3.],
       [   4.,    5.,    6.]])
```

#### 使用缺失值和填充值

我们尝试导入的数据集中可能缺少一些条目。在前面的例子中，我们使用转换器将空字符串转换为浮点。但是，用户定义的转换器可能会很快变得繁琐，难以管理。

``genfromtxt``函数提供了另外两种补充机制：``missing_values``参数用于识别丢失的数据，第二个参数``filling_values``用于处理这些缺失的数据。

#### ``missing_values``

默认情况下，任何空字符串都被标记为缺失。我们也可以考虑更复杂的字符串，比如``"N/A"``或``"???"``代表丢失或无效的数据。``missing_values``参数接受三种值：

- **单个字符串或逗号分隔的字符串** - 该字符串将用作所有列缺失数据的标记
- **字符串** - 在这种情况下，每个项目都按顺序与列关联。
- **字典类型** - 字典的值是字符串或字符串序列。相应的键可以是列索引（整数）或列名称（字符串）。另外，可以使用特殊键None来定义适用于所有列的默认值。

#### ``filling_values``

我们知道如何识别丢失的数据，但我们仍然需要为这些丢失的条目提供一个值。默认情况下，根据此表根据预期的dtype确定此值：

我们知道如何识别丢失的数据，但我们仍然需要为这些丢失的条目提供一个值。默认情况下，根据此表根据预期的dtype确定此值：

**预期类型** | **默认**
---|---
``bool`` | ``False``
``int`` | ``-1``
``float`` | ``np.nan``
``complex`` | ``np.nan+0j``
``string`` | ``'???'``

通过``filling_values``可选参数，我们可以更好地控制缺失值的转换。像``missing_values``一样，此参数接受不同类型的值：

- **单个值** - 这将是所有列的默认值
- **类数组类型** - 每个条目都是相应列的默认值
- **字典类型** - 每个键可以是列索引或列名称，并且相应的值应该是单个对象。我们可以使用特殊键None为所有列定义默认值。

在下面的例子中，我们假设缺少的值在第一列中用``"N/A"``标记，并由``"???"``在第三栏。如果它们出现在第一列和第二列中，我们希望将这些缺失值转换为0，如果它们出现在最后一列中，则将它们转换为-999：

```python
>>> data = "N/A, 2, 3\n4, ,???"
>>> kwargs = dict(delimiter=",",
...               dtype=int,
...               names="a,b,c",
...               missing_values={0:"N/A", 'b':" ", 2:"???"},
...               filling_values={0:0, 'b':0, 2:-999})
>>> np.genfromtxt(BytesIO(data), **kwargs)
array([(0, 2, 3), (4, 0, -999)],
      dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<i8')])
```

#### ``usemask``

我们也可能想通过构造一个布尔掩码来跟踪丢失数据的发生，其中``True``条目缺少数据，否则``False``。为此，我们只需将可选参数``usemask``设置为``True``（默认值为``False``）。输出数组将成为``MaskedArray``。

### 快捷方式函数

除了 [genfromtxt](https://numpy.org/devdocs/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt) 之外，numpy.lib.io模块还提供了几个从 
[genfromtxt](https://numpy.org/devdocs/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt) 派生的方便函数。这些函数的工作方式与原始函数相同，但它们具有不同的默认值。

- **recfromtxt** - 返回标准 [numpy.recarray](https://numpy.org/devdocs/reference/generated/numpy.recarray.html#numpy.recarray)（如果 ``usemask=False``）或 MaskedRecords数组（如果 ``usemaske=True``）。默认dtype是 ``dtype=None`` ，意味着将自动确定每列的类型。
- **recfromcsv** - 类似 recfromtxt，但有默认值 ``delimiter=","`` 。
