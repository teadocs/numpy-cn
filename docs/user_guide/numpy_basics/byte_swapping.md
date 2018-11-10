<title>numpy字节排序 - <%-__DOC_NAME__ %></title>
<meta name="keywords" content="numpy字节排序,numpy字节交换" />

# 字节交换

## 字节排序和ndarrays简介

``ndarray`` 是为内存中的数据提供python数组接口的对象。

经常发生的情况是，你想要使用数组查看的内存与你运行Python的计算机的字节顺序不同。

例如，我可能正在使用小端CPU的计算机（例如Intel Pentium），但是我已经从大端的计算机写入的文件中加载了一些数据。假设我已经从Sun（big-endian）计算机写入的文件中加载了4个字节。我知道这4个字节代表两个16位整数。在big-endian机器上，最高有效字节（MSB）首先存储一个双字节整数，然后存储最低有效字节（LSB）。因此，这些字节按内存顺序排列：

1. MSB整数1
1. LSB整数1
1. MSB整数2
1. LSB整数2

假设这两个整数实际上是1和770。由于770 = 256 * 3 + 2，内存中的4个字节将分别包含：0,1,3,2。我从文件中加载的字节将包含以下内容：

```python
>>> big_end_str = chr(0) + chr(1) + chr(3) + chr(2)
>>> big_end_str
'\x00\x01\x03\x02'
```

我们可能想要使用``ndarray``来访问这些整数。在这种情况下，我们可以围绕这个内存创建一个数组，并告诉numpy有两个整数，它们是16位和大端：

```python
>>> import numpy as np
>>> big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=big_end_str)
>>> big_end_arr[0]
1
>>> big_end_arr[1]
770
```

请注意``dtype``上的数据类型``>i2``，``>``表示'big-endian'（``<``是小端），``i2``表示'带符号的2字节整数'。例如，如果我们的数据表示一个无符号的4字节little-endian整数，则dtype字符串应该是``<u4``。

事实上，我们为什么不尝试呢？

```python
>>> little_end_u4 = np.ndarray(shape=(1,),dtype='<u4', buffer=big_end_str)
>>> little_end_u4[0] == 1 * 256**1 + 3 * 256**2 + 2 * 256**3
True
```

回到我们的``big_end_arr`` - 在这种情况下，我们的基础数据是big-endian（数据字节顺序），我们设置了dtype匹配（dtype也是big-endian）。但是，有时你需要翻转这些。

<div class="warning-warp">
<b>警告！</b>

<p>标量当前不包含字节顺序信息，因此从数组中提取标量将以本机字节顺序返回一个整数。因此：</p>

<pre class="prettyprint language-python">
<code class="hljs">>>> big_end_arr[0].dtype.byteorder == little_end_u4[0].dtype.byteorder
True</code>
</pre>
</div>



## 更改字节顺序

正如你从介绍可以想象的，有两种方式可以影响数组的字节顺序和它正在查看的底层内存之间的关系：

- 更改数组dtype中的字节顺序信息，以便它将未确定的数据解释为处于不同的字节顺序。这是``arr.newbyteorder()``的作用
- 改变底层数据的字节顺序，保持原来的dtype解释。这是``arr.byteswap()``所做的。

你需要更改字节顺序的常见情况是：

1. 你的数据和dtype字尾不匹配，并且你想要更改dtype以使其与数据匹配。
1. 你的数据和dtype字尾不匹配，你想要交换数据以便它们与dtype匹配
1. 你的数据和dtype的字节匹配，但你想要交换数据和dtype来反映这一点

### 数据和dtype字节顺序不匹配，将dtype更改为匹配数据

我们做一些他们不匹配的东西：

```python
>>> wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_str)
>>> wrong_end_dtype_arr[0]
256
```

这种情况的明显解决方法是更改​​dtype，以便提供正确的排列顺序：

```python
>>> fixed_end_dtype_arr = wrong_end_dtype_arr.newbyteorder()
>>> fixed_end_dtype_arr[0]
1
```

请注意数组在内存中没有改变：

```python
>>> fixed_end_dtype_arr.tobytes() == big_end_str
True
```

### 数据和类型字节顺序不匹配，更改数据以匹配dtype

如果你需要将内存中的数据设置为特定顺序，则可能需要执行此操作。例如，你可能将内存写入需要某个字节排序的文件。

```python
>>> fixed_end_mem_arr = wrong_end_dtype_arr.byteswap()
>>> fixed_end_mem_arr[0]
1
```

现在数组在内存中有更改：

```python
>>> fixed_end_mem_arr.tobytes() == big_end_str
False
```

### 数据和dtype字节顺序匹配，交换数据和dtype

你可能为一个数组指定了正确的dtype，但你需要数组在内存中有相反的字节顺序，你想让dtype匹配，所以数组值是有意义的。在这种情况下，你只需执行以前的两个操作：

```python
>>> swapped_end_arr = big_end_arr.byteswap().newbyteorder()
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_str
False
```

使用ndarray astype方法可以实现将数据转换为特定dtype和字节顺序的更简单的方法：

```python
>>> swapped_end_arr = big_end_arr.astype('<i2')
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_str
False
```