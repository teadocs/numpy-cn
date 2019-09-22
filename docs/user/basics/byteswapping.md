# 字节交换

## 字节排序和ndarrays简介

``ndarray``是一个为内存中的数据提供python数组接口的对象。

经常发生的情况是，要用数组查看的内存与运行Python的计算机的字节顺序不同。

例如，我可能正在使用带有 little-endian CPU 的计算机 - 例如Intel Pentium，但是我已经从一个由 big-endian计算机 编写的文件中加载了一些数据。假设我已经从Sun（big-endian）计算机写入的文件中加载了4个字节。我知道这4个字节代表两个16位整数。在 big-endian 机器上，首先以最高有效字节（MSB）存储双字节整数，然后存储最低有效字节（LSB）。因此字节按内存顺序排列：

1. MSB整数1
1. LSB整数1
1. MSB整数2
1. LSB整数2

假设两个整数实际上是1和770.因为770 = 256 * 3 + 2，内存中的4个字节将分别包含：0,1,3,2。我从文件加载的字节将包含这些内容：

``` python
>>> big_end_buffer = bytearray([0,1,3,2])
>>> big_end_buffer
bytearray(b'\x00\x01\x03\x02')
```

我们可能需要使用 ``ndarray`` 来访问这些整数。在这种情况下，我们可以围绕这个内存创建一个数组，并告诉numpy有两个整数，并且它们是16位和Big-endian：

``` python
>>> import numpy as np
>>> big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=big_end_buffer)
>>> big_end_arr[0]
1
>>> big_end_arr[1]
770
```

注意上面的数组``dtype > i2``。``>`` 表示 ``big-endian``( ``<`` 是 ``Little-endian`` )，``i2`` 表示‘有符号的2字节整数’。例如，如果我们的数据表示单个无符号4字节小端整数，则dtype字符串将为 ``<u4``。

事实上，为什么我们不尝试呢？

``` python
>>> little_end_u4 = np.ndarray(shape=(1,),dtype='<u4', buffer=big_end_buffer)
>>> little_end_u4[0] == 1 * 256**1 + 3 * 256**2 + 2 * 256**3
True
```

回到我们的 ``big_end_arr`` - 在这种情况下我们的基础数据是big-endian（数据字节序），我们设置dtype匹配（dtype也是big-endian）。但是，有时你需要翻转它们。

::: danger 警告

标量当前不包含字节顺序信息，因此从数组中提取标量将返回本机字节顺序的整数。因此：

``` python
>>> big_end_arr[0].dtype.byteorder == little_end_u4[0].dtype.byteorder
True
```

:::

## 更改字节顺序

从介绍中可以想象，有两种方法可以影响数组的字节顺序与它所查看的底层内存之间的关系：

- 更改数组dtype中的字节顺序信息，以便将基础数据解释为不同的字节顺序。这是作用 ``arr.newbyteorder()``
- 更改基础数据的字节顺序，保留dtype解释。这是做什么的 ``arr.byteswap()``。

您需要更改字节顺序的常见情况是：

1. 您的数据和dtype字节顺序不匹配，并且您希望更改dtype以使其与数据匹配。
1. 您的数据和dtype字节顺序不匹配，并且您希望交换数据以使它们与dtype匹配
1. 您的数据和dtype字节顺序匹配，但您希望交换数据和dtype来反映这一点

### 数据和dtype字节顺序不匹配，更改dtype以匹配数据

我们制作一些他们不匹配的东西：

``` python
>>> wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_buffer)
>>> wrong_end_dtype_arr[0]
256
```

这种情况的明显解决方法是更改​​dtype，以便它给出正确的字节顺序：

``` python
>>> fixed_end_dtype_arr = wrong_end_dtype_arr.newbyteorder()
>>> fixed_end_dtype_arr[0]
1
```

请注意，内存中的数组未更改：

``` python
>>> fixed_end_dtype_arr.tobytes() == big_end_buffer
True
```

### 数据和类型字节顺序不匹配，更改数据以匹配dtype

如果您需要内存中的数据是某种顺序，您可能希望这样做。例如，您可能正在将内存写入需要特定字节排序的文件。

``` python
>>> fixed_end_mem_arr = wrong_end_dtype_arr.byteswap()
>>> fixed_end_mem_arr[0]
1
```

现在数组 *已* 在内存中更改：

``` python
>>> fixed_end_mem_arr.tobytes() == big_end_buffer
False
```

### 数据和dtype字节序匹配，交换数据和dtype

您可能有一个正确指定的数组dtype，但是您需要数组在内存中具有相反的字节顺序，并且您希望dtype匹配以便数组值有意义。在这种情况下，您只需执行上述两个操作：

``` python
>>> swapped_end_arr = big_end_arr.byteswap().newbyteorder()
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_buffer
False
```

使用ndarray astype方法可以更简单地将数据转换为特定的dtype和字节顺序：

``` python
>>> swapped_end_arr = big_end_arr.astype('<i2')
>>> swapped_end_arr[0]
1
>>> swapped_end_arr.tobytes() == big_end_buffer
False
```
