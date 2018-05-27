==================================
数据类型
==================================
另见：
	Data type objects

----------------------------------
数组类型和类型之间的转换
----------------------------------
Numpy支持比Python更多的数字类型。本部分显示哪些是可用的，以及如何修改数组的数据类型。

+------------+------------+-----------+
| Header 1   | Header 2   | Header 3  |
+============+============+===========+
| body row 1 | column 2   | column 3  |
+------------+------------+-----------+
| body row 2 | Cells may span columns.|
+------------+------------+-----------+
| body row 3 | Cells may  | - Cells   |
+------------+ span rows. | - contain |
| body row 4 |            | - blocks. |
+------------+------------+-----------+

----------------------------------
数组标量
----------------------------------

Numpy通常返回数组的元素作为数组标量（与相关dtype的标量）。数组标量与Python标量不同，但大多数情况下它们可以互换使用（主要例外是Python版本比v2.x更早的版本，其中整数数组标量不能充当列表和元组的索引）。有一些例外情况，比如代码需要非常特定的标量属性，或者当它特别检查某个值是否为Python标量时。通常，使用相应的Python类型函数（例如int、float、complex、str，unicode）将数组标量显式转换为Python标量就很容易解决问题。

使用数组标量的主要优点是它们保留数组类型（Python可能没有可用的匹配标量类型，例如int16）。因此，使用数组标量可以确保数组和标量之间的相同行为，而不管该值是否在数组中。NumPy标量也有很多和数组相同的方法。

----------------------------------
扩展精度
----------------------------------

Python的浮点数通常是64位浮点数，几乎相当于np.float64。在某些不常见的情况下，使用Python的浮点数更精确。这在numpy是否可能取决于硬件和开发环境：具体来说，x86机器提供80位精度的硬件浮点数，大多数C编译器提供它为long double类型，MSVC（Windows版本的标准）让long double和double（64位）完全一样。Numpy使编译器的long double为np.longdouble（复数为np.clongdouble）。你可以用np.finfo(np.longdouble)找出你的numpy提供的是什么。

Numpy 不提供比 C long double 更高精度的数据类型，特别地 128 位的IEEE quad precision 数据类型（FORTRAN的 REAL*16） 不可用。

For efficient memory alignment, np.longdouble is usually stored padded with zero bits, either to 96 or 128 bits. 哪个更有效率取决于硬件和开发环境；通常在32位系统上它们被填充到96位，而在64位系统上它们通常被填充到128位。np.longdouble被填充到系统默认值；为需要特定填充的用户提供np.float96和np.float128。In spite of the names, np.float96 and np.float128 provide only as much precision as np.longdouble, that is, 80 bits on most x86 machines and 64 bits in standard Windows builds.

请注意，即使np.longdouble提供比python float更多的精度，也很容易失去额外的精度，因为python通常强制值通过float传递值。例如，%格式操作符要求将其参数转换为标准python类型，因此即使请求了许多小数位，也不可能保留扩展精度。使用值1 + np.finfo(np.longdouble).eps测试你的代码非常有用。