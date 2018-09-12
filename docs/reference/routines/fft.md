# 离散傅里叶变换

## 标准的 FFTs

- fft(a[, n, axis, norm])	计算一维离散傅立叶变换。
- ifft(a[, n, axis, norm])	计算一维离散傅立叶逆变换。
- fft2(a[, s, axes, norm])	计算二维离散傅立叶变换
- ifft2(a[, s, axes, norm])	计算二维逆离散傅立叶变换。
- fftn(a[, s, axes, norm])	计算二维逆离散傅立叶变换。
- ifftn(a[, s, axes, norm])	计算N维逆离散傅立叶变换。

## 真实的 FFTs

- rfft(a[, n, axis, norm])	计算一维离散傅立叶变换用于实际输入。
- irfft(a[, n, axis, norm])	计算实际输入的n点DFT的倒数。
- rfft2(a[, s, axes, norm])	计算实阵列的二维FFT.
- irfft2(a[, s, axes, norm])	计算实数组的二维逆FFT。
- rfftn(a[, s, axes, norm])	计算实输入的N维离散傅立叶变换.
- irfftn(a[, s, axes, norm])	求实输入的N维FFT的逆运算.

## 埃尔米特快速傅里叶变换

- hfft(a[, n, axis, norm])	计算具有厄米对称性的信号的FFT，即实际频谱。
- ihfft(a[, n, axis, norm])	计算具有Hermitian对称性的信号的反FFT。

## 辅助相关api

- fftfreq(n[, d])	返回离散傅里叶变换采样频率。
- rfftfreq(n[, d])	返回离散傅立叶变换采样频率(用于rfft、irfft)。
- fftshift(x[, axes])	将零频率分量移到频谱的中心。
- ifftshift(x[, axes])	反移的反义词。

## 背景资料

傅立叶分析基本上是一种将函数表示为周期性分量之和以及从这些分量中恢复函数的方法。当函数及其傅里叶变换都被离散化的对应物替换时，它被称为离散傅里叶变换（DFT）。DFT已经成为数值计算的支柱，部分原因在于它的计算速度非常快，称为快速傅里叶变换（FFT），高斯（1805）已知并且由Cooley以其当前形式揭示。 Tukey [CT309]。按等人。[NR309]提供了傅里叶分析及其应用的可访问介绍。

由于离散傅里叶变换将其输入分离为在离散频率下贡献的分量，因此它在数字信号处理中具有大量应用，例如用于滤波，并且在这种情况下，变换的离散化输入通常被称为信号。 ，存在于时域中。输出称为频谱或变换，存在于频域中。

## 实施细节

定义DFT的方法有很多种，如指数变化、归一化等。在这个实现中，DFT被定义为

![公式](/static/images/c3e12e4fbd5334e071b7dfdd4d059fc3584b81e8.svg)

DFT一般定义为复输入和复输出，线性频率f上的单频分量用复指数 ![公式](/static/images/9127ee37034ef9c70d96a488f67e0c82f9e92ff8.svg)  表示，其中Δt是采样间隔。

结果中的值遵循所谓的“标准”顺序：如果``A=fft(a, n)``，则``A[0]``包含零频率项(信号的和)，对于实际输入，该项总是纯实的。然后``A[1:n/2]``包含正频率项，``A[n/2+1:]``包含负频率项，按负频率递减的顺序排列。对于偶数个输入点，``A[n/2]``表示正负奈奎斯特频率，对于实际输入也是纯实的。对于奇数个输入点，``A[(n-1)/2]``的正频率最大，``A[(n+1)/2]``的负频率最大。方法 ``np.fft.fftfreq(N)`` 返回一个数组，给出输出中相应元素的频率。常规的``np.fft.fftShift(A)``变换和它们的频率将零频率分量放在中间，``np.fft.ifftShift(A)``取消这一移位。

当输入a为时域信号且``A=FFT(A)``时，``np.abs(A)``为其幅度谱，``np.abs(A)*2``为其功率谱。相位谱由 ``np.angle(A)``得到。

逆DFT定义为

![公式](/static/images/25d7a89b77473363cb4da8b11ca853073f63729f.svg)

与正向变换不同的是，它的符号是指数型参数，默认的归一化是1/n。

## 正常化

默认规范化具有未缩放的直接变换，并且逆变换按 1/n 缩放。 通过将关键字参数``norm``设置为``"ortho"``（默认为None）可以获得单一变换，这样直接变换和逆变换都将被![公式](/static/images/f780dc84ea49c387f9417b50f0619e404d91c28a.svg)缩放。

## 实变换和厄米特变换

当输入是纯实的时，它的变换是厄米变换，即fk频率上的分量是频率fk上分量的复共轭，这意味着对于实际输入，负频率分量中没有正频率分量不能提供的信息。rfft函数族被设计为对实际输入进行运算，并通过只计算正频率分量(直到并包括Nyquist频率)来利用这种对称性。因此，n个输入点产生``n/2+1``个复输出点。这个族的逆假设它的输入具有相同的对称性，并且对于n个点的输出使用``n/2+1``个输入点。

相应地，当光谱是纯实的时，信号是厄密的。hfft函数族利用了这种对称性，在输入(时间)域中使用``n/2+1``个复点作为频域上的n个实点。

在更高的维度上，使用FFT，例如用于图像分析和滤波。FFT的计算效率意味着它也可以是计算大卷积的一种更快的方法，它利用了时域卷积等效于频域逐点乘法的特性。

## 更高的维度

在二维中，DFT定义为

![公式](/static/images/abfed18222e3312d95d1597d08c38d85847a8ad5.svg)

它以明显的方式延伸到更高的尺寸，而更高尺寸的倒置也以相同的方式延伸。

## 参考文献

[CT309]	Cooley, James W., and John W. Tukey, 1965, “An algorithm for the machine calculation of complex Fourier series,” Math. Comput. 19: 297-301.
[NR309]	Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P., 2007, Numerical Recipes: The Art of Scientific Computing, ch. 12-13. Cambridge Univ. Press, Cambridge, UK.

## 例子

例如，请参阅各种功能api。