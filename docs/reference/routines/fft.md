# 离散傅立叶变换（``numpy.fft``）

## 标准 FFTs

方法 | 描述
---|---
[fft](https://numpy.org/devdocs/reference/generated/numpy.fft.fft.html#numpy.fft.fft)(a[, n, axis, norm]) | 计算一维离散傅立叶变换。
[ifft](https://numpy.org/devdocs/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft)(a[, n, axis, norm]) | 计算一维离散傅立叶逆变换。
[fft2](https://numpy.org/devdocs/reference/generated/numpy.fft.fft2.html#numpy.fft.fft2)(a[, s, axes, norm]) | 计算二维离散傅立叶变换
[ifft2](https://numpy.org/devdocs/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2)(a[, s, axes, norm]) | 计算二维离散傅立叶逆变换。
[fftn](https://numpy.org/devdocs/reference/generated/numpy.fft.fftn.html#numpy.fft.fftn)(a[, s, axes, norm]) | 计算N维离散傅立叶变换。
[ifftn](https://numpy.org/devdocs/reference/generated/numpy.fft.ifftn.html#numpy.fft.ifftn)(a[, s, axes, norm]) | 计算N维逆离散傅立叶变换。

## 实际 FFTs

方法 | 描述
---|---
[rfft](https://numpy.org/devdocs/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft)(a[, n, axis, norm]) | 计算实数输入的一维离散傅立叶变换。
[irfft](https://numpy.org/devdocs/reference/generated/numpy.fft.irfft.html#numpy.fft.irfft)(a[, n, axis, norm]) | 对于实输入，计算n点DFT的逆。
[rfft2](https://numpy.org/devdocs/reference/generated/numpy.fft.rfft2.html#numpy.fft.rfft2)(a[, s, axes, norm]) | 计算实数组的二维FFT。
[irfft2](https://numpy.org/devdocs/reference/generated/numpy.fft.irfft2.html#numpy.fft.irfft2)(a[, s, axes, norm]) | 计算实数组的二维逆FFT。
[rfftn](https://numpy.org/devdocs/reference/generated/numpy.fft.rfftn.html#numpy.fft.rfftn)(a[, s, axes, norm]) | 计算实输入的N维离散傅立叶变换。
[irfftn](https://numpy.org/devdocs/reference/generated/numpy.fft.irfftn.html#numpy.fft.irfftn)(a[, s, axes, norm]) | 计算实输入的N维FFT的逆。

## 厄米特 FFTs

方法 | 描述
---|---
[hfft](https://numpy.org/devdocs/reference/generated/numpy.fft.hfft.html#numpy.fft.hfft)(a[, n, axis, norm]) | 计算具有厄米特对称性的信号的FFT，即实谱。
[ihfft](https://numpy.org/devdocs/reference/generated/numpy.fft.ihfft.html#numpy.fft.ihfft)(a[, n, axis, norm]) | 计算具有厄米特对称性的信号的逆FFT。

## 帮助

方法 | 描述
---|---
[fftfreq](https://numpy.org/devdocs/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq)(n[, d]) | 返回离散傅立叶变换采样频率。
[rfftfreq](https://numpy.org/devdocs/reference/generated/numpy.fft.rfftfreq.html#numpy.fft.rfftfreq)(n[, d]) | 返回离散傅立叶变换采样频率(用于rfft、irfft)。
[fftshift](https://numpy.org/devdocs/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift)(x[, axes]) | 将零频率分量移至频谱中心。
[ifftshift](https://numpy.org/devdocs/reference/generated/numpy.fft.ifftshift.html#numpy.fft.ifftshift)(x[, axes]) | [fftshift](https://numpy.org/devdocs/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift)的逆。

## 背景资料

傅立叶分析基本上是一种方法，用于将函数表示为周期分量之和，
并用于从这些分量中恢复函数。当函数及其傅立叶变换都被离散化的对应物替换时，
它被称为离散傅立叶变换(DFT)。DFT已成为数值计算的支柱，
部分原因是因为有一种计算它的非常快的算法，称为快速傅立叶变换(FFT)，
高斯(1805)已知，并由Cooley和Tukey[CT]以其当前形式曝光。
Press et al.。[NR]提供傅立叶分析及其应用的易懂介绍。

因为离散傅立叶变换将其输入分离成在离散频率上贡献的分量，
所以它在数字信号处理中具有大量的应用，
例如用于滤波，并且在这种情况下，
对变换的离散化输入通常被称为*信号*，
其存在于时域中。输出被称为*频谱* 或 *变换*，并且存在于 *频域* 中。

## 实施细节

有许多方法来定义DFT，在指数符号、归一化等方面有所不同。在此实现中，DFT被定义为：

<center>
<img src="/static/images/math/c3e12e4fbd5334e071b7dfdd4d059fc3584b81e8.svg" alt="A_k =  \sum_{m=0}^{n-1} a_m \exp\left\{-2\pi i{mk \over n}\right\}
\qquad k = 0,\ldots,n-1.">
</center>

The DFT is in general defined for complex inputs and outputs, and a
single-frequency component at linear frequency  is
represented by a complex exponential <img class="math" src="/static/images/math/9127ee37034ef9c70d96a488f67e0c82f9e92ff8.svg" alt="a_m = \exp\{2\pi i\,f m\Delta t\}">
, where <img class="math" src="/static/images/math/ec002955bdf95ee9869878fbad4f80fc98539359.svg" alt="\Delta t">
is the sampling interval.

The values in the result follow so-called “standard” order: If ``A =
fft(a, n)``, then ``A[0]`` contains the zero-frequency term (the sum of
the signal), which is always purely real for real inputs. Then ``A[1:n/2]``
contains the positive-frequency terms, and ``A[n/2+1:]`` contains the
negative-frequency terms, in order of decreasingly negative frequency.
For an even number of input points, ``A[n/2]`` represents both positive and
negative Nyquist frequency, and is also purely real for real input.  For
an odd number of input points, ``A[(n-1)/2]`` contains the largest positive
frequency, while ``A[(n+1)/2]`` contains the largest negative frequency.
The routine ``np.fft.fftfreq(n)`` returns an array giving the frequencies
of corresponding elements in the output.  The routine
``np.fft.fftshift(A)`` shifts transforms and their frequencies to put the
zero-frequency components in the middle, and ``np.fft.ifftshift(A)`` undoes
that shift.

When the input *a* is a time-domain signal and ``A = fft(a)``, ``np.abs(A)``
is its amplitude spectrum and ``np.abs(A)**2`` is its power spectrum.
The phase spectrum is obtained by ``np.angle(A)``.

The inverse DFT is defined as

<center>
<img src="/static/images/math/25d7a89b77473363cb4da8b11ca853073f63729f.svg" alt="a_m = \frac{1}{n}\sum_{k=0}^{n-1}A_k\exp\left\{2\pi i{mk\over n}\right\}
\qquad m = 0,\ldots,n-1.">
</center>

It differs from the forward transform by the sign of the exponential
argument and the default normalization by 1/n.

## Normalization

The default normalization has the direct transforms unscaled and the inverse
transforms are scaled by 1/n. It is possible to obtain unitary
transforms by setting the keyword argument ``norm`` to ``"ortho"`` (default is
*None*) so that both direct and inverse transforms will be scaled by <img class="math" src="/static/images/math/f780dc84ea49c387f9417b50f0619e404d91c28a.svg" alt="1/\sqrt{n}">.

## Real and Hermitian transforms

When the input is purely real, its transform is Hermitian, i.e., the
component at frequency  is the complex conjugate of the
component at frequency , which means that for real
inputs there is no information in the negative frequency components that
is not already available from the positive frequency components.
The family of [``rfft``](https://numpy.org/devdocs/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft) functions is
designed to operate on real inputs, and exploits this symmetry by
computing only the positive frequency components, up to and including the
Nyquist frequency.  Thus, ``n`` input points produce ``n/2+1`` complex
output points.  The inverses of this family assumes the same symmetry of
its input, and for an output of ``n`` points uses ``n/2+1`` input points.

Correspondingly, when the spectrum is purely real, the signal is
Hermitian.  The [``hfft``](https://numpy.org/devdocs/reference/generated/numpy.fft.hfft.html#numpy.fft.hfft) family of functions exploits this symmetry by
using ``n/2+1`` complex points in the input (time) domain for ``n`` real
points in the frequency domain.

In higher dimensions, FFTs are used, e.g., for image analysis and
filtering.  The computational efficiency of the FFT means that it can
also be a faster way to compute large convolutions, using the property
that a convolution in the time domain is equivalent to a point-by-point
multiplication in the frequency domain.

## 高维度

在二维中，DFT定义为：

<center>
<img src="/static/images/math/abfed18222e3312d95d1597d08c38d85847a8ad5.svg" alt="a_m = \frac{1}{n}\sum_{k=0}^{n-1}A_k\exp\left\{2\pi i{mk\over n}\right\}
\qquad m = 0,\ldots,n-1.">
</center>

它以明显的方式延伸到更高的维度，并且在更高维度中的倒数也以同样的方式延伸。

## 参考文献

[[CT]](#id1)Cooley, James W., and John W. Tukey, 1965, “An algorithm for the machine calculation of complex Fourier series,” Math. Comput. 19: 297-301.

[[NR]](#id2)Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P., 2007, Numerical Recipes: The Art of Scientific Computing, ch. 12-13. Cambridge Univ. Press, Cambridge, UK.

## 示例

有关示例，请参见各种函数。
