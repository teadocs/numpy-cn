# Discrete Fourier Transform (``numpy.fft``)

## Standard FFTs

method | description
---|---
[fft](https://numpy.org/devdocs/reference/generated/numpy.fft.fft.html#numpy.fft.fft)(a[, n, axis, norm]) | Compute the one-dimensional discrete Fourier Transform.
[ifft](https://numpy.org/devdocs/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft)(a[, n, axis, norm]) | Compute the one-dimensional inverse discrete Fourier Transform.
[fft2](https://numpy.org/devdocs/reference/generated/numpy.fft.fft2.html#numpy.fft.fft2)(a[, s, axes, norm]) | Compute the 2-dimensional discrete Fourier Transform
[ifft2](https://numpy.org/devdocs/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2)(a[, s, axes, norm]) | Compute the 2-dimensional inverse discrete Fourier Transform.
[fftn](https://numpy.org/devdocs/reference/generated/numpy.fft.fftn.html#numpy.fft.fftn)(a[, s, axes, norm]) | Compute the N-dimensional discrete Fourier Transform.
[ifftn](https://numpy.org/devdocs/reference/generated/numpy.fft.ifftn.html#numpy.fft.ifftn)(a[, s, axes, norm]) | Compute the N-dimensional inverse discrete Fourier Transform.

## Real FFTs

method | description
---|---
[rfft](https://numpy.org/devdocs/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft)(a[, n, axis, norm]) | Compute the one-dimensional discrete Fourier Transform for real input.
[irfft](https://numpy.org/devdocs/reference/generated/numpy.fft.irfft.html#numpy.fft.irfft)(a[, n, axis, norm]) | Compute the inverse of the n-point DFT for real input.
[rfft2](https://numpy.org/devdocs/reference/generated/numpy.fft.rfft2.html#numpy.fft.rfft2)(a[, s, axes, norm]) | Compute the 2-dimensional FFT of a real array.
[irfft2](https://numpy.org/devdocs/reference/generated/numpy.fft.irfft2.html#numpy.fft.irfft2)(a[, s, axes, norm]) | Compute the 2-dimensional inverse FFT of a real array.
[rfftn](https://numpy.org/devdocs/reference/generated/numpy.fft.rfftn.html#numpy.fft.rfftn)(a[, s, axes, norm]) | Compute the N-dimensional discrete Fourier Transform for real input.
[irfftn](https://numpy.org/devdocs/reference/generated/numpy.fft.irfftn.html#numpy.fft.irfftn)(a[, s, axes, norm]) | Compute the inverse of the N-dimensional FFT of real input.

## Hermitian FFTs

method | description
---|---
[hfft](https://numpy.org/devdocs/reference/generated/numpy.fft.hfft.html#numpy.fft.hfft)(a[, n, axis, norm]) | Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum.
[ihfft](https://numpy.org/devdocs/reference/generated/numpy.fft.ihfft.html#numpy.fft.ihfft)(a[, n, axis, norm]) | Compute the inverse FFT of a signal that has Hermitian symmetry.

## Helper routines

method | description
---|---
[fftfreq](https://numpy.org/devdocs/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq)(n[, d]) | Return the Discrete Fourier Transform sample frequencies.
[rfftfreq](https://numpy.org/devdocs/reference/generated/numpy.fft.rfftfreq.html#numpy.fft.rfftfreq)(n[, d]) | Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft).
[fftshift](https://numpy.org/devdocs/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift)(x[, axes]) | Shift the zero-frequency component to the center of the spectrum.
[ifftshift](https://numpy.org/devdocs/reference/generated/numpy.fft.ifftshift.html#numpy.fft.ifftshift)(x[, axes]) | The inverse of [fftshift](https://numpy.org/devdocs/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift).

## Background information

Fourier analysis is fundamentally a method for expressing a function as a
sum of periodic components, and for recovering the function from those
components.  When both the function and its Fourier transform are
replaced with discretized counterparts, it is called the discrete Fourier
transform (DFT).  The DFT has become a mainstay of numerical computing in
part because of a very fast algorithm for computing it, called the Fast
Fourier Transform (FFT), which was known to Gauss (1805) and was brought
to light in its current form by Cooley and Tukey [[CT]](#rfb1dc64dd6a5-ct).  Press et al. [[NR]](#rfb1dc64dd6a5-nr)
provide an accessible introduction to Fourier analysis and its
applications.

Because the discrete Fourier transform separates its input into
components that contribute at discrete frequencies, it has a great number
of applications in digital signal processing, e.g., for filtering, and in
this context the discretized input to the transform is customarily
referred to as a *signal*, which exists in the *time domain*.  The output
is called a *spectrum* or *transform* and exists in the *frequency domain*.

## Implementation details

There are many ways to define the DFT, varying in the sign of the
exponent, normalization, etc.  In this implementation, the DFT is defined
as

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

## Higher dimensions

In two dimensions, the DFT is defined as

<center>
<img src="/static/images/math/abfed18222e3312d95d1597d08c38d85847a8ad5.svg" alt="a_m = \frac{1}{n}\sum_{k=0}^{n-1}A_k\exp\left\{2\pi i{mk\over n}\right\}
\qquad m = 0,\ldots,n-1.">
</center>

which extends in the obvious way to higher dimensions, and the inverses
in higher dimensions also extend in the same way.

## References

[[CT]](#id1)Cooley, James W., and John W. Tukey, 1965, “An algorithm for the machine calculation of complex Fourier series,” Math. Comput. 19: 297-301.

[[NR]](#id2)Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P., 2007, Numerical Recipes: The Art of Scientific Computing, ch. 12-13. Cambridge Univ. Press, Cambridge, UK.

## Examples

For examples, see the various functions.