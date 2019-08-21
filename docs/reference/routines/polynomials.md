# Polynomials

Polynomials in NumPy can be created, manipulated, and even fitted using the [Using the Convenience Classes](https://www.numpy.org/devdocs/reference/routines.polynomials.classes.html) of the [numpy.polynomial](https://www.numpy.org/devdocs/reference/routines.polynomials.package.html#module-numpy.polynomial) package, introduced in NumPy 1.4.

Prior to NumPy 1.4, [numpy.poly1d](https://www.numpy.org/devdocs/reference/generated/numpy.poly1d.html#numpy.poly1d) was the class of choice and it is still available in order to maintain backward compatibility. However, the newer Polynomial package is more complete than [numpy.poly1d](https://www.numpy.org/devdocs/reference/generated/numpy.poly1d.html#numpy.poly1d) and its convenience classes are better behaved in the numpy environment. Therefore Polynomial is recommended for new coding.

# Transition notice

The various routines in the Polynomial package all deal with series whose coefficients go from degree zero upward, which is the reverse order of the Poly1d convention. The easy way to remember this is that indexes correspond to degree, i.e., coef[i] is the coefficient of the term of degree i.

- [Polynomial Package](https://www.numpy.org/devdocs/reference/routines.polynomials.package.html)
  - [Using the Convenience Classes](https://www.numpy.org/devdocs/reference/routines.polynomials.classes.html)
  - [Polynomial Module (numpy.polynomial.polynomial)](https://www.numpy.org/devdocs/reference/routines.polynomials.polynomial.html)
  - [Chebyshev Module (numpy.polynomial.chebyshev)](https://www.numpy.org/devdocs/reference/routines.polynomials.chebyshev.html)
  - [Legendre Module (numpy.polynomial.legendre)](https://www.numpy.org/devdocs/reference/routines.polynomials.legendre.html)
  - [Laguerre Module (numpy.polynomial.laguerre)](https://www.numpy.org/devdocs/reference/routines.polynomials.laguerre.html)
  - [Hermite Module, “Physicists’” (numpy.polynomial.hermite)](https://www.numpy.org/devdocs/reference/routines.polynomials.hermite.html)
  - [HermiteE Module, “Probabilists’” (numpy.polynomial.hermite_e)](https://www.numpy.org/devdocs/reference/routines.polynomials.hermite_e.html)
  - [Polyutils](https://www.numpy.org/devdocs/reference/routines.polynomials.polyutils.html)
- [Poly1d](https://www.numpy.org/devdocs/reference/routines.polynomials.poly1d.html)
  - [Basics](https://www.numpy.org/devdocs/reference/routines.polynomials.poly1d.html#basics)
  - [Fitting](https://www.numpy.org/devdocs/reference/routines.polynomials.poly1d.html#fitting)
  - [Calculus](https://www.numpy.org/devdocs/reference/routines.polynomials.poly1d.html#calculus)
  - [Arithmetic](https://www.numpy.org/devdocs/reference/routines.polynomials.poly1d.html#arithmetic)
  - [Warnings](https://www.numpy.org/devdocs/reference/routines.polynomials.poly1d.html#warnings)
