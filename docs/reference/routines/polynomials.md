# 多项式(`Polynomials`)

NumPy中的多项式可以使用NumPy 1.4中引入的[numpy.polynomial](https://www.numpy.org/devdocs/reference/routines.polynomials.package.html#module-numpy.polynomial)程序包的[Using the Convenience Classes](https://www.numpy.org/devdocs/reference/routines.polynomials.classes.html)创建，操纵甚至拟合。

在NumPy 1.4之前，[numpy.poly1d](https://www.numpy.org/devdocs/reference/generated/numpy.poly1d.html#numpy.poly1d)是选择的类，为了保持向后兼容性，仍可以使用。 但是，较新的Polynomial软件包比[numpy.poly1d](https://www.numpy.org/devdocs/reference/generated/numpy.poly1d.html#numpy.poly1d)更完整，并且其便利类在numpy环境中表现更好。 因此，建议多项式用于新编码。

# 过渡通知

多项式程序包中的各种例程均处理其系数从零度开始向上的级数，这与Poly1d约定的顺序相反。 记住这一点的简单方法是索引对应于度，即coef [i]是度i项的系数。

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
