# Financial functions

## Simple financial functions

method | description
---|---
[fv](gene[rate](https://numpy.org/devdocs/reference/generated/numpy.rate.html#numpy.rate)d/numpy.fv.html#numpy.fv)(rate, [nper](https://numpy.org/devdocs/reference/generated/numpy.nper.html#numpy.nper), [pmt](https://numpy.org/devdocs/reference/generated/numpy.pmt.html#numpy.pmt), [pv](https://numpy.org/devdocs/reference/generated/numpy.pv.html#numpy.pv)[, when]) | Compute the future value.
[pv](https://numpy.org/devdocs/reference/generated/numpy.pv.html#numpy.pv)(rate, nper, pmt[, fv, when]) | Compute the present value.
[npv](https://numpy.org/devdocs/reference/generated/numpy.npv.html#numpy.npv)(rate, values) | Returns the NPV (Net Present Value) of a cash flow series.
[pmt](https://numpy.org/devdocs/reference/generated/numpy.pmt.html#numpy.pmt)(rate, nper, pv[, fv, when]) | Compute the payment against loan principal plus interest.
[ppmt](https://numpy.org/devdocs/reference/generated/numpy.ppmt.html#numpy.ppmt)(rate, per, nper, pv[, fv, when]) | Compute the payment against loan principal.
[ipmt](https://numpy.org/devdocs/reference/generated/numpy.ipmt.html#numpy.ipmt)(rate, per, nper, pv[, fv, when]) | Compute the interest portion of a payment.
[irr](https://numpy.org/devdocs/reference/generated/numpy.irr.html#numpy.irr)(values) | Return the Internal Rate of Return (IRR).
[mirr](https://numpy.org/devdocs/reference/generated/numpy.mirr.html#numpy.mirr)(values, finance_rate, reinvest_rate) | Modified internal rate of return.
[nper](https://numpy.org/devdocs/reference/generated/numpy.nper.html#numpy.nper)(rate, pmt, pv[, fv, when]) | Compute the number of periodic payments.
[rate](https://numpy.org/devdocs/reference/generated/numpy.rate.html#numpy.rate)(nper, pmt, pv, fv[, when, guess, tol, …]) | Compute the rate of interest per period.
