# Financial functions

## Simple financial functions

method | description
---|---
[fv](gene[rate](generated/numpy.rate.html#numpy.rate)d/numpy.fv.html#numpy.fv)(rate, [nper](generated/numpy.nper.html#numpy.nper), [pmt](generated/numpy.pmt.html#numpy.pmt), [pv](generated/numpy.pv.html#numpy.pv)[, when]) | Compute the future value.
[pv](generated/numpy.pv.html#numpy.pv)(rate, nper, pmt[, fv, when]) | Compute the present value.
[npv](generated/numpy.npv.html#numpy.npv)(rate, values) | Returns the NPV (Net Present Value) of a cash flow series.
[pmt](generated/numpy.pmt.html#numpy.pmt)(rate, nper, pv[, fv, when]) | Compute the payment against loan principal plus interest.
[ppmt](generated/numpy.ppmt.html#numpy.ppmt)(rate, per, nper, pv[, fv, when]) | Compute the payment against loan principal.
[ipmt](generated/numpy.ipmt.html#numpy.ipmt)(rate, per, nper, pv[, fv, when]) | Compute the interest portion of a payment.
[irr](generated/numpy.irr.html#numpy.irr)(values) | Return the Internal Rate of Return (IRR).
[mirr](generated/numpy.mirr.html#numpy.mirr)(values, finance_rate, reinvest_rate) | Modified internal rate of return.
[nper](generated/numpy.nper.html#numpy.nper)(rate, pmt, pv[, fv, when]) | Compute the number of periodic payments.
[rate](generated/numpy.rate.html#numpy.rate)(nper, pmt, pv, fv[, when, guess, tol, …]) | Compute the rate of interest per period.
