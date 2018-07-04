# 金融API

## Simple financial functions

- fv(rate, nper, pmt, pv[, when])	Compute the future value.
- pv(rate, nper, pmt[, fv, when])	Compute the present value.
- npv(rate, values)	Returns the NPV (Net Present Value) of a cash flow series.
- pmt(rate, nper, pv[, fv, when])	Compute the payment against loan principal plus interest.
- ppmt(rate, per, nper, pv[, fv, when])	Compute the payment against loan principal.
- ipmt(rate, per, nper, pv[, fv, when])	Compute the interest portion of a payment.
- irr(values)	Return the Internal Rate of Return (IRR).
- mirr(values, finance_rate, reinvest_rate)	Modified internal rate of return.
- nper(rate, pmt, pv[, fv, when])	Compute the number of periodic payments.
- rate(nper, pmt, pv, fv[, when, guess, tol, …])	Compute the rate of interest per period.