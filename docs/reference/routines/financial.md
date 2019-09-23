# 财金相关

## 简单的财金功能

方法 | 描述
---|---
[fv](gene[rate](https://numpy.org/devdocs/reference/generated/numpy.rate.html#numpy.rate)d/numpy.fv.html#numpy.fv)(rate, [nper](https://numpy.org/devdocs/reference/generated/numpy.nper.html#numpy.nper), [pmt](https://numpy.org/devdocs/reference/generated/numpy.pmt.html#numpy.pmt), [pv](https://numpy.org/devdocs/reference/generated/numpy.pv.html#numpy.pv)[, when]) | 计算未来值。
[pv](https://numpy.org/devdocs/reference/generated/numpy.pv.html#numpy.pv)(rate, nper, pmt[, fv, when]) | 计算现值。
[npv](https://numpy.org/devdocs/reference/generated/numpy.npv.html#numpy.npv)(rate, values) | 返回现金流系列的NPV(净现值)。
[pmt](https://numpy.org/devdocs/reference/generated/numpy.pmt.html#numpy.pmt)(rate, nper, pv[, fv, when]) | 根据贷款本金加利息计算付款。
[ppmt](https://numpy.org/devdocs/reference/generated/numpy.ppmt.html#numpy.ppmt)(rate, per, nper, pv[, fv, when]) | 根据贷款本金计算付款。
[ipmt](https://numpy.org/devdocs/reference/generated/numpy.ipmt.html#numpy.ipmt)(rate, per, nper, pv[, fv, when]) | 计算付款的利息部分。
[irr](https://numpy.org/devdocs/reference/generated/numpy.irr.html#numpy.irr)(values) | 返回内部收益率(IRR)。
[mirr](https://numpy.org/devdocs/reference/generated/numpy.mirr.html#numpy.mirr)(values, finance_rate, reinvest_rate) | 修改后的内部收益率。
[nper](https://numpy.org/devdocs/reference/generated/numpy.nper.html#numpy.nper)(rate, pmt, pv[, fv, when]) | 计算定期付款的数量。
[rate](https://numpy.org/devdocs/reference/generated/numpy.rate.html#numpy.rate)(nper, pmt, pv, fv[, when, guess, tol, …]) | 计算每个期间的利率。
