# 金融API

## 简单的财务功能

- fv(rate, nper, pmt, pv[, when])	计算未来价值。
- pv(rate, nper, pmt[, fv, when])	计算现值。
- npv(rate, values)	返回现金流序列的NPV（净现值）。
- pmt(rate, nper, pv[, fv, when])	计算贷款本金和利息的付款。
- ppmt(rate, per, nper, pv[, fv, when])	计算贷款本金的付款。
- ipmt(rate, per, nper, pv[, fv, when])	计算付款的利息部分。
- irr(values)	返回内部收益率（IRR）。
- mirr(values, finance_rate, reinvest_rate)	修改内部收益率。
- nper(rate, pmt, pv[, fv, when])	计算定期付款的数量。
- rate(nper, pmt, pv, fv[, when, guess, tol, …])	计算每个期间的利率。