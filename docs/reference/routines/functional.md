# 功能的编写

- apply_along_axis(func1d, axis, arr, *args, …)	将函数应用于沿给定轴的1-D切片。
- apply_over_axes(func, a, axes) 在多个轴上重复应用功能。
- vectorize(pyfunc[, otypes, doc, excluded, …])	广义函数类。
- frompyfunc(func, nin, nout)	采用任意Python函数并返回NumPy ufunc。
- piecewise(x, condlist, funclist, *args, **kw)	评估分段定义的函数。