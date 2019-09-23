---
meta:
  - name: keywords
    content: NumPy 参考手册
  - name: description
    content: 本参考手册详细介绍了NumPy中包含的函数、模块和对象，描述了它们是什么以及它们做什么。了解如何使用NumPy，另请参阅NumPy用户指南。
---

# NumPy 参考手册

- **发行版本**:	1.18.dev0
- **日期**:	2019年8月14日

本参考手册详细介绍了NumPy中包含的函数、模块和对象，描述了它们是什么以及它们做什么。了解如何使用NumPy，另请参阅[NumPy用户指南](/user/index.html)。

- [数组对象](arrays/index.html)
  - [N维数组(ndarray)](arrays/ndarray.html)
  - [标量](arrays/scalars.html)
  - [数据类型对象(dtype)](arrays/dtypes.html)
  - [索引](arrays/indexing.html)
  - [迭代数组](arrays/nditer.html)
  - [标准数组子类](arrays/classes.html)
  - [掩码数组](arrays/maskedarray.html)
  - [数组接口](arrays/interface.html)
  - [日期时间和时间增量](arrays/datetime.html)
- [常量](constants.html)
- [通函数(ufunc)](ufuncs/)
  - [广播](ufuncs.html#broadcasting)
  - [确定输出类型](ufuncs.html#output-type-determination)
  - [使用内部缓冲区](ufuncs.html#use-of-internal-buffers)
  - [错误处理](ufuncs.html#error-handling)
  - [映射规则](ufuncs.html#casting-rules)
  - [重写Ufunc行为](ufuncs.html#overriding-ufunc-behavior)
  - [ufunc](ufuncs.html#ufunc)
  - [可用的ufunc](ufuncs.html#available-ufuncs)
- [常用API](routines/index.html)
  - [创建数组](routines/array-creation.html)
  - [数组操作](routines/array-manipulation.html)
  - [二进制操作](routines/bitwise.html)
  - [字符串操作](routines/char.html)
  - [C-Types外部函数接口(numpy.ctypeslib)](routines/ctypeslib.html)
  - [时间日期相关](routines/datetime.html)
  - [数据类型相关](routines/dtype.html)
  - [可选的Scipy加速支持(numpy.dual)](routines/dual.html)
  - [具有自动域的数学函数(numpy.emath)](routines/emath.html)
  - [浮点错误处理](routines/err.html)
  - [离散傅立叶变换(numpy.fft)](routines/fft.html)
  - [财金相关](routines/financial.html)
  - [功能的编写](routines/functional.html)
  - [NumPy特别的帮助功能](routines/help.html)
  - [索引相关](routines/indexing.html)
  - [输入和输出](routines/io.html)
  - [线性代数(numpy.linalg)](routines/linalg.html)
  - [逻辑函数](routines/logic.html)
  - [操作掩码数组](routines/ma.html)
  - [数学函数](routines/math.html)
  - [矩阵库(numpy.matlib)](routines/matlib.html)
  - [杂项](routines/other.html)
  - [填充数组](routines/padding.html)
  - [多项式](routines/polynomials.html)
  - [随机抽样(numpy.random)](random/index.html)
  - [集合操作](routines/set.html)
  - [排序、搜索和计数](routines/sort.html)
  - [统计相关](routines/statistics.html)
  - [测试支持(numpy.testing)](routines/testing.html)
  - [窗口函数](routines/window.html)
- [打包(numpy.distutils)](distutils.html)
  - [numpy.distutils中的模块](distutils.html#modules-in-numpy-distutils)
  - [构建可安装的C库](distutils.html#building-installable-c-libraries)
  - [转换.src文件](distutils.html#conversion-of-src-files)
- [NumPy Distutils 的用户指南 ](distutils_guide.html)
  - [SciPy structure](distutils_guide.html#scipy-structure)
  - [Requirements for SciPy packages](distutils_guide.html#requirements-for-scipy-packages)
  - [The setup.py file](distutils_guide.html#the-setup-py-file)
  - [The __init__.py file](distutils_guide.html#the-init-py-file)
  - [Extra features in NumPy Distutils](distutils_guide.html#extra-features-in-numpy-distutils)
- [NumPy C-API](c-api/index.html)
  - [Python类型和C结构](c-api/types-and-structures.html)
  - [系统配置](c-api/config.html)
  - [数据类型API](c-api/dtype.html)
  - [数组API](c-api/array.html)
  - [数组迭代API](c-api/iterator.html)
  - [UFunc API](c-api/ufunc.html)
  - [通常通函数API](c-api/generalized-ufuncs.html)
  - [NumPy核心库](c-api/coremath.html)
  - [弃用的 C API](c-api/deprecations.html)
- [NumPy 内部](internals/index.html)
  - [NumPy C代码说明](internals/code-explanations.html)
  - [内存校准](internals/alignment.html)
  - [numpy数组的内部结构](internals/index.html#internal-organization-of-numpy-arrays)
  - [多维数组索引顺序问题](internals/index.html#multidimensional-array-indexing-order-issues)
- [NumPy 和 SWIG](swig/index.html)
  - [numpy.i：NumPy的SWIG接口文件](swig/interface-file.html)
  - [测试numpy.i Typemaps](swig/testing.html)

## 致谢

本手册的大部分内容源自Travis E. Oliphant的书[《NumPy指南》](https://archive.org/details/NumPyBook) 于2008年8月免费的地公开给公共）。许多函数的参考文档由NumPy的众多贡献者和开发人员编写。
