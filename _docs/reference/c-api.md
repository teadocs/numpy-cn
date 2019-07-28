# NumPy C-API

当心那个不愿为细节操心的人。
Beware of the man who won’t be bothered with details.

— _William Feather, Sr._（威廉·费瑟先生）

真相就在那里。
The truth is out there.

— _Chris Carter, The X Files_ （克里斯·卡特，“X档案”）

NumPy提供了一个C-API，使用户能够扩展系统并访问数组对象，以便在其他例程中使用 真正理解C-API的最佳方法是阅读源代码 但是，如果你不熟悉（C）源代码，那么一开始这可能是一项艰巨的经历 请放心，通过练习，任务变得更容易，你可能会对C代码的易懂性感到惊讶。即使你不认为你可以从头开始编写C代码，理解和修改已编写的源代码然后创建它_de novo_要容易得多，

Python扩展尤其容易理解，因为它们都有非常相似的结构不可否认，NumPy不是Python的一个简单扩展，确实可能需要更多的窥探才能掌握因为代码生成技术简化了非常相似的代码的维护，但会使初学者对代码的可读性有所降低尽管如此，只要稍加坚持，代码就可以让你理解我希望，下面这篇关于C-api的指南能够帮助你熟悉可以用numPy完成的编译级别的工作，以便从你的代码中挤出最后一点必要的速度，

*   [Python类型和C结构](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html)
    *   [定义的新Python类型](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#new-python-types-defined)
    *   [其他C-结构](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#other-c-structures)
*   [系统配置](https://docs.scipy.org/doc/numpy/reference/c-api.config.html)
    *   [数据类型大小](https://docs.scipy.org/doc/numpy/reference/c-api.config.html#data-type-sizes)
    *   [平台信息](https://docs.scipy.org/doc/numpy/reference/c-api.config.html#platform-information)
*   [数据类型API](https://docs.scipy.org/doc/numpy/reference/c-api.dtype.html)
    *   [枚举类型](https://docs.scipy.org/doc/numpy/reference/c-api.dtype.html#enumerated-types)
    *   [定义](https://docs.scipy.org/doc/numpy/reference/c-api.dtype.html#defines)
    *   [类型名称](https://docs.scipy.org/doc/numpy/reference/c-api.dtype.html#c-type-names)
    *   [Printf格式](https://docs.scipy.org/doc/numpy/reference/c-api.dtype.html#printf-formatting)
*   [数组API](https://docs.scipy.org/doc/numpy/reference/c-api.array.html)
    *   [数组结构与数据访问](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-structure-and-data-access)
    *   [创建数组](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#creating-arrays)
    *   [处理types](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#dealing-with-types)
    *   [数组flags](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-flags)
    *   [数组方法替代API](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-method-alternative-api)
    *   [功能函数大全](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#functions)
    *   [具有对象语义的辅助数据](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#auxiliary-data-with-object-semantics)
    *   [数组迭代器](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-iterators)
    *   [广播(多迭代器)](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#broadcasting-multi-iterators)
    *   [邻域迭代器](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#neighborhood-iterator)
    *   [数组标量](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-scalars)
    *   [数据类型descriptors](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#data-type-descriptors)
    *   [转换Utilities](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#conversion-utilities)
    *   [杂项](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#miscellaneous)
*   [数组迭代器API](https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html)
    *   [数组迭代器](https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#array-iterator)
    *   [简单迭代例子](https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#simple-iteration-example)
    *   [简单多迭代例子](https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#simple-multi-iteration-example)
    *   [迭代器数据类型](https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#iterator-data-types)
    *   [建设与销毁](https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#construction-and-destruction)
    *   [迭代函数](https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#functions-for-iteration)
    *   [从以前的NumPy迭代器转换](https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html#converting-from-previous-numpy-iterators)
    *   [通用函数API](https://docs.scipy.org/doc/numpy/reference/c-api.ufunc.html)
    *   [常量](https://docs.scipy.org/doc/numpy/reference/c-api.ufunc.html#constants)
    *   [宏](https://docs.scipy.org/doc/numpy/reference/c-api.ufunc.html#macros)
    *   [功能](https://docs.scipy.org/doc/numpy/reference/c-api.ufunc.html#functions)
    *   [通用functions](https://docs.scipy.org/doc/numpy/reference/c-api.ufunc.html#generic-functions)
    *   [导入API](https://docs.scipy.org/doc/numpy/reference/c-api.ufunc.html#importing-the-api)
    *   [广义泛函数API](https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html)
    *   [定义](https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html#definitions)
    *   [签名](https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html#details-of-signature)
    *   [C-用于实现基本函数](https://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html#c-api-for-implementing-elementary-functions)
    *   [NumPy核心库](https://docs.scipy.org/doc/numpy/reference/c-api.coremath.html)
    *   [NumPy核心数学库](https://docs.scipy.org/doc/numpy/reference/c-api.coremath.html#numpy-core-math-library)
    *   [CAPI 弃用](https://docs.scipy.org/doc/numpy/reference/c-api.deprecations.html)
    *   [背景来历](https://docs.scipy.org/doc/numpy/reference/c-api.deprecations.html#background)
    *   [弃用机制NPY\_NO\_DEPRECATED_API](https://docs.scipy.org/doc/numpy/reference/c-api.deprecations.html#deprecation-mechanism-npy-no-deprecated-api)
