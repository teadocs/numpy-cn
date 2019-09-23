# NumPy C-API

> Beware of the man who won’t be bothered with details.
> 
> — *William Feather, Sr.*

> The truth is out there.
> 
> — *Chris Carter, The X Files*

NumPy提供了一个C-API，使用户能够扩展系统并访问数组对象，以便在其他例程中使用。真正理解C-API的最好方法是阅读源代码。
但是，如果您不熟悉(C)源代码，一开始这可能是一种令人畏惧的体验。
请放心，随着练习的进行，任务会变得更容易，您可能会惊讶于C代码可以如此简单地理解。
即使您不认为可以从头开始编写C代码，理解和修改已经编写的源代码也要容易得多，然后从头开始创建它。

Python扩展特别容易理解，因为它们都具有非常相似的结构。
诚然，NumPy不是Python的一个微不足道的扩展，可能需要更多的窥探才能掌握。
这是由于代码生成技术的缘故，这些技术简化了非常相似的代码的维护，但会使代码对初学者的可读性稍差一些。
不过，只要稍加坚持，代码就可以开放给您理解。我希望这本C-API指南可以帮助您熟悉可以用NumPy完成的编译级工作，
以便从您的代码中挤出最后一点必要的速度。

- [Python类型和C结构](types-and-structures.html)
  - [New Python Types Defined](types-and-structures.html#new-python-types-defined)
  - [Other C-Structures](types-and-structures.html#other-c-structures)
- [系统配置](config.html)
  - [Data type sizes](config.html#data-type-sizes)
  - [Platform information](config.html#platform-information)
  - [Compiler directives](config.html#compiler-directives)
  - [Interrupt Handling](config.html#interrupt-handling)
- [数据类型API](dtype.html)
  - [Enumerated Types](dtype.html#enumerated-types)
  - [Defines](dtype.html#defines)
  - [C-type names](dtype.html#c-type-names)
  - [Printf Formatting](dtype.html#printf-formatting)
- [数组API](array.html)
  - [Array structure and data access](array.html#array-structure-and-data-access)
  - [Creating arrays](array.html#creating-arrays)
  - [Dealing with types](array.html#dealing-with-types)
  - [Array flags](array.html#array-flags)
  - [Array method alternative API](array.html#array-method-alternative-api)
  - [Functions](array.html#functions)
  - [Auxiliary Data With Object Semantics](array.html#auxiliary-data-with-object-semantics)
  - [Array Iterators](array.html#array-iterators)
  - [Broadcasting (multi-iterators)](array.html#broadcasting-multi-iterators)
  - [Neighborhood iterator](array.html#neighborhood-iterator)
  - [Array Scalars](array.html#array-scalars)
  - [Data-type descriptors](array.html#data-type-descriptors)
  - [Conversion Utilities](array.html#conversion-utilities)
  - [Miscellaneous](array.html#miscellaneous)
- [数组迭代API](iterator.html)
  - [Array Iterator](iterator.html#array-iterator)
  - [Simple Iteration Example](iterator.html#simple-iteration-example)
  - [Simple Multi-Iteration Example](iterator.html#simple-multi-iteration-example)
  - [Iterator Data Types](iterator.html#iterator-data-types)
  - [Construction and Destruction](iterator.html#construction-and-destruction)
  - [Functions For Iteration](iterator.html#functions-for-iteration)
  - [Converting from Previous NumPy Iterators](iterator.html#converting-from-previous-numpy-iterators)
- [UFunc API](ufunc.html)
  - [Constants](ufunc.html#constants)
  - [Macros](ufunc.html#macros)
  - [Functions](ufunc.html#functions)
  - [Generic functions](ufunc.html#generic-functions)
  - [Importing the API](ufunc.html#importing-the-api)
- [一般的通函数API](generalized-ufuncs.html)
  - [Definitions](generalized-ufuncs.html#definitions)
  - [Details of Signature](generalized-ufuncs.html#details-of-signature)
  - [C-API for implementing Elementary Functions](generalized-ufuncs.html#c-api-for-implementing-elementary-functions)
- [NumPy核心库](coremath.html)
  - [NumPy core math library](coremath.html#numpy-core-math-library)
- [弃用的 C API](deprecations.html)
  - [Background](deprecations.html#background)
  - [Deprecation Mechanism NPY_NO_DEPRECATED_API](deprecations.html#deprecation-mechanism-npy-no-deprecated-api)