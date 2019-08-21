# NumPy C-API

> Beware of the man who won’t be bothered with details.
> 
> — *William Feather, Sr.*

> The truth is out there.
> 
> — *Chris Carter, The X Files*

NumPy provides a C-API to enable users to extend the system and get access to the array object for use in other routines. The best way to truly understand the C-API is to read the source code. If you are unfamiliar with (C) source code, however, this can be a daunting experience at first. Be assured that the task becomes easier with practice, and you may be surprised at how simple the C-code can be to understand. Even if you don’t think you can write C-code from scratch, it is much easier to understand and modify already-written source code then create it de novo.

Python extensions are especially straightforward to understand because they all have a very similar structure. Admittedly, NumPy is not a trivial extension to Python, and may take a little more snooping to grasp. This is especially true because of the code-generation techniques, which simplify maintenance of very similar code, but can make the code a little less readable to beginners. Still, with a little persistence, the code can be opened to your understanding. It is my hope, that this guide to the C-API can assist in the process of becoming familiar with the compiled-level work that can be done with NumPy in order to squeeze that last bit of necessary speed out of your code.

- [Python Types and C-Structures](types-and-structures.html)
  - [New Python Types Defined](types-and-structures.html#new-python-types-defined)
  - [Other C-Structures](types-and-structures.html#other-c-structures)
- [System configuration](config.html)
  - [Data type sizes](config.html#data-type-sizes)
  - [Platform information](config.html#platform-information)
  - [Compiler directives](config.html#compiler-directives)
  - [Interrupt Handling](config.html#interrupt-handling)
- [Data Type API](dtype.html)
  - [Enumerated Types](dtype.html#enumerated-types)
  - [Defines](dtype.html#defines)
  - [C-type names](dtype.html#c-type-names)
  - [Printf Formatting](dtype.html#printf-formatting)
- [Array API](array.html)
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
- [Array Iterator API](iterator.html)
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
- [Generalized Universal Function API](generalized-ufuncs.html)
  - [Definitions](generalized-ufuncs.html#definitions)
  - [Details of Signature](generalized-ufuncs.html#details-of-signature)
  - [C-API for implementing Elementary Functions](generalized-ufuncs.html#c-api-for-implementing-elementary-functions)
- [NumPy core libraries](coremath.html)
  - [NumPy core math library](coremath.html#numpy-core-math-library)
- [C API Deprecations](deprecations.html)
  - [Background](deprecations.html#background)
  - [Deprecation Mechanism NPY_NO_DEPRECATED_API](deprecations.html#deprecation-mechanism-npy-no-deprecated-api)