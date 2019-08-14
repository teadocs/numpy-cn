# NumPy C-API

Beware of the man who won’t be bothered with details.
— William Feather, Sr.

The truth is out there.
— Chris Carter, The X Files

NumPy provides a C-API to enable users to extend the system and get access to the array object for use in other routines. The best way to truly understand the C-API is to read the source code. If you are unfamiliar with (C) source code, however, this can be a daunting experience at first. Be assured that the task becomes easier with practice, and you may be surprised at how simple the C-code can be to understand. Even if you don’t think you can write C-code from scratch, it is much easier to understand and modify already-written source code then create it de novo.

Python extensions are especially straightforward to understand because they all have a very similar structure. Admittedly, NumPy is not a trivial extension to Python, and may take a little more snooping to grasp. This is especially true because of the code-generation techniques, which simplify maintenance of very similar code, but can make the code a little less readable to beginners. Still, with a little persistence, the code can be opened to your understanding. It is my hope, that this guide to the C-API can assist in the process of becoming familiar with the compiled-level work that can be done with NumPy in order to squeeze that last bit of necessary speed out of your code.

Python Types and C-Structures
New Python Types Defined
Other C-Structures
System configuration
Data type sizes
Platform information
Compiler directives
Interrupt Handling
Data Type API
Enumerated Types
Defines
C-type names
Printf Formatting
Array API
Array structure and data access
Creating arrays
Dealing with types
Array flags
Array method alternative API
Functions
Auxiliary Data With Object Semantics
Array Iterators
Broadcasting (multi-iterators)
Neighborhood iterator
Array Scalars
Data-type descriptors
Conversion Utilities
Miscellaneous
Array Iterator API
Array Iterator
Simple Iteration Example
Simple Multi-Iteration Example
Iterator Data Types
Construction and Destruction
Functions For Iteration
Converting from Previous NumPy Iterators
UFunc API
Constants
Macros
Functions
Generic functions
Importing the API
Generalized Universal Function API
Definitions
Details of Signature
C-API for implementing Elementary Functions
NumPy core libraries
NumPy core math library
C API Deprecations
Background
Deprecation Mechanism NPY_NO_DEPRECATED_API