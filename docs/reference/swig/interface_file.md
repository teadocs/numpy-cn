# NumPy的SWIG接口文件

## Introduction

The Simple Wrapper and Interface Generator (or [SWIG](http://www.swig.org/)) is a powerful tool for generating wrapper code for interfacing to a wide variety of scripting languages. [SWIG](http://www.swig.org/) can parse header files, and using only the code prototypes, create an interface to the target language. But [SWIG](http://www.swig.org/) is not omnipotent. For example, it cannot know from the prototype:

```python
double rms(double* seq, int n);
```

what exactly ``seq`` is. Is it a single value to be altered in-place? Is it an array, and if so what is its length? Is it input-only? Output-only? Input-output? [SWIG](http://www.swig.org/) cannot determine these details, and does not attempt to do so.

If we designed rms, we probably made it a routine that takes an input-only array of length n of double values called seq and returns the root mean square. The default behavior of [SWIG](http://www.swig.org/), however, will be to create a wrapper function that compiles, but is nearly impossible to use from the scripting language in the way the C routine was intended.

For Python, the preferred way of handling contiguous (or technically, strided) blocks of homogeneous data is with NumPy, which provides full object-oriented access to multidimensial arrays of data. Therefore, the most logical Python interface for the rms function would be (including doc string):

```python
def rms(seq):
    """
    rms: return the root mean square of a sequence
    rms(numpy.ndarray) -> double
    rms(list) -> double
    rms(tuple) -> double
    """
```

where ``seq`` would be a NumPy array of ``double`` values, and its length ``n`` would be extracted from ``seq`` internally before being passed to the C routine. Even better, since NumPy supports construction of arrays from arbitrary Python sequences, ``seq`` itself could be a nearly arbitrary sequence (so long as each element can be converted to a double) and the wrapper code would internally convert it to a NumPy array before extracting its data and length.

[SWIG](http://www.swig.org/) allows these types of conversions to be defined via a mechanism called typemaps. This document provides information on how to use ``numpy.i``, a [SWIG](http://www.swig.org/) interface file that defines a series of typemaps intended to make the type of array-related conversions described above relatively simple to implement. For example, suppose that the ``rms`` function prototype defined above was in a header file named ``rms.h``. To obtain the Python interface discussed above, your [SWIG](http://www.swig.org/) interface file would need the following:

```C
%{
#define SWIG_FILE_WITH_INIT
#include "rms.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* seq, int n)};
%include "rms.h"
```

Typemaps are keyed off a list of one or more function arguments, either by type or by type and name. We will refer to such lists as signatures. One of the many typemaps defined by ``numpy.i`` is used above and has the signature ``(double* IN_ARRAY1, int DIM1)``. The argument names are intended to suggest that the ``double*`` argument is an input array of one dimension and that the ``int`` represents the size of that dimension. This is precisely the pattern in the ``rms`` prototype.

Most likely, no actual prototypes to be wrapped will have the argument names ``IN_ARRAY1`` and ``DIM1``. We use the [SWIG](http://www.swig.org/) ``%apply`` directive to apply the typemap for one-dimensional input arrays of type ``double`` to the actual prototype used by ``rms``. Using ``numpy.i`` effectively, therefore, requires knowing what typemaps are available and what they do.

A [SWIG](http://www.swig.org/) interface file that includes the [SWIG](http://www.swig.org/) directives given above will produce wrapper code that looks something like:

```
 1 PyObject *_wrap_rms(PyObject *args) {
 2   PyObject *resultobj = 0;
 3   double *arg1 = (double *) 0 ;
 4   int arg2 ;
 5   double result;
 6   PyArrayObject *array1 = NULL ;
 7   int is_new_object1 = 0 ;
 8   PyObject * obj0 = 0 ;
 9
10   if (!PyArg_ParseTuple(args,(char *)"O:rms",&obj0)) SWIG_fail;
11   {
12     array1 = obj_to_array_contiguous_allow_conversion(
13                  obj0, NPY_DOUBLE, &is_new_object1);
14     npy_intp size[1] = {
15       -1
16     };
17     if (!array1 || !require_dimensions(array1, 1) ||
18         !require_size(array1, size, 1)) SWIG_fail;
19     arg1 = (double*) array1->data;
20     arg2 = (int) array1->dimensions[0];
21   }
22   result = (double)rms(arg1,arg2);
23   resultobj = SWIG_From_double((double)(result));
24   {
25     if (is_new_object1 && array1) Py_DECREF(array1);
26   }
27   return resultobj;
28 fail:
29   {
30     if (is_new_object1 && array1) Py_DECREF(array1);
31   }
32   return NULL;
33 }
```

The typemaps from numpy.i are responsible for the following lines of code: 12–20, 25 and 30. Line 10 parses the input to the rms function. From the format string "O:rms", we can see that the argument list is expected to be a single Python object (specified by the O before the colon) and whose pointer is stored in obj0. A number of functions, supplied by numpy.i, are called to make and check the (possible) conversion from a generic Python object to a NumPy array. These functions are explained in the section [Helper Functions](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#helper-functions), but hopefully their names are self-explanatory. At line 12 we use obj0 to construct a NumPy array. At line 17, we check the validity of the result: that it is non-null and that it has a single dimension of arbitrary length. Once these states are verified, we extract the data buffer and length in lines 19 and 20 so that we can call the underlying C function at line 22. Line 25 performs memory management for the case where we have created a new array that is no longer needed.

This code has a significant amount of error handling. Note the SWIG_fail is a macro for goto fail, referring to the label at line 28. If the user provides the wrong number of arguments, this will be caught at line 10. If construction of the NumPy array fails or produces an array with the wrong number of dimensions, these errors are caught at line 17. And finally, if an error is detected, memory is still managed correctly at line 30.

Note that if the C function signature was in a different order:

```c
double rms(int n, double* seq);
```

that SWIG would not match the typemap signature given above with the argument list for rms. Fortunately, numpy.i has a set of typemaps with the data pointer given last:

```c
%apply (int DIM1, double* IN_ARRAY1) {(int n, double* seq)};
```

This simply has the effect of switching the definitions of arg1 and arg2 in lines 3 and 4 of the generated code above, and their assignments in lines 19 and 20.

## Using numpy.i

The ``numpy.i`` file is currently located in the tools/swig sub-directory under the numpy installation directory. Typically, you will want to copy it to the directory where you are developing your wrappers.

A simple module that only uses a single [SWIG](http://www.swig.org/) interface file should include the following:

```
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}
```

Within a compiled Python module, ``import_array()`` should only get called once. This could be in a C/C++ file that you have written and is linked to the module. If this is the case, then none of your interface files should ``#define SWIG_FILE_WITH_INIT`` or call ``import_array()``. Or, this initialization call could be in a wrapper file generated by [SWIG](http://www.swig.org/) from an interface file that has the %init block as above. If this is the case, and you have more than one [SWIG](http://www.swig.org/) interface file, then only one interface file should ``#define SWIG_FILE_WITH_INIT`` and call ``import_array()``.

## Available Typemaps

The typemap directives provided by numpy.i for arrays of different data types, say double and int, and dimensions of different types, say int or long, are identical to one another except for the C and NumPy type specifications. The typemaps are therefore implemented (typically behind the scenes) via a macro:

```c
%numpy_typemaps(DATA_TYPE, DATA_TYPECODE, DIM_TYPE)
```

that can be invoked for appropriate (DATA_TYPE, DATA_TYPECODE, DIM_TYPE) triplets. For example:

```c
%numpy_typemaps(double, NPY_DOUBLE, int)
%numpy_typemaps(int,    NPY_INT   , int)
```

The numpy.i interface file uses the %numpy_typemaps macro to implement typemaps for the following C data types and int dimension types:

- signed char
- unsigned char
- short
- unsigned short
- int
- unsigned int
- long
- unsigned long
- long long
- unsigned long long
- float
- double

In the following descriptions, we reference a generic DATA_TYPE, which could be any of the C data types listed above, and ``DIM_TYPE`` which should be one of the many types of integers.

The typemap signatures are largely differentiated on the name given to the buffer pointer. Names with ``FARRAY`` are for Fortran-ordered arrays, and names with ``ARRAY`` are for C-ordered (or 1D arrays).

