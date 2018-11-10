# NumPy的SWIG接口文件

## 介绍

Simple Wrapper和Interface Generator（或SWIG）是一个功能强大的工具，用于生成包装代码，以便与各种脚本语言进行交互。SWIG可以解析头文件，只使用代码原型，创建目标语言的接口。 但SWIG并非无所不能。 例如，它无法从原型中知道：

```python
double rms(double* seq, int n);
```

究竟是什么 ``seq``。它是一个可以就地改变的单一值吗？它是一个数组，如果是这样，它的长度是多少？ 它只是输入吗？仅输出?输入输出？SWIG无法确定这些细节，也不会尝试这样做。

如果我们设计了rms，我们可能会使它成为一个例程，它接受一个名为seq的长度为n的double值的输入数组，并返回均方根。 但是，SWIG的默认行为是创建一个编译的包装函数，但几乎不可能像C例程那样使用脚本语言。

对于Python，处理连续（或技术上，跨步）的同构数据块的首选方法是使用NumPy，它提供对多维数据数组的完全面向对象的访问。 因此，rms函数的最合理的Python接口将是（包括doc string）：

```python
def rms(seq):
    """
    rms: return the root mean square of a sequence
    rms(numpy.ndarray) -> double
    rms(list) -> double
    rms(tuple) -> double
    """
```

其中``seq``将是一个'`double``值的NumPy数组，其长度``n``将在内部从``seq``中提取，然后传递给C例程。更好的是，由于NumPy支持从任意Python序列构造数组，``seq``本身可能是一个几乎任意的序列（只要每个元素都可以转换为double），包装器代码将在内部将其转换为NumPy 在提取数据和长度之前的数组。

SWIG允许通过称为类型映射的机制定义这些类型的转换。 本文档提供了有关如何使用``numpy.i``的信息，这是一个SWIG接口文件，它定义了一系列类型映射，旨在使上述与数组相关的转换类型实现起来相对简单。 例如，假设上面定义的rms函数原型位于名为rms.h的头文件中。 要获得上面讨论的Python接口，你的SWIG接口文件需要以下内容：

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

类型映射通过类型或类型和名称键入一个或多个函数参数的列表。 我们将这些列表称为签名。``numpy.i``定义的许多类型映射之一在上面使用并具有签名``（double * IN_ARRAY1，int DIM1）``。参数名称旨在表明``double*``参数是一个维度的输入数组，而``int``表示该维度的大小。这正是``rms``原型中的模式。

最有可能的是，没有要包装的实际原型将具有参数名称 ``IN_ARRAY1`` 和 ``DIM1`` 。我们使用SWIG ``％apply`` 指令将类型为``double``的一维输入数组的typemap应用于``rms``使用的实际原型。 因此，有效地使用 ``numpy.i`` 需要知道可用的类型映射以及它们的作用。

包含上面给出的SWIG指令的SWIG接口文件将生成如下所示的包装器代码：

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

来自 ``numpy.i`` 的类型映射负责以下代码行：12-20,25和30.第10行将输入解析为 ``rms`` 函数。从格式字符串“O：rms”，我们可以看到参数列表应该是一个Python对象（由冒号前的O指定），其指针存储在obj0中。由numpy.i提供的许多函数被调用来制作和检查从通用Python对象到NumPy数组的（可能的）转换。[Helper Functions](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#helper-functions)一节中对这些函数进行了解释，但希望它们的名称不言自明。在第12行，我们使用obj0来构造NumPy数组。在第17行，我们检查结果的有效性：它是非null并且它具有任意长度的单个维度。验证这些状态后，我们在第19行和第20行中提取数据缓冲区和长度，以便我们可以在第22行调用底层C函数。第25行执行内存管理，以便我们创建一个不再有新数组的情况需要。

此代码具有大量错误处理。注意SWIG_fail是goto失败的宏，引用第28行的标签。如果用户提供了错误数量的参数，则会在第10行捕获。如果NumPy数组的构造失败或产生错误的数组 维数，这些错误在第17行捕获。最后，如果检测到错误，仍然在第30行正确管理内存。

请注意，如果C函数签名的顺序不同：

```c
double rms(int n, double* seq);
```

SWIG与上面给出的类型映射签名与rms的参数列表不匹配。幸运的是，numpy.i有一组带有最后给出的数据指针的文字映射：

```c
%apply (int DIM1, double* IN_ARRAY1) {(int n, double* seq)};
```

这简单地具有在上面生成的代码的第3行和第4行中切换arg1和arg2的定义的效果，以及它们在第19行和第20行中的赋值。

## 使用 numpy.i

``numpy.i``文件当前位于numpy安装目录下的tools / swig子目录中。 通常，您需要将其复制到开发包装器的目录中。

仅使用单个 SWIG 接口文件的简单模块应包括以下内容：

```
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}
```

在编译的Python模块中，``import_array()`` 应该只被调用一次。 这可能是您编写的C / C++ 文件，并链接到模块。如果是这种情况，那么你的接口文件都不应该 ``#define SWIG_FILE_WITH_INIT`` 或调用 ``import_array()``。或者，此初始化调用可以位于由 SWIG 从具有上述 ％init 块的接口文件生成的包装文件中。如果是这种情况，并且你有多个 SWIG 接口文件，那么只有一个接口文件应该 ``#define SWIG_FILE_WITH_INIT`` 并调用 ``import_array()``。

## 可用的字体映射

numpy.i为不同数据类型的数组提供的typemap指令，比如double和int，以及不同类型的维度，比如int或long，除了C和NumPy类型规范之外，它们彼此相同。 因此，通过宏实现（通常在幕后）类型图：

```c
%numpy_typemaps(DATA_TYPE, DATA_TYPECODE, DIM_TYPE)
```

可以为适当的 (DATA_TYPE, DATA_TYPECODE, DIM_TYPE) 三元组调用。 例如：

```c
%numpy_typemaps(double, NPY_DOUBLE, int)
%numpy_typemaps(int,    NPY_INT   , int)
```

numpy.i接口文件使用％numpy_typemaps宏来实现以下C数据类型和int维类型的类型映射：

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

在下面的描述中，我们引用了一个通用的DATA_TYPE，它可以是上面列出的任何C数据类型，以及``DIM_TYPE``，它应该是许多类型的整数之一。

类型映射签名在很大程度上区分给缓冲区指针的名称。带有``FARRAY``的名称用于Fortran排序的数组，带有``ARRAY``的名称用于C-ordered（或1D数组）。

### 输入 Arrays

输入数组被定义为传递到例程但不会就地更改或返回给用户的数据数组。 因此，Python输入数组几乎可以被任何Python序列（例如列表）转换为所请求的数组类型。输入数组签名是：

1D:

- ( DATA_TYPE IN_ARRAY1[ANY] )
- ( DATA_TYPE* IN_ARRAY1, int DIM1 )
- ( int DIM1, DATA_TYPE* IN_ARRAY1 )

2D:

- ( DATA_TYPE IN_ARRAY2[ANY][ANY] )
- ( DATA_TYPE* IN_ARRAY2, int DIM1, int DIM2 )
- ( int DIM1, int DIM2, DATA_TYPE* IN_ARRAY2 )
- ( DATA_TYPE* IN_FARRAY2, int DIM1, int DIM2 )
- ( int DIM1, int DIM2, DATA_TYPE* IN_FARRAY2 )

3D:

- ( DATA_TYPE IN_ARRAY3[ANY][ANY][ANY] )
- ( DATA_TYPE* IN_ARRAY3, int DIM1, int DIM2, int DIM3 )
- ( int DIM1, int DIM2, int DIM3, DATA_TYPE* IN_ARRAY3 )
- ( DATA_TYPE* IN_FARRAY3, int DIM1, int DIM2, int DIM3 )
- ( int DIM1, int DIM2, int DIM3, DATA_TYPE* IN_FARRAY3 )

4D:

- (DATA_TYPE IN_ARRAY4[ANY][ANY][ANY][ANY])
- (DATA_TYPE* IN_ARRAY4, DIM_TYPE DIM1, DIM_TYPE DIM2, DIM_TYPE DIM3, DIM_TYPE DIM4)
- (DIM_TYPE DIM1, DIM_TYPE DIM2, DIM_TYPE DIM3, , DIM_TYPE DIM4, DATA_TYPE* IN_ARRAY4)
- (DATA_TYPE* IN_FARRAY4, DIM_TYPE DIM1, DIM_TYPE DIM2, DIM_TYPE DIM3, DIM_TYPE DIM4)
- (DIM_TYPE DIM1, DIM_TYPE DIM2, DIM_TYPE DIM3, DIM_TYPE DIM4, DATA_TYPE* IN_FARRAY4)

The first signature listed, ``( DATA_TYPE IN_ARRAY[ANY] )`` is for one-dimensional arrays with hard-coded dimensions. Likewise, ``( DATA_TYPE IN_ARRAY2[ANY][ANY] )`` is for two-dimensional arrays with hard-coded dimensions, and similarly for three-dimensional.

### In-Place Arrays

In-place arrays are defined as arrays that are modified in-place. The input values may or may not be used, but the values at the time the function returns are significant. The provided Python argument must therefore be a NumPy array of the required type. The in-place signatures are

1D:

- ( DATA_TYPE INPLACE_ARRAY1[ANY] )
- ( DATA_TYPE* INPLACE_ARRAY1, int DIM1 )
- ( int DIM1, DATA_TYPE* INPLACE_ARRAY1 )

2D:

- ( DATA_TYPE INPLACE_ARRAY2[ANY][ANY] )
- ( DATA_TYPE* INPLACE_ARRAY2, int DIM1, int DIM2 )
- ( int DIM1, int DIM2, DATA_TYPE* INPLACE_ARRAY2 )
- ( DATA_TYPE* INPLACE_FARRAY2, int DIM1, int DIM2 )
- ( int DIM1, int DIM2, DATA_TYPE* INPLACE_FARRAY2 )

3D:

- ( DATA_TYPE INPLACE_ARRAY3[ANY][ANY][ANY] )
- ( DATA_TYPE* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3 )
- ( int DIM1, int DIM2, int DIM3, DATA_TYPE* INPLACE_ARRAY3 )
- ( DATA_TYPE* INPLACE_FARRAY3, int DIM1, int DIM2, int DIM3 )
- ( int DIM1, int DIM2, int DIM3, DATA_TYPE* INPLACE_FARRAY3 )

4D:

- (DATA_TYPE INPLACE_ARRAY4[ANY][ANY][ANY][ANY])
- (DATA_TYPE* INPLACE_ARRAY4, DIM_TYPE DIM1, DIM_TYPE DIM2, DIM_TYPE DIM3, DIM_TYPE DIM4)
- (DIM_TYPE DIM1, DIM_TYPE DIM2, DIM_TYPE DIM3, , DIM_TYPE DIM4, DATA_TYPE* INPLACE_ARRAY4)
- (DATA_TYPE* INPLACE_FARRAY4, DIM_TYPE DIM1, DIM_TYPE DIM2, DIM_TYPE DIM3, DIM_TYPE DIM4)
- (DIM_TYPE DIM1, DIM_TYPE DIM2, DIM_TYPE DIM3, DIM_TYPE DIM4, DATA_TYPE* INPLACE_FARRAY4)

These typemaps now check to make sure that the ``INPLACE_ARRAY`` arguments use native byte ordering. If not, an exception is raised.

There is also a “flat” in-place array for situations in which you would like to modify or process each element, regardless of the number of dimensions. One example is a “quantization” function that quantizes each element of an array in-place, be it 1D, 2D or whatever. This form checks for continuity but allows either C or Fortran ordering.

ND:

- (DATA_TYPE* INPLACE_ARRAY_FLAT, DIM_TYPE DIM_FLAT)

### Argout Arrays

Argout arrays are arrays that appear in the input arguments in C, but are in fact output arrays. This pattern occurs often when there is more than one output variable and the single return argument is therefore not sufficient. In Python, the conventional way to return multiple arguments is to pack them into a sequence (tuple, list, etc.) and return the sequence. This is what the argout typemaps do. If a wrapped function that uses these argout typemaps has more than one return argument, they are packed into a tuple or list, depending on the version of Python. The Python user does not pass these arrays in, they simply get returned. For the case where a dimension is specified, the python user must provide that dimension as an argument. The argout signatures are

1D:

- ( DATA_TYPE ARGOUT_ARRAY1[ANY] )
- ( DATA_TYPE* ARGOUT_ARRAY1, int DIM1 )
- ( int DIM1, DATA_TYPE* ARGOUT_ARRAY1 )

2D:

- ( DATA_TYPE ARGOUT_ARRAY2[ANY][ANY] )

3D:

- ( DATA_TYPE ARGOUT_ARRAY3[ANY][ANY][ANY] )

4D:

- ( DATA_TYPE ARGOUT_ARRAY4[ANY][ANY][ANY][ANY] )

These are typically used in situations where in C/C++, you would allocate a(n) array(s) on the heap, and call the function to fill the array(s) values. In Python, the arrays are allocated for you and returned as new array objects.

Note that we support ``DATA_TYPE*`` argout typemaps in 1D, but not 2D or 3D. This is because of a quirk with the [SWIG](http://www.swig.org/) typemap syntax and cannot be avoided. Note that for these types of 1D typemaps, the Python function will take a single argument representing ``DIM1``.

### Argout View Arrays

Argoutview arrays are for when your C code provides you with a view of its internal data and does not require any memory to be allocated by the user. This can be dangerous. There is almost no way to guarantee that the internal data from the C code will remain in existence for the entire lifetime of the NumPy array that encapsulates it. If the user destroys the object that provides the view of the data before destroying the NumPy array, then using that array may result in bad memory references or segmentation faults. Nevertheless, there are situations, working with large data sets, where you simply have no other choice.

The C code to be wrapped for argoutview arrays are characterized by pointers: pointers to the dimensions and double pointers to the data, so that these values can be passed back to the user. The argoutview typemap signatures are therefore

1D:

- ( DATA_TYPE** ARGOUTVIEW_ARRAY1, DIM_TYPE* DIM1 )
- ( DIM_TYPE* DIM1, DATA_TYPE** ARGOUTVIEW_ARRAY1 )

2D:

- ( DATA_TYPE** ARGOUTVIEW_ARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2 )
- ( DIM_TYPE* DIM1, DIM_TYPE* DIM2, DATA_TYPE** ARGOUTVIEW_ARRAY2 )
- ( DATA_TYPE** ARGOUTVIEW_FARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2 )
- ( DIM_TYPE* DIM1, DIM_TYPE* DIM2, DATA_TYPE** ARGOUTVIEW_FARRAY2 )

3D:

- ( DATA_TYPE** ARGOUTVIEW_ARRAY3, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3)
- ( DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DATA_TYPE** ARGOUTVIEW_ARRAY3)
- ( DATA_TYPE** ARGOUTVIEW_FARRAY3, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3)
- ( DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DATA_TYPE** ARGOUTVIEW_FARRAY3)

4D:

- (DATA_TYPE** ARGOUTVIEW_ARRAY4, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4)
- (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4, DATA_TYPE** ARGOUTVIEW_ARRAY4)
- (DATA_TYPE** ARGOUTVIEW_FARRAY4, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4)
- (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4, DATA_TYPE** ARGOUTVIEW_FARRAY4)

Note that arrays with hard-coded dimensions are not supported. These cannot follow the double pointer signatures of these typemaps.

Memory Managed Argout View Arrays
A recent addition to numpy.i are typemaps that permit argout arrays with views into memory that is managed. See the discussion here.

1D:

- (DATA_TYPE** ARGOUTVIEWM_ARRAY1, DIM_TYPE* DIM1)
- (DIM_TYPE* DIM1, DATA_TYPE** ARGOUTVIEWM_ARRAY1)

2D:

- (DATA_TYPE** ARGOUTVIEWM_ARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2)
- (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DATA_TYPE** ARGOUTVIEWM_ARRAY2)
- (DATA_TYPE** ARGOUTVIEWM_FARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2)
- (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DATA_TYPE** ARGOUTVIEWM_FARRAY2)

3D:

- (DATA_TYPE** ARGOUTVIEWM_ARRAY3, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3)
- (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DATA_TYPE** ARGOUTVIEWM_ARRAY3)
- (DATA_TYPE** ARGOUTVIEWM_FARRAY3, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3)
- (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DATA_TYPE** ARGOUTVIEWM_FARRAY3)

4D:

- (DATA_TYPE** ARGOUTVIEWM_ARRAY4, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4)
- (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4, DATA_TYPE** ARGOUTVIEWM_ARRAY4)
- (DATA_TYPE** ARGOUTVIEWM_FARRAY4, DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4)
- (DIM_TYPE* DIM1, DIM_TYPE* DIM2, DIM_TYPE* DIM3, DIM_TYPE* DIM4, DATA_TYPE** ARGOUTVIEWM_FARRAY4)

### Output Arrays

The ``numpy.i`` interface file does not support typemaps for output arrays, for several reasons. First, C/C++ return arguments are limited to a single value. This prevents obtaining dimension information in a general way. Second, arrays with hard-coded lengths are not permitted as return arguments. In other words:

```c
double[3] newVector(double x, double y, double z);
```

is not legal C/C++ syntax. Therefore, we cannot provide typemaps of the form:

```c
%typemap(out) (TYPE[ANY]);
```

If you run into a situation where a function or method is returning a pointer to an array, your best bet is to write your own version of the function to be wrapped, either with %extend for the case of class methods or ``%ignore`` and ``%rename`` for the case of functions.

### Other Common Types: bool

Note that C++ type ``bool`` is not supported in the list in the [Available Typemaps](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#available-typemaps) section. NumPy bools are a single byte, while the C++ bool is four bytes (at least on my system). Therefore:

```c
%numpy_typemaps(bool, NPY_BOOL, int)
```

will result in typemaps that will produce code that reference improper data lengths. You can implement the following macro expansion:

```c
%numpy_typemaps(bool, NPY_UINT, int)
```

to fix the data length problem, and [Input Arrays](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#input-arrays) will work fine, but [In-Place](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#in-place-arrays) Arrays might fail type-checking.

### Other Common Types: complex

Typemap conversions for complex floating-point types is also not supported automatically. This is because Python and NumPy are written in C, which does not have native complex types. Both Python and NumPy implement their own (essentially equivalent) ``struct`` definitions for complex variables:

```c
/* Python */
typedef struct {double real; double imag;} Py_complex;

/* NumPy */
typedef struct {float  real, imag;} npy_cfloat;
typedef struct {double real, imag;} npy_cdouble;
```

We could have implemented:

```c
%numpy_typemaps(Py_complex , NPY_CDOUBLE, int)
%numpy_typemaps(npy_cfloat , NPY_CFLOAT , int)
%numpy_typemaps(npy_cdouble, NPY_CDOUBLE, int)
```

which would have provided automatic type conversions for arrays of type Py_complex, npy_cfloat and npy_cdouble. However, it seemed unlikely that there would be any independent (non-Python, non-NumPy) application code that people would be using [SWIG](http://www.swig.org/) to generate a Python interface to, that also used these definitions for complex types. More likely, these application codes will define their own complex types, or in the case of C++, use std::complex. Assuming these data structures are compatible with Python and NumPy complex types, %numpy_typemap expansions as above (with the user’s complex type substituted for the first argument) should work.

## NumPy Array Scalars and SWIG

[SWIG](http://www.swig.org/) has sophisticated type checking for numerical types. For example, if your C/C++ routine expects an integer as input, the code generated by [SWIG](http://www.swig.org/) will check for both Python integers and Python long integers, and raise an overflow error if the provided Python integer is too big to cast down to a C integer. With the introduction of NumPy scalar arrays into your Python code, you might conceivably extract an integer from a NumPy array and attempt to pass this to a SWIG-wrapped C/C++ function that expects an int, but the [SWIG](http://www.swig.org/) type checking will not recognize the NumPy array scalar as an integer. (Often, this does in fact work – it depends on whether NumPy recognizes the integer type you are using as inheriting from the Python integer type on the platform you are using. Sometimes, this means that code that works on a 32-bit machine will fail on a 64-bit machine.)

If you get a Python error that looks like the following:

```python
TypeError: in method 'MyClass_MyMethod', argument 2 of type 'int'
```

and the argument you are passing is an integer extracted from a NumPy array, then you have stumbled upon this problem. The solution is to modify the [SWIG](http://www.swig.org/) type conversion system to accept NumPy array scalars in addition to the standard integer types. Fortunately, this capability has been provided for you. Simply copy the file:

```
pyfragments.swg
```

to the working build directory for you project, and this problem will be fixed. It is suggested that you do this anyway, as it only increases the capabilities of your Python interface.

### Why is There a Second File?

The [SWIG](http://www.swig.org/) type checking and conversion system is a complicated combination of C macros, [SWIG](http://www.swig.org/) macros, [SWIG](http://www.swig.org/) typemaps and [SWIG](http://www.swig.org/) fragments. Fragments are a way to conditionally insert code into your wrapper file if it is needed, and not insert it if not needed. If multiple typemaps require the same fragment, the fragment only gets inserted into your wrapper code once.

There is a fragment for converting a Python integer to a C ``long``. There is a different fragment that converts a Python integer to a C ``int``, that calls the routine defined in the ``long`` fragment. We can make the changes we want here by changing the definition for the ``long`` fragment. [SWIG](http://www.swig.org/) determines the active definition for a fragment using a “first come, first served” system. That is, we need to define the fragment for ``long`` conversions prior to [SWIG](http://www.swig.org/) doing it internally. [SWIG](http://www.swig.org/) allows us to do this by putting our fragment definitions in the file ``pyfragments.swg``. If we were to put the new fragment definitions in ``numpy.i``, they would be ignored.

## Helper Functions

The ``numpy.i`` file contains several macros and routines that it uses internally to build its typemaps. However, these functions may be useful elsewhere in your interface file. These macros and routines are implemented as fragments, which are described briefly in the previous section. If you try to use one or more of the following macros or functions, but your compiler complains that it does not recognize the symbol, then you need to force these fragments to appear in your code using:

```c
%fragment("NumPy_Fragments");
```

in your [SWIG](http://www.swig.org/) interface file.

### Macros

- is_array(a) Evaluates as true if ``a`` is non-``NULL`` and can be cast to a ``PyArrayObject*``.
- array_type(a) Evaluates to the integer data type code of a, assuming a can be cast to a ``PyArrayObject*``.
- array_numdims(a) Evaluates to the integer number of dimensions of a, assuming a can be cast to a ``PyArrayObject*``.
- array_dimensions(a) Evaluates to an array of type ``npy_intp`` and length ``array_numdims(a)``, giving the lengths of all of the dimensions of ``a``, assuming ``a`` can be cast to a ``PyArrayObject*``.
- array_size(a,i) Evaluates to the i-th dimension size of a, assuming a can be cast to a PyArrayObject*.
- array_strides(a) Evaluates to an array of type npy_intp and length array_numdims(a), giving the stridess of all of the dimensions of a, assuming a can be cast to a PyArrayObject*. A stride is the distance in bytes between an element and its immediate neighbor along the same axis.
- array_stride(a,i) Evaluates to the i-th stride of a, assuming a can be cast to a PyArrayObject*.
- array_data(a) Evaluates to a pointer of type void* that points to the data buffer of a, assuming a can be cast to a PyArrayObject*.
- array_descr(a) Returns a borrowed reference to the dtype property (PyArray_Descr*) of a, assuming a can be cast to a PyArrayObject*.
- array_flags(a) Returns an integer representing the flags of a, assuming a can be cast to a PyArrayObject*.
- array_enableflags(a,f) Sets the flag represented by f of a, assuming a can be cast to a PyArrayObject*.
- array_is_contiguous(a) Evaluates as true if a is a contiguous array. Equivalent to (PyArray_ISCONTIGUOUS(a)).
- array_is_native(a) Evaluates as true if the data buffer of a uses native byte order. Equivalent to (PyArray_ISNOTSWAPPED(a)).
- array_is_fortran(a) Evaluates as true if a is FORTRAN ordered.

### Routines

- pytype_string()
    - Return type: ``const char*``
    - Arguments:
        - ``PyObject* py_obj``, a general Python object.
    - Return a string describing the type of ``py_obj``.
- typecode_string()
    - Return type: ``const char*``
    - Arguments: 
        - ``int typecode``, a NumPy integer typecode.
    - Return a string describing the type corresponding to the NumPy ``typecode``.
- type_match()
    - Return type: int
    - Arguments:
        - ``int actual_type``, the NumPy typecode of a NumPy array.
        - ``int desired_type``, the desired NumPy typecode.
    - Make sure that ``actual_type`` is compatible with ``desired_type``. For example, this allows character and byte types, or int and long types, to match. This is now equivalent to ``PyArray_EquivTypenums()``.
- obj_to_array_no_conversion()
    - Return type: ``PyArrayObject*``
    - Arguments:
        - ``PyObject\* input``, a general Python object.
        - ``int typecode``, the desired NumPy typecode.
    - Cast input to a PyArrayObject* if legal, and ensure that it is of type typecode. If input cannot be cast, or the typecode is wrong, set a Python error and return NULL.
- obj_to_array_allow_conversion()
    - Return type: ``PyArrayObject*``
    - Arguments:
        - ``PyObject\* input``, a general Python object.
        - ``int typecode``, the desired NumPy typecode of the resulting array.
        - ``int* is_new_object``, returns a value of 0 if no conversion performed, else 1.
    - Convert ``input`` to a NumPy array with the given typecode. On success, return a valid ``PyArrayObject*`` with the correct type. On failure, the Python error string will be set and the routine returns ``NULL``.
- make_contiguous()
    - Return type: ``PyArrayObject*``
    - Arguments:
        - ``PyArrayObject* ary``, a NumPy array.
        - ``int* is_new_object``, returns a value of 0 if no conversion performed, else 1.
        - ``int min_dims``, minimum allowable dimensions.
        - ``int max_dims``, maximum allowable dimensions.
    - Check to see if ary is contiguous. If so, return the input pointer and flag it as not a new object. If it is not contiguous, create a new PyArrayObject* using the original data, flag it as a new object and return the pointer.
- make_fortran()
    - Return type: ``PyArrayObject*``
    - Arguments
        - ``PyArrayObject* ary``, a NumPy array.
        - ``int* is_new_object``, returns a value of 0 if no conversion performed, else 1.
    - Check to see if ary is Fortran contiguous. If so, return the input pointer and flag it as not a new object. If it is not Fortran contiguous, create a new PyArrayObject* using the original data, flag it as a new object and return the pointer.
- obj_to_array_contiguous_allow_conversion()
    - Return type: ``PyArrayObject*``
    - Arguments:
        - ``PyObject\* input``, a general Python object.
        - ``int typecode``, the desired NumPy typecode of the resulting array.
        - ``int* is_new_object``, returns a value of 0 if no conversion performed, else 1.
    - Convert ``input`` to a contiguous ``PyArrayObject*`` of the specified type. If the input object is not a contiguous ``PyArrayObject*``, a new one will be created and the new object flag will be set.
- obj_to_array_fortran_allow_conversion()
    - Return type: ``PyArrayObject*``
    - Arguments:
        - ``PyObject\* input``, a general Python object.
        - ``int typecode``, the desired NumPy typecode of the resulting array.
        - ``int* is_new_object``, returns a value of 0 if no conversion performed, else 1.
    - Convert ``input`` to a Fortran contiguous ``PyArrayObject*`` of the specified type. If the input object is not a Fortran contiguous ``PyArrayObject*``, a new one will be created and the new object flag will be set.
- require_contiguous()
    - Return type: ``int``
    - Arguments:
        - ``PyArrayObject* ary``, a NumPy array.
    - Test whether ary is contiguous. If so, return 1. Otherwise, set a Python error and return 0.
- require_native()
    - Return type: ``int``
    - Arguments:
        - ``PyArray_Object*`` ary, a NumPy array.
    - Require that ary is not byte-swapped. If the array is not byte-swapped, return 1. Otherwise, set a Python error and return 0.
- require_dimensions()
    - Return type: ``int``
    - Arguments:
        - ``PyArrayObject*`` ary, a NumPy array.
        - ``int exact_dimensions``, the desired number of dimensions.
    - Require ``ary`` to have a specified number of dimensions. If the array has the specified number of dimensions, return 1. Otherwise, set a Python error and return 0.
- require_dimensions_n()
    - Return type: ``int``
    - Arguments:
        - ``PyArrayObject* ary``, a NumPy array.
        - ``int* exact_dimensions``, an array of integers representing acceptable numbers of dimensions.
        - ``int n``, the length of ``exact_dimensions``.
    - Require ``ary`` to have one of a list of specified number of dimensions. If the array has one of the specified number of dimensions, return 1. Otherwise, set the Python error string and return 0.
- require_size()
    - Return type: ``int``
    - Arguments:
        - ``PyArrayObject* ary``, a NumPy array.
        - ``npy_int* size``, an array representing the desired lengths of each dimension.
        - ``int n``, the length of ``size``.
    - Require ``ary`` to have a specified shape. If the array has the specified shape, return 1. Otherwise, set the Python error string and return 0.
- require_fortran()
    - Return type: ``int``
    - Arguments:
        - ``PyArrayObject* ary``, a NumPy array.
    - Require the given ``PyArrayObject`` to to be Fortran ordered. If the ``PyArrayObject`` is already Fortran ordered, do nothing. Else, set the Fortran ordering flag and recompute the strides.

## Beyond the Provided Typemaps

There are many C or C++ array/NumPy array situations not covered by a simple %include "numpy.i" and subsequent %apply directives.

### A Common Example

Consider a reasonable prototype for a dot product function:

```c
double dot(int len, double* vec1, double* vec2);
```

The Python interface that we want is:

```python
def dot(vec1, vec2):
    """
    dot(PyObject,PyObject) -> double
    """
```

The problem here is that there is one dimension argument and two array arguments, and our typemaps are set up for dimensions that apply to a single array (in fact, [SWIG](http://www.swig.org/) does not provide a mechanism for associating len with vec2 that takes two Python input arguments). The recommended solution is the following:

```c
%apply (int DIM1, double* IN_ARRAY1) {(int len1, double* vec1),
                                      (int len2, double* vec2)}
%rename (dot) my_dot;
%exception my_dot {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
double my_dot(int len1, double* vec1, int len2, double* vec2) {
    if (len1 != len2) {
        PyErr_Format(PyExc_ValueError,
                     "Arrays of lengths (%d,%d) given",
                     len1, len2);
        return 0.0;
    }
    return dot(len1, vec1, vec2);
}
%}
```

If the header file that contains the prototype for **double dot()** also contains other prototypes that you want to wrap, so that you need to **%include** this header file, then you will also need a **%ignore dot**; directive, placed after the **%rename** and before the %include directives. Or, if the function in question is a class method, you will want to use **%extend** rather than **%inline** in addition to **%ignore**.

**A note on error handling**: Note that **my_dot** returns a **double** but that it can also raise a Python error. The resulting wrapper function will return a Python float representation of 0.0 when the vector lengths do not match. Since this is not **NULL**, the Python interpreter will not know to check for an error. For this reason, we add the **%exception** directive above for **my_dot** to get the behavior we want (note that **$action** is a macro that gets expanded to a valid call to **my_dot**). In general, you will probably want to write a SWIG macro to perform this task.

### Other Situations

There are other wrapping situations in which ``numpy.i`` may be helpful when you encounter them.

- In some situations, it is possible that you could use the ``%numpy_typemaps`` macro to implement typemaps for your own types. See the [Other Common Types: bool](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#other-common-types-bool) or [Other Common Types: complex](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#other-common-types-complex) sections for examples. Another situation is if your dimensions are of a type other than int (say long for example):
    ```c
    %numpy_typemaps(double, NPY_DOUBLE, long)
    ```
- You can use the code in ``numpy.i`` to write your own typemaps. For example, if you had a five-dimensional array as a function argument, you could cut-and-paste the appropriate four-dimensional typemaps into your interface file. The modifications for the fourth dimension would be trivial.
- Sometimes, the best approach is to use the ``%extend`` directive to define new methods for your classes (or overload existing ones) that take a ``PyObject*`` (that either is or can be converted to a PyArrayObject*) instead of a pointer to a buffer. In this case, the helper routines in numpy.i can be very useful.
- Writing typemaps can be a bit nonintuitive. If you have specific questions about writing SWIG typemaps for NumPy, the developers of numpy.i do monitor the Numpy-discussion and Swig-user mail lists.

### A Final Note

When you use the ``%apply`` directive, as is usually necessary to use ``numpy.i``, it will remain in effect until you tell [SWIG](http://www.swig.org/) that it shouldn’t be. If the arguments to the functions or methods that you are wrapping have common names, such as length or ``vector``, these typemaps may get applied in situations you do not expect or want. Therefore, it is always a good idea to add a ``%clear`` directive after you are done with a specific typemap:

```c
%apply (double* IN_ARRAY1, int DIM1) {(double* vector, int length)}
%include "my_header.h"
%clear (double* vector, int length);
```

In general, you should target these typemap signatures specifically where you want them, and then clear them after you are done.

## Summary

Out of the box, ``numpy.i`` provides typemaps that support conversion between NumPy arrays and C arrays:
- That can be one of 12 different scalar types: ``signed char``, ``unsigned char``, ``short``, ``unsigned short``, ``int``, ``unsigned int``, ``long``, ``unsigned long``, ``long long``, ``unsigned long long``, ``float`` and ``double``.
- That support 74 different argument signatures for each data type, including:
    - One-dimensional, two-dimensional, three-dimensional and four-dimensional arrays.
    - Input-only, in-place, argout, argoutview, and memory managed argoutview behavior.
    - Hard-coded dimensions, data-buffer-then-dimensions specification, and dimensions-then-data-buffer specification.
    - Both C-ordering (“last dimension fastest”) or Fortran-ordering (“first dimension fastest”) support for 2D, 3D and 4D arrays.

The ``numpy.i`` interface file also provides additional tools for wrapper developers, including:

- A [SWIG](http://www.swig.org/) macro (``%numpy_typemaps``) with three arguments for implementing the 74 argument signatures for the user’s choice of (1) C data type, (2) NumPy data type (assuming they match), and (3) dimension type.

- Fourteen C macros and fifteen C functions that can be used to write specialized typemaps, extensions, or inlined functions that handle cases not covered by the provided typemaps. Note that the macros and functions are coded specifically to work with the NumPy C/API regardless of NumPy version number, both before and after the deprecation of some aspects of the API after version 1.6.