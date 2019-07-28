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

``numpy.i``文件当前位于numpy安装目录下的tools / swig子目录中。 通常，你需要将其复制到开发包装器的目录中。

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

在编译的Python模块中，``import_array()`` 应该只被调用一次。 这可能是你编写的C / C++ 文件，并链接到模块。如果是这种情况，那么你的接口文件都不应该 ``#define SWIG_FILE_WITH_INIT`` 或调用 ``import_array()``。或者，此初始化调用可以位于由 SWIG 从具有上述 ％init 块的接口文件生成的包装文件中。如果是这种情况，并且你有多个 SWIG 接口文件，那么只有一个接口文件应该 ``#define SWIG_FILE_WITH_INIT`` 并调用 ``import_array()``。

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

### 输入数组

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

列出的第一个签名 ``( DATA_TYPE IN_ARRAY[ANY] )``  用于硬编码维度的一维数组。同样，``( DATA_TYPE IN_ARRAY2[ANY][ANY] )`` 是针对二维硬编码维数组的，同样也是针对三维数组的。 

### 就地数组

就地数组定义为就地修改的数组。可以使用也可以不使用输入值，但函数返回时的值很重要。因此，提供的Python参数必须是所需类型的NumPy数组。就地签名是：

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

现在检查这些类型映射以确保``INPLACE_ARRAY``参数使用本机字节排序。如果没有，则引发异常。

对于你希望修改或处理每个元素的想法，还有一个 “平面” 就地数组，无论维数是多少。一个例子是“量化”函数，其就地量化阵列的每个元素，无论是1D，2D还是其他。此表单检查连续性，但允许C或Fortran排序。

N维:

- (DATA_TYPE* INPLACE_ARRAY_FLAT, DIM_TYPE DIM_FLAT)

### Argout数组

Argout数组是出现在C中的输入参数中的数组，但实际上是输出数组。当存在多个输出变量且单个返回参数因此不足时，通常会出现此模式。在Python中，返回多个参数的传统方法是将它们打包成一个序列（元组，列表等）并返回序列。这就是argout类型映射的作用。如果使用这些argout类型映射的包装函数具有多个返回参数，则它们将打包到元组或列表中，具体取决于Python的版本。Python用户不会传递这些数组，只是返回它们。对于指定维度的情况，python用户必须将该维度作为参数提供。argout签名是：

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

这些通常用于以下情况：在 C/C++中，你将在堆上分配一个（n）数组，并调用该函数来填充数组值。在Python中，数组是为你分配的，并作为新的数组对象返回。

注意，我们支持1D中的 ``DATA_TYPE*`` argout的类型，但不支持2D或3D。这是因为[SWIG](http://www.swig.org/)类型图语法的怪癖，是无法避免的。请注意，对于这些类型的1D类型的映射，Python函数将接受一个表示 ``DIM1`` 的参数。

### Arguut 视图数组

Argoutview数组用于C代码向你提供其内部数据的视图，并且不需要用户分配任何内存。这可能很危险。几乎没有办法保证来自C代码的内部数据在封装它的NumPy数组的整个生存期内保持存在。如果用户在销毁NumPy数组之前销毁提供数据视图的对象，那么使用该数组可能会导致错误的内存引用或分段错误。然而，在某些情况下，使用大型数据集时，你没有其他选择。

为argoutview数组包装的C代码的特征是指针：指向维度的指针和指向数据的双指针，这样就可以将这些值传回给用户。因此，argoutview类型图签名就是：

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

请注意，不支持具有硬编码尺寸的数组。这些不能遵循这些类型映射的双指针签名。

内存管理Argout视图阵列

numpy.i最近添加了一些类型映射，它们允许argout数组具有对托管内存的视图。请参阅此处的讨论。

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

### 输出数组

``numpy.i`` 接口文件不支持输出数组的类型图，原因有几个。首先，C/C+返回参数仅限于一个值。这将防止以一般方式获取维度信息。其次，硬编码长度的数组不允许作为返回参数。换言之：

```c
double[3] newVector(double x, double y, double z);
```

不是合法的C/C+语法。因此，我们不能提供以下形式的类型图：

```c
%typemap(out) (TYPE[ANY]);
```

如果遇到函数或方法返回指向数组的指针的情况，最好的办法是编写自己要包装的函数版本，对于类方法的情况使用 %extend 或 ``%ignore`` 和对于函数的情况 ``%rename``。

### 其他常见类型：bool

Note that C++ type ``bool`` is not supported in the list in the [Available Typemaps](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#available-typemaps) section. NumPy bools are a single byte, while the C++ bool is four bytes (at least on my system). Therefore:

```c
%numpy_typemaps(bool, NPY_BOOL, int)
```

将导致生成将引用不正确数据长度的代码的类型映射。你可以实现以下宏扩展：

```c
%numpy_typemaps(bool, NPY_UINT, int)
```

修复数据长度问题，[输入数组](#输入数组)将正常工作，但[就地数组](#就地数组)可能无法进行类型检查。

### 其他常见类型：复杂类型

复杂浮点类型的类型图转换也不受自动支持。这是因为Python和NumPy是用C编写的，C没有本机复杂类型。Python和NumPy都为复杂变量实现了它们自己的(本质上等效的)“struct”定义：

```c
/* Python */
typedef struct {double real; double imag;} Py_complex;

/* NumPy */
typedef struct {float  real, imag;} npy_cfloat;
typedef struct {double real, imag;} npy_cdouble;
```

我们本可以这样：

```c
%numpy_typemaps(Py_complex , NPY_CDOUBLE, int)
%numpy_typemaps(npy_cfloat , NPY_CFLOAT , int)
%numpy_typemaps(npy_cdouble, NPY_CDOUBLE, int)
```

这将为Py_complex，npy_cfloat和npy_cdouble类型的数组提供自动类型转换。但是，似乎不太可能存在任何独立的（非Python，非NumPy）应用程序代码，人们将使用SWIG生成Python接口，这些代码也将这些定义用于复杂类型。 更可能的是，这些应用程序代码将定义自己的复杂类型，或者在 C++ 的情况下，使用std::complex。 假设这些数据结构与Python和NumPy复杂类型兼容，那么上面的 %numpy_typemap 扩展（使用用户的复杂类型替换第一个参数）应该有效。

## NumPy阵列标量和SWIG

SWIG对数值类型进行了复杂的类型检查。例如，如果你的C / C++例程需要一个整数作为输入，SWIG生成的代码将检查Python整数和Python长整数，如果提供的Python整数太大而无法转换为C，则会引发溢出错误 整数。通过在你的Python代码中引入NumPy标量数组，你可以想象从NumPy数组中提取一个整数并尝试将其传递给需要int的SWIG包装的 C/C++ 函数，但SWIG类型检查将无法识别 NumPy数组标量为整数。（通常，这确实有效 - 这取决于NumPy是否识别你正在使用的整数类型继承自你正在使用的平台上的Python整数类型。有时，这意味着在32位计算机上运行的代码将 在64位计算机上失败。）

如果你收到如下所示的Python错误：

```python
TypeError: in method 'MyClass_MyMethod', argument 2 of type 'int'
```

你传递的参数是从NumPy数组中提取的整数，然后你偶然发现了这个问题。解决方案是修改 SWIG 类型转换系统以接受除标准整数类型之外的NumPy数组标量。幸运的是，已经为你提供了此功能。只需复制文件：

```
pyfragments.swg
```

到你项目的工作构建目录，这个问题就能修复。建议你一定要这样做，因为它只会增加Python接口的功能。

### 为什么有第二个文件？

SWIG类型检查和转换系统是C宏，SWIG宏，SWIG类型映射和SWIG片段的复杂组合。 片段是一种在需要时有条件地将代码插入到包装器文件中的方法，如果不需要则不插入它。 如果多个类型映射需要相同的片段，则片段只会插入到包装器代码中一次。

有一个片段用于将Python整数转换为C``long``。 有一个不同的片段将Python整数转换为C``int``，它调用 ``long`` 片段中定义的例程。我们可以通过更改``long``片段的定义来进行我们想要的更改。 SWIG使用“先到先得”系统确定片段的活动定义。 也就是说，我们需要在SWIG内部执行之前为``long``转换定义片段。 SWIG允许我们通过将我们的片段定义放在文件``pyfragments.swg``中来实现这一点。 如果我们将新的片段定义放在``numpy.i``中，它们将被忽略。

## 帮助功能

``numpy.i`` 文件包含几个内部用于构建其类型映射的宏和例程。 但是，这些函数在接口文件的其他位置可能很有用。 这些宏和例程以片段形式实现，这在前一节中有简要描述。 如果你尝试使用以下一个或多个宏或函数，但你的编译器抱怨它无法识别该符号，那么你需要强制使用以下代码在代码中显示这些片段：

```c
%fragment("NumPy_Fragments");
```

在你的SWIG界面文件中。

### 宏

- is_array(a) 如果``a``是非``NULL``则为true，并且可以强制转换为``PyArrayObject*``。
- array_type(a) 计算a的整数数据类型代码，假设可以转换为``PyArrayObject*``。
- array_numdims(a) 计算a的整数维数，假设a可以强制转换为``PyArrayObject*``。
- array_dimensions(a) 计算一个类型为``npy_intp``和长度 ``array_numdims(a)`` 的数组，给出``a``的所有维度的长度，假设``a``可以强制转换为``PyArrayObject*``。
- array_size(a,i) 计算a的第i个维度大小，假设可以转换为PyArrayObject*。
- array_strides(a) 求值为npy_intp和length array_numdims（a）类型的数组，给出a的所有维度的stridess，假设a可以转换为PyArrayObject*。步幅是元素与其直接邻居沿同一轴的字节距离。
- array_stride(a,i) 评估a的第i步，假设可以转换为PyArrayObject*。
- array_data(a) 求值为指向a的数据缓冲区的void*类型的指针，假设可以转换为PyArrayObject*。
- array_descr(a) 返回对a的dtype属性（PyArray_Descr*）的借用引用，假设可以强制转换为PyArrayObject*。
- array_flags(a) 返回一个表示a的标志的整数，假设可以强制转换为PyArrayObject*。
- array_enableflags(a,f) 设置由f的f表示的标志，假设可以转换为PyArrayObject*。
- array_is_contiguous(a) 如果a是连续数组，则求值为true。 相当于(PyArray_ISCONTIGUOUS(a)）。
- array_is_native(a) 如果数据缓冲区使用本机字节顺序，则计算结果为true。等价于(PyArrayISNOTSWAPPED(A)。
- array_is_fortran(a) 如果a是FORTRAN顺序的，则计算为true。

### API

- pytype_string()
    - 返回类型： ``const char*``
    - 参数：
        - ``PyObject* py_obj``, 一个普通的Python对象。
    - 返回描述`py_obj`类型的字符串。
- typecode_string()
    - 返回类型： ``const char*``
    - 参数： 
        - ``int typecode``, NumPy整数类型码。
    - 返回一个字符串，该字符串描述与NumPy`Typeecode`相对应的类型。
- type_match()
    - 返回类型： int
    - 参数：
        - ``int actual_type``, NumPy数组的NumPy类型代码。
        - ``int desired_type``, 所需的NumPy类型码。
    - 确保``actual_type``与``desired_type``兼容。 例如，这允许匹配字符和字节类型，或int和long类型。这现在相当于``PyArray_EquivTypenums()``。
- obj_to_array_no_conversion()
    - 返回类型： ``PyArrayObject*``
    - 参数：
        - ``PyObject\* input``, 一般的Python对象。
        - ``int typecode``, 所需的NumPy类型代码。
    - 如果合法，将输入转换为PyArrayObject*，并确保它的类型为typecode。 如果无法转换输入，或者typecode错误，请设置Python错误并返回NULL。
- obj_to_array_allow_conversion()
    - 返回类型： ``PyArrayObject*``
    - 参数：
        - ``PyObject\* input``, 一般的Python对象。
        - ``int typecode``, 生成的数组所需的NumPy类型代码。
        - ``int* is_new_object``, 如果没有执行转换，则返回值0，否则返回1。
    - Convert ``input`` 到具有给定typecode的NumPy数组。 成功后，返回有效。
    ``PyArrayObject *``具有正确的类型。 失败时，将设置Python错误字符串，例程返回“NULL”。
- make_contiguous()
    - 返回类型： ``PyArrayObject*``
    - 参数：
        - ``PyArrayObject* ary``, 一种 NumPy 数组.
        - ``int* is_new_object``, 如果没有执行转换，则返回值0，否则返回1。
        - ``int min_dims``, 最小允许尺寸。
        - ``int max_dims``, 最大允许尺寸。
    - 检查ary是否连续。 如果是这样，则返回输入指针并将其标记为不是新对象。 如果它不连续，则使用原始数据创建一个新的PyArrayObject *，将其标记为新对象并返回指针。
- make_fortran()
    - 返回类型： ``PyArrayObject*``
    - Arguments
        - ``PyArrayObject* ary``,一个NumPy数组。
        - ``int* is_new_object``, 如果没有执行转换，则返回值0，否则返回1。
    - 检查ary是否是Fortran连续的。 如果是这样，则返回输入指针并将其标记为不是新对象。 如果它不是Fortran连续的，则使用原始数据创建一个新的PyArrayObject *，将其标记为新对象并返回指针。
- obj_to_array_contiguous_allow_conversion()
    - 返回类型： ``PyArrayObject*``
    - 参数：
        - ``PyObject\* input``, 一般的Python对象。
        - ``int typecode``, 生成的数组所需的NumPy类型代码。
        - ``int* is_new_object``, 如果没有执行转换，则返回值0，否则返回1。
    - 将``input``转换为指定类型的连续``PyArrayObject *``。 如果输入对象不是连续的``PyArrayObject *``，则将创建一个新对象，并设置新的对象标志。
- obj_to_array_fortran_allow_conversion()
    - 返回类型： ``PyArrayObject*``
    - 参数：
        - ``PyObject\* input``, 一般的Python对象。
        - ``int typecode``, 生成的数组所需的NumPy类型代码。
        - ``int* is_new_object``, 如果没有执行转换，则返回值0，否则返回1。
    - 将``input``转换为指定类型的Fortran连续``PyArrayObject*``。 如果输入对象不是Fortran连续的``PyArrayObject*``，则将创建一个新对象，并设置新的对象标志。
- require_contiguous()
    - 返回类型： ``int``
    - 参数：
        - ``PyArrayObject* ary``, 一个NumPy数组。
    - 测试ary是否连续。 如果是，则返回1.否则，设置Python错误并返回0。
- require_native()
    - 返回类型： ``int``
    - 参数：
        - ``PyArray_Object*`` ary，NumPy数组。
    - 要求ary不进行字节交换。 如果数组不是字节交换的，则返回1.否则，设置Python错误并返回0。
- require_dimensions()
    - 返回类型： ``int``
    - 参数：
        - ``PyArrayObject*`` ary，NumPy数组。
        - ``int exact_dimensions``, 所需的尺寸数量。
    - 要求``ary``具有指定数量的尺寸。 如果数组具有指定的维数，则返回1.否则，设置Python错误并返回0。
- require_dimensions_n()
    - 返回类型： ``int``
    - 参数：
        - ``PyArrayObject* ary``, 一个NumPy数组。
        - ``int* exact_dimensions``, 表示可接受维数的整数数组。
        - ``int n``, ``exact_dimensions``的长度。
    - 要求``ary``具有指定维数的列表之一。 如果数组具有指定数量的维度之一，则返回1.否则，设置Python错误字符串并返回 0。
- require_size()
    - 返回类型： ``int``
    - 参数：
        - ``PyArrayObject* ary``, 一个NumPy数组。
        - ``npy_int* size``, 表示每个维度的所需长度的数组。
        - ``int n``, the length of ``size``.
    - 要求``ary``具有指定的形状。 如果数组具有指定的形状，则返回1.否则，设置Python错误字符串并返回0。
- require_fortran()
    - 返回类型： ``int``
    - 参数：
        - ``PyArrayObject* ary``, 一个NumPy数组。
    - 要求给定的``PyArrayObject``是Fortran命令。 如果``PyArrayObject``已经被Fortran命令，那么什么都不做。 否则，设置Fortran排序标志并重新计算步幅。

## 超出所提供的类型

有许多C或C ++数组/ NumPy数组情况不包含在简单的 %include “numpy.i” 和后续的 %apply 指令中。

### 一个常见的例子

考虑点积函数的合理原型：

```c
double dot(int len, double* vec1, double* vec2);
```

我们想要的Python接口是：

```python
def dot(vec1, vec2):
    """
    dot(PyObject,PyObject) -> double
    """
```

这里的问题是有一个维度参数和两个数组参数，我们的类型映射是为适用于单个数组的维度设置的（事实上，SWIG 没有 提供一种机制，用于将len与带有两个Python输入参数的vec2相关联。 建议的解决方案如下：

这里的问题是有一个维度参数和两个数组参数，我们的类型映射是为适用于单个数组的维度设置的（事实上，SWIG没有提供一种机制，用于将len与vec2相关联，后者带有两个Python输入参数）。建议的解决方案如下：

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

如果包含**double dot()**原型的头文件还包含你要包装的其他原型，那么你需要 **%include** 这个头文件，那么你还需要一个 **%ignore dot**; 指令，放在 **%rename** 之后和 %include 指令之前。 或者，如果所讨论的函数是类方法，除了 **%ignore** 之外，你还需要使用 **%extend** 而不是 **%inline**。

**关于错误处理的注释**：注意 **my_dot** 返回 **double** 但它也会引发Python错误。当向量长度不匹配时，生成的包装函数将返回0.0的Python浮点表示形式。 由于这不是 **NULL**，因此Python解释器不会知道检查错误。出于这个原因，我们在 **my_dot** 上面添加 **%exception** 指令以获得我们想要的行为（注意 **$action** 是一个宏，它被扩展为对 **my_dot** 的有效调用）。通常，你可能希望编写SWIG宏来执行此任务。
 
### 其他情况

还有其他包装情况，当你遇到它们时，``numpy.i`` 可能会有所帮助。

- 在某些情况下，可以使用 ``%numpy_ypemaps`` 宏为你自己的类型实现类型映射。有关示例，请参阅[其他通用类型：bool](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#other-common-types-bool) 或 [其他通用类型：复杂部分](https://docs.scipy.org/doc/numpy/reference/swig.interface-file.html#other-common-types-complex) 。另一种情况是，如果维度的类型不是int(例如，Long)：
    ```c
    %numpy_typemaps(double, NPY_DOUBLE, long)
    ```
- 可以使用``numpy.i``中的代码编写自己的类型图。例如，如果你有一个五维数组作为函数参数，你可以剪切并粘贴适当的四维类型图到你的接口文件中。对第四维的修改将是微不足道的。
- 有时，最好的方法是使用 ``%ext`` 指令为你的类(或重载现有的)定义新方法，这些方法采用 ``PyObject*`` (既可以是或可以转换为PyArrayObject*)，而不是指向缓冲区的指针。在这种情况下，numpy.i 中的帮助程序例程非常有用。
- 编写类型图可能有点不直观。如果你有关于为NumPy编写SWIG类型图的具体问题，numpy.i的开发人员确实会监视Numpy讨论和SWIG用户邮件列表。

### 最后说明

当你使用 ``%apply`` 指令(这通常是使用 ``numpy.i`` 所必需的)时，它将一直有效，直到你告诉 SWIG 说不应该这样做)为止。如果要包装的函数或方法的参数具有公共名称，如长度或 ``vector``，则这些类型映射可能应用于你不希望或不希望出现的情况。因此，在完成特定的类型地图之后，添加一个 ``%clear`` 指令总是一个好主意：

```c
%apply (double* IN_ARRAY1, int DIM1) {(double* vector, int length)}
%include "my_header.h"
%clear (double* vector, int length);
```

通常，你应该针对这些类型地图签名，特别是你想要它们的地方，然后在你完成之后清除它们。

## 总结

开箱即用，`numpy.i`提供支持NumPy数组和C数组之间转换的类型图：
- T它可以是12种不同的标量类型之一：``signed char``, ``unsigned char``, ``short``, ``unsigned short``, ``int``, ``unsigned int``, ``long``, ``unsigned long``, ``long long``, ``unsigned long long``, ``float`` 和 ``double``.
- 它为每种数据类型支持74个不同的参数签名，包括：
    - 一维、二维、三维和四维阵列。
    - 仅输入、就地、argout、argoutview和内存管理的argoutview行为.
    - 硬编码尺寸数据缓冲器然后尺寸规范和尺寸然后数据缓冲区规格。
    - C排序(“最后维最快”)或Fortran排序(“第一维最快”)都支持2D、3D和4D数组。

``numpy.i`` 接口文件还为包装程序开发人员提供了其他工具，包括：

- 带有三个参数的SWIG宏(``%numpy_ypemap``)，用于实现用户选择的(1) C数据类型、(2)NumPy数据类型(假设它们匹配)和 (3)维度类型的74个参数签名。

- 14个C宏和15个C函数，可用于编写专门的类型图、扩展或内联函数，这些函数处理所提供的类型图未涵盖的情况。请注意，宏和函数是专门为与NumPy C/API一起工作而编写的，不管NumPy的版本号是多少，在1.6版之后API的某些方面被弃用之前和之后都是如此。