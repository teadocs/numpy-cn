# 如何扩展NumPy

> That which is static and repetitive is boring. That which is dynamic and random is confusing. In between lies art.
> 
> — John A. Locke

> Science is a differential equation. Religion is a boundary condition.
> 
> — Alan Turing

## 编写扩展模块

虽然ndarray对象旨在允许在Python中进行快速计算，但它也被设计为通用的并且满足各种各样的计算需求。因此，如果绝对速度是必不可少的，那么就不能替换特定于您的应用程序和硬件的精心编制的循环。这是numpy包含f2py的原因之一，因此可以使用易于使用的机制将（简单的）C / C ++和（任意）Fortran代码直接链接到Python中。我们鼓励您使用和改进此机制。本节的目的不是记录此工具，而是记录编写此工具所依赖的扩展模块的更基本步骤。

当扩展模块被编写，编译并安装到Python路径（sys.path）中的某个位置时，可以将代码导入到Python中，就好像它是标准的python文件一样。它将包含已在C代码中定义和编译的对象和方法。在Python中执行此操作的基本步骤已有详细记录，您可以在[www.python.org](https://www.python.org)上的在线文档中找到更多信息。

除了Python C-API之外，NumPy还有一个完整而丰富的C-API，允许在C级上进行复杂的操作。但是，对于大多数应用程序，通常只使用少量API调用。如果你需要做的就是提取一个指向内存的指针以及一些形状信息以传递给另一个计算例程，那么你将使用非常不同的调用，然后如果你试图创建一个类似于数组的新类型或添加一个新数据ndarrays的类型。本章介绍了最常用的API调用和宏。

## 必需的子程序

必须在C代码中定义一个函数才能使Python将其用作扩展模块。该函数必须被称为init {name}，其中{name}是Python中模块的名称。必须声明此函数，以便对例程外部的代码可见。除了添加您想要的方法和常量之外，此子例程还必须包含调用``import_array()``
和/或``import_ufunc()``取决于需要哪个C-API。只要实际调用任何C-API子例程，忘记放置这些命令就会将自身显示为一个丑陋的分段错误（崩溃）。实际上，在单个文件中可以有多个init {name}函数，在这种情况下，该文件将定义多个模块。但是，有一些技巧可以让它正常工作，这里没有涉及。

一个最小的``init{name}``方法看起来像：

``` c
PyMODINIT_FUNC
init{name}(void)
{
   (void)Py_InitModule({name}, mymethods);
   import_array();
}
```

mymethods必须是PyMethodDef结构的数组（通常是静态声明的），它包含方法名，实际的C函数，指示方法是否使用关键字参数的变量，以及docstrings。这些将在下一节中介绍。如果要向模块添加常量，则存储Py_InitModule的返回值，Py_InitModule是一个模块对象。向模块添加项目的最常用方法是使用PyModule_GetDict（模块）获取模块字典。使用模块字典，您可以手动将任何您喜欢的内容添加到模块中。向模块添加对象的更简单方法是使用三个额外的Python C-API调用之一，这些调用不需要单独提取模块字典。这些内容记录在Python文档中，但为方便起见，在此处重复：


int ``PyModule_AddObject``（[PyObject](https://docs.python.org/dev/c-api/structures.html#c.PyObject) *   *module* ，char *   *name* ，[PyObject](https://docs.python.org/dev/c-api/structures.html#c.PyObject) *   *value*  ）[¶](#c.PyModule_AddObject)


int ``PyModule_AddIntConstant``（[PyObject](https://docs.python.org/dev/c-api/structures.html#c.PyObject) *   *module* ，char *   *name* ，long   *value*  ）[¶](#c.PyModule_AddIntConstant)


int ``PyModule_AddStringConstant``（[PyObject](https://docs.python.org/dev/c-api/structures.html#c.PyObject) *   *module* ，char *   *name* ，char *   *value*  ）[¶](#c.PyModule_AddStringConstant)

所有这三个函数都需要 *模块* 对象（Py_InitModule的返回值）。该 *名称* 是标签模块中的值的字符串。根据调用的函数， *value* 参数是一般对象（[``PyModule_AddObject``](#c.PyModule_AddObject)窃取对它的引用），整数常量或字符串常量。

## 定义函数

传递给Py_InitModule函数的第二个参数是一个结构，可以很容易地在模块中定义函数。在上面给出的示例中，mymethods结构将在文件的早期（通常在init {name}子例程之前）定义为：

``` c
static PyMethodDef mymethods[] = {
    { nokeywordfunc,nokeyword_cfunc,
      METH_VARARGS,
      Doc string},
    { keywordfunc, keyword_cfunc,
      METH_VARARGS|METH_KEYWORDS,
      Doc string},
    {NULL, NULL, 0, NULL} /* Sentinel */
}
```

mymethods数组中的每个条目都是一个[``PyMethodDef``](https://docs.python.org/dev/c-api/structures.html#c.PyMethodDef)结构，包含1）Python名称，2）实现函数的C函数，3）指示是否接受此函数的关键字的标志，以及4）函数的文档字符串。通过向该表添加更多条目，可以为单个模块定义任意数量的功能。最后一个条目必须全部为NULL，如图所示充当哨兵。Python查找此条目以了解已定义模块的所有函数。

完成扩展模块必须做的最后一件事是实际编写执行所需功能的代码。有两种函数：不接受关键字参数的函数和那些函数。

### 没有关键字参数的函数

不接受关键字参数的函数应写为：

``` c
static PyObject*
nokeyword_cfunc (PyObject *dummy, PyObject *args)
{
    /* convert Python arguments */
    /* do function */
    /* return something */
}
```

伪参数不在此上下文中使用，可以安全地忽略。该 *ARGS* 参数包含所有的传递给函数作为一个元组的参数。此时您可以执行任何操作，但通常管理输入参数的最简单方法是调用[``PyArg_ParseTuple``](https://docs.python.org/dev/c-api/arg.html#c.PyArg_ParseTuple)（args，format_string，addresses_to_C_variables ...）或[``PyArg_UnpackTuple``](https://docs.python.org/dev/c-api/arg.html#c.PyArg_UnpackTuple)（元组，“名称”，分钟，最大，......）。有关如何使用第一个函数的详细说明，请参见Python C-API参考手册第5.5节（解析参数和构建值）。您应该特别注意使用转换器函数在Python对象和C对象之间进行的“O＆”格式。所有其他格式函数都可以（大部分）被认为是这个一般规则的特殊情况。NumPy C-API中定义了几种可能有用的转换器功能。特别是，该[``PyArray_DescrConverter``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_DescrConverter)
函数对于支持任意数据类型规范非常有用。此函数将任何有效的数据类型Python对象转换为
 对象。请记住传入应填写的C变量的地址。[``PyArray_Descr *``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Descr)

有很多关于如何在[``PyArg_ParseTuple``](https://docs.python.org/dev/c-api/arg.html#c.PyArg_ParseTuple)
整个NumPy源代码中使用的示例。标准用法是这样的：

``` c
PyObject *input;
PyArray_Descr *dtype;
if (!PyArg_ParseTuple(args, "OO&", &input,
                      PyArray_DescrConverter,
                      &dtype)) return NULL;
```

请务必记住，在使用“O”格式字符串时，您会获得对该对象的 *借用* 引用。但是，转换器功能通常需要某种形式的内存处理。在此示例中，如果转换成功，则 *dtype* 将保持对对象的新引用，而 *输入* 将保留借用的引用。因此，如果此转换与另一个转换（比如整数）混合并且数据类型转换成功但整数转换失败，那么您需要在返回之前将引用计数释放到数据类型对象。一种典型的方法是
在调用之前将 *dtype* 设置为，然后
在 *dtype上* 使用[``PyArray_Descr *``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Descr) **  ** ``NULL``[``PyArg_ParseTuple``](https://docs.python.org/dev/c-api/arg.html#c.PyArg_ParseTuple)[``Py_XDECREF``](https://docs.python.org/dev/c-api/refcounting.html#c.Py_XDECREF) **  回来之前。

处理完输入参数后，将编写实际完成工作的代码（可能会根据需要调用其他函数）。C函数的最后一步是返回一些东西。如果遇到错误，``NULL``则应返回（确保实际设置了错误）。如果不应返回任何内容，则递增
 [``Py_None``](https://docs.python.org/dev/c-api/none.html#c.Py_None)并返回它。如果应该返回单个对象，则返回它（确保您首先拥有对它的引用）。如果应该返回多个对象，那么您需要返回一个元组。该[``Py_BuildValue``](https://docs.python.org/dev/c-api/arg.html#c.Py_BuildValue)（format_string，c_variables ...）函数可以很容易地从C变量构建Python对象的元组。请特别注意格式字符串中“N”和“O”之间的区别，否则您可能很容易造成内存泄漏。'O'格式字符串增加它对应的C变量的引用计数，而'N'格式字符串窃取对相应C变量的引用。如果已经为对象创建了引用并且只想对元组进行引用，则应使用“N”。如果您只有一个对象的借用引用并且需要创建一个来提供元组，则应该使用“O”。[``PyObject *``](https://docs.python.org/dev/c-api/structures.html#c.PyObject)[``PyObject *``](https://docs.python.org/dev/c-api/structures.html#c.PyObject)

### 带关键字参数的函数

这些函数与没有关键字参数的函数非常相似。唯一的区别是函数签名是：

``` c
static PyObject*
keyword_cfunc (PyObject *dummy, PyObject *args, PyObject *kwds)
{
...
}
```

kwds参数包含一个Python字典，其键是关键字参数的名称，其值是相应的关键字参数值。无论你认为合适，都可以处理这本字典。然而，处理它的最简单方法是[``PyArg_ParseTuple``](https://docs.python.org/dev/c-api/arg.html#c.PyArg_ParseTuple)用[``PyArg_ParseTupleAndKeywords``](https://docs.python.org/dev/c-api/arg.html#c.PyArg_ParseTupleAndKeywords)（args，kwds，format_string，char * kwlist []，地址......）调用替换
 （args，format_string，addresses ...）函数。此函数的kwlist参数是一个``NULL``字符串数组，提供了预期的关键字参数。format_string中的每个条目都应该有一个字符串。如果传入无效的关键字参数，则使用此函数将引发TypeError。

有关此功能的更多帮助，请参阅Python文档中的扩展和嵌入教程的第1.8节（扩展函数的关键字参数）。

### 引用计数

编写扩展模块时最大的困难是引用计数。这是f2py，weave，Cython，ctypes等受欢迎的重要原因。如果您错误处理引用计数，则可能会出现从内存泄漏到分段错误的问题。我知道处理参考计数的唯一策略是血液，汗水和眼泪。首先，你强迫每个Python变量都有一个引用计数。然后，您可以准确了解每个函数对对象的引用计数的作用，以便在需要时可以正确使用DECREF和INCREF。引用计数可以真正测试您对编程工艺的耐心和勤奋程度。尽管形象严峻，大多数引用计数的情况非常简单，最常见的困难是由于某些错误而在从例程退出之前不在对象上使用DECREF。第二，是不会在传递给将要窃取引用的函数或宏的对象上拥有引用的常见错误（ *例如*  [``PyTuple_SET_ITEM``](https://docs.python.org/dev/c-api/tuple.html#c.PyTuple_SET_ITEM)，和大多数采取[``PyArray_Descr``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Descr)对象的功能）。

通常，在创建变量时会获得对变量的新引用，或者是某个函数的返回值（但是有一些突出的例外 - 例如从元组或字典中获取项目）。当您拥有引用时，您有责任确保[``Py_DECREF``](https://docs.python.org/dev/c-api/refcounting.html#c.Py_DECREF)在不再需要该变量时调用（var）（并且没有其他函数“窃取”其引用）。此外，如果您将Python对象传递给将“窃取”引用的函数，那么您需要确保拥有它（或用于[``Py_INCREF``](https://docs.python.org/dev/c-api/refcounting.html#c.Py_INCREF)获取自己的引用）。您还将遇到借用参考的概念。借用引用的函数不会改变对象的引用计数，也不会期望“保持”引用。它只是暂时使用该对象。当你使用[``PyArg_ParseTuple``](https://docs.python.org/dev/c-api/arg.html#c.PyArg_ParseTuple)或者
 [``PyArg_UnpackTuple``](https://docs.python.org/dev/c-api/arg.html#c.PyArg_UnpackTuple)您收到对元组中对象的借用引用，不应更改其函数内的引用计数。通过练习，您可以学会正确引用计数，但一开始可能会令人沮丧。

引用计数错误的一个常见来源是[``Py_BuildValue``](https://docs.python.org/dev/c-api/arg.html#c.Py_BuildValue)
函数。请特别注意'N'格式字符和'O'格式字符之间的区别。如果在子例程中创建一个新对象（例如输出数组），并且在返回值的元组中将其传回，则最应该使用“N”格式字符。[``Py_BuildValue``](https://docs.python.org/dev/c-api/arg.html#c.Py_BuildValue)。“O”字符将引用计数增加1。这将为调用者提供一个全新数组的两个引用计数。删除变量并且引用计数减1时，仍会有额外的引用计数，并且永远不会释放该数组。您将有一个引用计数引起的内存泄漏。使用'N'字符将避免这种情况，因为它将使用单个引用计数返回给调用者一个对象（在元组内）。

## 处理数组对象

NumPy的大多数扩展模块都需要访问ndarray对象（或其中一个子类）的内存。最简单的方法不需要您了解NumPy的内部结构。方法是

1. 确保处理的是行为良好的数组(按机器字节顺序和单段对齐)，具有正确的维数类型和数量。
    1. 通过使用PyArray_FromAny或在其上构建的宏将其从某个Python对象转换。
    1. 通过使用PyArray_NewFromDescr或基于它的更简单的宏或函数构建所需形状和类型的新ndarray。
1. 获取数组的形状和指向其实际数据的指针。
1. 将数据和形状信息传递给子例程或实际执行计算的其他代码部分。
1. 如果您正在编写算法，那么我建议您使用数组中包含的步幅信息来访问数组的元素(PyArray_GetPtr宏使这一过程变得轻松)。然后，您可以放松您的要求，这样就不会强制使用单段数组和可能导致的数据复制。

这些子主题中的每一个都会在下面的小节中介绍。

### 转换任意序列对象

从任何可以转换为数组的Python对象获取数组的主程序是[``PyArray_FromAny``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_FromAny)。这个函数非常灵活，有许多输入参数。几个宏使它更容易使用基本功能。[``PyArray_FROM_OTF``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_FROM_OTF)可以说是最常用的这些宏中最有用的。它允许您将任意Python对象转换为特定内置数据类型（ *例如*  float）的数组，同时指定一组特定的需求（ *例如，* 连续，对齐和可写）。语法是

- [``PyArray_FROM_OTF``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_FROM_OTF)

  从任何可以转换为数组的 Python 对象 *obj* 返回 ndarray。返回数组中的维数由对象确定。
  返回数组的所需数据类型在 *typenum* 中提供，它应该是枚举类型之一。
  对返回数组的*要求（requirements）*可以是标准数组标志的任意组合。下面将更详细地解释这些论点中的每一个。
  成功后，您将收到对数组的新引用。如果失败，则返回 ``NULL`` 并设置异常。

  - **obj**

    该对象可以是任何可转换为ndarray的Python对象。
    如果对象已经是满足要求的ndarray的子类，则返回一个新的引用。否则，构造一个新的数组。
    除非使用数组接口，否则将obj的内容复制到新数组，以便不必复制数据。
    可以转换为数组的对象包括：
    1)任何嵌套的Sequence对象，
    2)暴露数组接口的任何对象，
    3)具有[数组](/reference/arrays/classes.html#numpy.class.__array__)方法的任何对象(应该返回ndarray)，
    以及4)任何标量对象(变成零维数组)。
    否则符合要求的ndarray的子类将被传递。如果要确保基类ndarray，
    则在Requirements标志中使用 [NPY_ARRAY_ENSUREARRAY](https://www.numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_ENSUREARRAY)。
    只有在必要时才会制作副本。
    如果要保证复制，则将 [NPY_ARRAY_ENSURECOPY](https://www.numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_ENSURECOPY) 传递给Requirements标志。

  - **typenum**

    枚举类型之一或[``NPY_NOTYPE``](https://numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_NOTYPE)(如果数据类型应从对象本身确定)。可以使用基于C的名称：

    [NPY_BOOL](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_BOOL), [NPY_BYTE](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_BYTE), [NPY_UBYTE](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_UBYTE), [NPY_SHORT](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_SHORT), [NPY_USHORT](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_USHORT), [NPY_INT](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_INT), [NPY_UINT](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_UINT), [NPY_LONG](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_LONG), [NPY_ULONG](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_ULONG), [NPY_LONGLONG](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_LONGLONG), [NPY_ULONGLONG](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_ULONGLONG), [NPY_DOUBLE](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_DOUBLE), [NPY_LONGDOUBLE](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_LONGDOUBLE), [NPY_CFLOAT](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_CFLOAT), [NPY_CDOUBLE](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_CDOUBLE), [NPY_CLONGDOUBLE](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_CLONGDOUBLE), [NPY_OBJECT](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_OBJECT).

    或者，可以使用平台上支持的位宽名称。例如：

    [NPY_INT8](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_INT8), [NPY_INT16](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_INT16), [NPY_INT32](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_INT32), [NPY_INT64](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_INT64), [NPY_UINT8](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_UINT8), [NPY_UINT16](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_UINT16), [NPY_UINT32](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_UINT32), [NPY_UINT64](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_UINT64), [NPY_FLOAT32](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_FLOAT32), [NPY_FLOAT64](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_FLOAT64), [NPY_COMPLEX64](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_COMPLEX64), [NPY_COMPLEX128](https://www.numpy.org/devdocs/reference/c-api/dtype.html#c.NPY_COMPLEX128).

    仅当可以在不丢失精度的情况下完成时，对象才会转换为所需的类型。
    否则将返回 ``NULL`` 并引发错误。
    在Requirements标志中使用 [``NPY_ARRAY_FORCECAST``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_FORCECAST) 覆盖此行为。

  - **requirements**

    ndarray的存储器模型允许每个维度上的任意步幅前进到数组的下一个元素。
    然而，通常需要与需要C连续或Fortran连续内存布局的代码进行接口。
    此外，ndarray可能未对齐(元素的地址不是元素大小的整数倍)，如果您尝试将指针取消引用到数组数据，
    这可能会导致程序崩溃(或至少工作速度更慢)。
    这两个问题都可以通过将Python对象转换为一个数组来解决，
    该数组对于您的特定用法来说更“表现良好”。

    Requirements标志允许指定哪种类型的数组是可接受的。
    如果传入的对象不满足此要求，则创建一个副本，以便返回的对象将满足这些要求。
    这些ndarray可以使用非常通用的内存指针。此标志允许指定返回的数组对象的所需属性。
    在详细的API一章中解释了所有的标志。最常用的标志是 [``NPY_ARRAY_IN_ARRAY``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_IN_ARRAY)、[``NPY_OUT_ARRAY``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_OUT_ARRAY)、and [``NPY_ARRAY_INOUT_ARRAY``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_INOUT_ARRAY):

    [``NPY_ARRAY_IN_ARRAY``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_IN_ARRAY)
    
    此标志对于必须按C连续顺序和对齐的数组非常有用。这些类型的数组通常是某些算法的输入数组。

    [``NPY_ARRAY_OUT_ARRAY``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_OUT_ARRAY)

    此标志用于指定C连续顺序的数组，该数组是对齐的，并且也可以写入。这样的数组通常作为输出返回(尽管通常这样的输出数组是从头开始创建的)。

    [``NPY_ARRAY_INOUT_ARRAY``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_INOUT_ARRAY)
    
    此标志用于指定将用于输入和输出的数组。
    必须在接口例程末尾的 [Py_DECREF](https://docs.python.org/dev/c-api/refcounting.html#c.Py_DECREF) 之前调用 [PyArray_ResolveWritebackIfCopy](https://www.numpy.org/devdocs/reference/c-api/array.html#c.PyArray_ResolveWritebackIfCopy) ，
    以将临时数据写回传入的原始数组。
    使用 [NPY_ARRAY_WRITEBACKIFCOPY](https://www.numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_WRITEBACKIFCOPY) 或 
    [NPY_ARRAY_UPDATEIFCOPY](https://www.numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_UPDATEIFCOPY) 标志要求输入对象已经是数组(因为其他对象不能以这种方式自动更新)。
    如果发生错误，请在设置了这些标志的数组上使用 [PyArray_DiscardWritebackIfCopy](https://www.numpy.org/devdocs/reference/c-api/array.html#c.PyArray_DiscardWritebackIfCopy)(obj)。
    这将设置底层基本数组可写，而不会导致内容复制回原始数组。

    可以作为附加要求进行OR运算的其他有用标志包括：

    [``NPY_ARRAY_FORCECAST``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_FORCECAST)
    
    强制转换为所需的类型，即使在不丢失信息的情况下也是如此。
    
    [``NPY_ARRAY_ENSURECOPY``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_ENSURECOPY)

    确保生成的数组是原始数组的副本。

    [``NPY_ARRAY_ENSUREARRAY``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_ENSUREARRAY)

    确保生成的对象是实际的ndarray，而不是子类。


::: tip 注意

数组是否进行字节交换取决于数组的数据类型。始终请求本机字节顺序数组[``PyArray_FROM_OTF``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_FROM_OTF)，因此[``NPY_ARRAY_NOTSWAPPED``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_NOTSWAPPED)require参数中不需要标志。也无法从此例程中获取字节交换数组。

:::

### 创建一个全新的ndarray 

通常，必须在扩展模块代码中创建新数组。也许需要输出数组，
并且您不希望调用者必须提供它。
也许只需要一个临时数组来进行中间计算。
无论需要什么，都需要简单的方法来获得任何数据类型的ndarray对象。
这样做最常见的功能是[``PyArray_NewFromDescr``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_NewFromDescr)。
所有数组创建函数都经过这个重复使用的代码。由于其灵活性，使用起来可能有点混乱。
结果，存在更易于使用的更简单的形式。这些表单是[``PyArray_SimpleNew``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_SimpleNew)函数族的一部分
 ，它通过为常见用例提供默认值来简化界面。

### 获取ndarray内存并访问ndarray的元素

如果obj是一个 ndarray()，那么ndarray 的数据区域由void *指针（obj）或char *指针（obj）指向。请记住（通常）此数据区域可能未根据数据类型对齐，它可能表示字节交换数据，和/或可能无法写入。如果数据区域是对齐的并且是以本机字节顺序排列的，那么如何获取数组的特定元素只能由npy_intp变量数组（obj）确定。特别是，这个整数的c数组显示了必须向当前元素指针添加多少**字节**才能到达每个维度中的下一个元素。对于小于4维的数组，有[``PyArrayObject *``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArrayObject)[``PyArray_DATA``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_DATA)[``PyArray_BYTES``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_BYTES)[``PyArray_STRIDES``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_STRIDES)****``PyArray_GETPTR{k}``
（obj，...）宏，其中{k}是整数1,2,3或4，这使得使用数组步幅更容易。争论...... 将{k}非负整数索引表示到数组中。例如，假设``E``是一个三维的ndarray。``E[i,j,k]``
获得元素的（void *）指针作为[``PyArray_GETPTR3``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_GETPTR3)（E，i，j，k）。

如前所述，C风格的连续数组和Fortran风格的连续数组具有特定的跨步模式。两个数组标志（[``NPY_ARRAY_C_CONTIGUOUS``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_C_CONTIGUOUS)和[``NPY_ARRAY_F_CONTIGUOUS``](https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_ARRAY_F_CONTIGUOUS)）表示特定数组的跨步模式是否与C风格的连续或Fortran风格的连续匹配或两者都不匹配。可以使用[``PyArray_IS_C_CONTIGUOUS``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_IS_C_CONTIGUOUS)（obj）和()
 来测试跨步模式是否匹配标准C或Fortran[``PyArray_ISFORTRAN``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_ISFORTRAN)（obj）分别。大多数第三方库都期望连续的数组。但是，通常支持通用跨越并不困难。我鼓励您尽可能在自己的代码中使用跨步信息，并保留包裹第三方代码的单段要求。使用与ndarray一起提供的跨步信息而不是需要连续的跨步减少了必须进行的复制。

## 示例

以下示例显示了如何编写一个包含两个输入参数（将转换为数组）和输出参数（必须是数组）的包装器。该函数返回None并更新输出数组。请注意NumPy v1.14及更高版本的WRITEBACKIFCOPY语义的更新使用

``` c
static PyObject *
example_wrapper(PyObject *dummy, PyObject *args)
{
    PyObject *arg1=NULL, *arg2=NULL, *out=NULL;
    PyObject *arr1=NULL, *arr2=NULL, *oarr=NULL;

    if (!PyArg_ParseTuple(args, "OOO!", &arg1, &arg2,
        &PyArray_Type, &out)) return NULL;

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr2 == NULL) goto fail;
#if NPY_API_VERSION >= 0x0000000c
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (oarr == NULL) goto fail;

    /* code that makes use of arguments */
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
     */

    Py_DECREF(arr1);
    Py_DECREF(arr2);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_ResolveWritebackIfCopy(oarr);
#endif
    Py_DECREF(oarr);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_DiscardWritebackIfCopy(oarr);
#endif
    Py_XDECREF(oarr);
    return NULL;
}
```
