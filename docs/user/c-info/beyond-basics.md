# 深入的知识

> The voyage of discovery is not in seeking new landscapes but in having new eyes.
> 
> — Marcel Proust

> Discovery is seeing what everyone else has seen and thinking what no one else has thought.
> 
> — Albert Szent-Gyorgi

## 迭代数组中的元素

### 基本迭代

一种常见的算法要求是能够遍历多维数组中的所有元素。数组迭代器对象使这种方法易于以通用方式完成，适用于任何维度的数组。当然，如果您知道要使用的维数，那么您始终可以编写嵌套for循环来完成迭代。但是，如果要编写适用于任意数量维度的代码，则可以使用数组迭代器。访问数组的.flat属性时返回数组迭代器对象。

基本用法是调用[``PyArray_IterNew``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_IterNew)（``array``），其中array是ndarray对象（或其子类之一）。返回的对象是一个array-iterator对象（由ndarray的.flat属性返回的同一对象）。此对象通常强制转换为PyArrayIterObject *，以便可以访问其成员。所需的唯一成员``iter->size``包含数组的总大小``iter->index``，其中包含数组的当前1-d索引，以及``iter->dataptr``指向数组当前元素的数据的指针。有时，访问``iter->ao``哪个是指向底层ndarray对象的指针也很有用。

在数组的当前元素处理数据之后，可以使用macro [``PyArray_ITER_NEXT``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_ITER_NEXT)（``iter``）获取数组的下一个元素
 。迭代总是以C风格的连续方式进行（最后一个索引变化最快）。的
 [``PyArray_ITER_GOTO``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_ITER_GOTO)（``iter``，``destination``）可以用来跳到一个特定点的数组，其中在``destination``是npy_intp数据类型与空间的数组，以处理潜在的数组中的维度中的至少数。有时使用[``PyArray_ITER_GOTO1D``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_ITER_GOTO1D)（``iter``，``index``）将跳转到由值给出的1-d索引是有用的``index``。但是，最常见的用法在以下示例中给出。

``` c
PyObject *obj; /* assumed to be some ndarray object */
PyArrayIterObject *iter;
...
iter = (PyArrayIterObject *)PyArray_IterNew(obj);
if (iter == NULL) goto fail;   /* Assume fail has clean-up code */
while (iter->index < iter->size) {
    /* do something with the data at it->dataptr */
    PyArray_ITER_NEXT(it);
}
...
```

您还可以使用[``PyArrayIter_Check``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArrayIter_Check)（``obj``）来确保您拥有迭代器对象和[``PyArray_ITER_RESET``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_ITER_RESET)（``iter``）以将迭代器对象重置回数组的开头。

在这一点上应该强调的是，如果你的数组已经是连续的，你可能不需要数组迭代器（使用数组迭代器可以工作，但会比你写的最快的代码慢）。数组迭代器的主要目的是使用任意步长将迭代封装在N维数组上。它们在NumPy源代码本身的许多地方使用。如果您已经知道您的数组是连续的（Fortran或C），那么只需将元素大小添加到正在运行的指针变量就可以非常有效地引导您完成数组。换句话说，在连续的情况下（假设为双精度），这样的代码可能会更快。

``` c
npy_intp size;
double *dptr;  /* could make this any variable type */
size = PyArray_SIZE(obj);
dptr = PyArray_DATA(obj);
while(size--) {
   /* do something with the data at dptr */
   dptr++;
}
```

### 迭代除一个轴之外的所有轴

一种常见的算法是循环遍历数组的所有元素，并通过发出函数调用对每个元素执行一些函数。由于函数调用可能非常耗时，因此加速此类算法的一种方法是编写函数，使其获取数据向量，然后编写迭代，以便一次对整个数据维度执行函数调用。这增加了每个函数调用完成的工作量，从而将函数调用开头减少到总时间的一小部分。即使在没有函数调用的情况下执行循环的内部，在具有最大数量元素的维度上执行内循环也是有利的，以利用在使用流水线操作来增强基础操作的微处理器上可用的速度增强。

的[``PyArray_IterAllButAxis``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_IterAllButAxis)（``array``，``&dim``）构造被修改，使得它不会在由暗淡指示的尺寸迭代的迭代器对象。这个迭代器对象的唯一限制是不能使用``PyArray_Iter_GOTO1D``（``it``，``ind``）宏（因此，如果将此对象传递回Python，则平面索引将不起作用 - 所以你不应该这样做）。请注意，此例程中返回的对象仍然通常转换为PyArrayIterObject *。所做的就是修改返回迭代器的步幅和尺寸，以模拟迭代数组[...，0，...]，其中0放在
 维度上。如果dim为负，则找到并使用具有最大轴的尺寸。

### 迭代多个数组

通常，希望同时迭代几个数组。通用函数就是这种行为的一个例子。如果您只想迭代具有相同形状的数组，那么只需创建几个迭代器对象就是标准过程。例如，以下代码迭代两个假定具有相同形状和大小的数组（实际上obj1必须至少具有与obj2一样多的总元素）：

``` c
/* It is already assumed that obj1 and obj2
   are ndarrays of the same shape and size.
*/
iter1 = (PyArrayIterObject *)PyArray_IterNew(obj1);
if (iter1 == NULL) goto fail;
iter2 = (PyArrayIterObject *)PyArray_IterNew(obj2);
if (iter2 == NULL) goto fail;  /* assume iter1 is DECREF'd at fail */
while (iter2->index < iter2->size)  {
    /* process with iter1->dataptr and iter2->dataptr */
    PyArray_ITER_NEXT(iter1);
    PyArray_ITER_NEXT(iter2);
}
```

### 在多个数组上广播

当一个操作涉及多个数组时，您可能希望使用数学操作(即ufuncs)使用的相同广播规则。
这可以使用 [``PyArrayMultiIterObject``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArrayMultiIterObject) 轻松完成。
这是从Python命令numpy.Broadcast返回的对象，它几乎和C一样容易使用。
函数 [PyArray_MultiIterNew](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_MultiIterNew)(``n``, ``...``)。使用（``n``个输入对象代替）。
输入对象可以是数组或任何可以转换为数组的对象。
返回指向 [``PyArrayMultiIterObject``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArrayMultiIterObject) 的指针。
广播已经完成，它调整迭代器，以便为每个输入调用PyArray_ITER_NEXT，
以便前进到每个数组中的下一个元素。
这种递增由 [PyArray_MultiIter_NEXT](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_MultiIter_NEXT)(``Obj``)宏自动执行（它可以将乘法器 ``obj`` 处理为 PyArrayMultiObject \* 或 [PyObject \*](https://docs.python.org/dev/c-api/structures.html#c.PyObject)）。
输入编号 ``i`` 中的数据可使用[PyArray_MultiIter_DATA](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_MultiIter_DATA)(``obj``, ``i``)和总（广播）大小作为**PyArray_MultiIter_SIZE**(``Obj``)。
下面是使用此功能的示例。

``` c
mobj = PyArray_MultiIterNew(2, obj1, obj2);
size = PyArray_MultiIter_SIZE(obj);
while(size--) {
    ptr1 = PyArray_MultiIter_DATA(mobj, 0);
    ptr2 = PyArray_MultiIter_DATA(mobj, 1);
    /* code using contents of ptr1 and ptr2 */
    PyArray_MultiIter_NEXT(mobj);
}
```

function [``PyArray_RemoveSmallest``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_RemoveSmallest)(``multi``) 可用于获取多迭代器对象并调整所有迭代器，以便迭代不会发生在最大维度上（它使得该维度的大小为1）。
循环使用指针的代码很可能也需要每个迭代器的步幅数据。此信息存储在 multi->iters[i]->strides 中。

在NumPy源代码中使用多迭代器有几个例子，因为它使N维广播代码编写起来非常简单。浏览源代码以获取更多示例。

## 用户定义的数据类型

NumPy带有24种内置数据类型。虽然这涵盖了绝大多数可能的用例，但可以想象用户可能需要额外的数据类型。有一些支持在NumPy系统中添加额外的数据类型。此附加数据类型的行为与常规数据类型非常相似，只是ufunc必须具有1-d循环才能单独处理它。同时检查其他数据类型是否可以“安全”地转换到这种新类型或从这种新类型转换为“can cast”，除非您还注册了新数据类型可以转换为哪种类型。添加数据类型是NumPy 1.0中经过较少测试的领域之一，因此该方法可能存在漏洞。如果使用已有的OBJECT或VOID数据类型无法执行您想要执行的操作，则仅添加新数据类型。

### 添加新数据类型

要开始使用新的数据类型，您需要首先定义一个新的Python类型来保存新数据类型的标量。如果您的新类型具有二进制兼容布局，则可以接受从其中一个数组标量继承。这将允许您的新数据类型具有数组标量的方法和属性。新数据类型必须具有固定的内存大小（如果要定义需要灵活表示的数据类型，如变量精度数，则使用指向对象的指针作为数据类型）。新Python类型的对象结构的内存布局必须是PyObject_HEAD，后跟数据类型所需的固定大小的内存。例如，新Python类型的合适结构是：

``` c
typedef struct {
   PyObject_HEAD;
   some_data_type obval;
   /* the name can be whatever you want */
} PySomeDataTypeObject;
```

在定义了新的Python类型对象之后，必须定义一个新[``PyArray_Descr``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Descr)结构，其typeobject成员将包含指向您刚刚定义的数据类型的指针。此外，必须定义“.f”成员中的必需函数：nonzero，copyswap，copyswapn，setitem，getitem和cast。但是，您定义的“.f”成员中的函数越多，新数据类型就越有用。将未使用的函数初始化为NULL非常重要。这可以使用[``PyArray_InitArrFuncs``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_InitArrFuncs)（f）来实现。

一旦[``PyArray_Descr``](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Descr)创建了新结构并填充了您调用的所需信息和有用函数
 [``PyArray_RegisterDataType``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_RegisterDataType)（new_descr）。此调用的返回值是一个整数，为您提供指定数据类型的唯一type_number。此类型编号应存储并由您的模块提供，以便其他模块可以使用它来识别您的数据类型（查找用户定义的数据类型编号的另一种机制是根据类型的名称进行搜索 - 与数据类型相关联的对象``PyArray_TypeNumFromName``）。

### 注册投射功能

您可能希望允许内置（和其他用户定义的）数据类型自动转换为您的数据类型。为了实现这一点，您必须使用您希望能够从中投射的数据类型注册一个转换函数。这需要为要支持的每个转换编写低级转换函数，然后使用数据类型描述符注册这些函数。低级转换函数具有签名。


void ``castfunc``（ void *   *from* ，void *   *to* ，[npy_intp ](https://numpy.org/devdocs/reference/c-api/dtype.html#c.npy_intp)  *n* ，void *   *fromarr* ，void *   *toarr*  ）[¶](#c.castfunc)

铸``n``元件``from``一个键入``to``另一个。要转换的数据位于由from指向的连续，正确交换和对齐的内存块中。要转换为的缓冲区也是连续的，正确交换和对齐的。fromarr和toarr参数只应用于灵活元素大小的数组（字符串，unicode，void）。

一个示例castfunc是：

``` c
static void
double_to_float(double *from, float* to, npy_intp n,
       void* ig1, void* ig2);
while (n--) {
      (*to++) = (double) *(from++);
}
```

然后可以使用以下代码注册以将双精度转换为浮点数：

``` c
doub = PyArray_DescrFromType(NPY_DOUBLE);
PyArray_RegisterCastFunc(doub, NPY_FLOAT,
     (PyArray_VectorUnaryFunc *)double_to_float);
Py_DECREF(doub);
```

### 注册强制规则

默认情况下，不会假定所有用户定义的数据类型都可安全地转换为任何内置数据类型。此外，不假定内置数据类型可安全地转换为用户定义的数据类型。这种情况限制了用户定义的数据类型参与ufuncs使用的强制系统的能力，以及在NumPy中进行自动强制时的其他情况。这可以通过将数据类型注册为从特定数据类型对象安全地转换来更改。函数[``PyArray_RegisterCanCast``](https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_RegisterCanCast)（from_descr，totype_number，scalarkind）应该用于指定数据类型对象from_descr可以转换为类型号为totype_number的数据类型。如果您不想改变标量强制规则，那么请使用``NPY_NOSCALAR``scalarkind参数。

如果要允许新数据类型也能够共享标量强制规则，则需要在数据类型对象的“.f”成员中指定scalarkind函数，以返回新数据的标量类型-type应该被视为（标量的值可用于该函数）。然后，您可以为可以从用户定义的数据类型返回的每个标量类型注册可以单独转换的数据类型。如果您没有注册标量强制处理，那么所有用户定义的数据类型都将被视为``NPY_NOSCALAR``。

### 注册ufunc循环

您可能还希望为数据类型注册低级ufunc循环，以便数据类型的ndarray可以无缝地应用数学。注册具有完全相同的arg_types签名的新循环，静默替换该数据类型的任何先前注册的循环。

在为ufunc注册一维循环之前，必须预先创建ufunc。
然后调用 [PyUFunc_RegisterLoopForType](https://numpy.org/devdocs/reference/c-api/ufunc.html#c.PyUFunc_RegisterLoopForType)(…)。
以及循环所需的信息。
如果进程成功，则此函数的返回值为0；
如果进程不成功，则返回 ``-1``，并设置错误条件。

## 在C中对ndarray进行子类型化

自2.2以来一直潜伏在Python中的一个较少使用的功能是在C中子类类型的能力。这个设施是使NumPy脱离已经在C中的数字代码库的重要原因之一。 C中的子类型允许在内存管理方面具有更大的灵活性。即使您对如何为Python创建新类型有基本的了解，在C中进行子类型输入并不困难。虽然最简单的是从单个父类型进行子类型化，但也可以从多个父类型进行子类型化。C中的多重继承通常没有Python中那么有用，因为对Python子类型的限制是它们具有二进制兼容的内存布局。也许由于这个原因，从单个父类型子类型更容易一些。

与Python对象相对应的所有C结构必须以[``PyObject_HEAD``](https://docs.python.org/dev/c-api/structures.html#c.PyObject_HEAD)（或[``PyObject_VAR_HEAD``](https://docs.python.org/dev/c-api/structures.html#c.PyObject_VAR_HEAD)）开头
 。同样，任何子类型都必须具有C结构，该结构以与父类型完全相同的内存布局（或多重继承的情况下的所有父类型）开始。这样做的原因是Python可能会尝试访问子类型结构的成员，就像它具有父结构一样（ *即* 它会将指定的指针强制转换为指向父结构的指针，然后取消引用其中一个成员）。如果内存布局不兼容，则此尝试将导致不可预测的行为（最终导致内存冲突和程序崩溃）。

[PyObject_HEAD](https://docs.python.org/dev/c-api/structures.html#c.PyObject_HEAD)中的元素之一是指向  type-object 结构的指针。
通过创建一个新的类型-对象结构并用函数和指针填充它来创建一个新的Python类型，
以描述该类型的所需行为。
通常，还会创建一个新的C结构来包含该类型的每个对象所需的特定于实例的信息。
例如，[&PyArray_Type](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Type)是指向ndarray的类型-对象表的指针，
而[PyArrayObject \*](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArrayObject)变量是指向ndarray的特定实例的指针
（ndarray结构的成员之一反过来是指向类型-对象表 [&PyArray_Type](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Type) 的指针）。
最后，必须为每个新的Python类型调用 [PyType_Ready](https://docs.python.org/dev/c-api/type.html#c.PyType_Ready)(\<POINTER_TO_TYPE_OBJECT\>)。

### 创建子类型

要创建子类型，必须遵循类似的过程，除了只有不同的行为需要在类型 - 对象结构中使用新条目。所有其他条目都可以为NULL，并将使用[``PyType_Ready``](https://docs.python.org/dev/c-api/type.html#c.PyType_Ready)父类型中的相应函数填充。特别是，要在C中创建子类型，请按照下列步骤操作：

1. 如果需要，创建一个新的C结构来处理类型的每个实例。典型的 C 的结构是：

    ``` c
    typedef _new_struct {
        PyArrayObject base;
        /* new things here */
    } NewArrayObject;
    ```
    
    请注意，完整的PyArrayObject用作第一个条目，以确保新类型的实例的二进制布局与PyArrayObject相同。
1. 使用指向新函数的指针填充新的Python类型对象结构，这些新函数将覆盖默认行为，同时保留任何应该保持相同的未填充(或空)的函数。tp_name元素应该不同。
1. 用指向（Main）父类型对象的指针填充新类型对象结构的tp_base成员。对于多重继承，还要用一个元组填充tp_base成员，该元组包含所有父对象（按照它们用于定义继承的顺序）。请记住，所有父类型必须具有相同的C结构，才能使多重继承正常工作。
1. 调用[PyType_Ready](https://docs.python.org/dev/c-api/type.html#c.PyType_Ready)(\<pointer_to_new_type\>)。如果此函数返回负数，则表示发生故障，并且类型未初始化。否则，该类型就可以使用了。通常，将对新类型的引用放入模块字典中，以便可以从Python访问它，这一点通常很重要。

有关在 C 中创建子类型的更多信息，请参阅PEP 253（可从[https://www.python.org/dev/peps/pep-0253](https://www.python.org/dev/peps/pep-0253)获取）。

### ndarray子类型的特定功能

数组使用一些特殊的方法和属性，以便于子类型与基本ndarray类型的互操作。

#### __array_finalize__方法

- ndarray.``__array_finalize__``

    ndarray的几个数组创建函数允许创建特定子类型的规范。这允许在许多例程中无缝地处理子类型。
    但是，当以这种方式创建子类型时，__new__方法和__init__方法都不会被调用。
    而是分配子类型并填充适当的实例结构成员。
    最后，[``__array_finalize__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array_finalize__)
    在对象字典中查找属性。如果它存在而不是None，那么它可以是包含指向a的指针的CObject，``PyArray_FinalizeFunc``也可以是采用单个参数的方法（可以是None）。

    如果[``__array_finalize__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array_finalize__)属性是CObject，
    则指针必须是指向具有签名的函数的指针：

    ``` c
    (int) (PyArrayObject *, PyObject *)
    ```

    第一个参数是新创建的子类型。
    第二个参数（如果不是NULL）是“父”数组（如果数组是使用切片或其他操作创建的，其中存在明显可区分的父项）。
    这个例程可以做任何想做的事情。它应该在错误时返回-1，否则返回0。

    如果[``__array_finalize__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array_finalize__)属性不是None也不是CObject，
    那么它必须是一个Python方法，它将父数组作为参数（如果没有父元素，则可以是None），
    并且不返回任何内容。将捕获并处理此方法中的错误。

#### __array_priority__属性

- ndarray.``__array_priority__``

  当涉及两个或更多个子类型的操作出现时，该属性允许简单但灵活地确定哪个子类型应被视为“主要”。在使用不同子类型的操作中，具有最大[``__array_priority__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array_priority__)
  属性的子类型将确定输出的子类型。如果两个子类型相同，[``__array_priority__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array_priority__)则第一个参数的子类型确定输出。[``__array_priority__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array_priority__)对于基本ndarray类型，default
  属性返回值0.0，对于子类型，返回1.0。此属性也可以由不是ndarray的子​​类型的对象定义，并且可以用于确定[``__array_wrap__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array_wrap__)应该为返回输出调用哪个方法。

#### __array_wrap__方法

- ndarray.``__array_wrap__``

    任何类或类型都可以定义此方法，该方法应采用ndarray参数并返回该类型的实例。
    它可以看作是该[``__array__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array__)方法的反面。
    ufuncs（和其他NumPy函数）使用此方法允许其他对象通过。
    对于Python > 2.4，它也可以用来编写一个装饰器，
    它将一个仅适用于ndarrays的函数转换为一个可以使用[``__array__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array__)和[``__array_wrap__``](https://numpy.org/devdocs/reference/arrays.classes.html#numpy.class.__array_wrap__)方法处理任何类型的函数。
