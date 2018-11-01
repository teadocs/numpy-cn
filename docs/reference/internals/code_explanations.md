# NumPy C语言代码解释

> 名人名言：
> 狂热包括当你忘记了你的目标时加倍努力。-乔治·桑塔亚纳。
> 权威是一个人，他能告诉你比你真正想知道的更多的事情。-未知

本章试图解释一些新代码背后的逻辑。 这些解释背后的目的是让某人能够比仅仅盯着代码更容易理解实现背后的想法。 也许以这种方式，可以改进，借用和/或优化算法。

## 内存模型

ndarray的一个基本方面是数组被视为从某个位置开始的内存“块”。这种内存的解释取决于步幅信息。对于N维数组中的每个维度，整数（stride）指示必须跳过多少字节才能到达该维度中的下一个元素。 除非您有单段数组，否则在遍历数组时必须查阅此步幅信息。 编写接受strides的代码并不困难，只需使用（char *）指针，因为strides以字节为单位。 还要记住，步幅不必是元素大小的单位倍数。 另外，请记住，如果数组的维数为0（有时称为rank-0数组），则strides和dimension变量为NULL。

除了[PyArrayObject](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayObject)的步幅和维度成员中包含的结构信息之外，标志还包含有关如何访问数据的重要信息。 特别是，当内存根据数据类型数组位于合适的边界时，设置[NPY_ARRAY_ALIGNED](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_ARRAY_ALIGNED)标志。 即使你有一个连续的内存块，你也不能仅仅假设取消引用一个特定于数据类型的指向元素的指针是安全的。 只有设置了[NPY_ARRAY_ALIGNED](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_ARRAY_ALIGNED)标志才是安全操作（在某些平台上它可以工作，但在其他平台上，如Solaris，它会导致总线错误）。 如果您计划写入阵列的存储区，也应该确保[NPY_ARRAY_WRITEABLE](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_ARRAY_WRITEABLE)。 还可以获得指向不可写存储区的指针。 有时，当未设置[NPY_ARRAY_WRITEABLE](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_ARRAY_WRITEABLE)标志时写入存储区域将是粗鲁的。 其他时候它可能导致程序崩溃（例如，作为只读存储器映射文件的数据区）。

## 数据类型封装

数据类型是ndarray的重要抽象。 操作将查看数据类型以提供操作阵列所需的关键功能。 此功能在[PyArray_Descr](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr) 结构的'f'成员指向的函数指针列表中提供。 通过这种方式，可以简单地通过在'f'成员中提供具有合适的函数指针的[PyArray_Descr](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr) 结构来扩展数据类型的数量。 对于内置类型，有一些优化绕过这种机制，但数据类型抽象的要点是允许添加新的数据类型。

作为内置数据类型之一，void data-type允许包含1个或多个字段的任意结构化类型作为数组的元素。 字段只是另一个数据类型对象以及当前结构化类型的偏移量。 为了支持任意嵌套字段，为void类型实现了数据类型访问的几个递归实现。 常见的习语是循环遍历字典的元素并基于存储在给定偏移处的数据类型对象执行特定操作。 这些偏移可以是任意数字。 因此，必要时必须识别并考虑到遇到错位数据的可能性。

## N-D 迭代器

在许多NumPy代码中非常常见的操作是需要迭代一般的，跨步的N维数组的所有元素。 在迭代器对象的概念中抽象出通用N维循环的这种操作。 要编写N维循环，只需从ndarray创建迭代器对象，使用迭代器对象结构的dataptr成员，并在迭代器对象上调用宏[PyArray_ITER_NEXT](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_ITER_NEXT)（it）以移动到下一个元素。 “next”元素始终处于C连续顺序。 该宏通过第一个特殊外壳工作，C-contiguous，1-D和2-D情况非常简单。

对于一般情况，迭代通过跟踪迭代器对象中的坐标计数器列表来工作。在每次迭代时，最后一个坐标计数器增加（从0开始）。如果此计数器小于该维度中的数组大小（预先计算和存储的值），则计数器将增加，并且dataptr成员将通过该维度中的步幅增加，并且宏结束。如果到达维度的末尾，则将最后一个维度的计数器重置为零，并通过将步幅值减去一个小于该维度中元素数量的步数值，将数据带移回该维度的开头（这是也预先计算并存储在迭代器对象的backstrides成员中）。在这种情况下，宏不会结束，但是本地维度计数器会递减，以便倒数第二个维度替换最后一个维度所扮演的角色，并且在倒数第二个维度上再次执行先前描述的测试尺寸。通过这种方式，可以适当调整dataptr以实现任意跨步。

[PyArrayIterObject](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayIterObject) 结构的坐标成员维护当前的N-d计数器，除非基础数组是C-连续的，在这种情况下，坐标计数被旁路。[PyArrayIterObject](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayIterObject) 的索引成员跟踪迭代器的当前平坦索引。 它由[PyArray_ITER_NEXT](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_ITER_NEXT)宏更新。

## 广播

在Numeric中，广播是在深入ufuncobject.c的几行代码中实现的。 在NumPy中，广播的概念已被抽象化，以便可以在多个地方进行。 广播由[PyArray_Broadcast](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Broadcast)函数处理。 此函数需要传入[PyArrayMultiIterObject](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayMultiIterObject)（或等效二进制的东西）.[PyArrayMultiIterObject](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayMultiIterObject)跟踪每个维度中广播的维度和大小数量以及广播结果的总大小。 它还跟踪正在广播的数组的数量以及指向正在广播的每个阵列的迭代器的指针。

[PyArray_Broadcast](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Broadcast)函数获取已经定义的迭代器，并使用它们来确定每个维度中的广播形状（在广播发生的同时创建迭代器，然后使用PyMultiIter_New函数）。 然后，调整迭代器，以便每个迭代器都认为它正在迭代具有广播大小的数组。 这是通过调整迭代器的尺寸数量和每个尺寸的形状来完成的。 这是有效的，因为迭代器步幅也会调整。 广播仅调整（或添加）长度为1的维度。 对于这些维度，strides变量只是设置为0，因此当广播操作在扩展维度上操作时，该数组上迭代器的数据指针不会移动。

广播总是在Numeric中使用0值步幅实现扩展尺寸。它在NumPy中以完全相同的方式完成。 最大的区别在于，现在在[PyArrayIterObject](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayIterObject)中跟踪步长数组，广播结果中涉及的迭代器在[PyArrayMultiIterObject](https://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayMultiIterObject)中保持跟踪，**PyArray_BroadCast**调用实现广播规则。

## 数组标量

数组标量提供了Python类型的层次结构，允许存储在数组中的数据类型与从数组中提取元素时返回的Python类型进行一对一的对应。这个规则的一个例外是对象数组。对象数组是任意Python对象的异构集合。当您从一个对象数组中选择一个项目时，您将获得原始的Python对象(而不是一个对象数组标量，该标量确实存在，但很少用于实际目的)。

数组标量还提供与数组相同的方法和属性，目的是使用相同的代码支持任意维(包括0维)。除了空标量之外，数组标量是只读的(不可变的)，它也可以被写入，以便结构化数组字段设置更自然地工作 (a[0][‘f1’] = ``value`` ).

## 索引

所有python索引操作arr[index]都是通过首先准备索引并查找索引类型来组织的。支持的索引类型是：

- integer
- newaxis
- slice
- ellipsis
- integer arrays/array-likes (fancy)
- boolean (单布尔数组); 如果有多个布尔数组作为索引或形状不完全匹配，则布尔数组将转换为整数数组。
- 0-d 布尔值（也是整数）; 0-d布尔数组是一种特殊情况，必须在高级索引代码中处理。 它们表示必须将0-d布尔数组解释为整数数组。

除了标量数组特殊情况外，整数数组被解释为整数索引，这很重要，因为整数数组索引会强制复制，但如果返回标量则会被忽略（完整整数索引）。 除了超出限制值和高级索引的广播错误之外，准备好的索引保证有效。 这包括为不完整的索引添加省略号，例如当二维数组用单个整数索引时。

下一步取决于找到的索引类型。 如果所有维度都使用整数索引，则返回或设置标量。 单个布尔索引数组将调用专门的布尔函数。 包含省略号或切片但没有高级索引的索引将始终通过计算新的步幅和内存偏移量来创建旧数组的视图。 然后可以返回此视图，或者对于赋值，使用 **PyArray_CopyObject** 填充该视图。 请注意，*PyArray_CopyObject* 也可以在其他分支中的临时数组上调用，以在数组为对象dtype时支持复杂的赋值。

### 高级索引

到目前为止，最复杂的情况是高级索引，它可能与也可能不与典型的基于视图的索引相结合。 这里整数索引被解释为基于视图。 在尝试理解这一点之前，您可能想要熟悉它的细微之处。 高级索引代码有三个不同的分支和一个特例：

- 有一个索引数组，它以及赋值数组可以简单地迭代。 例如，它们可能是连续的。 索引数组也必须是 **intp** 类型，赋值中的值数组应该是正确的类型。这纯粹是一条快速的道路。
- 只有整数数组索引，因此不存在子数组。
- 基于视图和高级索引是混合的。 在这种情况下，基于视图的索引定义了由高级索引组合的子阵列集合。 例如，``arr[[1, 2, 3],,:]`` 是通过垂直堆叠子数组 ``arr[1, :]``、``arr[2,:]`` 和 ``arr[3, :]`` 来创建的。
- 有一个子阵列，但它只有一个元素。 这种情况可以像没有子阵列一样处理，但在安装过程中需要注意。

确定适用的情况，检查广播以及确定所需的转置类型都在 *PyArray_MapIterNew* 中完成。 设置完成后，有两种情况。 如果没有子数组或者只有一个元素，则不需要进行子数组迭代，并且准备迭代器，迭代所有索引数组以及结果或值数组。 如果存在子阵列，则准备三个迭代器。 一个用于索引数组，一个用于结果或值数组（减去其子数组），另一个用于原始数据和结果/赋值数组的子数组。 前两个迭代器将指针提供（或允许计算）到子数组的开头，然后允许重新启动子数组迭代。

当高级指数彼此相邻时，可能需要转置。 所有必要的转置都由 **PyArray_MapIterSwapAxes** 处理，并且必须由调用者处理，除非要求PyArray_MapIterNew分配结果。

在准备之后，获取和设置是相对简单的，尽管需要考虑不同的迭代模式。 除非在项目获取期间只有一个索引数组，否则将事先检查索引的有效性。 否则，它将在内部循环中进行处理以进行优化。

## 通用功能
通用函数是可调用对象，它通过将逐个元素工作的基本1-D循环包装成完全易于使用的函数来获取N个输入并产生M个输出，这些函数无缝地实现广播，类型检查和缓冲强制，以及输出参数处理。 新的通用函数通常在C中创建，尽管有一种从Python函数创建ufunc的机制([frompyfunc](https://docs.scipy.org/doc/numpy/reference/generated/numpy.frompyfunc.html#numpy.frompyfunc)))。 用户必须提供一个1-D循环，该循环实现基本函数，获取输入标量值并将生成的标量放入适当的输出槽中，如实现中所述。

## 建立

每个ufunc计算都涉及与设置计算相关的一些开销。这种开销的实际意义在于，即使ufunc的实际计算速度非常快，您也可以编写数组和特定于类型的代码，这些代码对于小型数组而言比ufunc更快。特别是，使用ufuncs在0-D阵列上执行许多计算将比其他基于Python的解决方案慢（静默导入的scalarmath模块精确存在，以使阵列标量具有基于ufunc的计算的外观和显着降低的开销）。

当调用ufunc时，必须完成许多事情。从这些设置操作收集的信息存储在循环对象中。这个循环对象是一个C结构（它可以成为一个Python对象，但不会这样初始化，因为它只在内部使用）。此循环对象具有需要与PyArray_Broadcast一起使用的布局，以便可以以与在其他代码段中处理的方式相同的方式处理广播。

首先要做的是在特定于线程的全局字典中查找缓冲区大小，错误掩码和关联的错误对象的当前值。错误掩码的状态控制在发现错误条件时发生的情况。应注意，仅在执行每个1-D循环之后才执行硬件错误标志的检查。这意味着如果输入和输出数组是连续的并且类型正确以便执行单个1-D循环，则在计算出阵列的所有元素之前可能不会检查标志。在特定于线程的字典中查找这些值需要花费时间，除了非常小的数组之外，很容易忽略这些时间。

检查后，特定于线程的全局变量，输入被评估以确定ufunc应如何进行，并在必要时构造输入和输出数组。任何非数组的输入都将转换为数组（必要时使用上下文）。注意哪些输入是标量（因此转换为0-D阵列）。

接下来，根据输入数组类型，从ufunc可用的1-D循环中选择适当的1-D循环。通过尝试将输入的数据类型的签名与可用签名进行匹配来选择该1-D循环。与内置类型对应的签名存储在ufunc结构的types成员中。对应于用户定义类型的签名存储在函数信息的链接列表中，head元素作为CObject存储在由数据类型编号键入的userloops字典中（参数列表中的第一个用户定义类型是用作关键）。搜索签名，直到找到可以安全地转换输入数组的签名（忽略任何不允许确定结果类型的标量参数）。这种搜索过程的含义是，当存储签名时，“较小类型”应放在“较大类型”之下。如果未找到1-D循环，则报告错误。否则，使用存储的签名更新argument_list - 如果需要进行转换并修复1-D循环假定的输出类型。

如果ufunc有2个输入和1个输出，第二个输入是Object数组，则执行特殊情况检查，如果第二个输入不是ndarray，则返回NotImplemented，具有__array_priority__属性，并且具有__r {op } __特殊方法。通过这种方式，Python被发信号通知其他对象有机会完成操作而不是使用通用对象数组计算。这允许（例如）稀疏矩阵覆盖乘法运算符1-D循环。

对于小于指定缓冲区大小的输入数组，副本由所有非连续，未对齐或不在字节顺序的数组组成，以确保对于小数组，使用单个循环。然后，为所有输入数组创建数组迭代器，并将生成的迭代器集合广播为单个形状。

然后处理输出参数（如果有的话）并构造任何缺少的返回数组。如果任何提供的输出数组没有正确的类型（或未对齐）且小于缓冲区大小，则构造一个新的输出数组，并设置特殊的 **WRITEBACKIFCOPY** 标志。在函数结束时，调用[PyArray_ResolveWritebackIfCopy](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_ResolveWritebackIfCopy)，以便将其内容复制回输出数组。然后处理输出参数的迭代器。

最后，决定如何执行循环机制以确保输入数组的所有元素组合在一起以生成正确类型的输出数组。 循环执行的选项是单循环（对于连续，对齐和正确的数据类型），跨循环（对于非连续但仍对齐且正确的数据类型）和缓冲循环（对于错误对齐或不正确的数据） 类型情况）。 根据要求的执行方法，然后设置并计算循环。

### Function call

This section describes how the basic universal function computation loop is setup and executed for each of the three different kinds of execution. If **NPY_ALLOW_THREADS** is defined during compilation, then as long as no object arrays are involved, the Python Global Interpreter Lock (GIL) is released prior to calling the loops. It is re-acquired if necessary to handle error conditions. The hardware error flags are checked only after the 1-D loop is completed.

#### One Loop

This is the simplest case of all. The ufunc is executed by calling the underlying 1-D loop exactly once. This is possible only when we have aligned data of the correct type (including byte-order) for both input and output and all arrays have uniform strides (either contiguous, 0-D, or 1-D). In this case, the 1-D computational loop is called once to compute the calculation for the entire array. Note that the hardware error flags are only checked after the entire calculation is complete.

#### Strided Loop

When the input and output arrays are aligned and of the correct type, but the striding is not uniform (non-contiguous and 2-D or larger), then a second looping structure is employed for the calculation. This approach converts all of the iterators for the input and output arguments to iterate over all but the largest dimension. The inner loop is then handled by the underlying 1-D computational loop. The outer loop is a standard iterator loop on the converted iterators. The hardware error flags are checked after each 1-D loop is completed.

#### Buffered Loop

This is the code that handles the situation whenever the input and/or output arrays are either misaligned or of the wrong data-type (including being byte-swapped) from what the underlying 1-D loop expects. The arrays are also assumed to be non-contiguous. The code works very much like the strided-loop except for the inner 1-D loop is modified so that pre-processing is performed on the inputs and post- processing is performed on the outputs in bufsize chunks (where bufsize is a user-settable parameter). The underlying 1-D computational loop is called on data that is copied over (if it needs to be). The setup code and the loop code is considerably more complicated in this case because it has to handle:

- memory allocation of the temporary buffers
- deciding whether or not to use buffers on the input and output data (mis-aligned and/or wrong data-type)
- copying and possibly casting data for any inputs or outputs for which buffers are necessary.
- special-casing Object arrays so that reference counts are properly handled when copies and/or casts are necessary.
- breaking up the inner 1-D loop into bufsize chunks (with a possible remainder).

Again, the hardware error flags are checked at the end of each 1-D loop.

### Final output manipulation

Ufuncs allow other array-like classes to be passed seamlessly through the interface in that inputs of a particular class will induce the outputs to be of that same class. The mechanism by which this works is the following. If any of the inputs are not ndarrays and define the [__array_wrap__](https://docs.scipy.org/doc/numpy/reference/arrays.classes.html#numpy.class.__array_wrap__) method, then the class with the largest [__array_priority__](https://docs.scipy.org/doc/numpy/reference/arrays.classes.html#numpy.class.__array_priority__) attribute determines the type of all the outputs (with the exception of any output arrays passed in). The [__array_wrap__](https://docs.scipy.org/doc/numpy/reference/arrays.classes.html#numpy.class.__array_wrap__) method of the input array will be called with the ndarray being returned from the ufunc as it’s input. There are two calling styles of the [__array_wrap__](https://docs.scipy.org/doc/numpy/reference/arrays.classes.html#numpy.class.__array_wrap__) function supported. The first takes the ndarray as the first argument and a tuple of “context” as the second argument. The context is (ufunc, arguments, output argument number). This is the first call tried. If a TypeError occurs, then the function is called with just the ndarray as the first argument.

### Methods

There are three methods of ufuncs that require calculation similar to the general-purpose ufuncs. These are reduce, accumulate, and reduceat. Each of these methods requires a setup command followed by a loop. There are four loop styles possible for the methods corresponding to no-elements, one-element, strided-loop, and buffered- loop. These are the same basic loop styles as implemented for the general purpose function call except for the no-element and one- element cases which are special-cases occurring when the input array objects have 0 and 1 elements respectively.

#### Setup

The setup function for all three methods is ``construct_reduce``. This function creates a reducing loop object and fills it with parameters needed to complete the loop. All of the methods only work on ufuncs that take 2-inputs and return 1 output. Therefore, the underlying 1-D loop is selected assuming a signature of [ ``otype``, ``otype``, ``otype`` ] where ``otype`` is the requested reduction data-type. The buffer size and error handling is then retrieved from (per-thread) global storage. For small arrays that are mis-aligned or have incorrect data-type, a copy is made so that the un-buffered section of code is used. Then, the looping strategy is selected. If there is 1 element or 0 elements in the array, then a simple looping method is selected. If the array is not mis-aligned and has the correct data-type, then strided looping is selected. Otherwise, buffered looping must be performed. Looping parameters are then established, and the return array is constructed. The output array is of a different shape depending on whether the method is reduce, accumulate, or reduceat. If an output array is already provided, then it’s shape is checked. If the output array is not C-contiguous, aligned, and of the correct data type, then a temporary copy is made with the WRITEBACKIFCOPY flag set. In this way, the methods will be able to work with a well-behaved output array but the result will be copied back into the true output array when [PyArray_ResolveWritebackIfCopy](https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_ResolveWritebackIfCopy) is called at function completion. Finally, iterators are set up to loop over the correct axis (depending on the value of axis provided to the method) and the setup routine returns to the actual computation routine.

#### Reduce

All of the ufunc methods use the same underlying 1-D computational loops with input and output arguments adjusted so that the appropriate reduction takes place. For example, the key to the functioning of reduce is that the 1-D loop is called with the output and the second input pointing to the same position in memory and both having a step- size of 0. The first input is pointing to the input array with a step- size given by the appropriate stride for the selected axis. In this way, the operation performed is

![公式](https://docs.scipy.org/doc/numpy/_images/math/64e8d1e06184815b84c4245104e73c4f040d0439.svg)

where ``N+1`` is the number of elements in the input, i, o is the output, and i[k] is the k^{\textrm{th}} element of i along the selected axis. This basic operations is repeated for arrays with greater than 1 dimension so that the reduction takes place for every 1-D sub-array along the selected axis. An iterator with the selected dimension removed handles this looping.

For buffered loops, care must be taken to copy and cast data before the loop function is called because the underlying loop expects aligned data of the correct data-type (including byte-order). The buffered loop must handle this copying and casting prior to calling the loop function on chunks no greater than the user-specified bufsize.

#### Accumulate

The accumulate function is very similar to the reduce function in that the output and the second input both point to the output. The difference is that the second input points to memory one stride behind the current output pointer. Thus, the operation performed is

![公式](https://docs.scipy.org/doc/numpy/_images/math/9f318f73ba08f65fc37acd490db5021bfc2a82be.svg)

The output has the same shape as the input and each 1-D loop operates over N elements when the shape in the selected axis is N+1. Again, buffered loops take care to copy and cast the data before calling the underlying 1-D computational loop.

#### Reduceat

The reduceat function is a generalization of both the reduce and accumulate functions. It implements a reduce over ranges of the input array specified by indices. The extra indices argument is checked to be sure that every input is not too large for the input array along the selected dimension before the loop calculations take place. The loop implementation is handled using code that is very similar to the reduce code repeated as many times as there are elements in the indices input. In particular: the first input pointer passed to the underlying 1-D computational loop points to the input array at the correct location indicated by the index array. In addition, the output pointer and the second input pointer passed to the underlying 1-D loop point to the same position in memory. The size of the 1-D computational loop is fixed to be the difference between the current index and the next index (when the current index is the last index, then the next index is assumed to be the length of the array along the selected dimension). In this way, the 1-D loop will implement a reduce over the specified indices.

Mis-aligned or a loop data-type that does not match the input and/or output data-type is handled using buffered code where-in data is copied to a temporary buffer and cast to the correct data-type if necessary prior to calling the underlying 1-D function. The temporary buffers are created in (element) sizes no bigger than the user settable buffer-size value. Thus, the loop must be flexible enough to call the underlying 1-D computational loop enough times to complete the total calculation in chunks no bigger than the buffer-size.