# 数组接口

::: tip 注意

本页描述了用于从其他C扩展访问numpy数组内容的numpy特定API。
[PEP 3118](https://www.python.org/dev/peps/pep-3118) - [``修订的缓冲区协议``](https://docs.python.org/dev/c-api/buffer.html#c.PyObject_GetBuffer)为 Python 2.6 和 3.0 引入了类似的标准化API，以供任何扩展模块使用。
[Cython](http://cython.org/) 的缓冲区数组支持使用 [PEP 3118](https://www.python.org/dev/peps/pep-3118) 的API；请参阅 [Cython 的 numpy 教程](https://github.com/cython/cython/wiki/tutorials-numpy)。
Cython 提供了一种方法来编写支持Python版本低于2.6的缓冲区协议的代码，
因为它具有向后兼容的实现，利用这里描述的数组接口。

:::

**版本号：** 3

数组接口（有时称为数组协议）是在2005年创建的，
它是类似于数组的Python对象在任何可能的情况下智能地重用彼此的数据缓冲区的一种方法。
同构N维数组接口是对象共享N维数组内存和信息的默认机制。
该接口由Python端和使用两个属性的C端组成。希望在应用程序代码中被视为N维数组的对象应支持这些属性中的至少一个。
希望在应用程序代码中支持N维数组的对象应该至少查找这些属性中的一个，
并适当地使用提供的信息。

该接口描述同构数组，因为数组的每一项都具有相同的“类型”。
这种类型可以非常简单，也可以是相当任意和复杂的类C结构。

有两种方式使用该接口：Python端和C端。两者都是独立的属性。

## Python 方法

这种接口方法由具有 [``__array_interface__``](#__array_interface__) 属性的对象组成。

- ##### ``__array_interface__``

  项目字典（必需的3个，可选的5个）。如果未提供字典中的可选键，则它们具有隐含的默认值。

  关键字是：

  - **shape**（必须的）

      其元素为每个维度中的数组大小的元组。每个条目都是一个整数（Python int 或 long）。
      请注意，这些整数可能比平台 “int” 或 “long” 所能容纳的大（Python int是C long）。
      由使用此属性的代码适当地处理此问题；或者在可能发生溢出时引发错误，或者使用Py_Long_Long作为形状的C类型。

  - **typestr**（必须的）

      提供同质数组基本类型的字符串基本字符串格式由3部分组成：
      描述数据字节顺序的字符（``<``: little-endian，``>``:
      big-endian，``|``: not-relevant），
      给出数组基本类型的字符代码，以及提供类型使用的字节数的整数。

      基本类型字符代码为：

      代码 | 描述
      ---|---
      t | 位字段（Bit field，后面的整数表示位字段中的位数）。
      b | Boolean（Boolean 整数类型，其中所有值仅为True或False）。
      i | Integer（整数）
      u | 无符号整数（Unsigned integer）
      f | 浮点数（Floating point）
      c | 复浮点数（Complex floating point）
      m | 时间增量（Timedelta）
      M | 日期增量（Datetime）
      O | 对象（即内存包含指向 [PyObject](https://docs.python.org/dev/c-api/structures.html#c.PyObject) 的指针）
      S | 字符串（固定长度的char序列）
      U | Unicode（[Py_UNICODE](https://docs.python.org/dev/c-api/unicode.html#c.Py_UNICODE)的固定长度序列）
      V | 其他（void *  - 每个项目都是固定大小的内存块）

  - **descr**（可选的）

      提供同构数组中每个项的存储器布局的更详细描述的元组列表。
      列表中的每个元组都有两个或三个元素。通常，当 *typestr* 为 ``V[0-9]+`` 时将使用此属性，但这不是必需的。
      唯一的要求是 *typestr* 键中表示的字节数与此处表示的总字节数相同。
      其思想是支持组成数组元素的类C结构的描述。列表中每个元组的元素是

      1. 提供与数据类型的此部分关联的名称的字符串。
      这也可以是 ``('full name',
      'basic_name')`` 的元组，其中 basic name 是表示字段全名的有效 Python 变量名。
      1. *typestr* 中的基本类型描述字符串或其他列表（对于嵌套结构化类型）。
      1. 一个可选的形状元组，提供结构的这一部分应该重复多少次。
      如果不给出这一点，则不假定重复。使用这个通用接口可以描述非常复杂的结构。
      但是，请注意，数组的每个元素仍然是相同的数据类型。下面给出了使用此接口的一些示例。

    **Default**: ``[('', typestr)]``

  - **data**（可选的）

      一个2元组，其第一个参数是一个整数（如果需要，可以是一个长整数）。
      该指针必须指向数据的第一个元素（换句话说，在这种情况下总是忽略任何偏移）。
      元组中的第二个条目是只读标志。

      因此，该属性可以应用于暴露[``缓冲区接口``](https://docs.python.org/dev/c-api/objbuffer.html#c.PyObject_AsCharBuffer)的对象。
      如果此键不存在，则将通过对象本身的缓冲区接口完成内存共享。
      在这种情况下，偏移键可用于指示缓冲区的开始。
      必须由新对象存储对公开数组接口的对象的引用。

    **Default**: ``None``

  - **strides**（可选的）
    
    `None`` 表示 C 风格（C-style）的连续数组或步长元组，它提供跳转到相应维度中下一个数组元素所需的字节数。
    每个条目必须是一个整数（Python int或long）。
    与形状一样，值可以用C``int`` 或 ``long`` 表示; 调用代码应该通过引发错误或在C中使用 ``Py_LONG_LONG`` 来处理它。
    默认值为 ``None``，这意味着 C 风格 的连续内存缓冲区。
    在此模型中，数组的最后一个维度会有所不同。
    例如，对于一个对象，其数组长度为8个字节且形状为 (10,20,30) 的对象 (4800, 240, 8)，
    默认为strup元组

    **Default**: ``None`` （C 风格连续）
  - **mask**（可选的）

    ``None`` 或暴露数组接口的对象。
    mask数组的所有元素都应该被解释为true或not true。
    该对象的形状应该是 *“broadcastable”* 到原始数组的形状。

    **Default**: ``None``（所有数组值都有效）
  - **offset**（可选的）
    
    数组数据区域中的整数偏移量。这只能在 ``None`` 或返回 ``buffer`` 对象时使用。

    **Default**: 0.
  - **version**（必须的）

    显示接口版本的整数（即此版本为3）。注意不要使用它来使暴露未来版本接口的对象无效。

## C-struct 访问

这种数组接口方法允许仅使用一个属性查找和明确定义的C结构更快地访问数组。

- #### ``__array_struct__``

  A :c:type: *PyCObject* ，其  ``voidptr`` 成员包含指向填充的 [``PyArrayInterface``](c-api/types-and-structures.html#c.PyArrayInterface) 结构的指针。
  结构的内存是动态创建的，``PyCObject`` 也是使用适当的析构函数创建的，
  因此该属性的检索器只需在完成时将 [``Py_DECREF``](https://docs.python.org/dev/c-api/refcounting.html#c.Py_DECREF) 应用于该属性返回的对象。
  此外，要么需要复制出数据，要么必须保留对公开此属性的对象的引用，以确保数据不会被释放。
  如果其他对象正在引用 ``__array_struct__`` 接口，则公开 ``__array_struct__`` 接口的对象也不得重新分配其内存。

PyArrayInterface结构在 ``numpy/ndarrayobject.h`` 中定义为：

``` python
typedef struct {
  int two;              /* contains the integer 2 -- simple sanity check */
  int nd;               /* number of dimensions */
  char typekind;        /* kind in array --- character code of typestr */
  int itemsize;         /* size of each element */
  int flags;            /* flags indicating how the data should be interpreted */
                        /*   must set ARR_HAS_DESCR bit to validate descr */
  Py_intptr_t *shape;   /* A length-nd array of shape information */
  Py_intptr_t *strides; /* A length-nd array of stride information */
  void *data;           /* A pointer to the first element of the array */
  PyObject *descr;      /* NULL or data-description (same as descr key
                                of __array_interface__) -- must set ARR_HAS_DESCR
                                flag or this will be ignored. */
} PyArrayInterface;
```

接口成员应包含5位数据应解释的内容。
数据位是 ``CONTIGUOUS`` (0x1)、``FORTRAN`` (0x2)、``ALIGNED`` (0x100)、``NOTSWAPPED`` (0x200) 和 ``WRITEABLE`` (0x400)。
最终标志 ``ARR_HAS_DESCR``(0x800) 表示该结构是否具有停止字段。
除非存在此标志，否则不应考虑该字段。

**自2006年6月16日起新增：**

在过去，``PyCObject`` 的 “desc” 成员本身使用上面 [``PyArrayInterface``](c-api/types-and-structures.html#c.PyArrayInterface) 结构的 “descr” 成员来公开指向暴露接口的对象的指针。这是界面的明确部分。使用``PyCObject_FromVoidPtrAndDesc`` 创建 ``PyCObject`` 时，请务必拥有对象的引用。

## 类型描述示例

为清楚起见，提供类型描述和相应的 [``__array_interface__``](#__array_interface__) ‘descr’ 条目的一些示例是有用的。
感谢Scott Gilbert的这些例子：

在每种情况下，‘descr’ 键是可选的，但当然提供了对于各种应用可能很重要的更多信息：

``` python
* Float data
    typestr == '>f4'
    descr == [('','>f4')]

* Complex double
    typestr == '>c8'
    descr == [('real','>f4'), ('imag','>f4')]

* RGB Pixel data
    typestr == '|V3'
    descr == [('r','|u1'), ('g','|u1'), ('b','|u1')]

* Mixed endian (weird but could happen).
    typestr == '|V8' (or '>u8')
    descr == [('big','>i4'), ('little','<i4')]

* Nested structure
    struct {
        int ival;
        struct {
            unsigned short sval;
            unsigned char bval;
            unsigned char cval;
        } sub;
    }
    typestr == '|V8' (or '<u8' if you want)
    descr == [('ival','<i4'), ('sub', [('sval','<u2'), ('bval','|u1'), ('cval','|u1') ]) ]

* Nested array
    struct {
        int ival;
        double data[16*4];
    }
    typestr == '|V516'
    descr == [('ival','>i4'), ('data','>f8',(16,4))]

* Padded structure
    struct {
        int ival;
        double dval;
    }
    typestr == '|V16'
    descr == [('ival','>i4'),('','|V4'),('dval','>f8')]
```

应该清楚的是，可以使用该接口描述任何结构化类型。

## 与数组接口（版本2）的差异

版本2界面非常相似。差异主要是审美。特别是：

1. PyArrayInterface 结构最后没有 descr 成员（因此没有标志ARR_HAS_DESCR）。
1. 未指定从 \_\_array_struct__ 返回的PyCObject的后代。
通常，它是暴露数组的对象（因此当C对象被销毁时，对它的引用可以被破坏）。
现在它必须是一个元组，其第一个元素是一个带有“PyArrayInterface Version＃”的字符串，
其第二个元素是暴露数组的对象。
1. 从 \_\_array_interface__['data'] 返回的元组曾经是一个十六进制字符串（现在它是一个整数或一个长整数）。
1. 没有 \_\_array_interface__ 属性而不是 \_\_array_interface__ 字典中的所有键（版本除外）都是它们自己的属性：因此获取python端信息
    - __array_data__
    - __array_shape__
    - __array_strides__
    - __array_typestr__
    - __array_descr__
    - __array_offset__
    - __array_mask__