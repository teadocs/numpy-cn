# 数组接口

> **注意**
> 本页描述了特定于numpy的API，用于从其他C扩展访问numpy数组的内容。PEP 3118-修订后的缓冲区协议为Python2.6和3.0引入了类似的标准化API，可供任何扩展模块使用。Cython的缓冲区数组支持使用PEP3118API；请参阅Cython numpy教程。Cython提供了一种编写支持缓冲区协议的代码的方法，它的Python版本超过了2.6版本，因为它有一个向后兼容的实现，它使用了这里描述的数组接口。

**版本: 3**

数组接口(有时称为数组协议)创建于2005年，作为类似数组的Python对象的一种手段，只要有可能，就可以明智地重用彼此的数据缓冲区。同构N维数组接口是对象共享N维数组内存和信息的默认机制。该接口由Python端和C端组成，使用两个属性。希望在应用程序代码中被视为N维数组的对象应该至少支持这些属性中的一个。希望在应用程序代码中支持N维数组的对象应该至少查找这些属性中的一个，并适当地使用所提供的信息。

此接口描述同构数组，即数组的每个项具有相同的“类型”。这种类型可以是非常简单的，也可以是相当任意和复杂的类C结构。

有两种使用接口的方法：Python端和C端。两者都是独立的属性。

## Python中

这种接口方法由具有 ``__array_interface__`` 属性的对象组成。

``__array_interface__``

项目字典（需要3个，5个可选）。 如果未提供字典，则字典中的可选键隐含默认值。

关键字是：

**shape** (必须)

> 元组，其元素是每个维度中的数组大小。每个条目都是一个整数(Python、int或Long)。请注意，这些整数可能大于平台“int”或“Long”所能容纳的大小(Pythonint是C长)。这取决于使用此属性的代码来适当地处理此问题；可以在可能发生溢出时引发错误，也可以使用Py_LONG_LONG作为形状的C类型。

**typestr** (必须)

> 提供同构数组基本类型的字符串基本字符串格式由3部分组成：描述数据字节顺序的字符 (<：little-endian>:big-endian,|:not-relevant)，字符代码 给出数组的基本类型，以及提供类型使用的字节数的整数。
> 
> 基本类型字符代码是：
> 
> - t	位字段(后面的整数给出位字段中的位数)。
> - b	布尔型(整数类型，其中所有值仅为True或false)
> - i	整数型
> - u	无符号整数型
> - f	浮点型
> - c	复杂浮点型
> - m	Timedelta型
> - M	日期时间型
> - O	对象（即内存包含指向PyObject的指针）
> - S	字符串（固定长度的char序列）
> - U	Unicode（Py_UNICODE的固定长度序列）
> - V	其他（void * - 每个项目都是固定大小的内存块）

**descr** (可选)

> 元组列表，提供同类数组中每个项的内存布局的更详细描述。 列表中的每个元组都有两个或三个元素。 通常，当typestr为V[0-9]+时，将使用此属性，但这不是必需的。 唯一的要求是typestr键中表示的字节数与此处表示的总字节数相同。 这个想法是支持构成数组元素的类C结构的描述。列表中每个元组的元素是：
> 
> 1. 提供与此数据类型部分关联的名称的字符串。 这也可以是（'full name'，'basic_name'）的元组，其中基本名称将是表示字段全名的有效Python变量名称。
> 1. 在typestr或另一个列表中的基本类型描述字符串（对于嵌套的结构化类型）
> 1. 一个可选的形状元组，提供应重复此部分结构的次数。 如果没有给出，则不假设重复。 使用这种通用接口可以描述非常复杂的结构。 但请注意，数组的每个元素仍然具有相同的数据类型。 下面给出了使用该接口的一些示例。
> **Default:** ``[('', typestr)]``

**data** (可选)

> 一个2元组，其第一个参数是一个整数（必要时是一个长整数），指向存储数组内容的数据区。 该指针必须指向数据的第一个元素（换句话说，在这种情况下总是忽略任何偏移）。 一个2元组，其第一个参数是一个整数（必要时是一个长整数），指向存储数组内容的数据区。 该指针必须指向数据的第一个元素（换句话说，在这种情况下总是忽略任何偏移）。 元组中的第二个条目是只读标志（true表示数据区域是只读的。）元组中的第二个条目是只读标志（true表示数据区域是只读的）。
> 
> 该属性也可以是暴露将用于共享数据的缓冲区接口的对象。 如果此键不存在（或返回None），则将通过对象本身的缓冲区接口完成内存共享。 在这种情况下，偏移键可用于指示缓冲区的开始。 如果要保护存储区，则必须由新对象存储对暴露数组接口的对象的引用。
> 
> **Default**: ``None``

**strides** (可选)

> ``None``表示C风格的连续数组或步长元组，它提供跳转到相应维度中下一个数组元素所需的字节数。 每个条目必须是整数（Python int或long）。 与形状一样，值可以大于可由C“int”或“long”表示的值; 调用代码应该通过引发错误或在C中使用Py_LONG_LONG来适当地处理它。默认值为None，这意味着C风格的连续内存缓冲区。 在此模型中，阵列的最后一个维度变化最快。 例如，对于数组条目长度为8个字节且形状为（10,20,30）的对象的默认步长元组将为（4800,240,8）
> 
> **Default:** ``None`` (C-style contiguous)

**mask** (可选)

> ``None``或暴露数组接口的对象。 掩码数组的所有元素应仅解释为true或不为true，指示此数组的哪些元素有效。 该对象的形状应该是“可广播”到原始数组的形状。
> 
> Default: None (所有数组值都有效)

**offset** (可选))

> 数组数据区域中的整数偏移量。 这只能在数据为“None”时返回或返回“缓冲区”对象时使用。
> 
> **Default**: 0.

**version** (必须)

> 显示接口版本的整数（即此版本为3）。 注意不要使用它来使暴露未来版本的接口的对象无效。

## C结构的访问

这种数组接口方法允许仅使用一个属性查找和明确定义的C结构能更快地访问数组。

``__array_struct__``

A :C:type:PyCObject，其voidptr成员包含一个指向已填充的PyArrayInterface结构的指针。结构的内存是动态创建的，PyCObject也是用适当的析构函数创建的，因此该属性的检索器只需在完成时将Py_DECREF应用于该属性返回的对象。另外，需要将数据复制出来，或者必须保留对暴露此属性的对象的引用，以确保数据未被释放。如果其他对象正在引用它们，则暴露__arraystruct__接口的对象也不能重新分配它们的内存。

PyArrayInterface结构在 ``numpy/ndarrayObject.`` 中定义为：

```c
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

标志成员可以由表示如何解释数据的5位和表示如何解释接口的一位组成。数据位是连续的(0x1)、FORTRAN(0x2)、对齐(0x100)、NOTSWAPPED(0x200)和可写(0x400)。最后一个标志ARR_HAS_DISDISR(0x800)指示该结构是否有arrdesr字段。除非出现此标志，否则不应访问该字段。

**2006年6月16日以来的新特性：**

在过去，大多数实现使用PyCObject本身的“desc”成员(不要将其与上面PyArrayInterface结构的“下降”成员混淆-它们是两个独立的东西)来保存指向暴露接口的对象的指针。这现在是接口的显式部分。当使用PyCObject_FromVoidPtrAndDesc创建PyCObject时，请确保拥有对该对象的引用。

## 类型描述实例

为清楚起见，提供类型描述和相应的 ``__array_interface__`` 'descr'条目的一些示例是有用的。感谢Scott Gilbert的这些例子：

在每种情况下，'descr'键都是可选的，但当然提供了对于各种应用可能很重要的更多信息：

```python
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

应该清楚的是，可以使用该接口来描述任何结构化类型。

## 与Array接口（版本2）的差异

版本2界面非常相似。差异主要是审美。尤其是：

1. PyArrayInterface结构最后没有descr成员（因此没有标志ARR_HAS_DESCR）
1. 未指定从__array_struct__返回的PyCObject的desc成员。 通常，它是暴露数组的对象（因此当C对象被销毁时，可以保留和销毁对它的引用）。 现在它必须是一个元组，其第一个元素是带有“PyArrayInterface Version＃”的字符串，其第二个元素是暴露数组的对象。
1. 从__array_interface __ ['data']返回的元组曾经是一个十六进制字符串（现在它是一个整数或一个长整数）。
1. 没有__array_interface__属性，而__array_interface__字典中的所有键（版本除外）都是它们自己的属性：因此要获取Python端信息，你必须单独访问属性：
    - __array_data__
    - __array_shape__
    - __array_strides__
    - __array_typestr__
    - __array_descr__
    - __array_offset__
    - __array_mask__