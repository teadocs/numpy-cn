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

> A 2-tuple whose first argument is an integer (a long integer if necessary) that points to the data-area storing the array contents. This pointer must point to the first element of data (in other words any offset is always ignored in this case). The second entry in the tuple is a read-only flag (true means the data area is read-only).
> 
> This attribute can also be an object exposing the buffer interface which will be used to share the data. If this key is not present (or returns None), then memory sharing will be done through the buffer interface of the object itself. In this case, the offset key can be used to indicate the start of the buffer. A reference to the object exposing the array interface must be stored by the new object if the memory area is to be secured.
> 
> **Default**: ``None``

**strides** (optional)

> Either ``None`` to indicate a C-style contiguous array or a Tuple of strides which provides the number of bytes needed to jump to the next array element in the corresponding dimension. Each entry must be an integer (a Python int or long). As with shape, the values may be larger than can be represented by a C “int” or “long”; the calling code should handle this appropriately, either by raising an error, or by using Py_LONG_LONG in C. The default is None which implies a C-style contiguous memory buffer. In this model, the last dimension of the array varies the fastest. For example, the default strides tuple for an object whose array entries are 8 bytes long and whose shape is (10,20,30) would be (4800, 240, 8)
> 
> **Default:** ``None`` (C-style contiguous)

**mask** (optional)

> ``None`` or an object exposing the array interface. All elements of the mask array should be interpreted only as true or not true indicating which elements of this array are valid. The shape of this object should be “broadcastable” to the shape of the original array.
> 
> Default: None (All array values are valid)

**offset** (optional)

> An integer offset into the array data region. This can only be used when data is ``None`` or returns a ``buffer`` object.
> 
> **Default**: 0.

**version** (required)

> An integer showing the version of the interface (i.e. 3 for this version). Be careful not to use this to invalidate objects exposing future versions of the interface.

## C-struct access

This approach to the array interface allows for faster access to an array using only one attribute lookup and a well-defined C-structure.

``__array_struct__``

A :c:type: PyCObject whose voidptr member contains a pointer to a filled PyArrayInterface structure. Memory for the structure is dynamically created and the PyCObject is also created with an appropriate destructor so the retriever of this attribute simply has to apply Py_DECREF to the object returned by this attribute when it is finished. Also, either the data needs to be copied out, or a reference to the object exposing this attribute must be held to ensure the data is not freed. Objects exposing the __array_struct__ interface must also not reallocate their memory if other objects are referencing them.

The PyArrayInterface structure is defined in ``numpy/ndarrayobject.h`` as:

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

The flags member may consist of 5 bits showing how the data should be interpreted and one bit showing how the Interface should be interpreted. The data-bits are CONTIGUOUS (0x1), FORTRAN (0x2), ALIGNED (0x100), NOTSWAPPED (0x200), and WRITEABLE (0x400). A final flag ARR_HAS_DESCR (0x800) indicates whether or not this structure has the arrdescr field. The field should not be accessed unless this flag is present.

**New since June 16, 2006:**

In the past most implementations used the “desc” member of the PyCObject itself (do not confuse this with the “descr” member of the PyArrayInterface structure above — they are two separate things) to hold the pointer to the object exposing the interface. This is now an explicit part of the interface. Be sure to own a reference to the object when the PyCObject is created using PyCObject_FromVoidPtrAndDesc.

## Type description examples

For clarity it is useful to provide some examples of the type description and corresponding ``__array_interface__`` ‘descr’ entries. Thanks to Scott Gilbert for these examples:

In every case, the ‘descr’ key is optional, but of course provides more information which may be important for various applications:

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

It should be clear that any structured type could be described using this interface.

## Differences with Array interface (Version 2)

The version 2 interface was very similar. The differences were largely aesthetic. In particular:

1. The PyArrayInterface structure had no descr member at the end (and therefore no flag ARR_HAS_DESCR)
1. The desc member of the PyCObject returned from __array_struct__ was not specified. Usually, it was the object exposing the array (so that a reference to it could be kept and destroyed when the C-object was destroyed). Now it must be a tuple whose first element is a string with “PyArrayInterface Version #” and whose second element is the object exposing the array.
1. The tuple returned from __array_interface__[‘data’] used to be a hex-string (now it is an integer or a long integer).
1. There was no __array_interface__ attribute instead all of the keys (except for version) in the __array_interface__ dictionary were their own attribute: Thus to obtain the Python-side information you had to access separately the attributes:
    - __array_data__
    - __array_shape__
    - __array_strides__
    - __array_typestr__
    - __array_descr__
    - __array_offset__
    - __array_mask__