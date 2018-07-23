# 子类化ndarray

## 介绍

子类化ndarray相对简单，但与其他Python对象相比，它有一些复杂性。 在这个页面上，我们解释了允许你子类化ndarray的机制，以及实现子类的含义。

### ndarrays和对象创建

ndarray的子类化很复杂，因为ndarray类的新实例可以以三种不同的方式出现。 这些是：

1. 显式构造函数调用 - 如``MySubClass（params）``。 这是创建Python实例的常用途径。
1. 查看转换 - 将现有的ndarray转换为给定的子类。
1. 从模板创建新实例-从模板实例创建新实例。示例包括从子类数组返回片、从uFuncs创建返回类型和复制数组。有关更多详细信息，请参见[从模板创建](#从模版创建)。

最后两个是ndarrays的特征 - 为了支持数组切片之类的东西。 子类化ndarray的复杂性是由于numpy必须支持后两种实例创建路径的机制。

## 视图投影

视图投影是标准的ndarray机制，通过它您可以获取任何子类的ndarray，并将该数组的视图作为另一个（指定的）子类返回：

```python
>>> import numpy as np
>>> # create a completely useless ndarray subclass
>>> class C(np.ndarray): pass
>>> # create a standard ndarray
>>> arr = np.zeros((3,))
>>> # take a view of it, as our useless subclass
>>> c_arr = arr.view(C)
>>> type(c_arr)
<class 'C'>
```

## 从模版创建

当numpy发现它需要从模板实例创建新实例时，ndarray子类的新实例也可以通过与视图转换非常相似的机制来实现。 这个情况的最明显的时候是你正为子类阵列切片的时候。例如：

```python
>>> v = c_arr[1:]
>>> type(v) # the view is of type 'C'
<class 'C'>
>>> v is c_arr # but it's a new instance
False
```

切片是原始 ``C_ARR`` 数据的视图。因此，当我们从ndarray获取视图时，我们返回一个新的ndarray，它属于同一个类，指向原始的数据。

在使用ndarray时，我们还需要这样的视图，比如复制数组(``C_arr.Copy()``)、创建ufunc输出数组(关于uFunc函数和其他函数，也请参阅_array_warp___ )和简化方法(比如 ``C_arr.Means()`` )。

## 视图投影和从模版创建的关系

这些路径都使用相同的机制。我们在这里进行区分，因为它们会为您的方法产生不同的输入。 具体来说，View转换意味着您已从ndarray的任何潜在子类创建了数组类型的新实例。从模板创建新意味着您已从预先存在的实例创建了类的新实例，例如，允许您跨特定于您的子类的属性进行复制。

## 子类化的含义

如果我们继承ndarray，我们不仅需要处理数组类型的显式构造，还需要处理视图投影或从模板创建。NumPy有这样的机制，这种机制使子类略微不标准。

ndarray用于支持视图和子类中的new-from-template（从模版创建）的机制有两个方面。

第一个是使用``ndarray .__ new__``方法进行对象初始化的主要工作，而不是更常用的``__init__``方法。 第二个是使用``__array_finalize__``方法，允许子类在创建视图和模板中的新实例后进行内存清除。

### 关于在Python中的 ``__new__`` 和 ``__init__`` 的简短入门

``__ new__``是一个标准的Python方法，如果存在，在创建类实例时在``__init__``之前调用。 有关更多详细信息，请参阅 [python __new__文档](https://docs.python.org/3/reference/datamodel.html#object.__new__)。

例如，请思考以下Python代码：

```python
class C(object):
    def __new__(cls, *args):
        print('Cls in __new__:', cls)
        print('Args in __new__:', args)
        return object.__new__(cls, *args)

    def __init__(self, *args):
        print('type(self) in __init__:', type(self))
        print('Args in __init__:', args)
```

思考后我们可以得到：

```python
>>> c = C('hello')
Cls in __new__: <class 'C'>
Args in __new__: ('hello',)
type(self) in __init__: <class 'C'>
Args in __init__: ('hello',)
```

当我们调用``C（'hello'）``时，``__ new__``方法将自己的类作为第一个参数，并传递参数，即字符串``'hello'``。 在python调用 ``__new__`` 之后，它通常（见下文）调用我们的 ``__init__`` 方法，将``__new__`` 的输出作为第一个参数（现在是一个类实例），然后传递参数。

正如你所看到的，对象可以在`__new__``方法或``__init__``方法中初始化，或两者兼而有之，实际上ndarray没有``__init__``方法，因为所有的初始化都是 在``__new__``方法中完成。

为什么要使用``__new__``而不是通常的``__init__``？ 因为在某些情况下，对于ndarray，我们希望能够返回其他类的对象。 考虑以下：

```python
class D(C):
    def __new__(cls, *args):
        print('D cls is:', cls)
        print('D args in __new__:', args)
        return C.__new__(C, *args)

    def __init__(self, *args):
        # we never get here
        print('In D __init__')
```

实践后：

```python
>>> obj = D('hello')
D cls is: <class 'D'>
D args in __new__: ('hello',)
Cls in __new__: <class 'C'>
Args in __new__: ('hello',)
>>> type(obj)
<class 'C'>
```

``C``的定义与之前相同，但对于``D``，``__new__``方法返回类``C``而不是``D``的实例。 请注意，``D``的``__init__``方法不会被调用。 通常，当``__new__``方法返回除定义它的类之外的类的对象时，不调用该类的``__init__``方法。

这就是ndarray类的子类如何能够返回保留类类型的视图。 在观察时，标准的ndarray机器创建了新的ndarray对象，例如：

```python
obj = ndarray.__new__(subtype, shape, ...
```

其中``subdtype``是子类。 因此，返回的视图与子类属于同一类，而不是类``ndarray``。

这解决了返回相同类型视图的问题，但现在我们遇到了一个新问题。 ndarray的机制可以用这种方式设置类，在它的标准方法中获取视图，但是ndarray``__new__``方法不知道我们在自己的``__new__``方法中做了什么来设置属性， 等等。 （旁白 - 为什么不调用``obj = subdtype .__ new __（...``那么？因为我们可能没有一个带有相同调用特征的`__new__``方法）。

### ``__array_finalize__`` 的作用

``__array_finalize__``是numpy提供的机制，允许子类处理创建新实例的各种方法。

请记住，子类实例可以通过以下三种方式实现：

1. 显式构造函数调用（``obj = MySubClass（params）``）。 这将调用通常的``MySubClass .__ new__``方法，然后再调用（如果存在）``MySubClass .__ init__``。
1. 视图投影。
1. 从模板创建。

我们的``MySubClass .__ new__``方法仅在显式构造函数调用的情况下被调用，因此我们不能依赖于``MySubClass .__ new__``或``MySubClass .__ init__``来处理视图投影和“从模板创建”。 事实证明，对于所有三种对象创建方法都会调用``MySubClass .__ array_finalize__``，所以这就是我们的对象从内部创建理通常会发生的情况。

- 对于显式构造函数调用，我们的子类需要创建自己的类的新ndarray实例。 在实践中，这意味着我们作为代码的编写者，需要调用``ndarray .__ new __（MySubClass，...）``，一个类的层次结构会调用``super（MySubClass，cls） .__ new __（cls，...``），或查看现有数组的视图投影（见下文）
- 对于视图投影和“从模板创建”，相当于``ndarray .__ new __（MySubClass，...``在C级的调用。
“__array_finalize__``收到的参数因上面三种实例创建方法而异。

以下代码允许我们查看调用顺序和参数：

```python
import numpy as np

class C(np.ndarray):
    def __new__(cls, *args, **kwargs):
        print('In __new__ with class %s' % cls)
        return super(C, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # in practice you probably will not need or want an __init__
        # method for your subclass
        print('In __init__ with class %s' % self.__class__)

    def __array_finalize__(self, obj):
        print('In array_finalize:')
        print('   self type is %s' % type(self))
        print('   obj type is %s' % type(obj))
```

现在看:

```python
>>> # Explicit constructor
>>> c = C((10,))
In __new__ with class <class 'C'>
In array_finalize:
   self type is <class 'C'>
   obj type is <type 'NoneType'>
In __init__ with class <class 'C'>
>>> # View casting
>>> a = np.arange(10)
>>> cast_a = a.view(C)
In array_finalize:
   self type is <class 'C'>
   obj type is <type 'numpy.ndarray'>
>>> # Slicing (example of new-from-template)
>>> cv = c[:1]
In array_finalize:
   self type is <class 'C'>
   obj type is <class 'C'>
```

``__array_finalize__ `` 的特征是:

```python
def __array_finalize__(self, obj):
```

可以看到``super``调用，它转到``ndarray .__ new__``，将``__array_finalize__``传递给我们自己的类（``self``）的新对象以及来自的对象 视图已被采用（``obj``）。 从上面的输出中可以看出，``self``始终是我们子类的新创建的实例，而``obj``的类型对于三个实例创建方法是不同的：

- 当从显式构造函数调用时，“obj”是“None”。
- 当从视图转换调用时，``obj``可以是ndarray的任何子类的实例，包括我们自己的子类。
- 当在新模板中调用时，``obj``是我们自己子类的另一个实例，我们可以用它来更新的``self``实例。

因为``__array_finalize__``是唯一始终看到正在创建新实例的方法，所以它是填充新对象属性的实例的默认值以及其他任务的合适的位置。

通过示例可能更清楚。

## 简单示例 - 向ndarray添加额外属性

```python
import numpy as np

class InfoArray(np.ndarray):

    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(InfoArray, subtype).__new__(subtype, shape, dtype,
                                                buffer, offset, strides,
                                                order)
        # set the new 'info' attribute to the value passed
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.info = getattr(obj, 'info', None)
        # We do not need to return anything
```

使用该对象如下所示：

```python
>>> obj = InfoArray(shape=(3,)) # explicit constructor
>>> type(obj)
<class 'InfoArray'>
>>> obj.info is None
True
>>> obj = InfoArray(shape=(3,), info='information')
>>> obj.info
'information'
>>> v = obj[1:] # new-from-template - here - slicing
>>> type(v)
<class 'InfoArray'>
>>> v.info
'information'
>>> arr = np.arange(10)
>>> cast_arr = arr.view(InfoArray) # view casting
>>> type(cast_arr)
<class 'InfoArray'>
>>> cast_arr.info is None
True
```

这个类不是很有用，因为它与裸ndarray对象具有相同的构造函数，包括传入缓冲区和形状等等。我们可能更偏向于希望构造函数能够将已经构成的ndarray类型通过常用的numpy的``np.array``来调用并返回一个对象。

## 更真实的示例 - 添加到现有数组的属性

这是一个类，它采用已经存在的标准ndarray，转换为我们的类型，并添加一个额外的属性。

```python
import numpy as np

class RealisticInfoArray(np.ndarray):

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)
```

所以:

```python
>>> arr = np.arange(5)
>>> obj = RealisticInfoArray(arr, info='information')
>>> type(obj)
<class 'RealisticInfoArray'>
>>> obj.info
'information'
>>> v = obj[1:]
>>> type(v)
<class 'RealisticInfoArray'>
>>> v.info
'information'
```

## ufuncs的``__array_ufunc__``

> 版本1.13中的新功能。

子类可以通过重写默认的``ndarray.__arrayufunc_``方法来重写在其上执行numpy uFunc函数时的行为。此方法将代替ufunc执行，如果未实现所请求的操作，则应返回操作结果或`NotImplemented``。

``__array_ufunc__`` 的特征是:

```python
def __array_ufunc__(ufunc, method, *inputs, **kwargs):

- *ufunc* 是被调用的ufunc对象。
- *method* 是一个字符串，指示如何调用Ufunc。“__call__” 表示它是直接调用的，或者是下面的其中一个：`methods <ufuncs.methods>`：“reduce”，“accumulate”，“reduceat”，“outer” 或 “at” 等属性。
- *inputs* 是 “ufunc” 的类型为元组的输入参数。
- *kwargs* A包含传递给函数的任何可选或关键字参数。 这包括任何``out``参数，并且它们总是包含在元组中。
```

典型的实现将转换任何作为自己类的实例的输入或输出，使用``super()``方法将所有内容传递给超类，最后在可能的反向转换后返回结果。 从``core / tests / test_umath.py``中的测试用例``test_ufunc_override_with_super``获取的示例如下。

```python
input numpy as np

class A(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, A):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, A):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super(A, self).__array_ufunc__(ufunc, method,
                                                 *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], A):
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(A)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], A):
            results[0].info = info

        return results[0] if len(results) == 1 else results
```

所以，这个类实际上没有做任何有趣的事情：它只是将它自己的任何实例转换为常规的ndarray（否则，我们会得到无限的递归！），并添加一个``info``字典，告诉哪些输入和输出它转换。因此，例如：

```python
>>> a = np.arange(5.).view(A)
>>> b = np.sin(a)
>>> b.info
{'inputs': [0]}
>>> b = np.sin(np.arange(5.), out=(a,))
>>> b.info
{'outputs': [0]}
>>> a = np.arange(5.).view(A)
>>> b = np.ones(1).view(A)
>>> c = a + b
>>> c.info
{'inputs': [0, 1]}
>>> a += b
>>> a.info
{'inputs': [0, 1], 'outputs': [0]}
```

Note that another approach would be to to use ``getattr(ufunc, methods)(*inputs, **kwargs)`` instead of the ``super`` call. For this example, the result would be identical, but there is a difference if another operand also defines ``__array_ufunc__``. E.g., lets assume that we evalulate ``np.add(a, b)``, where ``b`` is an instance of another class B that has an override. If you use ``super`` as in the example, ``ndarray.__array_ufunc__`` will notice that ``b`` has an override, which means it cannot evaluate the result itself. Thus, it will return NotImplemented and so will our class ``A``. Then, control will be passed over to ``b``, which either knows how to deal with us and produces a result, or does not and returns NotImplemented, raising a ``TypeError``.

If instead, we replace our ``super`` call with ``getattr(ufunc, method)``, we effectively do ``np.add(a.view(np.ndarray), b)``. Again, ``B.__array_ufunc__`` will be called, but now it sees an ndarray as the other argument. Likely, it will know how to handle this, and return a new instance of the ``B`` class to us. Our example class is not set up to handle this, but it might well be the best approach if, e.g., one were to re-implement ``MaskedArray`` using ``__array_ufunc__``.

As a final note: if the ``super`` route is suited to a given class, an advantage of using it is that it helps in constructing class hierarchies. E.g., suppose that our other class ``B`` also used the ``super`` in its ``__array_ufunc__`` implementation, and we created a class ``C`` that depended on both, i.e., ``class C(A, B)`` (with, for simplicity, not another ``__array_ufunc__`` override). Then any ufunc on an instance of ``C`` would pass on to ``A.__array_ufunc__``, the ``super`` call in ``A`` would go to ``B.__array_ufunc__``, and the ``super`` call in ``B`` would go to ``ndarray.__array_ufunc__``, thus allowing ``A`` and ``B`` to collaborate.

## ``__array_wrap__`` for ufuncs and other functions

Prior to numpy 1.13, the behaviour of ufuncs could only be tuned using ``__array_wrap__`` and ``__array_prepare__``. These two allowed one to change the output type of a ufunc, but, in constrast to ``__array_ufunc__``, did not allow one to make any changes to the inputs. It is hoped to eventually deprecate these, but ``__array_wrap__`` is also used by other numpy functions and methods, such as ``squeeze``, so at the present time is still needed for full functionality.

Conceptually, ``__array_wrap__`` “wraps up the action” in the sense of allowing a subclass to set the type of the return value and update attributes and metadata. Let’s show how this works with an example. First we return to the simpler example subclass, but with a different name and some print statements:

```python
import numpy as np

class MySubClass(np.ndarray):

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        print('In __array_finalize__:')
        print('   self is %s' % repr(self))
        print('   obj is %s' % repr(obj))
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def __array_wrap__(self, out_arr, context=None):
        print('In __array_wrap__:')
        print('   self is %s' % repr(self))
        print('   arr is %s' % repr(out_arr))
        # then just call the parent
        return super(MySubClass, self).__array_wrap__(self, out_arr, context)
```

We run a ufunc on an instance of our new array:

```python
>>> obj = MySubClass(np.arange(5), info='spam')
In __array_finalize__:
   self is MySubClass([0, 1, 2, 3, 4])
   obj is array([0, 1, 2, 3, 4])
>>> arr2 = np.arange(5)+1
>>> ret = np.add(arr2, obj)
In __array_wrap__:
   self is MySubClass([0, 1, 2, 3, 4])
   arr is array([1, 3, 5, 7, 9])
In __array_finalize__:
   self is MySubClass([1, 3, 5, 7, 9])
   obj is MySubClass([0, 1, 2, 3, 4])
>>> ret
MySubClass([1, 3, 5, 7, 9])
>>> ret.info
'spam'
```

Note that the ufunc (``np.add``) has called the ``__array_wrap__`` method with arguments ``self`` as ``obj``, and ``out_arr`` as the (ndarray) result of the addition. In turn, the default ``__array_wrap__`` (ndarray.``__array_wrap__``) has cast the result to class MySubClass, and called ``__array_finalize__`` - hence the copying of the ``info`` attribute. This has all happened at the C level.

But, we could do anything we wanted:

```python
class SillySubClass(np.ndarray):

    def __array_wrap__(self, arr, context=None):
        return 'I lost your data'
>>> arr1 = np.arange(5)
>>> obj = arr1.view(SillySubClass)
>>> arr2 = np.arange(5)
>>> ret = np.multiply(obj, arr2)
>>> ret
'I lost your data'
```

So, by defining a specific ``__array_wrap__`` method for our subclass, we can tweak the output from ufuncs. The ``__array_wrap__`` method requires ``self``, then an argument - which is the result of the ufunc - and an optional parameter context. This parameter is returned by ufuncs as a 3-element tuple: (name of the ufunc, arguments of the ufunc, domain of the ufunc), but is not set by other numpy functions. Though, as seen above, it is possible to do otherwise, ``__array_wrap__`` should return an instance of its containing class. See the masked array subclass for an implementation.

In addition to ``__array_wrap__``, which is called on the way out of the ufunc, there is also an ``__array_prepare__`` method which is called on the way into the ufunc, after the output arrays are created but before any computation has been performed. The default implementation does nothing but pass through the array. ``__array_prepare__`` should not attempt to access the array data or resize the array, it is intended for setting the output array type, updating attributes and metadata, and performing any checks based on the input that may be desired before computation begins. Like ``__array_wrap__``, ``__array_prepare__`` must return an ndarray or subclass thereof or raise an error.

## Extra gotchas - custom ``__del__`` methods and ndarray.base

One of the problems that ndarray solves is keeping track of memory ownership of ndarrays and their views. Consider the case where we have created an ndarray, ``arr`` and have taken a slice with ``v = arr[1:]``. The two objects are looking at the same memory. NumPy keeps track of where the data came from for a particular array or view, with the ``base`` attribute:

```python
>>> # A normal ndarray, that owns its own data
>>> arr = np.zeros((4,))
>>> # In this case, base is None
>>> arr.base is None
True
>>> # We take a view
>>> v1 = arr[1:]
>>> # base now points to the array that it derived from
>>> v1.base is arr
True
>>> # Take a view of a view
>>> v2 = v1[1:]
>>> # base points to the view it derived from
>>> v2.base is v1
True
```

In general, if the array owns its own memory, as for ``arr`` in this case, then ``arr.base`` will be None - there are some exceptions to this - see the numpy book for more details.

The ``base`` attribute is useful in being able to tell whether we have a view or the original array. This in turn can be useful if we need to know whether or not to do some specific cleanup when the subclassed array is deleted. For example, we may only want to do the cleanup if the original array is deleted, but not the views. For an example of how this can work, have a look at the ``memmap`` class in ``numpy.core``.

## Subclassing and Downstream Compatibility

When sub-classing ``ndarray`` or creating duck-types that mimic the ``ndarray`` interface, it is your responsibility to decide how aligned your APIs will be with those of numpy. For convenience, many numpy functions that have a corresponding ``ndarray`` method (e.g., ``sum``, ``mean``, ``take``, ``reshape``) work by checking if the first argument to a function has a method of the same name. If it exists, the method is called instead of coercing the arguments to a numpy array.

For example, if you want your sub-class or duck-type to be compatible with numpy’s sum function, the method signature for this object’s ``sum`` method should be the following:

```python
def sum(self, axis=None, dtype=None, out=None, keepdims=False):
...
```

This is the exact same method signature for ``np.sum``, so now if a user calls ``np.sum`` on this object, numpy will call the object’s own ``sum`` method and pass in these arguments enumerated above in the signature, and no errors will be raised because the signatures are completely compatible with each other.

If, however, you decide to deviate from this signature and do something like this:

```python
def sum(self, axis=None, dtype=None):
...
```

This object is no longer compatible with ``np.sum`` because if you call ``np.sum``, it will pass in unexpected arguments ``out`` and ``keepdims``, causing a TypeError to be raised.

If you wish to maintain compatibility with numpy and its subsequent versions (which might add new keyword arguments) but do not want to surface all of numpy’s arguments, your function’s signature should accept **kwargs. For example:

```python
def sum(self, axis=None, dtype=None, **unused_kwargs):
...
```

This object is now compatible with ``np.sum`` again because any extraneous arguments (i.e. keywords that are not ``axis`` or ``dtype``) will be hidden away in the ``**unused_kwargs`` parameter.