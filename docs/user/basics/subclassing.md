# 子类化ndarray 

## 介绍

子类化ndarray相对简单，但与其他Python对象相比，它有一些复杂性。在这个页面上，我们解释了允许你子类化ndarray的机制，以及实现子类的含义。

### ndarrays和对象创建

ndarray的子​​类化很复杂，因为ndarray类的新实例可以以三种不同的方式出现。这些是：

1. 显式构造函数调用 - 如 ``MySubClass(params)``。这是Python实例创建的常用途径。
1. 查看转换 - 将现有的ndarray转换为给定的子类
1. 模板中的新内容 - 从模板实例创建新实例。示例包括从子类化数组返回切片，从ufuncs创建返回类型以及复制数组。有关更多详细信息，请参阅
 [从模板创建](#从模板创建)

最后两个是ndarrays的特性 - 为了支持数组切片之类的东西。子类化ndarray的复杂性是由于numpy必须支持后两种实例创建路径的机制。

## 视图投影

*视图投影* 是标准的ndarray机制，通过它您可以获取任何子类的ndarray，并将该数组的视图作为另一个（指定的）子类返回：

``` python
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

## 从模板创建

当numpy发现它需要从模板实例创建新实例时，ndarray子类的新实例也可以通过与[视图投影](#视图投影)非常相似的机制来实现。
这个情况的最明显的时候是你正为子类数组切片的时候。例如：

``` python
>>> v = c_arr[1:]
>>> type(v) # the view is of type 'C'
<class 'C'>
>>> v is c_arr # but it's a new instance
False
```

切片是原始 ``c_arr`` 数据的 *视图* 。因此，当我们从ndarray中获取视图时，我们返回一个同一类的新ndarray，它指向原始数据。

在使用ndarrays时还有其它要点，我们需要这样的视图，例如复制数组（``c_arr.copy()``），创建ufunc输出数组（参见[__array_wrap__用于ufuncs和其他函数](#array-wrap-用于ufuncs和其他函数)），
以及减少方法（如``c_arr.mean()``。

## 视图投影与从模板创建的关系

这些路径都使用相同的机器。我们在这里进行区分，因为它们会为您的方法带来不同的输入。具体来说，
[视图投影](#视图投影)意味着您已从ndarray的任何潜在子类创建了数组类型的新实例。
[从模板创建](#从模板创建)意味着您已从预先存在的实例创建了类的新实例，例如，允许您跨特定于您的子类的属性进行复制。

## 子类化的含义

如果我们将 ndarray 子类化，我们不仅需要处理数组类型的显式构造，还需要处理[视图投影](#视图投影)或
[从模板创建](#从模板创建)。NumPy有这样的机制，这种机制使子类化略微不标准。

ndarray用于支持视图和子类中的从模板创建的机制有两个方面。

第一种是使用该``ndarray.__new__``方法进行对象初始化的主要工作，而不是更常用的``__init__``
方法。第二个是使用该``__array_finalize__``方法在模板创建视图和新实例后允许子类清理。

### 一个简短的Python入门``__new__``和``__init__``

``__new__``是一个标准的Python方法，如果存在，``__init__``在我们创建类实例之前调用它。
有关更多详细信息，请参阅[python \_\_new__ 文档](https://docs.python.org/reference/datamodel.html#object.__new__)。

例如，请考虑以下Python代码：

``` python
class C(object):
    def __new__(cls, *args):
        print('Cls in __new__:', cls)
        print('Args in __new__:', args)
        # The `object` type __new__ method takes a single argument.
        return object.__new__(cls)

    def __init__(self, *args):
        print('type(self) in __init__:', type(self))
        print('Args in __init__:', args)
```

它的意思是我们将会得到：

``` python
>>> c = C('hello')
Cls in __new__: <class 'C'>
Args in __new__: ('hello',)
type(self) in __init__: <class 'C'>
Args in __init__: ('hello',)
```

当我们调用时``C('hello')``，该``__new__``方法获得自己的类作为第一个参数，并传递参数，即字符串
 ``'hello'``。在python调用之后``__new__``，它通常（见下文）调用我​​们的``__init__``方法，输出``__new__``为第一个参数（现在是一个类实例），以及后面传递的参数。

如您所见，对象可以在``__new__``
方法或``__init__``方法中初始化，或者两者兼而有之，实际上ndarray没有``__init__``方法，因为所有初始化都是在``__new__``方法中完成的。

为什么要使用``__new__``而不仅仅是平常``__init__``？因为在某些情况下，对于ndarray，我们希望能够返回其他类的对象。考虑以下：

``` python
class D(C):
    def __new__(cls, *args):
        print('D cls is:', cls)
        print('D args in __new__:', args)
        return C.__new__(C, *args)

    def __init__(self, *args):
        # we never get here
        print('In D __init__')
```

意思是：

``` python
>>> obj = D('hello')
D cls is: <class 'D'>
D args in __new__: ('hello',)
Cls in __new__: <class 'C'>
Args in __new__: ('hello',)
>>> type(obj)
<class 'C'>
```

定义``C``与之前相同，但是，对于``D``，该
 ``__new__``方法返回类的实例``C``而不是
 ``D``。请注意，该``__init__``方法``D``不会被调用。通常，当``__new__``方法返回类的对象而不是定义``__init__``
它的类时，不调用该类的方法。

这就是ndarray类的子类如何能够返回保留类类型的视图。在进行视图时，标准的ndarray机器会创建新的ndarray对象，例如：

``` python
obj = ndarray.__new__(subtype, shape, ...
```

``subdtype``子类在哪里。因此，返回的视图与子类属于同一类，而不是类``ndarray``。

这解决了返回相同类型的视图的问题，但是现在我们有了一个新的问题。
ndarray的机制可以这样设置类，在其用于获取视图的标准方法中，
但是ndarray ``__new__`` 方法不知道我们在自己的 ``__new__`` 方法中为了设置属性所做的任何事情，
等等。(抛开-为什么不调用 ``obj = subdtype._new_(...`` 然后?。因为我们可能没有具有相同调用签名的 ``__new__`` 方法)。

### ``__array_finalize__`` 的作用

``__array_finalize__`` 是numpy提供的机制，允许子类处理创建新实例的各种方法。

请记住，子类实例可以通过以下三种方式实现：

1. 显式的调用构造函数（``obj = MySubClass（params）``）。 这将调用 ``MySubClass.__ new__`` 的常用序列，然后（如果存在）``MySubClass.__init__``。
1. [视图投影](#视图投影)
1. [从模板创建](#从模板创建)

我们的 ``MySubClass.__new__`` 方法只在显式构造函数调用的情况下被调用，
所以我们不能依赖 ``MySubClass.__new__`` 或 ``MySubClass.__init__`` 来处理视图转换和从模板创建。事实证明，
``MySubClass.__array_finalize__`` 确实为对象创建的所有三种方法都被调用，所以这是我们的对象创建内务通常去的地方。

- 对于显式构造函数调用，我们的子类需要创建自己的类的新ndarray实例。
在实践中，这意味着我们作为代码的作者将需要调用 ``ndarray.__new__(MySubClass,...)``, 一个类层次结构调用 ``super(MySubClass, cls).__new__(cls, ...)`` ，
或者查看现有数组的转换（见下文）
- 对于视图转换和从模板创建 ``ndarray.__new__(MySubClass,...``，在C级别调用等效项。

对于上述三种实例创建方法，``__array_finalize__`` 接收的参数不同。

以下代码允许我们查看调用序列和参数：

``` python
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

现在：

``` python
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
>>> # Slicing (example of 从模板创建)
>>> cv = c[:1]
In array_finalize:
   self type is <class 'C'>
   obj type is <class 'C'>
```

签名``__array_finalize__``是：

``` python
def __array_finalize__(self, obj):
```

可以看到进行的``super``调用
 ``ndarray.__new__``传递``__array_finalize__``了我们自己的class（``self``）的新对象以及从中获取视图的对象（``obj``）。从上面的输出可以看出，``self``它总是一个新创建的子类实例，并且``obj``
三种实例创建方法的类型不同：

- 从显式构造函数调用时，``obj``是``None``
- 从视图转换中调用时，``obj``可以是ndarray的任何子类的实例，包括我们自己的子类。
- 在从模板创建中调用时，``obj``是我们自己的子类的另一个实例，我们可能会用它来更新新``self``实例。

因为``__array_finalize__``是唯一始终看到正在创建新实例的方法，所以在其他任务中填充新对象属性的实例默认值是合理的。

通过一个例子，这可能更清楚。

## 简单示例 —— 向ndarray添加额外属性

``` python
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
        # From 从模板创建 - e.g infoarr[:3]
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

``` python
>>> obj = InfoArray(shape=(3,)) # explicit constructor
>>> type(obj)
<class 'InfoArray'>
>>> obj.info is None
True
>>> obj = InfoArray(shape=(3,), info='information')
>>> obj.info
'information'
>>> v = obj[1:] # 从模板创建 - here - slicing
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

这个类不是很有用，因为它与裸ndarray对象具有相同的构造函数，包括传入缓冲区和形状等等。我们可能更喜欢构造函数能够从通常的numpy调用中获取已经形成的ndarray ``np.array``并返回一个对象。

## 稍微更现实的例子 —— 添加到现有数组的属性

这是一个类，它采用已经存在的标准ndarray，转换为我们的类型，并添加一个额外的属性。

``` python
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

所以：

``` python
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

## ``__array_ufunc__`` 对于ufuncs 

*版本1.13中的新功能。* 

子类可以覆盖在通过覆盖默认``ndarray.__array_ufunc__``方法对其执行numpy ufuncs时发生的情况。执行此方法 *而不是*  ufunc，并且应该返回操作的结果，
或者[``NotImplemented``](https://docs.python.org/dev/library/constants.html#NotImplemented)如果未执行所请求的操作。

签名 ``__array_ufunc__`` 是：

``` python
def __array_ufunc__(ufunc, method, *inputs, **kwargs):

- *ufunc* is the ufunc object that was called.
- *method* is a string indicating how the Ufunc was called, either
  ``"__call__"`` to indicate it was called directly, or one of its
  :ref:`methods<ufuncs.methods>`: ``"reduce"``, ``"accumulate"``,
  ``"reduceat"``, ``"outer"``, or ``"at"``.
- *inputs* is a tuple of the input arguments to the ``ufunc``
- *kwargs* contains any optional or keyword arguments passed to the
  function. This includes any ``out`` arguments, which are always
  contained in a tuple.
```

典型的实现将转换作为一个人自己的类的实例的任何输入或输出，使用所有内容传递给超类
 ``super()``，并最终在可能的反向转换后返回结果。举例来说，来自测试案例采取
 ``test_ufunc_override_with_super``在``core/tests/test_umath.py``，如下。

``` python
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

所以，这个类实际上并没有做任何有趣的事情：它只是将它自己的任何实例转换为常规的ndarray（否则，我们将获得无限递归！），并添加一个``info``字典，告诉它转换了哪些输入和输出。因此，例如，

``` python
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

请注意，另一种方法是使用 ``getattr(ufunc，method)(*input，*kwargs)`` 而不是 ``super`` call。
对于本例，结果是相同的，但如果另一个操作数也定义了 ``__array_ufunc__`` ，则会有所不同。
例如，假设我们评估 ``np.add(a，b)``，其中b是具有覆盖的另一个类B的实例。
如果在示例中使用``super``，``ndarray.__array_ufunc__`` 会注意到b具有覆盖，这意味着它不能计算结果本身。
因此，它将返回 *NotImplemented* ，我们的类A也将如此。
然后，控制权将传递给 ``b``，``b`` 要么知道如何处理我们并产生结果，要么不知道并返回 *NotImplemented*，从而引发 ``TypeError``。

相反，如果我们用 ``getattr(ufunc，method)`` 替换 ``super`` call，我们将有效地执行 ``np.add(a.view(np.ndarray)，b)``。
同样，将调用 ``B.__array_ufunc__``，但现在它将 ``ndarray`` 视为另一个参数。
很可能，它将知道如何处理此问题，并将B类的新实例返回给我们。
我们的示例类没有设置为处理此问题，但如果例如使用 ``__array_ufunc__`` 重新实现 ``MaskedArray``，这可能是最好的方法。

最后要注意：如果 ``super`` 路由适合给定的类，使用它的一个优点是它有助于构造类层次结构。
例如，假设我们的其他类B在其 ``__array_ufunc__`` 实现中也使用了 ``super``，
并且我们创建了一个依赖于它们的类 ``C``，即 ``calss C(A, B)``（为简单起见，没有另一个 ``__array_ufunc__`` 覆盖）。 
然后，C实例上的任何ufunc都将传递给 ``A.__ array_ufunc__``，
``A`` 中的超级调用将转到 ``B.__ array_ufunc__``，
而 B 中的 ``super`` call 将转到 ``ndarray.__array_ufunc__`` ，从而允许 ``A`` 和 ``B`` 协作。

## ``__array_wrap__``用于ufuncs和其他函数

在numpy 1.13之前，ufuncs的行为只能使用 ``__array_wrap__`` 和  ``__array_prepare__`` 来调优。
这两个允许一个更改ufunc的输出类型，但与 ``__array_ufunc__`` 相反，不允许对输入进行任何更改。
希望最终淘汰这些功能，但是其他 numpy 函数和方法也使用 ``__array_wrap__`` ，例如 ``squeeze``，因此目前仍然需要完整的功能。

从概念上讲，``__array_wrap__`` “包装动作” 的意义是允许子类设置返回值的类型并更新属性和元数据。
让我们用一个例子来说明它是如何工作的。首先，我们返回到更简单的Example子类，但具有不同的名称和一些print语句：

``` python
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

我们在新数组的实例上运行ufunc：

``` python
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

注意，ufunc(``np.add``) 调用了 ``__array_WRAP__`` 方法，参数 ``self`` 作为 ``obj``，``out_arr``作为加法的(ndarray)结果。
反过来，默认 ``__array_wrap__(ndarray._array_warp__)`` 已将结果强制转换为类 ``MySubClass``，并调用 ``__array_finalize__``  - 因此复制了info属性。这一切都发生在C级。

但是，我们可以做任何我们想要的事情：

``` python
class SillySubClass(np.ndarray):

    def __array_wrap__(self, arr, context=None):
        return 'I lost your data'
```

``` python
>>> arr1 = np.arange(5)
>>> obj = arr1.view(SillySubClass)
>>> arr2 = np.arange(5)
>>> ret = np.multiply(obj, arr2)
>>> ret
'I lost your data'
```

因此，通过``__array_wrap__``为我们的子类定义一个特定的方法，我们可以调整ufuncs的输出。
该``__array_wrap__``方法需要``self``，然后是一个参数 - 这是ufunc的结果 - 和一个可选的参数 *上下文* 。
ufuncs 将此参数作为 3 元素元组返回:( ufunc的名称，ufunc的参数，ufunc的域），
但不是由其他numpy函数设置的。但是，如上所述，可以做其他事情，``__array_wrap__``应该返回其包含类的实例。
请参阅 masked 数组子类以获取实现。

除了 ``__array_wrap__`` 在ufunc 之外调用之外，
还有一个 ``__array_prepare__`` 方法在创建输出数组之后但在执行任何计算之前调用ufunc。
默认实现除了通过数组之外什么都不做。
``__array_prepare__`` 不应尝试访问数组数据或调整数组大小，
它用于设置输出数组类型，更新属性和元数据，以及根据计算开始之前可能需要的输入执行任何检查。
比如``__array_wrap__``，``__array_prepare__``必须返回一个ndarray或其子类或引发错误。

## 额外的坑 —— 自定义的 ``__del__`` 方法和 ndarray.base 

ndarray解决的问题之一是跟踪ndarray的内存所有权及其视图。
考虑这样的情况，我们已经创建了ndarray，``arr`` 并使用 ``v = arr[1:]``获取了一个切片。
这两个对象看的是相同的内存。NumPy使用base属性跟踪特定数组或视图的数据来自何处：

``` python
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

一般来说，如果数组拥有自己的内存，
就像``arr``在这种情况下那样，
那么``arr.base`` 将是None - 有一些例外 -—— 请参阅numpy书了解更多细节。

该``base``属性可用于判断我们是否有视图或原始数组。
如果我们需要知道在删除子类数组时是否进行某些特定的清理，这反过来会很有用。
例如，如果删除原始数组，我们可能只想进行清理，而不是视图。有关如何工作的示例，请查看 ``numpy.core`` 中的 ``memmap`` 类。

## 子类和下游兼容性

当子类化 ``ndarray`` 或创建模仿 ``ndarray`` 接口的 duck-types 时，
您的任务是决定您的API与numpy的API将如何对齐。
为方便起见，许多具有相应ndarray方法(例如，``sum``，``mean``，``take``，``reshape``)的Numpy函数通过检查函数的第一个参数是否具有同名的方法来工作。
如果存在，则调用该方法，而不是将参数强制到numpy数组。

例如，如果您希望子类或 duck-type 与 numpy 的 ``sum`` 函数兼容，则此对象``sum``方法的方法签名应如下所示：

``` python
def sum(self, axis=None, dtype=None, out=None, keepdims=False):
...
```

这是 ``np.sum`` 的完全相同的方法签名，
所以现在如果用户在这个对象上调用 ``np.sum``，numpy 将调用该对象自己的 ``sum`` 方法，
并在签名中传递上面枚举的这些参数，并且不会引发错误，因为签名彼此完全兼容。

但是，如果您决定偏离此签名并执行以下操作：

``` python
def sum(self, axis=None, dtype=None):
...
```

此对象不再兼容，``np.sum``因为如果调用``np.sum``，它将传递意外的参数，``out``并``keepdims``导致引发 TypeError。

如果你希望保持与 numpy 及其后续版本（可能添加新的关键字参数）的兼容性，
但又不想显示所有numpy的参数，那么你的函数的签名应该接受``**kwargs``。例如：

``` python
def sum(self, axis=None, dtype=None, **unused_kwargs):
...
```

此对象现在再次与 ``np.sum`` 兼容，因为任何无关的参数（即不是 ``axis`` 或 ``dtype`` 的关键字）都将隐藏在 ``*unused_kwargs`` 参数中。
