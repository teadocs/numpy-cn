<title>numpy子类化ndarray数组 - <%-__DOC_NAME__ %></title>
<meta name="keywords" content="numpy子类化ndarray数组" />

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

视图投影是标准的ndarray机制，通过它你可以获取任何子类的ndarray，并将该数组的视图作为另一个（指定的）子类返回：

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

这些路径都使用相同的机制。我们在这里进行区分，因为它们会为你的方法产生不同的输入。 具体来说，View转换意味着你已从ndarray的任何潜在子类创建了数组类型的新实例。从模板创建新意味着你已从预先存在的实例创建了类的新实例，例如，允许你跨特定于你的子类的属性进行复制。

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
注意，另一种方法是使用 ``getattr(ufunc, methods)(*inputs, **kwargs)`` 而不是调用 ``super`` 。对于此示例，结果将是相同的，但如果另一个运算数也定义 ``__array_ufunc__`` 则会存在差异。 例如，假设我们执行 ``np.add(a, b)``，其中b是另一个重写功能的b类的实例。如果你在示例中调用 ``super``， ``ndarray .__ array_ufunc__`` 会注意到b有一个覆盖，这意味着它无法评估结果本身。因此，它将返回NotImplemented，我们的类``A``也将返回。然后，控制权将传递给``b``，它知道如何处理我们并生成结果，或者不知道并返回NotImplemented，从而引发``TypeError``。

如果相反，我们用 ``getattr(ufunc, method)`` 替换我们的 ``Super`` 调用，那么我们实际上执行了``np.add(a.view(np.ndarray), b)``。同样，``b.arrayufunc_``将被调用，但现在它将一个ndarray作为另一个参数。很可能，它将知道如何处理这个问题，并将``B``类的一个新实例返回给我们。我们的示例类并不是为处理这个问题而设置的，但是，如果要使用``___array_`___ufunc_``重新实现 ``MaskedArray``，那么它很可能是最好的方法。

最后要注意：如果走``super``的路线适合给定的类，使用它的一个优点是它有助于构造类层次结构。例如，假设我们的其他类``B``在其``__array_ufunc__``实现中也使用了``super``，我们创建了一个依赖于它们的类``C``，即``class C （A，B）``（为简单起见，不是另一个``__array_ufunc__``覆盖）。 那么``C``实例上的任何ufunc都会传递给``A .__ array_ufunc__``，``A``中的``super``调用将转到``B .__ array_ufunc__``，并且 ``B``中的``super``调用将转到``ndarray .__ array_ufunc__``，从而允许``A``和`````进行协作。

## ``__array_wrap__``用于ufuncs和其他函数

在numpy 1.13之前，ufuncs的行为只能使用`__array_wrap__``和`__array_prepare__``进行调整。这两个允许一个更改ufunc的输出类型，但是，与前两者相反，`__array_ufunc__``，它不允许对输入进行任何更改。它希望最终弃能弃用这些功能，但是``__array_wrap__``也被其他numpy函数和方法使用，比如``squeeze``，所以目前仍需要完整的功能。

从概念上讲，`__array_wrap__``“包含动作”是允许子类设置返回值的类型并更新属性和元数据。让我们用一个例子来说明这是如何工作的。 首先，我们返回更简单的示例子类，但使用不同的名称和一些print语句：

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

我们在新数组的实例上运行ufunc：

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

请注意，ufunc (``np.add``) 调用了```__array_wack__``方法，其参数 ``self`` 作为 ``obj``，``out_arr```为该加法的(ndarray)结果。反过来，默认的 ``__array_wirp_`` (ndarray.``arraray_wirp_``) 已将结果转换为类MySubClass，名为 ``_array_radline_``` - 因此复制了``info`` 属性。这一切都发生在C级。

但是，我们可以做任何我们想做的事：

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

因此，通过为我们的子类定义一个特定的``__array_wrap__``方法，我们可以调整ufuncs的输出。 ``__array_wrap__``方法需要``self``，然后是一个参数 - 这是ufunc的结果 - 和一个可选的参数上下文。 ufuncs将此参数作为3元素元组返回:( ufunc的名称，ufunc的参数，ufunc的域），但不是由其他numpy函数设置的。 但是，如上所述，可以这样做，``__ array_wrap__``应返回其包含类的实例。 有关实现，请参阅masked数组子类。

除了在退出ufunc时调用的 ``__array_wrap__`` 之外，还存在一个 ``__array_prepare__`` 方法，该方法在创建输出数组之后但在执行任何计算之前，在进入ufunc的过程中被调用。默认实现除了传递数组之外什么都不做。``__array_prepare__`` 不应该尝试访问数组数据或调整数组大小，它的目的是设置输出数组类型，更新属性和元数据，并根据在计算开始之前需要的输入执行任何检查。与 ``__array_wrap__`` 一样，``__array_prepare__`` 必须返回一个ndarray或其子类，或引发一个错误。

## 额外的坑 - 自定义``__del__``方法 和 ndarray.base

darray解决的问题之一是跟踪ndarray的内存所有权和它们的视图。考虑这样一个例子：我们创建了一个ndarray，``arr``，并用 ``v=arr[1：]`` 取了一个切片。这两个对象看到的是相同的内存。NumPy使用 ``base`` 属性跟踪特定数组或视图的数据来源：

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

一般来说，如果数组拥有自己的内存，就像在这种情况下的 ``arr`` 一样，那么 ``arr.base`` 将是None - 这方面有一些例外 - 更多细节请参见 Numpy 的书籍。

``base``属性可以告诉我们是否有视图或原始数组。 如果我们需要知道在删除子类数组时是否进行某些特定的清理，这反过来会很有用。 例如，如果删除原始数组，我们可能只想进行清理，而不是视图。 有关它如何工作的示例，请查看``numpy.core``中的``memmap``类。

## 子类和下游兼容性

当对``ndarray``进行子类化或创建模仿``ndarray``接口的duck-types时，你有责任决定你的API与numpy的对齐方式。 为方便起见，许多具有相应``ndarray``方法的numpy函数（例如，``sum``，``mean``，``take``，``reshape``）都会检查第一个参数，看是否一个函数有一个同名的方法。如果是，则调用该方法，反之则将参数强制转换为numpy数组。

例如，如果你希望子类或duck-type与numpy的sum函数兼容，则此对象的`sum``方法的方法特征应如下所示：

```python
def sum(self, axis=None, dtype=None, out=None, keepdims=False):
...
```

这是``np.sum``的完全相同的方法特征，所以现在如果用户在这个对象上调用``np.sum``，numpy将调用该对象自己的``sum``方法并传入这些参数，在特征上枚举，并且不会引起任何错误，因为他们的特征彼此完全兼容。

但是，如果你决定偏离相关特征并执行以下操作：

```python
def sum(self, axis=None, dtype=None):
...
```

这个对象不再与``np.sum``兼容，因为如果你调用``np.sum``，它将传递意外的参数``out``和``keepdims``，导致引发TypeError的错误。

如果你希望保持与numpy及其后续版本（可能会添加新的关键字参数）的兼容性，但又不想显示所有numpy的参数，那么你的函数的特征应该接受 ** kwargs。 例如：

```python
def sum(self, axis=None, dtype=None, **unused_kwargs):
...
```

此对象现在再次与``np.sum``兼容，因为任何无关的参数（即不是``axis``或``dtype``的关键字）将被隐藏在``** unused_kwargs``参数中。