# 使用Python作为粘合剂

> There is no conversation more boring than the one where everybody
agrees.
> 
> *— Michel de Montaigne*

> Duct tape is like the force. It has a light side, and a dark side, and
it holds the universe together.
> 
> *— Carl Zwanzig*

很多人都喜欢说Python是一种很棒的粘合语言。希望本章能说服你这是真的。Python的第一批科学家通常使用它来粘合在超级计算机上运行的大型应用程序代码。在Python中编写代码比在shell脚本或Perl中编写代码更好，此外，轻松扩展Python的能力使得创建专门适应所解决问题的新类和类型变得相对容易。从这些早期贡献者的交互中，Numeric出现了一个类似于数组的对象，可用于在这些应用程序之间传递数据。

随着Numeric的成熟和发展成为NumPy，人们已经能够在NumPy中直接编写更多代码。通常，此代码对于生产使用来说足够快，但仍有时需要访问已编译的代码。要么从算法中获得最后一点效率，要么更容易访问用C / C ++或Fortran编写的广泛可用的代码。

本章将回顾许多可用于访问以其他编译语言编写的代码的工具。有许多资源可供学习从Python调用其他编译库，本章的目的不是让你成为专家。主要目标是让您了解一些可能性，以便您知道“Google”的内容，以便了解更多信息。

## 从Python调用其他编译库

虽然Python是一种很好的语言并且很乐意编写代码，但它的动态特性会导致开销，从而导致一些代码（ *即* 
for循环中的原始计算）比用静态编译语言编写的等效代码慢10-100倍。此外，由于在计算过程中创建和销毁临时数组，因此可能导致内存使用量大于必要值。对于许多类型的计算需求，通常不能节省额外的速度和内存消耗（至少对于代码的时间或内存关键部分而言）。因此，最常见的需求之一是从Python代码调用快速的机器代码例程（例如使用C / C ++或Fortran编译）。这相对容易做的事实是Python成为科学和工程编程的优秀高级语言的一个重要原因。

它们是调用编译代码的两种基本方法：编写扩展模块，然后使用import命令将其导入Python，或者使用[ctypes](https://docs.python.org/3/library/ctypes.html) 
模块直接从Python调用共享库子例程。编写扩展模块是最常用的方法。

::: danger 警告

如果你不小心，从Python调用C代码会导致Python崩溃。本章中没有一种方法可以免疫。您必须了解NumPy和正在使用的第三方库处理数据的方式。

:::

## 手工生成的包装器

在[编写扩展模块](how-to-extend.html#编写扩展模块)中讨论了扩展模块。与编译代码接口的最基本方法是编写扩展模块并构造调用编译代码的模块方法。为了提高可读性，您的方法应该利用 ``PyArg_ParseTuple`` 调用在Python对象和C数据类型之间进行转换。对于标准的C数据类型，可能已经有一个内置的转换器。对于其他人，您可能需要编写自己的转换器并使用``"O&"``格式字符串，该字符串允许您指定一个函数，该函数将用于执行从Python对象到所需的任何C结构的转换。

一旦执行了对适当的C结构和C数据类型的转换，包装器中的下一步就是调用底层函数。如果底层函数是C或C ++，这很简单。但是，为了调用Fortran代码，您必须熟悉如何使用编译器和平台从C / C ++调用Fortran子例程。这可能会有所不同的平台和编译器（这是f2py使接口Fortran代码的生活变得简单的另一个原因）但通常涉及下划线修改名称以及所有变量都通过引用传递的事实（即所有参数都是指针）。

手工生成的包装器的优点是您可以完全控制C库的使用和调用方式，从而可以实现精简且紧凑的界面，并且只需最少的开销。缺点是您必须编写，调试和维护C代码，尽管大多数代码都可以使用其他扩展模块中“剪切粘贴和修改”这种历史悠久的技术进行调整。因为，调用额外的C代码的过程是相当规范的，所以已经开发了代码生成过程以使这个过程更容易。其中一种代码生成技术与NumPy一起分发，可以轻松地与Fortran和（简单）C代码集成。这个软件包f2py将在下一节中简要介绍。

## f2py 

F2py允许您自动构建一个扩展模块，该模块与Fortran 77/90/95代码中的例程相连。它能够解析Fortran 77/90/95代码并自动为它遇到的子程序生成Python签名，或者你可以通过构造一个接口定义文件（或修改f2py生成的文件）来指导子程序如何与Python接口。 ）。

### 创建基本扩展模块的源

引入f2py最简单的方法可能是提供一个简单的例子。这是一个名为的文件中包含的子程序之一
 ``add.f``：

``` 
C
      SUBROUTINE ZADD(A,B,C,N)
C
      DOUBLE COMPLEX A(*)
      DOUBLE COMPLEX B(*)
      DOUBLE COMPLEX C(*)
      INTEGER N
      DO 20 J = 1, N
         C(J) = A(J)+B(J)
 20   CONTINUE
      END
```

此例程只是将元素添加到两个连续的数组中，并将结果放在第三个数组中。所有三个数组的内存必须由调用例程提供。f2py可以自动生成此例程的一个非常基本的接口：

``` python
f2py -m add add.f
```

假设您的搜索路径设置正确，您应该能够运行此命令。此命令将在当前目录中生成名为addmodule.c的扩展模块。现在可以像使用任何其他扩展模块一样从Python编译和使用此扩展模块。

### 创建编译的扩展模块

您还可以获取f2py来编译add.f并编译其生成的扩展模块，只留下可以从Python导入的共享库扩展文件：

``` python
f2py -c -m add add.f
```

此命令在当前目录中留下名为add。{ext}的文件（其中{ext}是平台上python扩展模块的相应扩展名 - 所以，pyd  *等* ）。然后可以从Python导入该模块。它将包含添加的每个子例程的方法（zadd，cadd，dadd，sadd）。每个方法的docstring包含有关如何调用模块方法的信息：

``` python
>>> import add
>>> print add.zadd.__doc__
zadd - Function signature:
  zadd(a,b,c,n)
Required arguments:
  a : input rank-1 array('D') with bounds (*)
  b : input rank-1 array('D') with bounds (*)
  c : input rank-1 array('D') with bounds (*)
  n : input int
```

### 改善基本界面

默认界面是fortran代码到Python的非常直译。Fortran数组参数现在必须是NumPy数组，整数参数应该是整数。接口将尝试将所有参数转换为其所需类型（和形状），如果不成功则发出错误。但是，因为它对参数的语义一无所知（因此C是输出而n应该与数组大小完全匹配），所以可能会以导致Python崩溃的方式滥用此函数。例如：

``` python
>>> add.zadd([1,2,3], [1,2], [3,4], 1000)
```

将导致程序在大多数系统上崩溃。在封面下，列表被转换为正确的数组，但随后底层的添加循环被告知超出分配的内存的边界。

为了改进界面，应提供指令。这是通过构造接口定义文件来完成的。通常最好从f2py可以生成的接口文件开始（从中获取其默认行为）。要获取f2py以生成接口文件，请使用-h选项：

``` python
f2py -h add.pyf -m add add.f
```

此命令将文件add.pyf保留在当前目录中。与zadd对应的此文件部分为：

``` 
subroutine zadd(a,b,c,n) ! in :add:add.f
   double complex dimension(*) :: a
   double complex dimension(*) :: b
   double complex dimension(*) :: c
   integer :: n
end subroutine zadd
```

通过放置intent指令和检查代码，可以清理接口，直到Python模块方法更易于使用且更健壮。

``` 
subroutine zadd(a,b,c,n) ! in :add:add.f
   double complex dimension(n) :: a
   double complex dimension(n) :: b
   double complex intent(out),dimension(n) :: c
   integer intent(hide),depend(a) :: n=len(a)
end subroutine zadd
```

intent指令，intent（out）用于告诉``c``作为输出变量的f2py，并且应该在传递给底层代码之前由接口创建。intent（hide）指令告诉f2py不允许用户指定变量``n``，而是从大小中获取它``a``。depend（``a``）指令必须告诉f2py n的值取决于输入a（因此在创建变量a之前它不会尝试创建变量n）。

修改后``add.pyf``，可以通过编译``add.f95``和生成新的python模块文件``add.pyf``：

``` python
f2py -c add.pyf add.f95
```

新界面有docstring：

``` python
>>> import add
>>> print add.zadd.__doc__
zadd - Function signature:
  c = zadd(a,b)
Required arguments:
  a : input rank-1 array('D') with bounds (n)
  b : input rank-1 array('D') with bounds (n)
Return objects:
  c : rank-1 array('D') with bounds (n)
```

现在，可以以更加健壮的方式调用该函数：

``` python
>>> add.zadd([1,2,3],[4,5,6])
array([ 5.+0.j,  7.+0.j,  9.+0.j])
```

请注意自动转换为正确的格式。

### 在Fortran源中插入指令

通过将变量指令作为特殊注释放在原始fortran代码中，也可以自动生成nice接口。因此，如果我修改源代码包含：

``` 
C
      SUBROUTINE ZADD(A,B,C,N)
C
CF2PY INTENT(OUT) :: C
CF2PY INTENT(HIDE) :: N
CF2PY DOUBLE COMPLEX :: A(N)
CF2PY DOUBLE COMPLEX :: B(N)
CF2PY DOUBLE COMPLEX :: C(N)
      DOUBLE COMPLEX A(*)
      DOUBLE COMPLEX B(*)
      DOUBLE COMPLEX C(*)
      INTEGER N
      DO 20 J = 1, N
         C(J) = A(J) + B(J)
 20   CONTINUE
      END
```

然后，我可以使用以下命令编译扩展模块：

``` python
f2py -c -m add add.f
```

函数add.zadd的结果签名与之前创建的签名完全相同。如果原来的源代码已经包含``A(N)``，而不是``A(*)``等以``B``和``C``，然后我可以得到（几乎）相同的接口简单地通过将
 注释行的源代码。唯一的区别是，这是一个默认为长度的可选输入。``INTENT(OUT) :: C``N``A``

### 过滤示例

用于与将要讨论的其他方法进行比较。下面是使用固定平均滤波器过滤二维精度浮点数的二维数组的函数的另一个示例。从这个例子中可以清楚地看到使用Fortran索引到多维数组的优势。

``` 
SUBROUTINE DFILTER2D(A,B,M,N)
C
      DOUBLE PRECISION A(M,N)
      DOUBLE PRECISION B(M,N)
      INTEGER N, M
CF2PY INTENT(OUT) :: B
CF2PY INTENT(HIDE) :: N
CF2PY INTENT(HIDE) :: M
      DO 20 I = 2,M-1
         DO 40 J=2,N-1
            B(I,J) = A(I,J) +
     $           (A(I-1,J)+A(I+1,J) +
     $            A(I,J-1)+A(I,J+1) )*0.5D0 +
     $           (A(I-1,J-1) + A(I-1,J+1) +
     $            A(I+1,J-1) + A(I+1,J+1))*0.25D0
 40      CONTINUE
 20   CONTINUE
      END
```

此代码可以编译并链接到名为filter的扩展模块中，使用：

``` python
f2py -c -m filter filter.f
```

这将在当前目录中生成一个名为filter.so的扩展模块，其中包含一个名为dfilter2d的方法，该方法返回输入的过滤版本。

### 从Python中调用f2py 

f2py程序是用Python编写的，可以在代码中运行，以便在运行时编译Fortran代码，如下所示：

``` python
from numpy import f2py
with open("add.f") as sourcefile:
    sourcecode = sourcefile.read()
f2py.compile(sourcecode, modulename='add')
import add
```

源字符串可以是任何有效的Fortran代码。如果要保存扩展模块源代码，则``source_fn``关键字可以为编译函数提供合适的文件名。

### 自动扩展模块生成

如果要分发f2py扩展模块，则只需要包含.pyf文件和Fortran代码。NumPy中的distutils扩展允许您完全根据此接口文件定义扩展模块。``setup.py``允许分发``add.f``模块的有效文件（作为包的一部分，
 ``f2py_examples``以便将其加载为``f2py_examples.add``）：

``` python
def configuration(parent_package='', top_path=None)
    from numpy.distutils.misc_util import Configuration
    config = Configuration('f2py_examples',parent_package, top_path)
    config.add_extension('add', sources=['add.pyf','add.f'])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
```

安装新包装很容易使用：

``` python
pip install .
```

假设您具有写入主要site-packages目录的正确权限，以获取您正在使用的Python版本。要使生成的包工作，您需要创建一个名为``__init__.py``
（与目录相同``add.pyf``）的文件。请注意，扩展模块完全根据``add.pyf``和``add.f``文件定义。.pyf文件到.c文件的转换由 *numpy.disutils* 处理。

### 结论

接口定义文件（.pyf）是如何微调Python和Fortran之间的接口的。在numpy / f2py / docs目录中找到了适合f2py的文档，其中NumPy安装在你的系统上（通常在site-packages下）。有关使用f2py（包括如何使用它来包装C代码）的更多信息，请参见[https://scipy-cookbook.readthedocs.io](https://scipy-cookbook.readthedocs.io) “与其他语言接口”标题下的信息。

连接编译代码的f2py方法是目前最复杂和最集成的方法。它允许使用已编译的代码清晰地分离Python，同时仍允许单独分发扩展模块。唯一的缺点是它需要Fortran编译器的存在才能让用户安装代码。然而，随着免费编译器g77，gfortran和g95以及高质量商业编译器的存在，这种限制并不是特别繁重。在我看来，Fortran仍然是编写快速而清晰的科学计算代码的最简单方法。它以最直接的方式处理复杂的数字和多维索引。但请注意，某些Fortran编译器无法优化代码以及良好的手写C代码。

## 用Cython 

[Cython](http://cython.org)是Python方言的编译器，它为速度添加（可选）静态类型，并允许将C或C ++代码混合到模块中。它生成C或C ++扩展，可以在Python代码中编译和导入。

如果您正在编写一个扩展模块，其中包含相当多的自己的算法代码，那么Cython是一个很好的匹配。其功能之一是能够轻松快速地处理多维数组。

请注意，Cython只是一个扩展模块生成器。与f2py不同，它不包括用于编译和链接扩展模块的自动工具（必须以通常的方式完成）。它确实提供了一个修改过的distutils类``build_ext``，它允许您从``.pyx``源构建扩展模块。因此，您可以写入``setup.py``文件：

``` python
from Cython.Distutils import build_ext
from distutils.extension import Extension
from distutils.core import setup
import numpy

setup(name='mine', description='Nothing',
      ext_modules=[Extension('filter', ['filter.pyx'],
                             include_dirs=[numpy.get_include()])],
      cmdclass = {'build_ext':build_ext})
```

当然，只有在扩展模块中使用NumPy数组时才需要添加NumPy包含目录（我们假设您使用的是Cython）。NumPy中的distutils扩展还包括支持自动生成扩展模块并将其从``.pyx``文件链接。它的工作原理是，如果用户没有安装Cython，那么它会查找具有相同文件名但``.c``扩展名的文件，然后使用该``.c``文件而不是尝试再次生成文件。

如果你只是使用Cython来编译一个标准的Python模块，那么你将得到一个C扩展模块，它通常比同等的Python模块运行得快一点。通过使用``cdef``关键字静态定义C变量，可以进一步提高速度。

让我们看一下我们之前看过的两个例子，看看如何使用Cython实现它们。这些示例使用Cython 0.21.1编译为扩展模块。

### Cython中的复杂添加

这是一个名为Cython的模块的一部分，``add.pyx``它实现了我们之前使用f2py实现的复杂加法函数：

``` 
cimport cython
cimport numpy as np
import numpy as np

# We need to initialize NumPy.
np.import_array()

#@cython.boundscheck(False)
def zadd(in1, in2):
    cdef double complex[:] a = in1.ravel()
    cdef double complex[:] b = in2.ravel()

    out = np.empty(a.shape[0], np.complex64)
    cdef double complex[:] c = out.ravel()

    for i in range(c.shape[0]):
        c[i].real = a[i].real + b[i].real
        c[i].imag = a[i].imag + b[i].imag

    return out
```

此模块显示使用该``cimport``语句从``numpy.pxd``Cython附带的标头加载定义。看起来NumPy是两次进口的; ``cimport``只使NumPy C-API可用，而常规``import``在运行时导致Python样式导入，并且可以调用熟悉的NumPy Python API。

该示例还演示了Cython的“类型化内存视图”，它类似于C级别的NumPy数组，因为它们是形状和跨步的数组，知道它们自己的范围（不同于通过裸指针寻址的C数组）。语法表示具有任意步幅的双精度的一维数组（向量）。一个连续的整数数组将是，而浮点矩阵将是。``double complex[:]``int[::1]``float[:, :]``

显示的注释是``cython.boundscheck``装饰器，它基于每个函数打开或关闭内存视图访问的边界检查。我们可以使用它来进一步加速我们的代码，但代价是安全性（或者在进入循环之前进行手动检查）。

除了视图语法之外，该函数可立即被Python程序员读取。变量的静态类型``i``是隐式的。我们也可以使用Cython的特殊NumPy数组语法代替视图语法，但首选视图语法。

### Cython中的图像过滤器

我们使用Fortran创建的二维示例与在Cython中编写一样容易：

``` 
cimport numpy as np
import numpy as np

np.import_array()

def filter(img):
    cdef double[:, :] a = np.asarray(img, dtype=np.double)
    out = np.zeros(img.shape, dtype=np.double)
    cdef double[:, ::1] b = out

    cdef np.npy_intp i, j

    for i in range(1, a.shape[0] - 1):
        for j in range(1, a.shape[1] - 1):
            b[i, j] = (a[i, j]
                       + .5 * (  a[i-1, j] + a[i+1, j]
                               + a[i, j-1] + a[i, j+1])
                       + .25 * (  a[i-1, j-1] + a[i-1, j+1]
                                + a[i+1, j-1] + a[i+1, j+1]))

    return out
```

这个2-d平均滤波器运行得很快，因为循环在C中，并且指针计算仅在需要时完成。如果将上面的代码编译为模块``image``，则``img``可以使用以下代码非常快速地过滤2-d图像：

``` python
import image
out = image.filter(img)
```

关于代码，有两点需要注意：首先，不可能将内存视图返回给Python。相反，``out``首先创建NumPy数组，然后使用``b``该数组的视图进行计算。其次，视图``b``是键入的。这意味着具有连续行的2-d数组，即C矩阵顺序。明确指定顺序可以加速某些算法，因为它们可以跳过步幅计算。``double[:, ::1]``

### Cython结论

Cython是几个科学Python库的首选扩展机制，包括Scipy，Pandas，SAGE，scikit-image和scikit-learn，以及XML处理库LXML。语言和编译器维护良好。

使用Cython有几个缺点：

1. 在编写自定义算法时，有时在包装现有C库时，需要熟悉C语言。特别是，当使用C内存管理（``malloc``和朋友）时，很容易引入内存泄漏。但是，只是编译重命名的Python模块``.pyx``
已经可以加快速度，并且添加一些类型声明可以在某些代码中提供显着的加速。
1. 很容易在Python和C之间失去一个清晰的分离，这使得重用你的C代码用于其他非Python相关项目变得更加困难。
1. Cython生成的C代码难以阅读和修改（并且通常编译有令人烦恼但无害的警告）。

Cython生成的扩展模块的一大优势是它们易于分发。总之，Cython是一个非常强大的工具，可以粘合C代码或快速生成扩展模块，不应该被忽视。它对于不能或不会编写C或Fortran代码的人特别有用。

## ctypes

[ctypes](https://docs.python.org/3/library/ctypes.html)
是一个包含在stdlib中的Python扩展模块，它允许您直接从Python调用共享库中的任意函数。这种方法允许您直接从Python接口C代码。这开辟了大量可供Python使用的库。然而，缺点是编码错误很容易导致丑陋的程序崩溃（就像C中可能发生的那样），因为对参数进行的类型或边界检查很少。当数组数据作为指向原始内存位置的指针传入时尤其如此。那么你应该负​​责子程序不会访问实际数组区域之外的内存。但，

因为ctypes方法将原始接口暴露给已编译的代码，所以它并不总是容忍用户错误。强大地使用ctypes模块通常需要额外的Python代码层，以便检查传递给底层子例程的对象的数据类型和数组边界。这个额外的检查层（更不用说从ctypes对象到ctypes本身执行的C-data类型的转换）将使接口比手写的扩展模块接口慢。但是，如果被调用的C例程正在执行任何大量工作，则此开销应该可以忽略不计。如果你是一个具有弱C技能的优秀Python程序员，那么ctypes是一种为编译代码的（共享）库编写有用接口的简单方法。

要使用ctypes，你必须

1. 有一个共享的库。
1. 加载共享库。
1. 将python对象转换为ctypes理解的参数。
1. 使用ctypes参数从库中调用函数。

### 加载共享库

共享库有几个要求，可以与特定于平台的ctypes一起使用。本指南假设您熟悉在系统上创建共享库（或者只是为您提供共享库）。要记住的项目是：

- 必须以特殊方式编译共享库（ *例如，* 使用``-shared``带有gcc 的标志）。
- 在某些平台（ *例如*  Windows）上，共享库需要一个.def文件，该文件指定要导出的函数。例如，mylib.def文件可能包含：

``` python
LIBRARY mylib.dll
EXPORTS
cool_function1
cool_function2
```

或者，您可以``__declspec(dllexport)``在函数的C定义中使用存储类说明符
 ，以避免需要此``.def``文件。

Python distutils中没有标准的方法来以跨平台的方式创建标准共享库（扩展模块是Python理解的“特殊”共享库）。因此，在编写本书时，ctypes的一大缺点是难以以跨平台的方式分发使用ctypes的Python扩展并包含您自己的代码，这些代码应编译为用户系统上的共享库。

### 加载共享库

加载共享库的一种简单但强大的方法是获取绝对路径名并使用ctypes的cdll对象加载它：

``` python
lib = ctypes.cdll[<full_path_name>]
```

但是，在Windows上，访问该``cdll``方法的属性将按当前目录或PATH中的名称加载第一个DLL。加载绝对路径名称需要一点技巧才能进行跨平台工作，因为共享库的扩展会有所不同。有一个``ctypes.util.find_library``实用程序可以简化查找库加载的过程，但它不是万无一失的。更复杂的是，不同平台具有共享库使用的不同默认扩展名（例如.dll  -  Windows，.so  -  Linux，.dylib  -  Mac OS X）。如果您使用ctypes包装需要在多个平台上工作的代码，则还必须考虑这一点。

NumPy提供称为``ctypeslib.load_library``（名称，路径）的便利功能
 。此函数采用共享库的名称（包括任何前缀，如'lib'但不包括扩展名）和共享库所在的路径。它返回一个ctypes库对象，或者``OSError``如果找不到库则引发一个或者``ImportError``如果ctypes模块不可用则引发一个。（Windows用户：使用加载的ctypes库对象
 ``load_library``总是在假定cdecl调用约定的情况下加载。请参阅下面的ctypes文档``ctypes.windll``和/或``ctypes.oledll``
了解在其他调用约定下加载库的方法）。

共享库中的函数可用作ctypes库对象的属性（从中返回``ctypeslib.load_library``）或使用``lib['func_name']``语法作为项目。如果函数名包含Python变量名中不允许的字符，则后一种检索函数名的方法特别有用。

### 转换参数

Python int / long，字符串和unicode对象会根据需要自动转换为等效的ctypes参数None对象也会自动转换为NULL指针。必须将所有其他Python对象转换为特定于ctypes的类型。围绕此限制有两种方法允许ctypes与其他对象集成。

1. 不要设置函数对象的argtypes属性，并``_as_parameter_``为要传入的对象定义
 方法。该
 ``_as_parameter_``方法必须返回一个Python int，它将直接传递给函数。
1. 将argtypes属性设置为一个列表，其条目包含具有名为from_param的类方法的对象，该类方法知道如何将对象转换为ctypes可以理解的对象（具有该``_as_parameter_``属性的int / long，字符串，unicode或对象）。

NumPy使用两种方法，优先选择第二种方法，因为它可以更安全。ndarray的ctypes属性返回一个对象，该对象具有一个``_as_parameter_``返回整数的属性，该整数表示与之关联的ndarray的地址。因此，可以将此ctypes属性对象直接传递给期望指向ndarray中数据的指针的函数。调用者必须确保ndarray对象具有正确的类型，形状，并且设置了正确的标志，否则如果传入指向不适当数组的数据指针则会导致令人讨厌的崩溃。

为了实现第二种方法，NumPy [``ndpointer``](#ndpointer)在[``numpy.ctypeslib``](https://numpy.org/devdocs/reference/routines.ctypeslib.html#module-numpy.ctypeslib)模块中提供了类工厂函数。此类工厂函数生成一个适当的类，可以放在ctypes函数的argtypes属性条目中。该类将包含一个from_param方法，ctypes将使用该方法将传入函数的任何ndarray转换为ctypes识别的对象。在此过程中，转换将执行检查用户在调用中指定的ndarray的任何属性[``ndpointer``](#ndpointer)。可以检查的ndarray的方面包括数据类型，维度的数量，形状和/或传递的任何数组上的标志的状态。from_param方法的返回值是数组的ctypes属性（因为它包含``_as_parameter_``
ctypes可以直接使用指向数组数据区域的属性。

ndarray的ctypes属性还赋予了额外的属性，这些属性在将有关数组的其他信息传递给ctypes函数时可能很方便。属性**数据**，
 **形状**和**步幅**可以提供与数据区域，形状和数组步幅相对应的ctypes兼容类型。data属性返回``c_void_p``表示指向数据区域的指针。shape和strides属性各自返回一个ctypes整数数组（如果是0-d数组，则返回None表示NULL指针）。数组的基本ctype是与平台上的指针大小相同的ctype整数。还有一些方法
 ``data_as({ctype})``，和``shape_as()``strides_as()``。它们将数据作为您选择的ctype对象返回，并使用您选择的基础类型返回shape / strides数组。为方便起见，该``ctypeslib``模块还包含``c_intp``一个ctypes整数数据类型，其大小``c_void_p``与平台上的大小相同
 （如果未安装ctypes，则其值为None）。

### 调用函数

该函数作为加载的共享库的属性或项目进行访问。因此，如果``./mylib.so``有一个名为的函数
 ``cool_function1``，我可以访问此函数：

``` python
lib = numpy.ctypeslib.load_library('mylib','.')
func1 = lib.cool_function1  # or equivalently
func1 = lib['cool_function1']
```

在ctypes中，函数的返回值默认设置为“int”。可以通过设置函数的restype属性来更改此行为。如果函数没有返回值（'void'），则使用None作为restype：

``` python
func1.restype = None
```

如前所述，您还可以设置函数的argtypes属性，以便在调用函数时让ctypes检查输入参数的类型。使用[``ndpointer``](#ndpointer)工厂函数生成现成的类，以便对新函数进行数据类型，形状和标志检查。该[``ndpointer``](#ndpointer)功能具有签名


``ndpointer``（ *dtype*   *= None* ， *ndim = None* ， *shape = None* ， *flags = None*  ）[¶](#ndpointer)

``None``不检查具有该值的关键字参数。指定关键字会强制在转换为与ctypes兼容的对象时检查ndarray的该方面。dtype关键字可以是任何被理解为数据类型对象的对象。ndim关键字应为整数，shape关键字应为整数或整数序列。flags关键字指定传入的任何数组所需的最小标志。这可以指定为逗号分隔要求的字符串，指示需求位OR'd在一起的整数，或者从flags的flags属性返回的flags对象。具有必要要求的数组。

在argtypes方法中使用ndpointer类可以使用ctypes和ndarray的数据区调用C函数更加安全。您可能仍希望将该函数包装在另一个Python包装器中，以使其对用户友好（隐藏一些明显的参数并使一些参数输出参数）。在此过程中，``requires``NumPy中的函数可能对从给定输入返回正确类型的数组很有用。

### 完整的例子

在这个例子中，我将展示如何使用其他方法实现的加法函数和过滤函数可以使用ctypes实现。第一，它实现了算法的C代码所包含的功能``zadd``，``dadd``，``sadd``，``cadd``，和``dfilter2d``。该``zadd``功能是：

``` c
/* Add arrays of contiguous data */
typedef struct {double real; double imag;} cdouble;
typedef struct {float real; float imag;} cfloat;
void zadd(cdouble *a, cdouble *b, cdouble *c, long n)
{
    while (n--) {
        c->real = a->real + b->real;
        c->imag = a->imag + b->imag;
        a++; b++; c++;
    }
}
```

用类似的代码``cadd``，``dadd``以及``sadd``用于处理复杂的浮点，双精度和浮点数据类型，分别为：

``` c
void cadd(cfloat *a, cfloat *b, cfloat *c, long n)
{
        while (n--) {
                c->real = a->real + b->real;
                c->imag = a->imag + b->imag;
                a++; b++; c++;
        }
}
void dadd(double *a, double *b, double *c, long n)
{
        while (n--) {
                *c++ = *a++ + *b++;
        }
}
void sadd(float *a, float *b, float *c, long n)
{
        while (n--) {
                *c++ = *a++ + *b++;
        }
}
```

该``code.c``文件还包含以下功能``dfilter2d``：

``` c
/*
 * Assumes b is contiguous and has strides that are multiples of
 * sizeof(double)
 */
void
dfilter2d(double *a, double *b, ssize_t *astrides, ssize_t *dims)
{
    ssize_t i, j, M, N, S0, S1;
    ssize_t r, c, rm1, rp1, cp1, cm1;

    M = dims[0]; N = dims[1];
    S0 = astrides[0]/sizeof(double);
    S1 = astrides[1]/sizeof(double);
    for (i = 1; i < M - 1; i++) {
        r = i*S0;
        rp1 = r + S0;
        rm1 = r - S0;
        for (j = 1; j < N - 1; j++) {
            c = j*S1;
            cp1 = j + S1;
            cm1 = j - S1;
            b[i*N + j] = a[r + c] +
                (a[rp1 + c] + a[rm1 + c] +
                 a[r + cp1] + a[r + cm1])*0.5 +
                (a[rp1 + cp1] + a[rp1 + cm1] +
                 a[rm1 + cp1] + a[rm1 + cp1])*0.25;
        }
    }
}
```

此代码相对于Fortran等效代码的一个可能的优点是它需要任意跨越（即非连续数组），并且还可能运行得更快，具体取决于编译器的优化功能。但是，它显然比简单的代码更复杂``filter.f``。必须将此代码编译到共享库中。在我的Linux系统上，这是使用以下方法完成

``` python
gcc -o code.so -shared code.c
```

这会在当前目录中创建名为code.so的shared_library。在Windows上，不要忘记``__declspec(dllexport)``在每个函数定义之前的行上添加void，或者写一个
 ``code.def``列出要导出的函数名称的文件。

应构建适用于此共享库的Python接口。为此，请在顶部创建一个名为interface.py的文件，其中包含以下行：

``` python
__all__ = ['add', 'filter2d']

import numpy as np
import os

_path = os.path.dirname('__file__')
lib = np.ctypeslib.load_library('code', _path)
_typedict = {'zadd' : complex, 'sadd' : np.single,
             'cadd' : np.csingle, 'dadd' : float}
for name in _typedict.keys():
    val = getattr(lib, name)
    val.restype = None
    _type = _typedict[name]
    val.argtypes = [np.ctypeslib.ndpointer(_type,
                      flags='aligned, contiguous'),
                    np.ctypeslib.ndpointer(_type,
                      flags='aligned, contiguous'),
                    np.ctypeslib.ndpointer(_type,
                      flags='aligned, contiguous,'\
                            'writeable'),
                    np.ctypeslib.c_intp]
```

此代码加载名为``code.{ext}``位于与此文件相同的路径中的共享库。然后，它会向库中包含的函数添加返回类型的void。它还将参数检查添加到库中的函数，以便ndarrays可以作为前三个参数与一个整数（大到足以在平台上保存指针）作为第四个参数传递。

设置过滤函数是类似的，并允许使用ndarray参数作为前两个参数调用过滤函数，并使用指向整数的指针（大到足以处理ndarray的步幅和形状）作为最后两个参数：

``` python
lib.dfilter2d.restype=None
lib.dfilter2d.argtypes = [np.ctypeslib.ndpointer(float, ndim=2,
                                       flags='aligned'),
                          np.ctypeslib.ndpointer(float, ndim=2,
                                 flags='aligned, contiguous,'\
                                       'writeable'),
                          ctypes.POINTER(np.ctypeslib.c_intp),
                          ctypes.POINTER(np.ctypeslib.c_intp)]
```

接下来，定义一个简单的选择函数，根据数据类型选择在共享库中调用哪个添加函数：

``` python
def select(dtype):
    if dtype.char in ['?bBhHf']:
        return lib.sadd, single
    elif dtype.char in ['F']:
        return lib.cadd, csingle
    elif dtype.char in ['DG']:
        return lib.zadd, complex
    else:
        return lib.dadd, float
    return func, ntype
```

最后，接口导出的两个函数可以简单地写成：

``` python
def add(a, b):
    requires = ['CONTIGUOUS', 'ALIGNED']
    a = np.asanyarray(a)
    func, dtype = select(a.dtype)
    a = np.require(a, dtype, requires)
    b = np.require(b, dtype, requires)
    c = np.empty_like(a)
    func(a,b,c,a.size)
    return c
```

和：

``` python
def filter2d(a):
    a = np.require(a, float, ['ALIGNED'])
    b = np.zeros_like(a)
    lib.dfilter2d(a, b, a.ctypes.strides, a.ctypes.shape)
    return b
```

### ctypes结论

使用ctypes是一种将Python与任意C代码连接起来的强大方法。它扩展Python的优点包括

- 从Python代码中清除C代码的分离
    - 除了Python和C之外，无需学习新的语法
    - 允许重复使用C代码
    - 可以使用简单的Python包装器获取为其他目的编写的共享库中的功能并搜索库。
- 通过ctypes属性轻松与NumPy集成
- 使用ndpointer类工厂进行完整的参数检查

它的缺点包括

- 由于缺乏在distutils中构建共享库的支持，很难分发使用ctypes创建的扩展模块（但我怀疑这会随着时间的推移而改变）。
- 您必须拥有代码的共享库（没有静态库）。
- 很少支持C ++代码及其不同的库调用约定。你可能需要一个围绕C ++代码的C包装器来与ctypes一起使用（或者只是使用Boost.Python）。

由于难以分发使用ctypes创建的扩展模块，因此f2py和Cython仍然是扩展Python以创建包的最简单方法。但是，在某些情况下，ctypes是一种有用的替代品。这应该为ctypes带来更多功能，这将消除扩展Python和使用ctypes分发扩展的难度。

## 您可能会觉得有用的其他工具

使用Python的其他人发现这些工具很有用，因此包含在这里。它们是分开讨论的，因为它们要么是现在由f2py，Cython或ctypes（SWIG，PyFort）处理的旧方法，要么是因为我对它们不太了解（SIP，Boost）。我没有添加这些方法的链接，因为我的经验是您可以使用Google或其他搜索引擎更快地找到最相关的链接，此处提供的任何链接都会很快过时。不要以为仅仅因为它包含在此列表中，我认为该软件包不值得您关注。我包含了有关这些软件包的信息，因为很多人发现它们很有用，我想尽可能多地为您提供解决易于集成代码问题的选项。

### SWIG

简化的包装器和接口生成器（SWIG）是一种古老且相当稳定的方法，用于将C / C ++  - 库包装到各种其他语言中。它并不特别了解NumPy数组，但可以通过使用类型映射与NumPy一起使用。numpy.i下的numpy / tools / swig目录中有一些示例类型映射以及一个使用它们的示例模块。SWIG擅长包装大型C / C ++库，因为它可以（几乎）解析其头文件并自动生成一个接口。从技术上讲，您需要生成``.i``
定义接口的文件。但是，这通常是这样``.i``file可以是标题本身的一部分。界面通常需要一些调整才能非常有用。这种解析C / C ++头文件和自动生成界面的能力仍然使SWIG成为一种有用的方法，可以将C / C ++中的functionalilty添加到Python中，尽管已经出现了更多针对Python的其他方法。SWIG实际上可以定位多种语言的扩展，但是这些类型映射通常必须是特定于语言的。尽管如此，通过修改特定于Python的类型映射，SWIG可用于将库与其他语言（如Perl，Tcl和Ruby）连接。

我对SWIG的体验总体上是积极的，因为它相对容易使用且非常强大。在更熟练地编写C扩展之前，我经常使用它。但是，我很难用SWIG编写自定义接口，因为它必须使用非Python特定的类型映射的概念来完成，并且使用类似C的语法编写。因此，我倾向于选择其他粘合策略，并且只会尝试使用SWIG来包装一个非常大的C / C ++库。尽管如此，还有其他人非常愉快地使用SWIG。

### SIP 

SIP是另一种用于包装特定于Python的C / C ++库的工具，似乎对C ++有很好的支持。Riverbank Computing开发了SIP，以便为QT库创建Python绑定。必须编写接口文件以生成绑定，但接口文件看起来很像C / C ++头文件。虽然SIP不是一个完整的C ++解析器，但它理解了相当多的C ++语法以及它自己的特殊指令，这些指令允许修改Python绑定的完成方式。它还允许用户定义Python类型和C / C ++结构和类之间的映射。

### 提升Python

Boost是C ++库的存储库，Boost.Python是其中一个库，它提供了一个简洁的接口，用于将C ++类和函数绑定到Python。Boost.Python方法的神奇之处在于它完全在纯C ++中工作而不引入新语法。许多C ++用户报告称，Boost.Python可以无缝地结合两者的优点。我没有使用过Boost.Python，因为我不是C ++的大用户，并且使用Boost来包装简单的C子例程通常都是过度杀戮。它的主要目的是使Python中的C ++类可用。因此，如果您有一组需要完全集成到Python中的C ++类，请考虑学习并使用Boost.Python。

### PyFort 

PyFort是一个很好的工具，可以将Fortran和类似Fortran的C代码包装到Python中，并支持数值数组。它由着名计算机科学家Paul Dubois编写，是Numeric（现已退休）的第一个维护者。值得一提的是希望有人会更新PyFort以使用NumPy数组，现在支持Fortran或C风格的连续数组。
