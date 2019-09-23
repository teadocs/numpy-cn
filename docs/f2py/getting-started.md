---
meta:
  - name: keywords
    content: NumPy 打包的三种方法 - 入门
  - name: description
    content: 使用F2PY将Fortran或C函数包装到Python包含以下步骤：
---

# 打包的三种方法 - 入门

使用F2PY将Fortran或C函数包装到Python包含以下步骤：

- 创建所谓的签名文件，其中包含对Fortran或C函数的包装器的描述，也称为函数的签名。对于Fortran例程，F2PY可以通过扫描Fortran源代码并捕获创建包装函数所需的所有相关信息来创建初始签名文件。
- 可选地，可以编辑F2PY创建的签名文件以优化包装器功能，使它们“更智能”和更“Pythonic”。
- F2PY读取签名文件并编写包含Fortran / C / Python绑定的Python C / API模块。
- F2PY编译所有源并构建包含包装器的扩展模块。在构建扩展模块时，F2PY使用
 ``numpy_distutils``它支持许多Fortran 77/90/95编译器，包括Gnu，Intel，Sun Fortre，SGI MIPSpro，Absoft，NAG，Compaq等编译器。

根据具体情况，这些步骤可以通过一个命令或一步一步执行，一些步骤可以省略或与其他步骤组合。

下面我将描述使用F2PY的三种典型方法。以下示例 ``Fortran 77代码`` 将说明：

``` python
C FILE: FIB1.F
      SUBROUTINE FIB(A,N)
C
C     CALCULATE FIRST N FIBONACCI NUMBERS
C
      INTEGER N
      REAL*8 A(N)
      DO I=1,N
         IF (I.EQ.1) THEN
            A(I) = 0.0D0
         ELSEIF (I.EQ.2) THEN
            A(I) = 1.0D0
         ELSE 
            A(I) = A(I-1) + A(I-2)
         ENDIF
      ENDDO
      END
C END FILE FIB1.F
```

## 快捷的方式

将Fortran子例程包装``FIB``到Python 的最快方法是运行

``` python
python -m numpy.f2py -c fib1.f -m fib1
```

此命令构建（参见``-c``flag，不带参数执行以查看命令行选项的说明）扩展模块（请参阅标志）到当前目录。现在，在Python中，可以通过以下方式访问Fortran子例程：``python -m numpy.f2py``fib1.so``-m``FIB``fib1.fib``

``` python
>>> import numpy
>>> import fib1
>>> print fib1.fib.__doc__
fib - Function signature:
  fib(a,[n])
Required arguments:
  a : input rank-1 array('d') with bounds (n)
Optional arguments:
  n := len(a) input int

>>> a = numpy.zeros(8,'d')
>>> fib1.fib(a)
>>> print a
[  0.   1.   1.   2.   3.   5.   8.  13.]
```

::: tip 注意

- 请注意，F2PY发现第二个参数``n``是第一个数组参数的维度``a``。由于默认情况下所有参数都是仅输入参数，因此F2PY ``n``可以使用默认值作为可选参数``len(a)``。
- 可以使用不同的值来选择``n``：

``` python
>>> a1 = numpy.zeros(8,'d')
>>> fib1.fib(a1,6)
>>> print a1
[ 0.  1.  1.  2.  3.  5.  0.  0.]
```

但是当它与输入数组不兼容时会引发异常``a``：

``` python
>>> fib1.fib(a,10)
fib:n=10
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
fib.error: (len(a)>=n) failed for 1st keyword n
>>>
```

这展示了F2PY中的一个有用功能，即F2PY实现相关参数之间的基本兼容性检查，以避免任何意外崩溃。
- 当一个NumPy数组（即Fortran连续且具有与假定的Fortran类型相对应的dtype）用作输入数组参数时，其C指针将直接传递给Fortran。

否则，F2PY会生成输入数组的连续副本（具有正确的dtype），并将副本的C指针传递给Fortran子例程。因此，对输入数组（副本）的任何可能更改都不会影响原始参数，如下所示：

``` python
>>> a = numpy.ones(8,'i')
>>> fib1.fib(a)
>>> print a
[1 1 1 1 1 1 1 1]
```

显然，这不是预期的行为。上述示例使用的事实``dtype=float``被认为是偶然的。

F2PY提供``intent(inplace)``了将修改输入数组属性的属性，以便Fortran例程所做的任何更改也将在输入参数中生效。例如，如果指定（见下文，如何），则上面的示例将为：``intent(inplace) a``

``` python
>>> a = numpy.ones(8,'i')
>>> fib1.fib(a)
>>> print a
[  0.   1.   1.   2.   3.   5.   8.  13.]
```

但是，将Fortran子例程所做的更改返回到python的推荐方法是使用``intent(out)``属性。它更有效，更清洁。
- ``fib1.fib``Python中的用法与``FIB``在Fortran中使用非常相似
 。但是，在Python中使用 *原位* 输出参数表明样式很差，因为Python中没有关于错误参数类型的安全机制。使用Fortran或C时，编译器自然会在编译期间发现任何类型不匹配，但在Python中，必须在运行时检查类型。因此，在Python中使用 *原位* 输出参数可能会导致难以发现错误，更不用说在实现所有必需的类型检查时代码将不太可读。

虽然将Fortran例程包装到Python的演示方法非常简单，但它有几个缺点（参见上面的注释）。这些缺点是由于F2PY无法确定一个或另一个参数的实际意图，输入或输出参数，或两者，或其他东西。因此，F2PY保守地假定所有参数都是默认的输入参数。

但是，有一些方法（见下文）如何“教导”F2PY关于函数参数的真实意图（以及其他内容）; 然后F2PY能够为Fortran函数生成更多Pythonic（更明确，更易于使用，更不容易出错）的包装器。

:::

## 聪明的方式

让我们逐个应用将Fortran函数包装到Python的步骤。

- 首先，我们``fib1.f``通过运行创建一个签名文件

    ``` python
    python -m numpy.f2py fib1.f -m fib2 -h fib1.pyf
    ```

    签名文件保存到``fib1.pyf``（见``-h``标志），其内容如下所示。

    ``` python
    !    -*- f90 -*-
    python module fib2 ! in 
        interface  ! in :fib2
            subroutine fib(a,n) ! in :fib2:fib1.f
                real*8 dimension(n) :: a
                integer optional,check(len(a)>=n),depend(a) :: n=len(a)
            end subroutine fib
        end interface 
    end python module fib2

    ! This file was auto-generated with f2py (version:2.28.198-1366).
    ! See http://cens.ioc.ee/projects/f2py2e/
    ```

- 接下来，我们将教导F2PY参数``n``是一个输入参数（use ``intent(in)``属性），结果，即``a``调用Fortran函数后的内容``FIB``，应该返回给Python（use ``intent(out)``属性）。此外，``a``应使用input参数给出的大小动态创建数组``n``（use ``depend(n)``属性表示依赖关系）。

    修改后的版本``fib1.pyf``（保存为
    ``fib2.pyf``）的内容如下：

    ``` python
    !    -*- f90 -*-
    python module fib2 
        interface
            subroutine fib(a,n)
                real*8 dimension(n),intent(out),depend(n) :: a
                integer intent(in) :: n
            end subroutine fib
        end interface 
    end python module fib2
    ```

- 最后，我们通过运行构建扩展模块

    ``` python
    python -m numpy.f2py -c fib2.pyf fib1.f
    ```

在Python中：

``` python
>>> import fib2
>>> print fib2.fib.__doc__
fib - Function signature:
  a = fib(n)
Required arguments:
  n : input int
Return objects:
  a : rank-1 array('d') with bounds (n)

>>> print fib2.fib(8)
[  0.   1.   1.   2.   3.   5.   8.  13.]
```

::: tip 注意

- 显然，``fib2.fib``现在的签名``FIB``更接近Fortran子程序的意图：给定数字``n``，``fib2.fib``将第一个``n``Fibonacci数作为NumPy数组返回。此外，新的Python签名``fib2.fib``
排除了我们遇到的任何意外``fib1.fib``。
- 请注意，默认情况下使用single ``intent(out)``也意味着
 ``intent(hide)``。具有``intent(hide)``指定属性的参数将不会列在包装函数的参数列表中。

:::

## 快捷而聪明的方式

如上所述，包装Fortran函数的“智能方法”适用于包装（例如第三方）Fortran代码，对其源代码的修改是不可取的，甚至也不可能。

但是，如果编辑Fortran代码是可以接受的，则在大多数情况下可以跳过生成中间签名文件。即，可以使用所谓的F2PY指令将F2PY特定属性直接插入到Fortran源代码中。F2PY指令定义了特殊注释行（``Cf2py``例如，从Fortran编译器开始），但是F2PY将它们解释为普通行。

下面显示了示例Fortran代码的修改版本，保存为``fib3.f``：

``` python
C FILE: FIB3.F
      SUBROUTINE FIB(A,N)
C
C     CALCULATE FIRST N FIBONACCI NUMBERS
C
      INTEGER N
      REAL*8 A(N)
Cf2py intent(in) n
Cf2py intent(out) a
Cf2py depend(n) a
      DO I=1,N
         IF (I.EQ.1) THEN
            A(I) = 0.0D0
         ELSEIF (I.EQ.2) THEN
            A(I) = 1.0D0
         ELSE 
            A(I) = A(I-1) + A(I-2)
         ENDIF
      ENDDO
      END
C END FILE FIB3.F
```

现在可以在一个命令中执行构建扩展模块：

``` python
python -m numpy.f2py -c -m fib3 fib3.f
```

请注意，生成的包装器与``FIB``前一种情况一样“智能”：

``` python
>>> import fib3
>>> print fib3.fib.__doc__
fib - Function signature:
  a = fib(n)
Required arguments:
  n : input int
Return objects:
  a : rank-1 array('d') with bounds (n)

>>> print fib3.fib(8)
[  0.   1.   1.   2.   3.   5.   8.  13.]
```