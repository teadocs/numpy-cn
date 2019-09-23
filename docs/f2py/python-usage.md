---
meta:
  - name: keywords
    content: 在Python中使用F2PY构建 NumPy
  - name: description
    content: Fortran / C例程，公共块或F2PY生成的Fortran 90模块数据的所有包装器都作为fortran 类型对象公开给Python。
---

# 在Python中使用F2PY构建

Fortran / C例程，公共块或F2PY生成的Fortran 90模块数据的所有包装器都作为``fortran`` 类型对象公开给Python 。
例程包装器是可调用``fortran``类型对象，而Fortran数据包装器具有引用数据对象的属性。

所有``fortran``类型对象都具有``_cpointer``包含CObject的属性，该CObject引用相应Fortran / C函数的C指针或C级别的变量。当这些函数的计算部分在C或Fortran中实现并用F2PY（或任何其他工具）包装时，这些CObject可以用作F2PY生成函数的回调参数，以绕过从Fortran或C调用Python函数的Python C / API层。能够提供功能的CObject）。

考虑一个Fortran 77文件``ftype.f``：

``` python
C FILE: FTYPE.F
      SUBROUTINE FOO(N)
      INTEGER N
Cf2py integer optional,intent(in) :: n = 13
      REAL A,X
      COMMON /DATA/ A,X(3)
      PRINT*, "IN FOO: N=",N," A=",A," X=[",X(1),X(2),X(3),"]"
      END
C END OF FTYPE.F
```

并使用。构建一个包装器。``f2py -c ftype.f -m ftype``

在Python中：

``` python
>>> import ftype
>>> print ftype.__doc__
This module 'ftype' is auto-generated with f2py (version:2.28.198-1366).
Functions:
  foo(n=13)
COMMON blocks:
  /data/ a,x(3)
.
>>> type(ftype.foo),type(ftype.data)
(<type 'fortran'>, <type 'fortran'>)
>>> ftype.foo()
 IN FOO: N= 13 A=  0. X=[  0.  0.  0.]
>>> ftype.data.a = 3
>>> ftype.data.x = [1,2,3]
>>> ftype.foo()
 IN FOO: N= 13 A=  3. X=[  1.  2.  3.]
>>> ftype.data.x[1] = 45  
>>> ftype.foo(24)
 IN FOO: N= 24 A=  3. X=[  1.  45.  3.]
>>> ftype.data.x
array([  1.,  45.,   3.],'f')
```

## 标量参数

通常，F2PY生成的包装函数的标量参数可以是普通的Python标量（整数，浮点数，复数）以及标量的任意序列对象（列表，元组，数组，字符串）。在后一种情况下，序列对象的第一个元素作为标量参数传递给Fortran例程。

请注意，当需要进行类型转换并且可能丢失信息时（例如，当将类型转换浮动为整数或复数转换为浮动时），F2PY不会引发任何异常。在复杂到真实的类型转换中，仅使用复数的实部。

``intent(inout)``标量参数被假定为数组对象，以便 *原位* 更改生效。建议使用具有适当类型的数组，但也可以使用其他类型的数组。

考虑以下Fortran 77代码：

``` python
C FILE: SCALAR.F
      SUBROUTINE FOO(A,B)
      REAL*8 A, B
Cf2py intent(in) a
Cf2py intent(inout) b
      PRINT*, "    A=",A," B=",B
      PRINT*, "INCREMENT A AND B"
      A = A + 1D0
      B = B + 1D0
      PRINT*, "NEW A=",A," B=",B
      END
C END OF FILE SCALAR.F
```

并使用它包装。``f2py -c -m scalar scalar.f``

在Python中：

``` python
>>> import scalar
>>> print scalar.foo.__doc__
foo - Function signature:
  foo(a,b)
Required arguments:
  a : input float
  b : in/output rank-0 array(float,'d')
 
>>> scalar.foo(2,3)   
     A=  2. B=  3.
 INCREMENT A AND B
 NEW A=  3. B=  4.
>>> import numpy
>>> a=numpy.array(2)   # these are integer rank-0 arrays
>>> b=numpy.array(3)
>>> scalar.foo(a,b)
     A=  2. B=  3.
 INCREMENT A AND B
 NEW A=  3. B=  4.
>>> print a,b            # note that only b is changed in situ
2 4
```

## 字符串参数

F2PY生成的包装函数接受（几乎）任何Python对象作为字符串参数，``str``应用于非字符串对象。例外是NumPy数组必须具有类型代码``'c'``或
 ``'1'``用作字符串参数。

当将字符串用作F2PY生成的包装函数的字符串参数时，字符串可以具有任意长度。如果长度大于预期，则字符串将被截断。如果长度小于预期，则分配并填充额外的内存``\0``。

因为Python字符串是不可变的，所以``intent(inout)``参数需要字符串的数组版本才能使 *原位* 更改生效。

考虑以下Fortran 77代码：

``` python
C FILE: STRING.F
      SUBROUTINE FOO(A,B,C,D)
      CHARACTER*5 A, B
      CHARACTER*(*) C,D
Cf2py intent(in) a,c
Cf2py intent(inout) b,d
      PRINT*, "A=",A
      PRINT*, "B=",B
      PRINT*, "C=",C
      PRINT*, "D=",D
      PRINT*, "CHANGE A,B,C,D"
      A(1:1) = 'A'
      B(1:1) = 'B'
      C(1:1) = 'C'
      D(1:1) = 'D'
      PRINT*, "A=",A
      PRINT*, "B=",B
      PRINT*, "C=",C
      PRINT*, "D=",D
      END
C END OF FILE STRING.F
```

并使用它包装。``f2py -c -m mystring string.f``

Python会话：

``` python
>>> import mystring
>>> print mystring.foo.__doc__
foo - Function signature:
  foo(a,b,c,d)
Required arguments:
  a : input string(len=5)
  b : in/output rank-0 array(string(len=5),'c')
  c : input string(len=-1)
  d : in/output rank-0 array(string(len=-1),'c')

>>> import numpy
>>> a=numpy.array('123')
>>> b=numpy.array('123')
>>> c=numpy.array('123')
>>> d=numpy.array('123')
>>> mystring.foo(a,b,c,d)
 A=123
 B=123
 C=123
 D=123
 CHANGE A,B,C,D
 A=A23
 B=B23
 C=C23
 D=D23
>>> a.tostring(),b.tostring(),c.tostring(),d.tostring()
('123', 'B23', '123', 'D23')
```

## 数组参数

通常，F2PY生成的包装函数的数组参数接受可以转换为NumPy数组对象的任意序列。一个例外是``intent(inout)``数组参数，它们必须始终是正确连续的并且具有正确的类型，否则会引发异常。另一个例外是``intent(inplace)``数组参数，如果参数的类型与预期不同，则属性将在原位更改（``intent(inplace)``有关更多信息，请参阅属性）。

通常，如果NumPy数组是正确连续的并且具有适当的类型，那么它将直接传递给包装的Fortran / C函数。否则，将生成输入数组的元素副本，并将正确连续且具有适当类型的副本用作数组参数。

有两种类型的适当连续的NumPy数组：

- 当数据按列存储时，Fortran连续数组，即存储在存储器中的数据索引从最低维开始;
- 当数据以行方式存储时，C-连续或简单连续的数组，即存储在存储器中的数据的索引从最高维度开始。

对于一维数组，这些概念重合。

例如，如果2x2数组``A``的元素按以下顺序存储在内存中，则它是Fortran连续的：

``` python
A[0,0] A[1,0] A[0,1] A[1,1]
```

如果订单如下，则为C-contiguous：

``` python
A[0,0] A[0,1] A[1,0] A[1,1]
```

要测试数组是否为C-contiguous，请使用``.iscontiguous()``
NumPy数组的方法。为了测试Fortran连续性，所有F2PY生成的扩展模块都提供了一个功能
 ``has_column_major_storage()``。此功能相当于
 ``.flags.f_contiguous``但效率更高。

通常不需要担心数组如何存储在内存中以及包装函数（Fortran函数还是C函数）是否采用一个或另一个存储顺序。F2PY自动确保包装函数以适当的存储顺序获取参数; 相应的算法被设计为仅在绝对必要时才复制数组。但是，当处理大小接近计算机中物理内存大小的非常大的多维输入数组时，必须注意始终使用适当连续且正确的类型参数。

要在将输入数组传递给Fortran例程之前将其转换为列主存储顺序，请使用``as_column_major_storage()``由所有F2PY生成的扩展模块提供的函数
 。

考虑Fortran 77代码：

``` python
C FILE: ARRAY.F
      SUBROUTINE FOO(A,N,M)
C
C     INCREMENT THE FIRST ROW AND DECREMENT THE FIRST COLUMN OF A
C
      INTEGER N,M,I,J
      REAL*8 A(N,M)
Cf2py intent(in,out,copy) a
Cf2py integer intent(hide),depend(a) :: n=shape(a,0), m=shape(a,1)
      DO J=1,M
         A(1,J) = A(1,J) + 1D0
      ENDDO
      DO I=1,N
         A(I,1) = A(I,1) - 1D0
      ENDDO
      END
C END OF FILE ARRAY.F
```

并使用它包装。``f2py -c -m arr array.f -DF2PY_REPORT_ON_ARRAY_COPY=1``

在Python中：

``` python
>>> import arr
>>> from numpy import array
>>> print arr.foo.__doc__
foo - Function signature:
  a = foo(a,[overwrite_a])
Required arguments:
  a : input rank-2 array('d') with bounds (n,m)
Optional arguments:
  overwrite_a := 0 input int
Return objects:
  a : rank-2 array('d') with bounds (n,m)

>>> a=arr.foo([[1,2,3],
...            [4,5,6]])
copied an array using PyArray_CopyFromObject: size=6, elsize=8
>>> print a
[[ 1.  3.  4.]
 [ 3.  5.  6.]]
>>> a.iscontiguous(), arr.has_column_major_storage(a)
(0, 1)
>>> b=arr.foo(a)              # even if a is proper-contiguous
...                           # and has proper type, a copy is made
...                           # forced by intent(copy) attribute
...                           # to preserve its original contents
... 
copied an array using copy_ND_array: size=6, elsize=8
>>> print a
[[ 1.  3.  4.]
 [ 3.  5.  6.]]
>>> print b
[[ 1.  4.  5.]
 [ 2.  5.  6.]]
>>> b=arr.foo(a,overwrite_a=1) # a is passed directly to Fortran
...                            # routine and its contents is discarded
... 
>>> print a
[[ 1.  4.  5.]
 [ 2.  5.  6.]]
>>> print b
[[ 1.  4.  5.]
 [ 2.  5.  6.]]
>>> a is b                       # a and b are actually the same objects
1
>>> print arr.foo([1,2,3])       # different rank arrays are allowed
copied an array using PyArray_CopyFromObject: size=3, elsize=8
[ 1.  1.  2.]
>>> print arr.foo([[[1],[2],[3]]])
copied an array using PyArray_CopyFromObject: size=3, elsize=8
[ [[ 1.]
  [ 3.]
  [ 4.]]]
>>>
>>> # Creating arrays with column major data storage order:
...
>>> s = arr.as_column_major_storage(array([[1,2,3],[4,5,6]]))
copied an array using copy_ND_array: size=6, elsize=4
>>> arr.has_column_major_storage(s)
1
>>> print s
[[1 2 3]
 [4 5 6]]
>>> s2 = arr.as_column_major_storage(s)
>>> s2 is s    # an array with column major storage order 
               # is returned immediately
1
```

## 回调参数

F2PY支持从Fortran或C代码调用Python函数。

考虑以下Fortran 77代码：

``` python
C FILE: CALLBACK.F
      SUBROUTINE FOO(FUN,R)
      EXTERNAL FUN
      INTEGER I
      REAL*8 R
Cf2py intent(out) r
      R = 0D0
      DO I=-5,5
         R = R + FUN(I)
      ENDDO
      END
C END OF FILE CALLBACK.F
```

并使用它包装。``f2py -c -m callback callback.f``

在Python中：

``` python
>>> import callback
>>> print callback.foo.__doc__
foo - Function signature:
  r = foo(fun,[fun_extra_args])
Required arguments:
  fun : call-back function
Optional arguments:
  fun_extra_args := () input tuple
Return objects:
  r : float
Call-back functions:
  def fun(i): return r
  Required arguments:
    i : input int
  Return objects:
    r : float

>>> def f(i): return i*i
... 
>>> print callback.foo(f)     
110.0
>>> print callback.foo(lambda i:1)
11.0
```

在上面的示例中，F2PY能够准确地猜测回叫函数的签名。但是，有时F2PY无法按照人们的意愿建立签名，然后必须手动修改签名文件中的回调函数的签名。即，签名文件可以包含特殊模块（这些模块的名称包含子串``__user__``），其收集回调函数的各种签名。例程签名中的回调参数具有属性``external``（另请参见``intent(callback)``属性）。要在``__user__``模块块中关联回调参数及其签名，请使用``use``如下所示的语句。回调参数的相同签名可以在不同的例程签名中引用。

我们使用与前面示例相同的Fortran 77代码，但现在我们假装F2PY无法正确猜测回调参数的签名。首先，我们``callback2.pyf``使用F2PY 创建一个初始签名文件：

``` python
f2py -m callback2 -h callback2.pyf callback.f
```

然后按如下方式修改它

``` python
!    -*- f90 -*-
python module __user__routines 
    interface
        function fun(i) result (r)
            integer :: i
            real*8 :: r
        end function fun
    end interface
end python module __user__routines

python module callback2
    interface
        subroutine foo(f,r)
            use __user__routines, f=>fun
            external f
            real*8 intent(out) :: r
        end subroutine foo
    end interface 
end python module callback2
```

最后，使用构建扩展模块。``f2py -c callback2.pyf callback.f``

示例Python会话与前面的示例相同，只是参数名称会有所不同。

有时，Fortran程序包可能要求用户提供程序包将使用的例程。F2PY可以构造这种例程的接口，以便可以从Fortran调用Python函数。

考虑以下Fortran 77子例程，它接受一个数组并将一个函数``func``应用于其元素。

``` python
subroutine calculate(x,n)
cf2py intent(callback) func
      external func
c     The following lines define the signature of func for F2PY:
cf2py real*8 y
cf2py y = func(y)
c
cf2py intent(in,out,copy) x
      integer n,i
      real*8 x(n)
      do i=1,n
         x(i) = func(x(i))
      end do
      end
```

预计功能``func``已在外部定义。为了使用Python函数``func``，它必须具有一个属性``intent(callback)``（必须在``external``语句之前指定）。

最后，使用构建扩展模块 ``f2py -c -m foo calculate.f``

在Python中：

``` python
>>> import foo
>>> foo.calculate(range(5), lambda x: x*x)
array([  0.,   1.,   4.,   9.,  16.])
>>> import math
>>> foo.calculate(range(5), math.exp)
array([  1.        ,   2.71828175,   7.38905621,  20.08553696,  54.59814835])
```

该函数作为参数包含在对Fortran子例程的python函数调用中，即使它 *不在*  Fortran子例程参数列表中。“外部”是指由f2py生成的C函数，而不是python函数本身。必须将python函数提供给C函数。

回调函数也可以在模块中明确设置。然后，没有必要将参数列表中的函数传递给Fortran函数。如果调用python回调函数的Fortran函数本身由另一个Fortran函数调用，则可能需要这样做。

考虑以下Fortran 77子例程：

``` python
subroutine f1()
         print *, "in f1, calling f2 twice.."
         call f2()
         call f2()
         return
      end
      
      subroutine f2()
cf2py    intent(callback, hide) fpy
         external fpy
         print *, "in f2, calling f2py.."
         call fpy()
         return
      end
```

并使用它包装。``f2py -c -m pfromf extcallback.f``

在Python中：

``` python
>>> import pfromf
>>> pfromf.f2()
Traceback (most recent call last):
  File "<stdin>", line 1, in ?
pfromf.error: Callback fpy not defined (as an argument or module pfromf attribute).

>>> def f(): print "python f"
... 
>>> pfromf.fpy = f
>>> pfromf.f2()
 in f2, calling f2py..
python f
>>> pfromf.f1()
 in f1, calling f2 twice..
 in f2, calling f2py..
python f
 in f2, calling f2py..
python f
>>>
```

### 解决回调函数的参数

F2PY生成的接口在回调参数方面非常灵活。对于每个回调参数``_extra_args``，F2PY引入了另一个可选参数。此参数可用于将额外参数传递给用户提供的回调参数。

如果F2PY生成的包装函数需要以下回调参数：

``` python
def fun(a_1,...,a_n):
   ...
   return x_1,...,x_k
```

但是以下Python函数

``` python
def gun(b_1,...,b_m):
   ...
   return y_1,...,y_l
```

由用户提供，此外，

``` python
fun_extra_args = (e_1,...,e_p)
```

如果使用Fortran或C函数调用回调参数，则应用以下规则``gun``：

- 如果那时被调用，这里
 。``p == 0``gun(a_1, ..., a_q)``q = min(m, n)``
- 如果那时被召唤。``n + p <= m``gun(a_1, ..., a_n, e_1, ..., e_p)``
- 如果那时被调用，这里
 。``p <= m < n + p``gun(a_1, ..., a_q, e_1, ..., e_p)``q=m-p``
- 如果那时被召唤。``p > m``gun(e_1, ..., e_m)``
- 如果小于所需参数的数量，
则引发异常。``n + p``gun``

该函数``gun``可以将任意数量的对象作为元组返回。然后应用以下规则：

- 如果，则忽略。``k < l``y_{k + 1}, ..., y_l``
- 如果，那么只设置。``k > l``x_1, ..., x_l``

## 常用块

F2PY ``common``为例程签名块中定义的块生成包装器。所有与当前扩展模块链接的Fortran代码都可以看到公共块，但不能看到其他扩展模块（这种限制是由于Python导入共享库的原因）。在Python中，``common``块的F2PY包装器``fortran``是具有与公共块的数据成员相关的（动态）属性的类型对象。访问时，这些属性作为NumPy数组对象（多维数组是Fortran连续）返回，这些对象直接链接到公共块中的数据成员。可以通过直接赋值或通过对相应数组对象的就地更改来更改数据成员。

考虑以下Fortran 77代码：

``` python
C FILE: COMMON.F
      SUBROUTINE FOO
      INTEGER I,X
      REAL A
      COMMON /DATA/ I,X(4),A(2,3)
      PRINT*, "I=",I
      PRINT*, "X=[",X,"]"
      PRINT*, "A=["
      PRINT*, "[",A(1,1),",",A(1,2),",",A(1,3),"]"
      PRINT*, "[",A(2,1),",",A(2,2),",",A(2,3),"]"
      PRINT*, "]"
      END
C END OF COMMON.F
```

并使用它包装。``f2py -c -m common common.f``

在Python中：

``` python
>>> import common
>>> print common.data.__doc__
i - 'i'-scalar
x - 'i'-array(4)
a - 'f'-array(2,3)

>>> common.data.i = 5
>>> common.data.x[1] = 2 
>>> common.data.a = [[1,2,3],[4,5,6]]
>>> common.foo()
 I= 5
 X=[ 0 2 0 0]
 A=[
 [  1.,  2.,  3.]
 [  4.,  5.,  6.]
 ]
>>> common.data.a[1] = 45
>>> common.foo()
 I= 5
 X=[ 0 2 0 0]
 A=[
 [  1.,  2.,  3.]
 [  45.,  45.,  45.]
 ]
>>> common.data.a                 # a is Fortran-contiguous
array([[  1.,   2.,   3.],
       [ 45.,  45.,  45.]],'f')
```

## Fortran 90模块数据

Fortran 90模块数据的F2PY接口类似于Fortran 77公共块。

考虑以下Fortran 90代码：

``` python
module mod
  integer i
  integer :: x(4)
  real, dimension(2,3) :: a
  real, allocatable, dimension(:,:) :: b 
contains
  subroutine foo
    integer k
    print*, "i=",i
    print*, "x=[",x,"]"
    print*, "a=["
    print*, "[",a(1,1),",",a(1,2),",",a(1,3),"]"
    print*, "[",a(2,1),",",a(2,2),",",a(2,3),"]"
    print*, "]"
    print*, "Setting a(1,2)=a(1,2)+3"
    a(1,2) = a(1,2)+3
  end subroutine foo
end module mod
```

并使用它包装。``f2py -c -m moddata moddata.f90``

在Python中：

``` python
>>> import moddata
>>> print moddata.mod.__doc__
i - 'i'-scalar
x - 'i'-array(4)
a - 'f'-array(2,3)
foo - Function signature:
  foo()


>>> moddata.mod.i = 5  
>>> moddata.mod.x[:2] = [1,2]
>>> moddata.mod.a = [[1,2,3],[4,5,6]]
>>> moddata.mod.foo()                
 i=           5
 x=[           1           2           0           0 ]
 a=[
 [   1.000000     ,   2.000000     ,   3.000000     ]
 [   4.000000     ,   5.000000     ,   6.000000     ]
 ]
 Setting a(1,2)=a(1,2)+3
>>> moddata.mod.a               # a is Fortran-contiguous
array([[ 1.,  5.,  3.],
       [ 4.,  5.,  6.]],'f')
```

### 可分配数组

F2PY对Fortran 90模块可分配阵​​列提供基本支持。

考虑以下Fortran 90代码：

``` python
module mod
  real, allocatable, dimension(:,:) :: b 
contains
  subroutine foo
    integer k
    if (allocated(b)) then
       print*, "b=["
       do k = 1,size(b,1)
          print*, b(k,1:size(b,2))
       enddo
       print*, "]"
    else
       print*, "b is not allocated"
    endif
  end subroutine foo
end module mod
```

并使用它包装。``f2py -c -m allocarr allocarr.f90``

在Python中：

``` python
>>> import allocarr 
>>> print allocarr.mod.__doc__
b - 'f'-array(-1,-1), not allocated
foo - Function signature:
  foo()

>>> allocarr.mod.foo()  
 b is not allocated
>>> allocarr.mod.b = [[1,2,3],[4,5,6]]         # allocate/initialize b
>>> allocarr.mod.foo()
 b=[
   1.000000       2.000000       3.000000    
   4.000000       5.000000       6.000000    
 ]
>>> allocarr.mod.b                             # b is Fortran-contiguous
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]],'f')
>>> allocarr.mod.b = [[1,2,3],[4,5,6],[7,8,9]] # reallocate/initialize b
>>> allocarr.mod.foo()
 b=[
   1.000000       2.000000       3.000000    
   4.000000       5.000000       6.000000    
   7.000000       8.000000       9.000000    
 ]
>>> allocarr.mod.b = None                      # deallocate array
>>> allocarr.mod.foo()
 b is not allocated
```
