# Numpy对于Matlab用户

## 介绍

MATLAB®和 NumPy/SciPy 有很多共同之处。但是也有很多不同之处。创建NumPy和SciPy是为了用Python最自然的方式进行数值和科学计算，而不是 MATLAB® 的克隆版。本章节旨在收集有关两者的差异，主要是为了帮助熟练的MATLAB®用户成为熟练的NumPy和SciPy用户。

## 一些关键的差异

MATLAB | NumPy
---|---
在MATLAB®中，基本数据类型是双精度浮点数的多维数组。大多数表达式采用这样的数组并且也返回这样的数据类型，操作这些数组的2-D实例的的方式被设计成或多或少地像线性代数中的矩阵运算一样。 | 在NumPy中，基本类型是多维``数组``。在所有维度(包括2D)上对这些数组的操作都是元素级的操作。但是，有一种特殊的``矩阵``类型用于做线性代数，它只是``array``类的一个子类。矩阵类数组的运算是线性代数运算.
MATLAB®是使用基于1（1）的索引。 使用（1）下标作为元素的初始位置。 请参阅 索引 | Python使用基于0(零)的索引。序列的初始元素使用[0]来查找。
MATLAB的脚本语言是为做线性代数而创建的。基本矩阵操作的语法很好也很干净，但是用于添加GUI和制作成熟应用程序的API或多或少是事后才想到的。| NumPy基于Python，它从一开始就被设计为一种优秀的通用编程语言。 虽然Matlab的一些数组操作的语法比NumPy更紧凑，但NumPy（由于是Python的附加组件）可以做许多Matlab所不能做的事情，例如将主数组类型子类化为干净地进行数组和矩阵数学运算。
在 MATLAB® 中，数组具有按值传递的语义，并有一个懒散的写拷贝方案，以防止在真正需要副本之前实际创建副本。使用切片操作复制数组的部分。 | 在NumPy数组，数组只是内存中的引用而已。 切片只是操作数组的视图层面而已。

## ‘array’ 和 ‘matrix’? 我应该选谁?

除了 ``np.ndarray`` 之外，NumPy还提供了一种额外的矩阵类型，您可以在一些现有代码中看到这种类型。想用哪一个呢？

### 简要的回答

**使用 arrays.**

- 它们是numpy的标准向量/矩阵/张量类型。许多numpy函数返回的是数组，而不是矩阵。
- 元素级运算和线性代数运算之间有着明确的区别。
- 如果你愿意，可以使用标准向量或行/列向量。

直到Python3.5，使用数组类型的唯一缺点是您必须使用 ``dot`` 而不是 ``*`` 来将两个张量(标量乘积、矩阵向量乘法等)相乘(减少)。从 Python3.5 之后，您就可以使用矩阵乘法 ``@`` 运算符。

### 详细的回答

NumPy包含 ``array`` 类和 ``Matrix`` 类。 ``array`` 类旨在成为用于多种数值计算的通用n维数组，而 ``matrix`` 类则专门用于促进线性代数计算。实际上，两者之间只有少数几个关键区别。

- 操作符 ``*``, ``dot()``, 和 ``multiply()``:
    - 对于数组， ``*`` 表示逐元素乘法，而 ``dot()`` 函数用于矩阵乘法。
    - 对于矩阵， ``*`` 表示矩阵乘法， ``multiply()`` 函数用于逐元素乘法。
- 矢量处理（一维数组）
    - 对于数组，向量形状1xN，Nx1和N都是不同的东西。像A [:,1]这样的操作返回形状N的一维数组，而不是形状Nx1的二维数组。
一维数组上的转置不起任何作用。
    - 对于矩阵，一维数组总是向上转换为1xN或Nx1矩阵（行或列向量）。A [:,1] 返回形状为Nx1的二维矩阵。
- 处理更高维数组（ndim> 2）
    - 数组对象的维数可以> 2;
    - 矩阵对象总是具有两个维度。
- 便捷的属性
    - ``array``有一个.T属性，它返回数据的转置。
    - ``矩阵``还具有.H，.I和.A属性，分别返回矩阵的共轭转置，反转和 ``asarray()``。
- 便捷的构造器
    - ``数组``构造函数接受(嵌套的)Pythonseqxues作为初始化器。如 ``array([1,2,3], [4,5,6])``。
    - ``矩阵``构造器另外采用方便的字符串初始化器（传入的参数是字符串）。如 ``matrix("[1 2 3; 4 5 6]")``。

使用这两种方法有好处也有坏处：

- 数组
    - :) 你可以将一维数组视为行或列向量。dot(A, v)将v视为列向量，而 ``dot(v，A)``将``v``视为行向量。这可以让你少传入许多的转置。
    - <:( 必须使用 dot() 函数进行矩阵乘法是很麻烦的 – ``dot(dot(A,B),C)`` vs. ``A*B*C``. 这不是Python> = 3.5的问题，因为``@``运算符允许它写成``A @ B @ C``.
    - :) 元素乘法很容易，直接：``A*B`` 就好.
    - :) ``array``是 "默认" 的NumPy类型，因此它获得的支持最多，并且大量的NumPy的第三方包都使用了这个类型。
    - :) 它可以更稳定的处理任意数量级的数据。
    - :) 如果是熟悉的话，其实它的语义更接近张量代数。
    - :) 所有的运算符 (``*``, ``/``, ``+``, ``-``) 都是元素级别的。
- 矩阵
    - :\\ 行为更像MATLAB®矩阵。
    - <:( 它的维度最大为二维，如果你想保存三维数据，你需要数组或者一个矩阵列表。
    - <:( 它的维度最少也是二维，你不能有向量还必须将它们转换为单列或单行矩阵。
    - <:( 由于 ``array`` 类型是NumPy中的默认类型，因此即使你将``矩阵``作为参数传入，某些函数也可能返回一个 ``array``类型。
在NumPy的内部函数中应该没有出现可以接受矩阵作为参数的情况（如果有，那它可能是一个bug），但基于NumPy的第三方包可能不像NumPy那样遵守类型规则。
    - :) ``A*B`` 是矩阵乘法，因此线性代数更方便(对于Python >= 3.5 的版本，普通数组与 ``@`` 运算符具有同样的方便性)。
    - <:( 元素级乘法要求调用乘法函数： ``multiply(A,B)``。
    - <:( 操作符重载的使用有点不合逻辑：元素级别中``*``运算符并不会生效， 但是``/``运算符却可以。

因此使用``array``是更为可取的。

## 使用Matrix的用户的福音

NumPy有一些特性，可以方便地使用``matrix``类型，这有望使Matlab的用户转化过来更为容易。

- 新增了一个``matlib``模块，该模块包含常用数组构造函数的矩阵版本，如one()、zeros()、empty()、view()、rand()、repmat()等。通常，这些函数返回数组，但matlib版本返回矩阵对象。
- ``mat`` 已被更改为``asmatrix``的同义词，而不是矩阵，从而使其成为将数组转换为矩阵而不复制数据的简洁方法。
- 一些顶级的函数已被删除。例如，``numpy.rand()``现在需要作为``numpy.random.rand()``访问。或者使用``matlib``模块中的``rand()``。但是“numpythonic”方式是使用``numpy.random.random()``，它为元组数据类型作为shape（形状），就像其他numpy函数一样。

## Table of Rough MATLAB-NumPy Equivalents

The table below gives rough equivalents for some common MATLAB® expressions. **These are not exact equivalents**, but rather should be taken as hints to get you going in the right direction. For more detail read the built-in documentation on the NumPy functions.

Some care is necessary when writing functions that take arrays or matrices as arguments — if you are expecting an ``array`` and are given a ``matrix``, or vice versa, then ‘*’ (multiplication) will give you unexpected results. You can convert back and forth between arrays and matrices using

- ``asarray``: always returns an object of type ``array``
- ``asmatrix`` or ``mat``: always return an object of type ``matrix``
- ``asanyarray``: always returns an array object or a subclass derived from it, depending on the input. For instance if you pass in a ``matrix`` it returns a ``matrix``.

These functions all accept both arrays and matrices (among other things like Python lists), and thus are useful when writing functions that should accept any array-like object.

In the table below, it is assumed that you have executed the following commands in Python:

```python
from numpy import *
import scipy.linalg
```

Also assume below that if the Notes talk about “matrix” that the arguments are two-dimensional entities.

### General Purpose Equivalents

MATLAB | NumPy | Notes
---|---|---
``help func`` | ``info(func)`` or ``help(func)`` or ``func?`` (in Ipython) | get help on the function func
``which func`` | see note HELP | find out where func is defined 
``type func`` | ``source(func)`` or ``func??`` (in Ipython) | print source for func (if not a native function)
``a && b`` | ``a and b`` | short-circuiting logical AND operator (Python native operator); scalar arguments only
``a \|\| b`` | ``a or b`` | short-circuiting logical OR operator (Python native operator); scalar arguments only
``1*i, 1*j, 1i, 1j`` | ``1j`` | complex numbers
``eps`` | ``np.spacing(1)`` | Distance between 1 and the nearest floating point number.
``ode45`` | ``scipy.integrate.solve_ivp(f)`` | integrate an ODE with Runge-Kutta 4,5
``ode15s`` | ``scipy.integrate.solve_ivp(f, method='BDF')`` | integrate an ODE with BDF method

### Linear Algebra Equivalents

MATLAB | NumPy | Notes
---|---|---
ndims(a) | ndim(a) or a.ndim | get the number of dimensions of an array
numel(a) | size(a) or a.size | get the number of elements of an array
size(a) | shape(a) or a.shape | get the “size” of the matrix
size(a,n) | a.shape[n-1] | get the number of elements of the n-th dimension of array a. (Note that MATLAB® uses 1 based indexing while Python uses 0 based indexing, See note INDEXING)
[ 1 2 3; 4 5 6 ] | array([[1.,2.,3.], [4.,5.,6.]]) | 2x3 matrix literal
[ a b; c d ] | vstack([hstack([a,b]), hstack([c,d])]) or bmat('a b; c d').A | construct a matrix from blocks a, b, c, and d
a(end) | a[-1] | access last element in the 1xn matrix a
a(2,5) | a[1,4] | access element in second row, fifth column
a(2,:) | a[1] or a[1,:] | entire second row of a
a(1:5,:) | a[0:5] or a[:5] or a[0:5,:] | the first five rows of a
a(end-4:end,:) | a[-5:] | the last five rows of a
a(1:3,5:9) | a[0:3][:,4:9] | rows one to three and columns five to nine of a. This gives read-only access.
a([2,4,5],[1,3]) | a[ix_([1,3,4],[0,2])] | rows 2,4 and 5 and columns 1 and 3. This allows the matrix to be modified, and doesn’t require a regular slice.
a(3:2:21,:) | a[ 2:21:2,:] | every other row of a, starting with the third and going to the twenty-first
a(1:2:end,:) | a[ ::2,:] | every other row of a, starting with the first
a(end:-1:1,:) or flipud(a) | a[ ::-1,:] | a with rows in reverse order
a([1:end 1],:) | a[r_[:len(a),0]] | a with copy of the first row appended to the end
a.' | a.transpose() or a.T | transpose of a
a' | a.conj().transpose() or a.conj().T | conjugate transpose of a
a * b | a.dot(b) | matrix multiply
a .* b | a * b | element-wise multiply
a./b | a/b | element-wise divide
a.^3 | a**3 | element-wise exponentiation
(a>0.5) | (a>0.5) | matrix whose i,jth element is (a_ij > 0.5). The Matlab result is an array of 0s and 1s. The NumPy result is an array of the boolean values False and True.
find(a>0.5) | nonzero(a>0.5) | find the indices where (a > 0.5)
a(:,find(v>0.5)) | a[:,nonzero(v>0.5)[0]] | extract the columms of a where vector v > 0.5
a(:,find(v>0.5)) | a[:,v.T>0.5] | extract the columms of a where column vector v > 0.5
a(a<0.5)=0 | a[a<0.5]=0 | a with elements less than 0.5 zeroed out
a .* (a>0.5) | a * (a>0.5) | a with elements less than 0.5 zeroed out
a(:) = 3 | a[:] = 3 | set all values to the same scalar value
y=x | y = x.copy() | numpy assigns by reference
y=x(2,:) | y = x[1,:].copy() | numpy slices are by reference
y=x(:) | y = x.flatten() | turn array into vector (note that this forces a copy)
1:10 | arange(1.,11.) or r_[1.:11.] or r_[1:10:10j] | create an increasing vector (see note RANGES)
0:9 | arange(10.) or r_[:10.] or r_[:9:10j] | create an increasing vector (see note RANGES)
[1:10]' | arange(1.,11.)[:, newaxis] | create a column vector
zeros(3,4) | zeros((3,4)) | 3x4 two-dimensional array full of 64-bit floating point zeros
zeros(3,4,5) | zeros((3,4,5)) | 3x4x5 three-dimensional array full of 64-bit floating point zeros
ones(3,4) | ones((3,4)) | 3x4 two-dimensional array full of 64-bit floating point ones
eye(3) | eye(3) | 3x3 identity matrix
diag(a) | diag(a) | vector of diagonal elements of a
diag(a,0) | diag(a,0) | square diagonal matrix whose nonzero values are the elements of a
rand(3,4) | random.rand(3,4) | random 3x4 matrix
linspace(1,3,4) | linspace(1,3,4) | 4 equally spaced samples between 1 and 3, inclusive
[x,y]=meshgrid(0:8,0:5) | mgrid[0:9.,0:6.] or meshgrid(r_[0:9.],r_[0:6.] | two 2D arrays: one of x values, the other of y values
  | ogrid[0:9.,0:6.] or ix_(r_[0:9.],r_[0:6.] | the best way to eval functions on a grid
[x,y]=meshgrid([1,2,4],[2,4,5]) | meshgrid([1,2,4],[2,4,5]) |  
  | ix_([1,2,4],[2,4,5]) | the best way to eval functions on a grid
repmat(a, m, n) | tile(a, (m, n)) | create m by n copies of a
[a b] | concatenate((a,b),1) or hstack((a,b)) or column_stack((a,b)) or c_[a,b] | concatenate columns of a and b
[a; b] | concatenate((a,b)) or vstack((a,b)) or r_[a,b] | concatenate rows of a and b
max(max(a)) | a.max() | maximum element of a (with ndims(a)<=2 for matlab)
max(a) | a.max(0) | maximum element of each column of matrix a
max(a,[],2) | a.max(1) | maximum element of each row of matrix a
max(a,b) | maximum(a, b) | compares a and b element-wise, and returns the maximum value from each pair
norm(v) | sqrt(dot(v,v)) or np.linalg.norm(v) | L2 norm of vector v
a & b | logical_and(a,b) | element-by-element AND operator (NumPy ufunc) See note LOGICOPS
a | b | logical_or(a,b) | element-by-element OR operator (NumPy ufunc) See note LOGICOPS
bitand(a,b) | a & b | bitwise AND operator (Python native and NumPy ufunc)
bitor(a,b) | a | b | bitwise OR operator (Python native and NumPy ufunc)
inv(a) | linalg.inv(a) | inverse of square matrix a
pinv(a) | linalg.pinv(a) | pseudo-inverse of matrix a
rank(a) | linalg.matrix_rank(a) | matrix rank of a 2D array / matrix a
a\b | linalg.solve(a,b) if a is square; linalg.lstsq(a,b) otherwise | solution of a x = b for x
b/a | Solve a.T x.T = b.T instead | solution of x a = b for x
[U,S,V]=svd(a) | U, S, Vh = linalg.svd(a), V = Vh.T | singular value decomposition of a
chol(a) | linalg.cholesky(a).T | cholesky factorization of a matrix (chol(a) in matlab returns an upper triangular matrix, but linalg.cholesky(a) returns a lower triangular matrix)
[V,D]=eig(a) | D,V = linalg.eig(a) | eigenvalues and eigenvectors of a
[V,D]=eig(a,b) | V,D = np.linalg.eig(a,b) | eigenvalues and eigenvectors of a, b
[V,D]=eigs(a,k) |   | find the k largest eigenvalues and eigenvectors of a
[Q,R,P]=qr(a,0) | Q,R = scipy.linalg.qr(a) | QR decomposition
[L,U,P]=lu(a) | L,U = scipy.linalg.lu(a) or LU,P=scipy.linalg.lu_factor(a) | LU decomposition (note: P(Matlab) == transpose(P(numpy)) )
conjgrad | scipy.sparse.linalg.cg | Conjugate gradients solver
fft(a) | fft(a) | Fourier transform of a
ifft(a) | ifft(a) | inverse Fourier transform of a
sort(a) | sort(a) or a.sort() | sort the matrix
[b,I] = sortrows(a,i) | I = argsort(a[:,i]), b=a[I,:] | sort the rows of the matrix
regress(y,X) | linalg.lstsq(X,y) | multilinear regression
decimate(x, q) | scipy.signal.resample(x, len(x)/q) | downsample with low-pass filtering
``unique(a)`` | ``unique(a)`` |  
``squeeze(a)`` | ``a.squeeze()`` |  

## Notes

**Submatrix**: Assignment to a submatrix can be done with lists of indexes using the ix_ command. E.g., for 2d array a, one might do: ``ind=[1,3]; a[np.ix_(ind,ind)]+=100``.

**HELP**: There is no direct equivalent of MATLAB’s ``which`` command, but the commands ``help`` and ``source`` will usually list the filename where the function is located. Python also has an inspect module (do ``import inspect``) which provides a ``getfile`` that often works.

**INDEXING**: MATLAB® uses one based indexing, so the initial element of a sequence has index 1. Python uses zero based indexing, so the initial element of a sequence has index 0. Confusion and flamewars arise because each has advantages and disadvantages. One based indexing is consistent with common human language usage, where the “first” element of a sequence has index 1. Zero based indexing simplifies indexing. See also a text by prof.dr. Edsger W. Dijkstra.

**RANGES**: In MATLAB®, 0:5 can be used as both a range literal and a ‘slice’ index (inside parentheses); however, in Python, constructs like 0:5 can only be used as a slice index (inside square brackets). Thus the somewhat quirky r_ object was created to allow numpy to have a similarly terse range construction mechanism. Note that r_ is not called like a function or a constructor, but rather indexed using square brackets, which allows the use of Python’s slice syntax in the arguments.

**LOGICOPS**: & or | in NumPy is bitwise AND/OR, while in Matlab & and | are logical AND/OR. The difference should be clear to anyone with significant programming experience. The two can appear to work the same, but there are important differences. If you would have used Matlab’s & or | operators, you should use the NumPy ufuncs logical_and/logical_or. The notable differences between Matlab’s and NumPy’s & and | operators are:

Non-logical {0,1} inputs: NumPy’s output is the bitwise AND of the inputs. Matlab treats any non-zero value as 1 and returns the logical AND. For example (3 & 4) in NumPy is 0, while in Matlab both 3 and 4 are considered logical true and (3 & 4) returns 1.
Precedence: NumPy’s & operator is higher precedence than logical operators like < and >; Matlab’s is the reverse.
If you know you have boolean arguments, you can get away with using NumPy’s bitwise operators, but be careful with parentheses, like this: z = (x > 1) & (x < 2). The absence of NumPy operator forms of logical_and and logical_or is an unfortunate consequence of Python’s design.

**RESHAPE and LINEAR INDEXING**: Matlab always allows multi-dimensional arrays to be accessed using scalar or linear indices, NumPy does not. Linear indices are common in Matlab programs, e.g. find() on a matrix returns them, whereas NumPy’s find behaves differently. When converting Matlab code it might be necessary to first reshape a matrix to a linear sequence, perform some indexing operations and then reshape back. As reshape (usually) produces views onto the same storage, it should be possible to do this fairly efficiently. Note that the scan order used by reshape in NumPy defaults to the ‘C’ order, whereas Matlab uses the Fortran order. If you are simply converting to a linear sequence and back this doesn’t matter. But if you are converting reshapes from Matlab code which relies on the scan order, then this Matlab code: z = reshape(x,3,4); should become z = x.reshape(3,4,order=’F’).copy() in NumPy.

## Customizing Your Environment

In MATLAB® the main tool available to you for customizing the environment is to modify the search path with the locations of your favorite functions. You can put such customizations into a startup script that MATLAB will run on startup.

NumPy, or rather Python, has similar facilities.

- To modify your Python search path to include the locations of your own modules, define the ``PYTHONPATH`` environment variable.
- To have a particular script file executed when the interactive Python interpreter is started, define the ``PYTHONSTARTUP`` environment variable to contain the name of your startup script.

Unlike MATLAB®, where anything on your path can be called immediately, with Python you need to first do an ‘import’ statement to make functions in a particular file accessible.

For example you might make a startup script that looks like this (Note: this is just an example, not a statement of “best practices”):

```python
# Make all numpy available via shorter 'num' prefix
import numpy as num
# Make all matlib functions accessible at the top level via M.func()
import numpy.matlib as M
# Make some matlib functions accessible directly at the top level via, e.g. rand(3,3)
from numpy.matlib import rand,zeros,ones,empty,eye
# Define a Hermitian function
def hermitian(A, **kwargs):
    return num.transpose(A,**kwargs).conj()
# Make some shortcuts for transpose,hermitian:
#    num.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
T = num.transpose
H = hermitian
```

## Links

See [http://mathesaurus.sf.net/](http://mathesaurus.sf.net/) for another MATLAB®/NumPy cross-reference.

An extensive list of tools for scientific work with python can be found in the topical [software page](http://scipy.org/topical-software.html).

MATLAB® and SimuLink® are registered trademarks of The MathWorks.