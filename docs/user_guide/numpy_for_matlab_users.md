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

## MATLAB和NumPy粗略的功能对应表

下表粗略的反应了一些常见MATLAB表达式的大致对应关系。**但这些并不完全等同**，而是作为一种指引，指引读者一个正确的反向。有关更多详细信息，请参阅NumPy函数的内置文档。

在编写以数组或矩阵作为参数的函数时，需要注意一点，就算如果您期望函数返回一个 ``array``， 传入的参数却是一个 ``matrix``，或反之亦然，那么 ``*`` (乘法运算符) 会给您带来意想不到的惊喜。它可以在数组和矩阵之间来回转换。

- ``asarray``: 总是返回一个 ``array`` 类型的对象。
- ``asmatrix`` 或 ``mat``: 总是返回一个  ``matrix`` 类型的对象。
- ``asanyarray``: 始终返回数组对象或数组对象派生的子类，具体取决于传入的类型。例如，如果传入一个``matrix``，它将返回一个``matrix``。

这些函数都接受数组和矩阵(除了别的类型哈，比如Python的list类型之类的)，因此在编写应该接受任何类似数组的对象的函数时很有用。

在下表中，假设你已经在Python中执行了以下命令：

```python
from numpy import *
import scipy.linalg
```

另外如果下表中的``注释``这一列的内容是和``matrix``有关的话，那么参数一定是二维的形式。

### 一般功能的对应表

MATLAB | NumPy | 注释
---|---|---
``help func`` | ``info(func)`` 或 ``help(func)`` 或 ``func?`` (在 Ipython 中) | 获得函数func的帮助。
``which func`` | see note HELP（译者注：在里的原链接已经失效。） | 找出func定义的位置。
``type func`` | ``source(func)`` 或 ``func??`` (在 Ipython 中) | 打印func的源代码(如果不是原生函数的话)。
``a && b`` | ``a and b`` | 短路逻辑 AND 运算符 (Python 原生运算符); 仅限标量参数。
``a \|\| b`` | ``a or b`` | 短路逻辑 OR 运算符 (Python 原生运算符); 仅限标量参数。
``1*i, 1*j, 1i, 1j`` | ``1j`` | 复数。
``eps`` | ``np.spacing(1)`` | 数字1和最近的浮点数之间的距离。
``ode45`` | ``scipy.integrate.solve_ivp(f)`` | 将ODE与Runge-Kutta 4,5整合在一起。
``ode15s`` | ``scipy.integrate.solve_ivp(f, method='BDF')`` | 用BDF方法整合ODE。

### 线性代数功能对应表

MATLAB | NumPy | 注释
---|---|---
ndims(a) | ndim(a) 或 a.ndim | 获取数组的维数。
numel(a) | size(a) 或 a.size | 获取数组的元素个数。
size(a) | shape(a) 或 a.shape | 求矩阵的“大小”
size(a,n) | a.shape[n-1] | 获取数组a的n维元素数量。(请注意，MATLAB使用基于1的索引，而Python使用基于0的索引，请参见注释索引)。
[ 1 2 3; 4 5 6 ] | array([[1.,2.,3.], [4.,5.,6.]]) | 一个 2x3 矩阵的字面量。
[ a b; c d ] | vstack([hstack([a,b]), hstack([c,d])]) 或 bmat('a b; c d').A | 从快a、b、c和d构造矩阵。
a(end) | a[-1] | 访问1xn矩阵a中的最后一个元素。
a(2,5) | a[1,4] | 访问第二行，第五列中的元素。
a(2,:) | a[1] 或 a[1,:] | 取得a数组第二个元素全部（译者注：第二个元素如果是数组，则返回这个数组）
a(1:5,:) | a[0:5] 或 a[:5] 或 a[0:5,:] | 取得a数组的前五行。
a(end-4:end,:) | a[-5:] | 取得a数组的后五行。
a(1:3,5:9) | a[0:3][:,4:9] | a数组的第1行到第3行和第5到第9列，这种方式只允许读取。
a([2,4,5],[1,3]) | a[ix_([1,3,4],[0,2])] | 行2,4和5以及第1列和第3列。这允许修改矩阵，并且不需要常规切片方式。
a(3:2:21,:) | a[ 2:21:2,:] | a数组每隔一行，从第三行开始，一直到第二十一行。
a(1:2:end,:) | a[ ::2,:] | a数组从第一行开始，每隔一行。
a(end:-1:1,:) 或 flipud(a) | a[ ::-1,:] | 反转a数组的顺序。
a([1:end 1],:) | a[r_[:len(a),0]] | 将a数组的第一行的副本添加到数组末尾。
a.' | a.transpose() 或 a.T | a数组的转置。
a' | a.conj().transpose() 或 a.conj().T | a数组的共轭转置。
a * b | a.dot(b) | 矩阵乘法。
a .* b | a * b | 元素相乘。
a./b | a/b | 元素相除。
a.^3 | a**3 | 元素指数运算。
(a>0.5) | (a>0.5) | 其i, th元素为(a_ij > 0.5)的矩阵。Matlab的结果是一个0和1的数组。NumPy结果是一个布尔值false和True的数组。
find(a>0.5) | nonzero(a>0.5) | 找到条件满足 (a > 0.5) 的索引。
a(:,find(v>0.5)) | a[:,nonzero(v>0.5)[0]] | 找到满足条件 向量v > 0.5 的列。
a(:,find(v>0.5)) | a[:,v.T>0.5] |  找到满足条件 列向量v > 0.5 的列。
a(a<0.5)=0 | a[a<0.5]=0 | 元素小于0.5 赋为 0。
a .* (a>0.5) | a * (a>0.5) | （译者注：应该是a乘上a中大于0.5的值的矩阵）
a(:) = 3 | a[:] = 3 | 将所有值设置为相同的标量值
y=x | y = x.copy() | numpy 通过拷贝引用来赋值。
y=x(2,:) | y = x[1,:].copy() | numpy 通过拷贝引用来切片操作。
y=x(:) | y = x.flatten() | 将数组转换为向量(请注意，这将强制拷贝)。
1:10 | arange(1.,11.) 或 r_[1.:11.] 或 r_[1:10:10j] | 创建一个可增长的向量 (参见下面的[注释](#note)章节)
0:9 | arange(10.) 或 r_[:10.] 或 r_[:9:10j] | 创建一个可增长的向量 (参见下面的[注释](#note)章节)
[1:10]' | arange(1.,11.)[:, newaxis] | 创建一个列向量。
zeros(3,4) | zeros((3,4)) | 创建一个全是0的填充的 3x4 的64位浮点类型的二维数组。
zeros(3,4,5) | zeros((3,4,5)) | 创建一个全是0的填充的 3x4x5 的64位浮点类型的三维数组。
ones(3,4) | ones((3,4)) | 创建一个全是 1 的填充的 3x4 的64位浮点类型的二维数组。
eye(3) | eye(3) | 创建一个3x3恒等矩阵。
diag(a) | diag(a) | 创建a数组的对角元素向量。
diag(a,0) | diag(a,0) | 创建方形对角矩阵，其非零值是a的所有元素。
rand(3,4) | random.rand(3,4) | 创建一个随机的 3x4 矩阵
linspace(1,3,4) | linspace(1,3,4) | 创建4个等间距的样本，介于1和3之间。
[x,y]=meshgrid(0:8,0:5) | mgrid[0:9.,0:6.] 或 meshgrid(r_[0:9.],r_[0:6.] | 两个2维数组：一个是x值，另一个是y值。
 - | ogrid[0:9.,0:6.] 或 ix_(r_[0:9.],r_[0:6.] | 最好的方法是在一个网格上执行函数。
[x,y]=meshgrid([1,2,4],[2,4,5]) | meshgrid([1,2,4],[2,4,5]) |  - 
 - | ix_([1,2,4],[2,4,5]) | 最好的方法是在网格上执行函数。
repmat(a, m, n) | tile(a, (m, n)) | 通过n份a的拷贝创建m。
[a b] | concatenate((a,b),1) 或 hstack((a,b)) 或 column_stack((a,b)) or c_[a,b] | 连接a和b的列。
[a; b] | concatenate((a,b)) 或 vstack((a,b)) 或 r_[a,b] | 连接a和b的行。
max(max(a)) | a.max() | 取a数组的中的最大元素（对于matlab来说，ndims(a) <= 2）
max(a) | a.max(0) | 求各列的最大值。
max(a,[],2) | a.max(1) | 求各行最大值。
max(a,b) | maximum(a, b) | 比较a和b元素，并返回每对中的最大值。
norm(v) | sqrt(dot(v,v)) 或 np.linalg.norm(v) | 向量v的L2范数
a & b | logical_and(a,b) | 逐元素使用 AND 运算符 (NumPy ufunc) (参见下面的[注释](#note)章节)
a | b | logical_or(a,b) | 逐元素使用 OR 运算符 (NumPy ufunc) (参见下面的[注释](#note)章节)
bitand(a,b) | a & b | 按位AND运算符  (Python原生 和 NumPy ufunc)
bitor(a,b) | a | b | 按位OR运算符 (Python原生 和 NumPy ufunc)
inv(a) | linalg.inv(a) | 矩阵a的逆运算
pinv(a) | linalg.pinv(a) | 矩阵a的反逆运算
rank(a) | linalg.matrix_rank(a) | 二维数组或者矩阵的矩阵rank。
a\b | 如果a是方形矩阵 linalg.solve(a,b) ，否则：linalg.lstsq(a,b)  | 对于x，x = b的解
b/a | Solve a.T x.T = b.T instead | 对于x，x a = b的解
[U,S,V]=svd(a) | U, S, Vh = linalg.svd(a), V = Vh.T | a数组的奇值分解
chol(a) | linalg.cholesky(a).T | 矩阵的cholesky分解（matlab中的``chol(a)``返回一个上三角矩阵，但``linalg.cholesky(a)``返回一个下三角矩阵）
[V,D]=eig(a) | D,V = linalg.eig(a) | a数组的特征值和特征向量
[V,D]=eig(a,b) | V,D = np.linalg.eig(a,b) | a，b数组的特征值和特征向量
[V,D]=eigs(a,k) | - | 找到a的k个最大特征值和特征向量
[Q,R,P]=qr(a,0) | Q,R = scipy.linalg.qr(a) | Q，R 的分解
[L,U,P]=lu(a) | L,U = scipy.linalg.lu(a) or LU,P=scipy.linalg.lu_factor(a) | LU 分解 (注意: P(Matlab) == transpose(P(numpy)) )
conjgrad | scipy.sparse.linalg.cg | 共轭渐变求解器
fft(a) | fft(a) | a数组的傅立叶变换
ifft(a) | ifft(a) | a的逆逆傅里叶变换
sort(a) | sort(a) or a.sort() | 对矩阵或者数组进行排序
[b,I] = sortrows(a,i) | I = argsort(a[:,i]), b=a[I,:] | 对矩阵或数组的行进行排序
regress(y,X) | linalg.lstsq(X,y) | 多线性回归
decimate(x, q) | scipy.signal.resample(x, len(x)/q) | 采用低通滤波的下采样
``unique(a)`` | ``unique(a)`` |  -
``squeeze(a)`` | ``a.squeeze()`` | -  

## Notes

**子矩阵**: 可以使用ix_命令使用索引列表完成对子矩阵的分配。例如，对于2d阵列a，有人可能会这样做： ``ind=[1,3]; a[np.ix_(ind,ind)]+=100``.

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