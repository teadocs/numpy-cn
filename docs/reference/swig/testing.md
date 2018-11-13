<title>numpy.i的Typemaps - <%-__DOC_NAME__ %></title>
<meta name="keywords" content="numpy测试numpy.i" />

# 测试 numpy.i 的 Typemaps

## 介绍

为numpy.i 的 SWIG接口文件编写测试是一个复杂的难题。目前，支持12种不同的数据类型，每种数据类型都有74个不同的参数签名，因此总共有888个类型图支持“开箱即用”。反过来，这些类型图中的每一个都可能需要几个单元测试，以便验证正确输入和不正确输入的预期行为。目前，当maketest在numpy/tools/swg子目录中运行时，这会导致执行1,000多个单独的单元测试。

为了方便许多类似的单元测试，使用了一些高级编程技术，包括C和SWIG宏，以及Python继承。本文档的目的是描述用于验证numpy.i类型图是否按预期工作的测试基础结构。

## 检测机构

支持三种独立的测试框架，分别用于一维，二维和三维阵列。对于一维数组，有两个C++文件，一个头和一个源，名为：

```
Vector.h
Vector.cxx
```

包含具有一维数组作为函数参数的各种函数的原型和代码。文件：

```
Vector.i
```

是一个SWIG接口文件，它定义了一个python模块 ``Vector``，它包含了 ``Vector.h`` 中的函数，同时利用 ``numpy.i`` 中的类型映射来正确处理C数组。

Makefile调用 ``swig`` 生成 ``Vector.py`` 和 ``Vector_wrap.cxx``，并执行编译 ``Vector_wrap.cxx`` 的 ``setup.py`` 脚本，并将扩展模块 ``_Vector.so`` 或 ``_Vector.dylib`` 链接在一起，具体取决于平台。 此扩展模块和代理文件 ``Vector.py`` 都放在构建目录下的子目录中。

实际测试使用名为的Python脚本进行：

```
testVector.py
```

它使用标准的Python库模块 ``unittest``，它对所支持的每种数据类型执行``Vector.h``中定义的每个函数的几个测试。

以完全相同的方式测试二维阵列。以上描述适用，但用``Matrix``代替``Vector``。对于三维测试，将``Tensor``替换为``Vector``。对于四维测试，将SuperTensor替换为Vector。对于平面就地阵列测试，将``Flat``替换为``Vector``。 对于下面的描述，我们将引用``Vector``测试，但相同的信息适用于``Matrix``，``Tensor``和``SuperTensor``测试。

命令``make test``将确保构建所有测试软件，然后运行所有三个测试脚本。

## 测试头文件

``Vector.h`` 是一个C ++头文件，它定义了一个名为 ``TEST_FUNC_PROTOS`` 的C宏，它带有两个参数：``TYPE``，它是一个数据类型名，例如``unsigned int``; 和``SNAME``，它是相同数据类型的短名称，没有空格，例如``uint``。 这个宏定义了几个函数原型，这些函数原型具有前缀“SNAME”并且至少有一个参数是类型为``TYPE``的数组。那些具有返回参数的函数返回一个``TYPE``值。

然后为numpy.i支持的所有数据类型实现``TEST_FUNC_PROTOS``：

- signed char
- unsigned char
- short
- unsigned short
- int
- unsigned int
- long
- unsigned long
- long long
- unsigned long long
- float
- double

## 测试源码文件

``Vector.cxx``是一个C ++源文件，它为``Vector.h``中指定的每个函数原型实现了可编译的代码。 它定义了一个C宏``TEST_FUNCS``，它具有相同的参数，其工作方式与``TEST_FUNC_PROTOS``在``Vector.h``中的工作方式相同.``TEST_FUNCS``为12个数据中的每一个实现 类型如上。

## 测试SWIG接口文件

``Vector.i``是一个 SWIG 接口文件，它定义了python模块 ``Vector``。 它遵循本章所述的使用 ``numpy.i`` 的约定。它定义了一个 SWIG 宏``％apply_numpy_typemaps``，它有一个参数``TYPE``。 它使用 SWIG 指令 %apply 将提供的类型映射应用于 ``Vector.h`` 中的参数签名。然后为 ``numpy.i`` 支持的所有数据类型实现该宏。然后它使用 ``numpy.i`` 中的类型映射将 ``%include'Vector.h`` 包装在 ``Vector.h`` 中的所有函数原型。

## 测试Python脚本

在``make``用于构建测试扩展模块之后，可以运行``testVector.py``来执行测试。与使用``unittest``来促进单元测试的其他脚本一样，``testVector.py``定义了一个继承自``unittest.TestCase``的类：

但是，此类不直接运行。 相反，它作为几个其他python类的基类，每个类都特定于特定的数据类型。``VectorTestCase``类存储两个用于输入信息的字符串：

- self.typeStr
    - 一个字符串，匹配``Vector.h``和``Vector.cxx``中使用的``SNAME``前缀之一。 例如，``double``。
- self.typeCode
    - 一个短（通常是单字符）字符串，表示numpy中的数据类型，对应于``self.typeStr``。 例如，如果``self.typeStr``是``double``，那么``self.typeCode``应该是``d``。

由 ``VectorTestCase`` 类定义的每个测试通过访问 ``Vector`` 模块的字典来提取它试图测试的python函数：

```python
length = Vector.__dict__[self.typeStr + "Length"]
```

在双精度测试的情况下，这将返回python函数``Vector.doubleLength``。

然后，我们为每个支持的数据类型定义一个新的测试用例类，其中包含一个简短的定义：

```python
class doubleTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"
```

这12个类中的每一个都被收集到一个 ``unittest.TestSuite`` 中，然后执行。 将错误和失败相加在一起并作为退出参数返回。 任何非零结果表明至少有一个测试没有通过。
