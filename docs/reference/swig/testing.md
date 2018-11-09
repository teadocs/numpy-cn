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

is a [SWIG](http://www.swig.org/) interface file that defines a python module ``Vector`` that wraps the functions in ``Vector.h`` while utilizing the typemaps in ``numpy.i`` to correctly handle the C arrays.

The ``Makefile`` calls ``swig`` to generate ``Vector.py`` and ``Vector_wrap.cxx``, and also executes the ``setup.py`` script that compiles ``Vector_wrap.cxx`` and links together the extension module ``_Vector.so`` or ``_Vector.dylib``, depending on the platform. This extension module and the proxy file ``Vector.py`` are both placed in a subdirectory under the build directory.

The actual testing takes place with a Python script named:

```
testVector.py
```

that uses the standard Python library module ``unittest``, which performs several tests of each function defined in ``Vector.h`` for each data type supported.

Two-dimensional arrays are tested in exactly the same manner. The above description applies, but with ``Matrix`` substituted for ``Vector``. For three-dimensional tests, substitute ``Tensor`` for ``Vector``. For four-dimensional tests, substitute SuperTensor for Vector. For flat in-place array tests, substitute ``Flat`` for ``Vector``. For the descriptions that follow, we will reference the ``Vector`` tests, but the same information applies to ``Matrix``, ``Tensor`` and ``SuperTensor`` tests.

The command ``make test`` will ensure that all of the test software is built and then run all three test scripts.

## 测试头文件

``Vector.h`` is a C++ header file that defines a C macro called ``TEST_FUNC_PROTOS`` that takes two arguments: ``TYPE``, which is a data type name such as ``unsigned int``; and ``SNAME``, which is a short name for the same data type with no spaces, e.g. ``uint``. This macro defines several function prototypes that have the prefix ``SNAME`` and have at least one argument that is an array of type ``TYPE``. Those functions that have return arguments return a ``TYPE`` value.

``TEST_FUNC_PROTOS`` is then implemented for all of the data types supported by numpy.i:

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

``Vector.cxx`` is a C++ source file that implements compilable code for each of the function prototypes specified in ``Vector.h``. It defines a C macro ``TEST_FUNCS`` that has the same arguments and works in the same way as ``TEST_FUNC_PROTOS`` does in ``Vector.h.`` ``TEST_FUNCS`` is implemented for each of the 12 data types as above.

## 测试SWIG接口文件

``Vector.i`` is a [SWIG](http://www.swig.org/) interface file that defines python module ``Vector``. It follows the conventions for using ``numpy.i`` as described in this chapter. It defines a [SWIG](http://www.swig.org/) macro ``%apply_numpy_typemaps`` that has a single argument ``TYPE``. It uses the [SWIG](http://www.swig.org/) directive %apply to apply the provided typemaps to the argument signatures found in ``Vector.h``. This macro is then implemented for all of the data types supported by ``numpy.i``. It then does a ``%include "Vector.h"`` to wrap all of the function prototypes in ``Vector.h`` using the typemaps in ``numpy.i``.

## 测试Python脚本

After ``make`` is used to build the testing extension modules, ``testVector.py`` can be run to execute the tests. As with other scripts that use ``unittest`` to facilitate unit testing, ``testVector.py`` defines a class that inherits from ``unittest.TestCase``:

However, this class is not run directly. Rather, it serves as a base class to several other python classes, each one specific to a particular data type. The ``VectorTestCase`` class stores two strings for typing information:

- self.typeStr
    - A string that matches one of the ``SNAME`` prefixes used in ``Vector.h`` and ``Vector.cxx``. For example, ``"double"``.
- self.typeCode
    - A short (typically single-character) string that represents a data type in numpy and corresponds to ``self.typeStr``. For example, if ``self.typeStr`` is ``"double"``, then ``self.typeCode`` should be ``"d"``.

Each test defined by the ``VectorTestCase`` class extracts the python function it is trying to test by accessing the ``Vector`` module’s dictionary:

```python
length = Vector.__dict__[self.typeStr + "Length"]
```

In the case of double precision tests, this will return the python function ``Vector.doubleLength``.

We then define a new test case class for each supported data type with a short definition such as:

```python
class doubleTestCase(VectorTestCase):
    def __init__(self, methodName="runTest"):
        VectorTestCase.__init__(self, methodName)
        self.typeStr  = "double"
        self.typeCode = "d"
```

Each of these 12 classes is collected into a ``unittest.TestSuite``, which is then executed. Errors and failures are summed together and returned as the exit argument. Any non-zero result indicates that at least one test did not pass.
