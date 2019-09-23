# 编写您自己的ufunc 

> I have the Power!
> 
> *— He-Man*

## 创建一个新的ufunc

在阅读本文之前，通过阅读/略读[扩展和嵌入Python解释器的](https://docs.python.org/extending/index.html)第1部分中的教程以及[如何扩展NumPy](how-to-extend.html)，可以帮助您熟悉Python的C扩展基础知识。

umath模块是一个计算机生成的C模块，可以创建许多ufunc。它提供了许多如何创建通用函数的示例。使用ufunc机制创建自己的ufunc也不困难。假设您有一个函数，您想要在其输入上逐个元素地操作。通过创建一个新的ufunc，您将获得一个处理的函数

- 广播
- N维循环
- 自动类型转换，内存使用量最少
- 可选的输出数组

创建自己的ufunc并不困难。所需要的只是您想要支持的每种数据类型的1-d循环。每个1-d循环必须具有特定签名，并且只能使用固定大小数据类型的ufunc。下面给出了用于创建新的ufunc以处理内置数据类型的函数调用。使用不同的机制为用户定义的数据类型注册ufunc。

在接下来的几节中，我们提供了可以轻松修改的示例代码，以创建自己的ufunc。这些示例是logit函数的连续更完整或复杂版本，这是统计建模中的常见功能。Logit也很有趣，因为由于IEEE标准（特别是IEEE 754）的神奇之处，下面创建的所有logit函数都自动具有以下行为。

``` python
>>> logit(0)
-inf
>>> logit(1)
inf
>>> logit(2)
nan
>>> logit(-2)
nan
```

这很好，因为函数编写器不必手动传播infs或nans。

## 示例非ufunc扩展名

为了比较和阅读器的一般启发，我们提供了一个简单的logit C扩展实现，它没有使用numpy。

为此，我们需要两个文件。第一个是包含实际代码的C文件，第二个是用于创建模块的setup.py文件。

``` c
#include <Python.h>
#include <math.h>

/*
 * spammodule.c
 * This is the C code for a non-numpy Python extension to
 * define the logit function, where logit(p) = log(p/(1-p)).
 * This function will not work on numpy arrays automatically.
 * numpy.vectorize must be called in python to generate
 * a numpy-friendly function.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */


/* This declares the logit function */
static PyObject* spam_logit(PyObject *self, PyObject *args);


/*
 * This tells Python what methods this module has.
 * See the Python-C API for more information.
 */
static PyMethodDef SpamMethods[] = {
    {"logit",
        spam_logit,
        METH_VARARGS, "compute logit"},
    {NULL, NULL, 0, NULL}
};


/*
 * This actually defines the logit function for
 * input args from Python.
 */

static PyObject* spam_logit(PyObject *self, PyObject *args)
{
    double p;

    /* This parses the Python argument into a double */
    if(!PyArg_ParseTuple(args, "d", &p)) {
        return NULL;
    }

    /* THE ACTUAL LOGIT FUNCTION */
    p = p/(1-p);
    p = log(p);

    /*This builds the answer back into a python object */
    return Py_BuildValue("d", p);
}


/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spam",
    NULL,
    -1,
    SpamMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_spam(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initspam(void)
{
    PyObject *m;

    m = Py_InitModule("spam", SpamMethods);
    if (m == NULL) {
        return;
    }
}
#endif
```

要使用setup.py文件，请将setup.py和spammodule.c放在同一文件夹中。然后python setup.py build将构建要导入的模块，或者setup.py install将模块安装到您的site-packages目录。

``` python
'''
    setup.py file for spammodule.c

    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.

    Calling
    $python setup.py build
    will build a file that looks like ./build/lib*, where
    lib* is a file that begins with lib. The library will
    be in this file and end with a C library extension,
    such as .so

    Calling
    $python setup.py install
    will install the module in your site-packages file.

    See the distutils section of
    'Extending and Embedding the Python Interpreter'
    at docs.python.org for more information.
'''


from distutils.core import setup, Extension

module1 = Extension('spam', sources=['spammodule.c'],
                        include_dirs=['/usr/local/lib'])

setup(name = 'spam',
        version='1.0',
        description='This is my spam package',
        ext_modules = [module1])
```

将垃圾邮件模块导入python后，您可以通过spam.logit调用logit。请注意，上面使用的函数不能按原样应用于numpy数组。为此，我们必须在其上调用numpy.vectorize。例如，如果在包含垃圾邮件库或垃圾邮件的文件中打开了python解释器，则可以执行以下命令：

``` python
>>> import numpy as np
>>> import spam
>>> spam.logit(0)
-inf
>>> spam.logit(1)
inf
>>> spam.logit(0.5)
0.0
>>> x = np.linspace(0,1,10)
>>> spam.logit(x)
TypeError: only length-1 arrays can be converted to Python scalars
>>> f = np.vectorize(spam.logit)
>>> f(x)
array([       -inf, -2.07944154, -1.25276297, -0.69314718, -0.22314355,
    0.22314355,  0.69314718,  1.25276297,  2.07944154,         inf])
```

结果编辑功能并不快！numpy.vectorize只是循环遍历spam.logit。循环在C级完成，但numpy数组不断被解析并重新构建。这很贵。当作者将numpy.vectorize（spam.logit）与下面构造的logit ufuncs进行比较时，logit ufuncs几乎快4倍。当然，取决于功能的性质，可以实现更大或更小的加速。

## 一种dtype的NumPy ufunc示例

为简单起见，我们为单个dtype提供了一个ufunc，即'f8'双精度型。与前一节一样，我们首先给出.c文件，然后是用于创建包含ufunc的模块的setup.py文件。

代码中与ufunc的实际计算相对应的位置标有/ * BEGIN main ufunc computation * /和/ * END main ufunc computation * /。这些行之间的代码是必须更改以创建自己的ufunc的主要事物。

``` c
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/*
 * single_type_logit.c
 * This is the C code for creating your own
 * NumPy ufunc for a logit function.
 *
 * In this code we only define the ufunc for
 * a single dtype. The computations that must
 * be replaced to create a ufunc for
 * a different function are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static PyMethodDef LogitMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void double_logit(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(double *)in;
        tmp /= 1-tmp;
        *((double *)out) = log(tmp);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&double_logit};

/* These are the input and return dtypes of logit.*/
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};

static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    LogitMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *logit, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    Py_DECREF(logit);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *logit, *d;


    m = Py_InitModule("npufunc", LogitMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    Py_DECREF(logit);
}
#endif
```

这是上面代码的setup.py文件。和以前一样，可以通过在命令提示符下调用python setup.py build来构建模块，也可以通过python setup.py install将其安装到site-packages。

``` python
'''
    setup.py file for logit.c
    Note that since this is a numpy extension
    we use numpy.distutils instead of
    distutils from the python standard library.

    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.

    Calling
    $python setup.py build
    will build a file that looks like ./build/lib*, where
    lib* is a file that begins with lib. The library will
    be in this file and end with a C library extension,
    such as .so

    Calling
    $python setup.py install
    will install the module in your site-packages file.

    See the distutils section of
    'Extending and Embedding the Python Interpreter'
    at docs.python.org  and the documentation
    on numpy.distutils for more information.
'''


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('npufunc_directory',
                           parent_package,
                           top_path)
    config.add_extension('npufunc', ['single_type_logit.c'])

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
```

安装完上述内容后，可以按如下方式导入和使用。

``` python
>>> import numpy as np
>>> import npufunc
>>> npufunc.logit(0.5)
0.0
>>> a = np.linspace(0,1,5)
>>> npufunc.logit(a)
array([       -inf, -1.09861229,  0.        ,  1.09861229,         inf])
```

## 示例具有多个dtypes的NumPy ufunc

我们最后给出了一个完整的ufunc示例，内部循环用于半浮点数，浮点数，双精度数和长双精度数。与前面的部分一样，我们首先给出.c文件，然后是相应的setup.py文件。

代码中与ufunc的实际计算相对应的位置标有/ * BEGIN main ufunc computation * /和/ * END main ufunc computation * /。这些行之间的代码是必须更改以创建自己的ufunc的主要事物。

``` c
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

/*
 * multi_type_logit.c
 * This is the C code for creating your own
 * NumPy ufunc for a logit function.
 *
 * Each function of the form type_logit defines the
 * logit function for a different numpy dtype. Each
 * of these functions must be modified when you
 * create your own ufunc. The computations that must
 * be replaced to create a ufunc for
 * a different function are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 */


static PyMethodDef LogitMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */

static void long_double_logit(char **args, npy_intp *dimensions,
                              npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out=args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    long double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(long double *)in;
        tmp /= 1-tmp;
        *((long double *)out) = logl(tmp);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void double_logit(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(double *)in;
        tmp /= 1-tmp;
        *((double *)out) = log(tmp);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void float_logit(char **args, npy_intp *dimensions,
                        npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in=args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    float tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(float *)in;
        tmp /= 1-tmp;
        *((float *)out) = logf(tmp);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}


static void half_float_logit(char **args, npy_intp *dimensions,
                             npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    float tmp;

    for (i = 0; i < n; i++) {

        /*BEGIN main ufunc computation*/
        tmp = *(npy_half *)in;
        tmp = npy_half_to_float(tmp);
        tmp /= 1-tmp;
        tmp = logf(tmp);
        *((npy_half *)out) = npy_float_to_half(tmp);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}


/*This gives pointers to the above functions*/
PyUFuncGenericFunction funcs[4] = {&half_float_logit,
                                   &float_logit,
                                   &double_logit,
                                   &long_double_logit};

static char types[8] = {NPY_HALF, NPY_HALF,
                NPY_FLOAT, NPY_FLOAT,
                NPY_DOUBLE,NPY_DOUBLE,
                NPY_LONGDOUBLE, NPY_LONGDOUBLE};
static void *data[4] = {NULL, NULL, NULL, NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    LogitMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *logit, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(funcs, data, types, 4, 1, 1,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    Py_DECREF(logit);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *logit, *d;


    m = Py_InitModule("npufunc", LogitMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(funcs, data, types, 4, 1, 1,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    Py_DECREF(logit);
}
#endif
```

这是上面代码的setup.py文件。和以前一样，可以通过在命令提示符下调用python setup.py build来构建模块，也可以通过python setup.py install将其安装到site-packages。

``` python
'''
    setup.py file for logit.c
    Note that since this is a numpy extension
    we use numpy.distutils instead of
    distutils from the python standard library.

    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.

    Calling
    $python setup.py build
    will build a file that looks like ./build/lib*, where
    lib* is a file that begins with lib. The library will
    be in this file and end with a C library extension,
    such as .so

    Calling
    $python setup.py install
    will install the module in your site-packages file.

    See the distutils section of
    'Extending and Embedding the Python Interpreter'
    at docs.python.org  and the documentation
    on numpy.distutils for more information.
'''


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.misc_util import get_info

    #Necessary for the half-float d-type.
    info = get_info('npymath')

    config = Configuration('npufunc_directory',
                            parent_package,
                            top_path)
    config.add_extension('npufunc',
                            ['multi_type_logit.c'],
                            extra_info=info)

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
```

安装完上述内容后，可以按如下方式导入和使用。

``` python
>>> import numpy as np
>>> import npufunc
>>> npufunc.logit(0.5)
0.0
>>> a = np.linspace(0,1,5)
>>> npufunc.logit(a)
array([       -inf, -1.09861229,  0.        ,  1.09861229,         inf])
```

## 示例具有多个参数/返回值的NumPy ufunc

我们的最后一个例子是一个带有多个参数的ufunc。它是对具有单个dtype的数据的logit ufunc的代码的修改。我们计算 (A\*B, logit(A\*B))。

我们只给出 C 代码，因为setup.py文件与[一种dtype的NumPy ufunc示例](#一种dtype的numpy-ufunc示例)中的setup.py文件完全相同，只是一行

``` python
config.add_extension('npufunc', ['single_type_logit.c'])
```

被替换为

``` python
config.add_extension('npufunc', ['multi_arg_logit.c'])
```

C文件如下。生成的ufunc接受两个参数A和B.它返回一个元组，其第一个元素是A * B，第二个元素是logit（A * B）。请注意，它会自动支持广播以及ufunc的所有其他属性。

``` c
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

/*
 * multi_arg_logit.c
 * This is the C code for creating your own
 * NumPy ufunc for a multiple argument, multiple
 * return value ufunc. The places where the
 * ufunc computation is carried out are marked
 * with comments.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 */


static PyMethodDef LogitMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void double_logitprod(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2], *out2 = args[3];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2], out2_step = steps[3];

    double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(double *)in1;
        tmp *= *(double *)in2;
        *((double *)out1) = tmp;
        *((double *)out2) = log(tmp/(1-tmp));
        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
        out2 += out2_step;
    }
}


/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&double_logitprod};

/* These are the input and return dtypes of logit.*/

static char types[4] = {NPY_DOUBLE, NPY_DOUBLE,
                        NPY_DOUBLE, NPY_DOUBLE};


static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    LogitMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *logit, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 2,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    Py_DECREF(logit);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *logit, *d;


    m = Py_InitModule("npufunc", LogitMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    logit = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 2,
                                    PyUFunc_None, "logit",
                                    "logit_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "logit", logit);
    Py_DECREF(logit);
}
#endif
```

## 示例带有结构化数组dtype参数的NumPy ufunc

此示例显示如何为结构化数组dtype创建ufunc。在这个例子中，我们展示了一个简单的ufunc，用于添加两个带有dtype'u8，u8，u8'的数组。该过程与其他示例略有不同，因为调用[``PyUFunc_FromFuncAndData``](https://numpy.org/devdocs/reference/c-api/ufunc.html#c.PyUFunc_FromFuncAndData)不会为自定义dtypes和结构化数组dtypes完全注册ufunc。我们还需要调用
 [``PyUFunc_RegisterLoopForDescr``](https://numpy.org/devdocs/reference/c-api/ufunc.html#c.PyUFunc_RegisterLoopForDescr)完成设置ufunc。

我们只提供C代码，因为setup.py文件与[一种dtype的NumPy ufunc示例](#一种dtype的numpy-ufunc示例)的setup.py文件完全相同，只有一行。

``` python
config.add_extension('npufunc', ['single_type_logit.c'])
```

被替换为

``` python
config.add_extension('npufunc', ['add_triplet.c'])
```

C文件如下。

``` c
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"


/*
 * add_triplet.c
 * This is the C code for creating your own
 * NumPy ufunc for a structured array dtype.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static PyMethodDef StructUfuncTestMethods[] = {
    {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void add_uint64_triplet(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp is1=steps[0];
    npy_intp is2=steps[1];
    npy_intp os=steps[2];
    npy_intp n=dimensions[0];
    uint64_t *x, *y, *z;

    char *i1=args[0];
    char *i2=args[1];
    char *op=args[2];

    for (i = 0; i < n; i++) {

        x = (uint64_t*)i1;
        y = (uint64_t*)i2;
        z = (uint64_t*)op;

        z[0] = x[0] + y[0];
        z[1] = x[1] + y[1];
        z[2] = x[2] + y[2];

        i1 += is1;
        i2 += is2;
        op += os;
    }
}

/* This a pointer to the above function */
PyUFuncGenericFunction funcs[1] = {&add_uint64_triplet};

/* These are the input and return dtypes of add_uint64_triplet. */
static char types[3] = {NPY_UINT64, NPY_UINT64, NPY_UINT64};

static void *data[1] = {NULL};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "struct_ufunc_test",
    NULL,
    -1,
    StructUfuncTestMethods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit_struct_ufunc_test(void)
#else
PyMODINIT_FUNC initstruct_ufunc_test(void)
#endif
{
    PyObject *m, *add_triplet, *d;
    PyObject *dtype_dict;
    PyArray_Descr *dtype;
    PyArray_Descr *dtypes[3];

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("struct_ufunc_test", StructUfuncTestMethods);
#endif

    if (m == NULL) {
#if defined(NPY_PY3K)
        return NULL;
#else
        return;
#endif
    }

    import_array();
    import_umath();

    /* Create a new ufunc object */
    add_triplet = PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0, 2, 1,
                                    PyUFunc_None, "add_triplet",
                                    "add_triplet_docstring", 0);

    dtype_dict = Py_BuildValue("[(s, s), (s, s), (s, s)]",
        "f0", "u8", "f1", "u8", "f2", "u8");
    PyArray_DescrConverter(dtype_dict, &dtype);
    Py_DECREF(dtype_dict);

    dtypes[0] = dtype;
    dtypes[1] = dtype;
    dtypes[2] = dtype;

    /* Register ufunc for structured dtype */
    PyUFunc_RegisterLoopForDescr(add_triplet,
                                dtype,
                                &add_uint64_triplet,
                                dtypes,
                                NULL);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "add_triplet", add_triplet);
    Py_DECREF(add_triplet);
#if defined(NPY_PY3K)
    return m;
#endif
}
```

返回的ufunc对象是一个可调用的Python对象。它应该放在一个（模块）字典中，其名称与ufunc-creation例程的name参数中使用的字典相同。以下示例是从umath模块改编而来的

``` c
static PyUFuncGenericFunction atan2_functions[] = {
                      PyUFunc_ff_f, PyUFunc_dd_d,
                      PyUFunc_gg_g, PyUFunc_OO_O_method};
static void* atan2_data[] = {
                      (void *)atan2f,(void *) atan2,
                      (void *)atan2l,(void *)"arctan2"};
static char atan2_signatures[] = {
              NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
              NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
              NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE
              NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};
...
/* in the module initialization code */
PyObject *f, *dict, *module;
...
dict = PyModule_GetDict(module);
...
f = PyUFunc_FromFuncAndData(atan2_functions,
    atan2_data, atan2_signatures, 4, 2, 1,
    PyUFunc_None, "arctan2",
    "a safe and correct arctan(x1/x2)", 0);
PyDict_SetItemString(dict, "arctan2", f);
Py_DECREF(f);
...
```
