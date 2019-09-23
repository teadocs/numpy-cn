---
meta:
  - name: keywords
    content: 使用 ``numpy.distutils`` 模块
  - name: description
    content: numpy.distutils 是NumPy扩展标准Python distutils的一部分，用于处理Fortran源代码和F2PY签名文件...
---

# 高级F2PY用法

## 将自编写函数添加到F2PY生成的模块

可以使用 ``usercode`` 和 ``pymethoddef`` 语句在签名文件中定义自编的Python C / API函数（它们必须在 ``python模块`` 块中使用）。 例如，以下签名文件``spam.pyf``。

``` python
!    -*- f90 -*-
python module spam
    usercode '''
  static char doc_spam_system[] = "Execute a shell command.";
  static PyObject *spam_system(PyObject *self, PyObject *args)
  {
    char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return Py_BuildValue("i", sts);
  }
    '''
    pymethoddef '''
    {"system",  spam_system, METH_VARARGS, doc_spam_system},
    '''
end python module spam
```

包装C库函数``system()``：

``` python
f2py -c spam.pyf
```

在Python中：

``` python
>>> import spam
>>> status = spam.system('whoami')
pearu
>> status = spam.system('blah')
sh: line 1: blah: command not found
```

## 修改F2PY生成模块的字典

以下示例说明如何将用户定义的变量添加到F2PY生成的扩展模块。给出以下签名文件：

``` python
!    -*- f90 -*-
python module var
  usercode '''
    int BAR = 5;
  '''
  interface
    usercode '''
      PyDict_SetItemString(d,"BAR",PyInt_FromLong(BAR));
    '''
  end interface
end python module
```

将其编译为：``f2py -c var.pyf``

请注意，必须在 ``interface（接口）`` 块内定义第二个 ``usercode`` 语句，并且通过变量 ``d`` 可以获得模块字典（有关其他详细信息，请参阅``f2py var.pyf`` 生成的 ``varmodule.c``）。

在Python中：

``` python
>>> import var
>>> var.BAR
5
```
