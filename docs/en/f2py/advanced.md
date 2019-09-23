# Advanced F2PY usages

## Adding self-written functions to F2PY generated modules

Self-written Python C/API functions can be defined inside
signature files using ``usercode`` and ``pymethoddef`` statements
(they must be used inside the ``python module`` block). For
example, the following signature file ``spam.pyf``

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

wraps the C library function ``system()``:

``` python
f2py -c spam.pyf
```

In Python:

``` python
>>> import spam
>>> status = spam.system('whoami')
pearu
>> status = spam.system('blah')
sh: line 1: blah: command not found
```

## Modifying the dictionary of a F2PY generated module

The following example illustrates how to add a user-defined
variables to a F2PY generated extension module. Given the following
signature file

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

compile it as ``f2py -c var.pyf``.

Notice that the second ``usercode`` statement must be defined inside
an ``interface`` block and where the module dictionary is available through
the variable ``d`` (see ``f2py var.pyf``-generated ``varmodule.c`` for
additional details).

In Python:

``` python
>>> import var
>>> var.BAR
5
```