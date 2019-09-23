# Using F2PY

F2PY can be used either as a command line tool ``f2py`` or as a Python
module ``numpy.f2py``. While we try to install the command line tool as part
of the numpy setup, some platforms like Windows make it difficult to
reliably put the executable on the ``PATH``. We will refer to ``f2py``
in this document but you may have to run it as a module

```
python -m numpy.f2py
```

If you run ``f2py`` with no arguments, and the line ``numpy Version`` at the
end matches the NumPy version printed from ``python -m numpy.f2py``, then you
can use the shorter version. If not, or if you cannot run ``f2py``, you should
replace all calls to ``f2py`` here with the longer version.

## Command ``f2py``

When used as a command line tool, ``f2py`` has three major modes,
distinguished by the usage of ``-c`` and ``-h`` switches:
To scan Fortran sources and generate a signature file, use

1. To scan Fortran sources and generate a signature file, use

    ``` python
    f2py -h <filename.pyf> <options> <fortran files>   \
      [[ only: <fortran functions>  : ]                \
      [ skip: <fortran functions>  : ]]...            \
      [<fortran files> ...]
    ```

    Note that a Fortran source file can contain many routines, and not necessarily all routines are needed to be used from Python. So, you can either specify which routines should be wrapped (in ``only: .. :`` part) or which routines F2PY should ignored (in ``skip: .. :`` part).

    If ``<filename.pyf>`` is specified as ``stdout`` then signatures are send to standard output instead of a file.

    Among other options (see below), the following options can be used in this mode:

    ``--overwrite-signature``

    Overwrite existing signature file.
1. To construct an extension module, use

    ``` python
    f2py <options> <fortran files>          \
      [[ only: <fortran functions>  : ]     \
      [ skip: <fortran functions>  : ]]... \
      [<fortran files> ...]
    ```

    The constructed extension module is saved as ``<modulename>module.c`` to the current directory.

    Here ``<fortran files>`` may also contain signature files. Among other options (see below), the following options can be used in this mode:

    - ``--debug-capi``

        Add debugging hooks to the extension module. When using this extension module, various information about the wrapper is printed to standard output, for example, the values of variables, the steps taken, etc.

    - ``-include'<includefile>'``

        Add a CPP ``#include`` statement to the extension module source. ``<includefile>`` should be given in one of the following forms:

        ```
        "filename.ext"
        <filename.ext>
        ```

        The include statement is inserted just before the wrapper functions. This feature enables using arbitrary C functions (defined in ``<includefile>``) in F2PY generated wrappers.

        This option is deprecated. Use ``usercode`` statement to specify C code snippets directly in signature files

    - ``--[no-]wrap-functions``

        Create Fortran subroutine wrappers to Fortran functions. ``--wrap-functions`` is default because it ensures maximum portability and compiler independence.

    - ``--include-paths <path1>:<path2>:..``

        Search include files from given directories.

    - ``--help-link [<list of resources names>]``

        List system resources found by numpy_distutils/system_info.py. For example, try ``f2py --help-link lapack_opt``.
1. To build an extension module, use

    ``` python
    f2py -c <options> <fortran files>       \
      [[ only: <fortran functions>  : ]     \
      [ skip: <fortran functions>  : ]]... \
      [ <fortran/c source files> ] [ <.o, .a, .so files> ]
    ```

    If ``<fortran files>`` contains a signature file, then a source for an extension module is constructed, all Fortran and C sources are compiled, and finally all object and library files are linked to the extension module ``<modulename>``.so which is saved into the current directory.

    If ``<fortran files>`` does not contain a signature file, then an extension module is constructed by scanning all Fortran source codes for routine signatures.

    Among other options (see below) and options described in previous mode, the following options can be used in this mode:

    - ``--help-fcompiler``

        List available Fortran compilers.

    - ``--help-compiler [depreciated]``

        List available Fortran compilers.

    - ``--fcompiler=<Vendor>``

        Specify Fortran compiler type by vendor.

    - ``--f77exec=<path>``
        
        Specify the path to F77 compiler

    - ``--fcompiler-exec=<path> [depreciated]``

        Specify the path to F77 compiler

    - ``--f90exec=<path>``

        Specify the path to F90 compiler

    - ``--f90compiler-exec=<path> [depreciated]``

        Specify the path to F90 compiler

    - ``--f77flags=<string>``

        Specify F77 compiler flags

    -  ``--f90flags=<string>``

        Specify F90 compiler flags

    - ``--opt=<string>``

        Specify optimization flags

    - ``--arch=<string>``

        Specify architecture specific optimization flags

    - ``--noopt``

        Compile without optimization

    - ``--noarch``

        Compile without arch-dependent optimization

    - ``--debug``

        Compile with debugging information

    - ``-l<libname>``

        Use the library ``<libname>`` when linking.

    - ``-D<macro>[=<defn=1>]``

        Define macro ``<macro>`` as ``<defn>``.

    - ``-U<macro>``

        Define macro ``<macro>``

    - ``-I<dir>``

        Append directory ``<dir>`` to the list of directories searched for include files.

    - ``-L<dir>``
        
        Add directory ``<dir>`` to the list of directories to be searched for -l.

    - ``link-<resource>``

        Link extension module with ``<resource>`` as defined by numpy_distutils/system_info.py. E.g. to link with optimized LAPACK libraries (vecLib on MacOSX, ATLAS elsewhere), use --link-lapack_opt. See also --help-link switch.

    When building an extension module, a combination of the following macros may be required for non-gcc Fortran compilers:

    ```
    -DPREPEND_FORTRAN
    -DNO_APPEND_FORTRAN
    -DUPPERCASE_FORTRAN
    ```

    To test the performance of F2PY generated interfaces, use -DF2PY_REPORT_ATEXIT. Then a report of various timings is printed out at the exit of Python. This feature may not work on all platforms, currently only Linux platform is supported.

    To see whether F2PY generated interface performs copies of array arguments, use ``-DF2PY_REPORT_ON_ARRAY_COPY=<int>``. When the size of an array argument is larger than ``<int>``, a message about the coping is sent to stderr.

Other options:

- ``-m <modulename>``

    Name of an extension module. Default is ``untitled``. Don’t use this option if a signature file (*.pyf) is used.

- ``--[no-]lower``

    Do [not] lower the cases in ``<fortran files>``. By default, ``--lower`` is assumed with ``-h`` switch, and ``--no-lower`` without the -h switch.

- ``--build-dir <dirname>``

    All F2PY generated files are created in ``<dirname>``. Default is ``tempfile.mkdtemp()``.

- ``--quiet``

    Run quietly.

- ``--verbose``

    Run with extra verbosity.

- ``-v``

    Print f2py version ID and exit.

Execute ``f2py`` without any options to get an up-to-date list of available options.

## Python module ``numpy.f2py``

::: danger Warning

The current Python interface to the ``f2py`` module is not mature and
may change in the future.

:::

Fortran to Python Interface Generator.

- ``numpy.f2py.``run_main``( *comline_list* )[[source]](https://github.com/numpy/numpy/blob/master/numpy/f2py/f2py2e.py#L398-L461)

    Equivalent to running:

    ``` python
    f2py <args>
    ```

    where ``=string.join(,' ')``, but in Python.  Unless
    ``-h`` is used, this function returns a dictionary containing
    information on generated modules and their dependencies on source
    files.  For example, the command ``f2py -m scalar scalar.f`` can be
    executed from Python as follows

    You cannot build extension modules with this function, that is,
    using ``-c`` is not allowed. Use ``compile`` command instead

    **Examples:**

    ``` python
    >>> import numpy.f2py
    >>> r = numpy.f2py.run_main(['-m','scalar','doc/source/f2py/scalar.f'])
    Reading fortran codes...
            Reading file 'doc/source/f2py/scalar.f' (format:fix,strict)
    Post-processing...
            Block: scalar
                            Block: FOO
    Building modules...
            Building module "scalar"...
            Wrote C/API module "scalar" to file "./scalarmodule.c"
    >>> print(r)
    {'scalar': {'h': ['/home/users/pearu/src_cvs/f2py/src/fortranobject.h'],
            'csrc': ['./scalarmodule.c', 
                      '/home/users/pearu/src_cvs/f2py/src/fortranobject.c']}}
    ```

- ``numpy.f2py.``compile``( *source* ,  *modulename='untitled'* ,  *extra_args=''* ,  *verbose=True* ,  *source_fn=None* ,  *extension='.f'* )[[source]](https://github.com/numpy/numpy/blob/master/numpy/f2py/__init__.py#L23-L117)

    Build extension module from a Fortran 77 source string with f2py.

    **Parameters:**

    type | description
    ---|---
    source : str or bytes | Fortran source of module / subroutine to compile. *Changed in version 1.16.0:* Accept str as well as bytes
    modulename : str, optional | The name of the compiled python module
    extra_args : str or list, optional | Additional parameters passed to f2py. *Changed in version 1.16.0:* A list of args may also be provided.
    verbose : bool, optional | Print f2py output to screen
    source_fn : str, optional | Name of the file where the fortran source is written. The default is to use a temporary file with the extension provided by the extension parameter
    extension : {‘.f’, ‘.f90’}, optional | Filename extension if source_fn is not provided. The extension tells which fortran standard is used. The default is f, which implies F77 standard. *New in version 1.11.0.*

    **Returns:** 

    type | description
    ---|---
    result : int | 0 on success

    **Examples:**

    ``` python
    >>> import numpy.f2py
    >>> fsource = '''
    ...       subroutine foo
    ...       print*, "Hello world!"
    ...       end 
    ... '''
    >>> numpy.f2py.compile(fsource, modulename='hello', verbose=0)
    0
    >>> import hello
    >>> hello.foo()
    Hello world!
    ```
