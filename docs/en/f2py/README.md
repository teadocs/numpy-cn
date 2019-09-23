# F2PY Users Guide and Reference Manual

The purpose of the ``F2PY`` – *Fortran to Python interface generator* –
is to provide a connection between Python and Fortran
languages.  F2PY is a part of [NumPy](https://www.numpy.org/) (``numpy.f2py``) and also available as a
standalone command line tool ``f2py`` when ``numpy`` is installed that
facilitates creating/building Python C/API extension modules that make it
possible

- to call Fortran 77/90/95 external subroutines and Fortran 90/95
module subroutines as well as C functions;
- to access Fortran 77 ``COMMON`` blocks and Fortran 90/95 module data,
including allocatable arrays

from Python.

- [Three ways to wrap - getting started](getting-started.html)
  - [The quick way](getting-started.html#the-quick-way)
  - [The smart way](getting-started.html#the-smart-way)
  - [The quick and smart way](getting-started.html#the-quick-and-smart-way)
- [Signature file](signature-file.html)
  - [Python module block](signature-file.html#python-module-block)
  - [Fortran/C routine signatures](signature-file.html#fortran-c-routine-signatures)
  - [Extensions](signature-file.html#extensions)
- [Using F2PY bindings in Python](python-usage.html)
  - [Scalar arguments](python-usage.html#scalar-arguments)
  - [String arguments](python-usage.html#string-arguments)
  - [Array arguments](python-usage.html#array-arguments)
  - [Call-back arguments](python-usage.html#call-back-arguments)
  - [Common blocks](python-usage.html#common-blocks)
  - [Fortran 90 module data](python-usage.html#fortran-90-module-data)
- [Using F2PY](usage.html)
  - [Command `f2py`](usage.html#command-f2py)
  - [Python module `numpy.f2py`](usage.html#python-module-numpy-f2py)
- [Using via `numpy.distutils`](distutils.html)
- [Advanced F2PY usages](advanced.html)
  - [Adding self-written functions to F2PY generated modules](advanced.html#adding-self-written-functions-to-f2py-generated-modules)
  - [Modifying the dictionary of a F2PY generated module](advanced.html#modifying-the-dictionary-of-a-f2py-generated-module)