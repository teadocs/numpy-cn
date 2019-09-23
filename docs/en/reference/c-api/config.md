# System configuration

When NumPy is built, information about system configuration is
recorded, and is made available for extension modules using NumPy’s C
API.  These are mostly defined in ``numpyconfig.h`` (included in
``ndarrayobject.h``). The public symbols are prefixed by ``NPY_*``.
NumPy also offers some functions for querying information about the
platform in use.

For private use, NumPy also constructs a ``config.h`` in the NumPy
include directory, which is not exported by NumPy (that is a python
extension which use the numpy C API will not see those symbols), to
avoid namespace pollution.

## Data type sizes

The ``NPY_SIZEOF_{CTYPE}`` constants are defined so that sizeof
information is available to the pre-processor.

- ``NPY_SIZEOF_SHORT``

    sizeof(short)

- ``NPY_SIZEOF_INT``

    sizeof(int)

- ``NPY_SIZEOF_LONG``

    sizeof(long)

- ``NPY_SIZEOF_LONGLONG``

    sizeof(longlong) where longlong is defined appropriately on the
    platform.

- ``NPY_SIZEOF_PY_LONG_LONG``

- ``NPY_SIZEOF_FLOAT``

    sizeof(float)

- ``NPY_SIZEOF_DOUBLE``

    sizeof(double)

- ``NPY_SIZEOF_LONG_DOUBLE``

    sizeof(longdouble) (A macro defines **NPY_SIZEOF_LONGDOUBLE** as well.)

- ``NPY_SIZEOF_PY_INTPTR_T``

    Size of a pointer on this platform (sizeof(void *)) (A macro defines
    NPY_SIZEOF_INTP as well.)

## Platform information

- ``NPY_CPU_X86``
- ``NPY_CPU_AMD64``
- ``NPY_CPU_IA64``
- ``NPY_CPU_PPC``
- ``NPY_CPU_PPC64``
- ``NPY_CPU_SPARC``
- ``NPY_CPU_SPARC64``
- ``NPY_CPU_S390``
- ``NPY_CPU_PARISC``

    *New in version 1.3.0.* 

    CPU architecture of the platform; only one of the above is
    defined.

    Defined in ``numpy/npy_cpu.h``
- ``NPY_LITTLE_ENDIAN``
- ``NPY_BIG_ENDIAN``
- ``NPY_BYTE_ORDER``

    *New in version 1.3.0.* 

    Portable alternatives to the ``endian.h`` macros of GNU Libc.
    If big endian, [``NPY_BYTE_ORDER``](#c.NPY_BYTE_ORDER) == [``NPY_BIG_ENDIAN``](#c.NPY_BIG_ENDIAN), and
    similarly for little endian architectures.

    Defined in ``numpy/npy_endian.h``.
- ``PyArray_GetEndianness``()

    *New in version 1.3.0.* 

    Returns the endianness of the current platform.
    One of ``NPY_CPU_BIG``, ``NPY_CPU_LITTLE``,
    or ``NPY_CPU_UNKNOWN_ENDIAN``.

## Compiler directives

- ``NPY_LIKELY``
- ``NPY_UNLIKELY``
- ``NPY_UNUSED``

## Interrupt Handling

- ``NPY_INTERRUPT_H``
- ``NPY_SIGSETJMP``
- ``NPY_SIGLONGJMP``
- ``NPY_SIGJMP_BUF``
- ``NPY_SIGINT_ON``
- ``NPY_SIGINT_OFF``