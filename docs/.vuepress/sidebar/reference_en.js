module.exports = function () {
  return [{
    title: '数组对象',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/arrays/', 'Index'],
      ['/en/reference/arrays/ndarray', 'The N-dimensional array (ndarray)'],
      ['/en/reference/arrays/scalars', 'Scalars'],
      ['/en/reference/arrays/dtypes', 'Data type objects (dtype)'],
      ['/en/reference/arrays/indexing', 'Indexing'],
      ['/en/reference/arrays/nditer', 'Iterating Over Arrays'],
      ['/en/reference/arrays/classes', 'Standard array subclasses'],
      ['/en/reference/arrays/maskedarray', 'Masked arrays'],
      ['/en/reference/arrays/interface', 'The Array Interface'],
      ['/en/reference/arrays/datetime', 'Datetimes and Timedeltas']
    ]
  }, {
    title: 'Constants',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/constants', 'Constants']
    ]
  }, {
    title: 'Universal functions(ufunc)',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/ufuncs', 'Universal functions(ufunc)']
    ]
  }, {
    title: 'Routines',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/routines/', 'Index'],
      ['/en/reference/routines/array-creation', 'Array creation routines'],
      ['/en/reference/routines/array-manipulation', 'Array manipulation routines'],
      ['/en/reference/routines/bitwise', 'Binary operations'],
      ['/en/reference/routines/char', 'String operations'],
      ['/en/reference/routines/ctypeslib', 'C-Types Foreign Function Interface (numpy.ctypeslib)'],
      ['/en/reference/routines/datetime', 'Datetime Support Functions'],
      ['/en/reference/routines/dtype', 'Data type routines'],
      ['/en/reference/routines/dual', 'Optionally Scipy-accelerated routines (numpy.dual)'],
      ['/en/reference/routines/emath', 'Mathematical functions with automatic domain (numpy.emath)'],
      ['/en/reference/routines/err', 'Floating point error handling'],
      ['/en/reference/routines/fft', 'Discrete Fourier Transform (numpy.fft)'],
      ['/en/reference/routines/financial', 'Financial functions'],
      ['/en/reference/routines/functional', 'Functional programming'],
      ['/en/reference/routines/help', 'NumPy-specific help functions'],
      ['/en/reference/routines/indexing', 'Indexing routines'],
      ['/en/reference/routines/io', 'Input and output'],
      ['/en/reference/routines/linalg', 'Linear algebra (numpy.linalg)'],
      ['/en/reference/routines/logic', 'Logic functions'],
      ['/en/reference/routines/ma', 'Masked array operations'],
      ['/en/reference/routines/math', 'Mathematical functions'],
      ['/en/reference/routines/matlib', 'Matrix library (numpy.matlib)'],
      ['/en/reference/routines/other', 'Miscellaneous routines'],
      ['/en/reference/routines/padding', 'Padding Arrays'],
      ['/en/reference/routines/polynomials', 'Polynomials'],
      ['/en/reference/routines/random', 'Random sampling (numpy.random)'],
      ['/en/reference/routines/set', 'Set routines'],
      ['/en/reference/routines/sort', 'Sorting, searching, and counting'],
      ['/en/reference/routines/statistics', 'Statistics'],
      ['/en/reference/routines/testing', 'Test Support (numpy.testing)'],
      ['/en/reference/routines/window', 'Window functions']
    ]
  }, {
    title: 'Packaging(numpy.distutils)',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/distutils', 'Packaging(numpy.distutils)']
    ]
  }, {
    title: 'NumPy Distutils - Users Guide',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/distutils_guide', 'NumPy Distutils - Users Guide']
    ]
  }, {
    title: 'NumPy C-API',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/c-api/', 'Index'],
      ['/en/reference/c-api/types-and-structures', 'Python Types and C-Structures'],
      ['/en/reference/c-api/config', 'System configuration'],
      ['/en/reference/c-api/dtype', 'Data Type API'],
      ['/en/reference/c-api/array', 'Array API'],
      ['/en/reference/c-api/iterator', 'Array Iterator API'],
      ['/en/reference/c-api/ufunc', 'UFunc API'],
      ['/en/reference/c-api/generalized-ufuncs', 'Generalized Universal Function API'],
      ['/en/reference/c-api/coremath', 'NumPy core libraries'],
      ['/en/reference/c-api/deprecations', 'C API Deprecations']
    ]
  }, {
    title: 'NumPy internals',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/internals/', 'Index'],
      ['/en/reference/internals/code-explanations', 'NumPy C Code Explanations'],
      ['/en/reference/internals/alignment', 'Memory Alignment']
    ]
  }, {
    title: 'NumPy and SWIG',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/en/reference/swig/', 'Index'],
      ['/en/reference/swig/interface-file', 'numpy.i: a SWIG Interface File for NumPy'],
      ['/en/reference/swig/testing', 'Testing the numpy.i Typemaps']
    ]
  }]
}