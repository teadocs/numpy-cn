module.exports = function () {
  return [{
    title: '数组对象',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/arrays/', '目录'],
      ['/reference/arrays/ndarray', 'N维数组(ndarray)'],
      ['/reference/arrays/scalars', 'Scalars'],
      ['/reference/arrays/dtypes', 'Data type objects (dtype)'],
      ['/reference/arrays/indexing', 'Indexing'],
      ['/reference/arrays/nditer', 'Iterating Over Arrays'],
      ['/reference/arrays/classes', 'Standard array subclasses'],
      ['/reference/arrays/maskedarray', 'Masked arrays'],
      ['/reference/arrays/interface', 'The Array Interface'],
      ['/reference/arrays/datetime', 'Datetimes and Timedeltas']
    ]
  }, {
    title: '常量',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/constants', '常量']
    ]
  }, {
    title: 'Universal functions(ufunc)',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/ufuncs', 'Universal functions(ufunc)']
    ]
  }, {
    title: 'Routines',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/routines/', 'index'],
      ['/reference/routines/array-creation', 'Array creation routines'],
      ['/reference/routines/array-manipulation', 'Array manipulation routines'],
      ['/reference/routines/bitwise', 'Binary operations'],
      ['/reference/routines/char', 'String operations'],
      ['/reference/routines/ctypeslib', 'C-Types Foreign Function Interface (numpy.ctypeslib)'],
      ['/reference/routines/datetime', 'Datetime Support Functions'],
      ['/reference/routines/dtype', 'Data type routines'],
      ['/reference/routines/dual', 'Optionally Scipy-accelerated routines (numpy.dual)'],
      ['/reference/routines/emath', 'Mathematical functions with automatic domain (numpy.emath)'],
      ['/reference/routines/err', 'Floating point error handling'],
      ['/reference/routines/fft', 'Discrete Fourier Transform (numpy.fft)'],
      ['/reference/routines/financial', 'Financial functions'],
      ['/reference/routines/functional', 'Functional programming'],
      ['/reference/routines/help', 'NumPy-specific help functions'],
      ['/reference/routines/indexing', 'Indexing routines'],
      ['/reference/routines/io', 'Input and output'],
      ['/reference/routines/linalg', 'Linear algebra (numpy.linalg)'],
      ['/reference/routines/logic', 'Logic functions'],
      ['/reference/routines/ma', 'Masked array operations'],
      ['/reference/routines/math', 'Mathematical functions'],
      ['/reference/routines/matlib', 'Matrix library (numpy.matlib)'],
      ['/reference/routines/other', 'Miscellaneous routines'],
      ['/reference/routines/padding', 'Padding Arrays'],
      ['/reference/routines/polynomials', 'Polynomials'],
      ['/reference/routines/random', 'Random sampling (numpy.random)'],
      ['/reference/routines/set', 'Set routines'],
      ['/reference/routines/sort', 'Sorting, searching, and counting'],
      ['/reference/routines/statistics', 'Statistics'],
      ['/reference/routines/testing', 'Test Support (numpy.testing)'],
      ['/reference/routines/window', 'Window functions']
    ]
  }, {
    title: 'Packaging(numpy.distutils)',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/distutils', 'Packaging(numpy.distutils)']
    ]
  }, {
    title: 'NumPy Distutils - Users Guide',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/distutils_guide', 'NumPy Distutils - Users Guide']
    ]
  }, {
    title: 'NumPy C-API',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/c-api/', 'index'],
      ['/reference/c-api/types-and-structures', 'Python Types and C-Structures'],
      ['/reference/c-api/config', 'System configuration'],
      ['/reference/c-api/dtype', 'Data Type API'],
      ['/reference/c-api/array', 'Array API'],
      ['/reference/c-api/iterator', 'Array Iterator API'],
      ['/reference/c-api/ufunc', 'UFunc API'],
      ['/reference/c-api/generalized-ufuncs', 'Generalized Universal Function API'],
      ['/reference/c-api/coremath', 'NumPy core libraries'],
      ['/reference/c-api/deprecations', 'C API Deprecations']
    ]
  }, {
    title: 'NumPy internals',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/internals/', 'index'],
      ['/reference/internals/code-explanations', 'NumPy C Code Explanations'],
      ['/reference/internals/alignment', 'Memory Alignment']
    ]
  }, {
    title: 'NumPy and SWIG',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/swig/', 'index'],
      ['/reference/swig/interface-file', 'numpy.i: a SWIG Interface File for NumPy'],
      ['/reference/swig/testing', 'Testing the numpy.i Typemaps']
    ]
  }]
}