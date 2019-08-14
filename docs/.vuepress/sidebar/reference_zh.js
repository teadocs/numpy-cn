module.exports = function () {
  return [{
    title: '数组对象',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/arrays/', '目录'],
      ['/reference/arrays/ndarray', 'N维数组(ndarray)'],
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
      ['/reference/c-api/', 'index']
    ]
  }, {
    title: 'NumPy internals',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/internals/', 'index']
    ]
  }, {
    title: 'NumPy and SWIG',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/swig/', 'index']
    ]
  }]
}