module.exports = function () {
  return [
    {
      title: 'Setting up',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/setting-up', 'Setting up']
      ]
    },
    {
      title: 'Quickstart tutorial',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/quickstart', 'Quickstart tutorial']
      ]
    },
    {
      title: 'NumPy basics',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/basics/types', 'Data types'],
        ['/user/basics/creation', 'Array creation'],
        ['/user/basics/io', 'I/O with NumPy'],
        ['/user/basics/indexing', 'Indexing'],
        ['/user/basics/broadcasting', 'Broadcasting'],
        ['/user/basics/byteswapping', 'Byte-swapping'],
        ['/user/basics/rec', 'Structured arrays'],
        ['/user/basics/dispatch', 'Writing custom array containers'],
        ['/user/basics/subclassing', 'Subclassing ndarray']
      ]
    },
    {
      title: 'Miscellaneous',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/misc', 'Miscellaneous']
      ]
    },
    {
      title: 'NumPy for Matlab users',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/numpy_for_matlab_users', 'NumPy for Matlab users']
      ]
    },
    {
      title: 'Building from source',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/building', 'Building from source']
      ]
    },
    {
      title: 'Using NumPy C-API',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/c-info/', 'Index'],
        ['/user/c-info/how-to-extend', 'How to extend NumPy'],
        ['/user/c-info/python-as-glue', 'Using Python as glue'],
        ['/user/c-info/ufunc-tutorial', 'Writing your own ufunc'],
        ['/user/c-info/beyond-basics', 'Beyond the Basics']
      ]
    }
  ]
}