module.exports = function () {
  return [
    {
      title: 'Setting up',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/user/setting-up', 'Setting up']
      ]
    },
    {
      title: 'Quickstart tutorial',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/user/quickstart', 'Quickstart tutorial']
      ]
    },
    {
      title: 'NumPy basics',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/user/basics/types', 'Data types'],
        ['/en/user/basics/creation', 'Array creation'],
        ['/en/user/basics/io', 'I/O with NumPy'],
        ['/en/user/basics/indexing', 'Indexing'],
        ['/en/user/basics/broadcasting', 'Broadcasting'],
        ['/en/user/basics/byteswapping', 'Byte-swapping'],
        ['/en/user/basics/rec', 'Structured arrays'],
        ['/en/user/basics/dispatch', 'Writing custom array containers'],
        ['/en/user/basics/subclassing', 'Subclassing ndarray']
      ]
    },
    {
      title: 'Miscellaneous',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/user/misc', 'Miscellaneous']
      ]
    },
    {
      title: 'NumPy for Matlab users',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/user/numpy_for_matlab_users', 'NumPy for Matlab users']
      ]
    },
    {
      title: 'Building from source',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/user/building', 'Building from source']
      ]
    },
    {
      title: 'Using NumPy C-API',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/en/user/c-info/', 'Index'],
        ['/en/user/c-info/how-to-extend', 'How to extend NumPy'],
        ['/en/user/c-info/python-as-glue', 'Using Python as glue'],
        ['/en/user/c-info/ufunc-tutorial', 'Writing your own ufunc'],
        ['/en/user/c-info/beyond-basics', 'Beyond the Basics']
      ]
    }
  ]
}