import type { SidebarConfigArray } from 'vuepress-theme-teadocs'

export default [
  {
    text: 'Setting up',
    collapsible: true,
    children: [
      '/en/user/setting-up', // { link: '/en/user/setting-up', text: 'Setting up' },
    ]
  },
  {
    text: 'Quickstart tutorial',
    collapsible: true,
    children: [
      '/en/user/quickstart', // { link: '/en/user/quickstart', text: 'Quickstart tutorial' },
    ]
  },
  {
    text: 'NumPy basics',
    collapsible: true,
    children: [
      '/en/user/basics/types', // { link: '/en/user/basics/types', text: 'Data types' },
      '/en/user/basics/creation', // { link: '/en/user/basics/creation', text: 'Array creation' },
      '/en/user/basics/io', // { link: '/en/user/basics/io', text: 'I/O with NumPy' },
      '/en/user/basics/indexing', // { link: '/en/user/basics/indexing', text: 'Indexing' },
      '/en/user/basics/broadcasting', // { link: '/en/user/basics/broadcasting', text: 'Broadcasting' },
      '/en/user/basics/byteswapping', // { link: '/en/user/basics/byteswapping', text: 'Byte-swapping' },
      '/en/user/basics/rec', // { link: '/en/user/basics/rec', text: 'Structured arrays' },
      '/en/user/basics/dispatch', // { link: '/en/user/basics/dispatch', text: 'Writing custom array containers' },
      '/en/user/basics/subclassing', // { link: '/en/user/basics/subclassing', text: 'Subclassing ndarray' },
    ]
  },
  {
    text: 'Miscellaneous',
    collapsible: true,
    children: [
      '/en/user/misc', // { link: '/en/user/misc', text: 'Miscellaneous' },
    ]
  },
  {
    text: 'NumPy for Matlab users',
    collapsible: true,
    children: [
      '/en/user/numpy_for_matlab_users', // { link: '/en/user/numpy_for_matlab_users', text: 'NumPy for Matlab users' },
    ]
  },
  {
    text: 'Building from source',
    collapsible: true,
    children: [
      '/en/user/building', // { link: '/en/user/building', text: 'Building from source' },
    ]
  },
  {
    text: 'Using NumPy C-API',
    collapsible: true,
    children: [
      { link: '/en/user/c-info/', text: 'Index' },  // { link: '/en/user/c-info/', text: 'Index' },
      '/en/user/c-info/how-to-extend', // { link: '/en/user/c-info/how-to-extend', text: 'How to extend NumPy' },
      '/en/user/c-info/python-as-glue', // { link: '/en/user/c-info/python-as-glue', text: 'Using Python as glue' },
      '/en/user/c-info/ufunc-tutorial', // { link: '/en/user/c-info/ufunc-tutorial', text: 'Writing your own ufunc' },
      '/en/user/c-info/beyond-basics', // { link: '/en/user/c-info/beyond-basics', text: 'Beyond the Basics' },
    ]
  }
]
