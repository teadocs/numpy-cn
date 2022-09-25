export default [
  {
    text: 'NumPy 介绍',
    collapsible: true,
    children: [
      '/user/setting-up',
    ]
  },
  {
    text: '快速入门教程',
    collapsible: true,
    children: [
      '/user/quickstart',
    ]
  },
  {
    text: 'NumPy 基础知识',
    collapsible: true,
    children: [
      { link: '/user/basics/', text: '目录' },
      '/user/basics/types', // { link: '/user/basics/types', text: '数据类型' },
      '/user/basics/creation', // { link: '/user/basics/creation', text: '创建数组' },
      '/user/basics/io', // { link: '/user/basics/io', text: 'NumPy与输入输出' },
      '/user/basics/indexing', // { link: '/user/basics/indexing', text: '索引' },
      '/user/basics/broadcasting', // { link: '/user/basics/broadcasting', text: '广播' },
      '/user/basics/byteswapping', // { link: '/user/basics/byteswapping', text: '字节交换' },
      '/user/basics/rec', // { link: '/user/basics/rec', text: '结构化数组' },
      '/user/basics/dispatch', // { link: '/user/basics/dispatch', text: '编写自定义数组容器' },
      '/user/basics/subclassing', // { link: '/user/basics/subclassing', text: '子类化ndarray' },
    ]
  },
  {
    text: '其他杂项',
    collapsible: true,
    children: [
      '/user/misc', // { link: '/user/misc', text: '其他杂项' },
    ]
  },
  {
    text: '与 Matlab 比较',
    collapsible: true,
    children: [
      '/user/numpy_for_matlab_users', // { link: '/user/numpy_for_matlab_users', text: '与 Matlab 比较' },
    ]
  },
  {
    text: '从源代码构建',
    collapsible: true,
    children: [
      '/user/building', // { link: '/user/building', text: '从源代码构建' },
    ]
  },
  {
    text: '使用NumPy的C-API',
    collapsible: true,
    children: [
      { link: '/user/c-info/', text: '目录' },
      '/user/c-info/how-to-extend', // { link: '/user/c-info/how-to-extend', text: '如何扩展NumPy' },
      '/user/c-info/python-as-glue', // { link: '/user/c-info/python-as-glue', text: '使用Python作为胶水' },
      '/user/c-info/ufunc-tutorial', // { link: '/user/c-info/ufunc-tutorial', text: '编写自己的ufunc' },
      '/user/c-info/beyond-basics', // { link: '/user/c-info/beyond-basics', text: '深入的知识' },
    ]
  }
]
