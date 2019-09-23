module.exports = function () {
  return [
    {
      title: 'NumPy 介绍',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/setting-up', 'NumPy 介绍']
      ]
    },
    {
      title: '快速入门教程',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/quickstart', '快速入门教程']
      ]
    },
    {
      title: 'NumPy 基础知识',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/basics/', '目录'],
        ['/user/basics/types', '数据类型'],
        ['/user/basics/creation', '创建数组'],
        ['/user/basics/io', 'NumPy与输入输出'],
        ['/user/basics/indexing', '索引'],
        ['/user/basics/broadcasting', '广播'],
        ['/user/basics/byteswapping', '字节交换'],
        ['/user/basics/rec', '结构化数组'],
        ['/user/basics/dispatch', '编写自定义数组容器'],
        ['/user/basics/subclassing', '子类化ndarray']
      ]
    },
    {
      title: '其他杂项',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/misc', '其他杂项']
      ]
    },
    {
      title: '与 Matlab 比较',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/numpy_for_matlab_users', '与 Matlab 比较']
      ]
    },
    {
      title: '从源代码构建',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/building', '从源代码构建']
      ]
    },
    {
      title: '使用NumPy的C-API',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/c-info/', '目录'],
        ['/user/c-info/how-to-extend', '如何扩展NumPy'],
        ['/user/c-info/python-as-glue', '使用Python作为胶水'],
        ['/user/c-info/ufunc-tutorial', '编写自己的ufunc'],
        ['/user/c-info/beyond-basics', '深入的知识']
      ]
    }
  ]
}