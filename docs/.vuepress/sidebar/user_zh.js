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
      title: '基础知识',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/user/basics/types', '数据类型'],
        ['/user/basics/creation', '创建数组'],
        ['/user/basics/io', '输入输出'],
        ['/user/basics/indexing', '索引'],
        ['/user/basics/broadcasting', '广播'],
        ['/user/basics/byteswapping', '字节交换'],
        ['/user/basics/rec', '结构化数组'],
        ['/user/basics/dispatch', '编写自定义数组容器'],
        ['/user/basics/subclassing', '子类化ndarray']
      ]
    }
  ]
}