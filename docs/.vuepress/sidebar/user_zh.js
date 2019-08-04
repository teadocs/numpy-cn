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
        ['/user/basics/indexing', '索引']
      ]
    }
  ]
}