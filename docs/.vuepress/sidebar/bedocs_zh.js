module.exports = function () {
  return [
    {
      title: 'NumPy 的文档相关',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/bedocs/', '目录'],
        ['/bedocs/howto_document', '一份给NumPy/SciPy的文档做贡献的指南'],
        ['/bedocs/example_source', '示例来源'],
        ['/bedocs/example_rendered', '渲染示例'],
        ['/bedocs/howto_build_docs', '构建NumPy API和参考文档']
      ]
    }
  ]
}