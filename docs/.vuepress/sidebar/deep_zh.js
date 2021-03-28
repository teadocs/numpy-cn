module.exports = function () {
  return [
    {
      title: '为深度学习新手准备的教程',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/deep/beginner/megengine_basic_concepts', '天元 MegEngine 基础概念'],
        ['/deep/beginner/learning_from_linear_regression', '一个稍微复杂些的线性回归模型'],
        ['/deep/beginner/from_linear_regression_to_linear_classification', '从线性回归到线性分类']
      ]
    }
  ]
}
