module.exports = function () {
  return [
    {
      title: '深度学习基础教程',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/deep/basics/', '前言'],
        ['/deep/basics/fit_a_line', '线性回归'],
        ['/deep/basics/recognize_digits', '数字识别'],
        ['/deep/basics/image_classification', '图像分类'],
        ['/deep/basics/word2vec', '词向量'],
        ['/deep/basics/recommender_system', '个性化推荐'],
        ['/deep/basics/understand_sentiment', '情感分析'],
        ['/deep/basics/label_semantic_roles', '语义角色标注'],
        ['/deep/basics/machine_translation', '机器翻译'],
        ['/deep/basics/gan', '生成对抗网络']
      ]
    },
    {
      title: '七日入门深度学习',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/deep/videos/day1_01', 'Day 1：初识深度学习'],
        ['/deep/videos/day1_02', 'Day 1：如何快速入门深度学习？'],
        ['/deep/videos/day2_01', 'Day 2：图像识别基础与实战'],
        ['/deep/videos/day3_01', 'Day 3：目标检测基础与实践（一）'],
        ['/deep/videos/day3_02', 'Day 3：目标检测实战-RCNN算法讲解'],
        ['/deep/videos/day3_03', 'Day 3：目标检测实战-YOLOv3检测物体']
      ]
    }
  ]
}
