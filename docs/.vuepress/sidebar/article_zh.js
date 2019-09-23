module.exports = function () {
  return [
    {
      title: '基础篇',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/article/basics/understanding_numpy', '理解 NumPy'],
        ['/article/basics/an_introduction_to_scientific_python_numpy', 'NumPy 简单入门教程'],
        ['/article/basics/python_numpy_tutorial', 'Python Numpy 教程'],
        ['/article/basics/different_ways_create_numpy_arrays', '创建 NumPy 数组的不同方式'],
        ['/article/basics/numpy_matrices_vectors', 'NumPy 中的矩阵和向量']
      ]
    },
    {
      title: '进阶篇',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/article/advanced/numpy_exercises_for_data_analysis', 'NumPy 数据分析练习'],
        ['/article/advanced/neural_network_with_numpy', 'NumPy 神经网络'],
        ['/article/advanced/numpy_array_programming', '使用 NumPy 进行数组编程'],
        ['/article/advanced/numpy_kmeans', 'NumPy 实现k均值聚类算法'],
        ['/article/advanced/dnc_rnn_lstm', 'NumPy 实现DNC、RNN和LSTM神经网络算法']
      ]
    },
    {
      title: '其他篇',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/article/other/py_basic_ops', 'OpenCV中的图像的基本操作'],
        ['/article/other/minpy-the-numpy-interface-upon-mxnets-backend', 'MinPy：MXNet后端的NumPy接口']
      ]
    }
  ]
}
