module.exports = function () {
  return [
    {
      title: '基础篇',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/articles/basics/understanding_numpy', '理解 NumPy'],
        ['/articles/basics/an_introduction_to_scientific_python_numpy', 'NumPy 简单入门教程'],
        ['/articles/basics/python_numpy_tutorial', 'Python Numpy 教程'],
        ['/articles/basics/different_ways_create_numpy_arrays', '创建 NumPy 数组的不同方式'],
        ['/articles/basics/numpy_matrices_vectors', 'NumPy 中的矩阵和向量']
      ]
    },
    {
      title: '进阶篇',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/articles/advanced/numpy_exercises_for_data_analysis', 'NumPy 数据分析练习'],
        ['/articles/advanced/neural_network_with_numpy', 'NumPy 神经网络'],
        ['/articles/advanced/numpy_array_programming', '使用 NumPy 进行数组编程'],
        ['/articles/advanced/numpy_kmeans', 'NumPy 实现k均值聚类算法'],
        ['/articles/advanced/dnc_rnn_lstm', 'NumPy 实现DNC、RNN和LSTM神经网络算法']
      ]
    },
    {
      title: '其他篇',
      collapsable: true,
      sidebarDepth: 3,
      children: [
        ['/articles/other/py_basic_ops', 'OpenCV中的图像的基本操作'],
        ['/articles/other/minpy-the-numpy-interface-upon-mxnets-backend', 'MinPy：MXNet后端的NumPy接口']
      ]
    }
  ]
}
