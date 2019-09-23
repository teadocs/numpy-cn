module.exports = function () {
  return [{
    title: '数组对象',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/arrays/', '目录'],
      ['/reference/arrays/ndarray', 'N维数组(ndarray)'],
      ['/reference/arrays/scalars', '标量'],
      ['/reference/arrays/dtypes', '数据类型对象(dtype)'],
      ['/reference/arrays/indexing', '索引'],
      ['/reference/arrays/nditer', '迭代数组'],
      ['/reference/arrays/classes', '标准数组子类'],
      ['/reference/arrays/maskedarray', '掩码数组'],
      ['/reference/arrays/interface', '数组接口'],
      ['/reference/arrays/datetime', '日期时间和时间增量']
    ]
  }, {
    title: '常量',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/constants', '常量']
    ]
  }, {
    title: '通函数(ufunc)',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/ufuncs', '通函数(ufunc)']
    ]
  }, {
    title: '常用API',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/routines/', '目录'],
      ['/reference/routines/array-creation', '创建数组'],
      ['/reference/routines/array-manipulation', '数组操作'],
      ['/reference/routines/bitwise', '二进制操作'],
      ['/reference/routines/char', '字符串操作'],
      ['/reference/routines/ctypeslib', 'C-Types外部函数接口(numpy.ctypeslib)'],
      ['/reference/routines/datetime', '时间日期相关'],
      ['/reference/routines/dtype', '数据类型相关'],
      ['/reference/routines/dual', '可选的Scipy加速支持(numpy.dual)'],
      ['/reference/routines/emath', '具有自动域的数学函数(numpy.emath)'],
      ['/reference/routines/err', '浮点错误处理'],
      ['/reference/routines/fft', '离散傅立叶变换(numpy.fft)'],
      ['/reference/routines/financial', '财金相关'],
      ['/reference/routines/functional', '实用的功能'],
      ['/reference/routines/help', '特殊的NumPy帮助功能'],
      ['/reference/routines/indexing', '索引相关'],
      ['/reference/routines/io', '输入和输出'],
      ['/reference/routines/linalg', '线性代数(numpy.linalg)'],
      ['/reference/routines/logic', '逻辑函数'],
      ['/reference/routines/ma', '操作掩码数组'],
      ['/reference/routines/math', '数学函数'],
      ['/reference/routines/matlib', '矩阵库(numpy.matlib)'],
      ['/reference/routines/other', '杂项'],
      ['/reference/routines/padding', '填充数组'],
      ['/reference/routines/polynomials', '多项式'],
      ['/reference/routines/random', '随机抽样(numpy.random)'],
      ['/reference/routines/set', '集合操作'],
      ['/reference/routines/sort', '排序、搜索和计数'],
      ['/reference/routines/statistics', '统计相关'],
      ['/reference/routines/testing', '测试支持(numpy.testing)'],
      ['/reference/routines/window', '窗口函数']
    ]
  }, {
    title: '打包（numpy.distutils）',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/distutils', '打包（numpy.distutils）']
    ]
  }, {
    title: 'NumPy Distutils 的用户指南',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/distutils_guide', 'NumPy Distutils 的用户指南']
    ]
  }, {
    title: 'NumPy C-API',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/c-api/', '目录'],
      ['/reference/c-api/types-and-structures', 'Python类型和C结构'],
      ['/reference/c-api/config', '系统配置'],
      ['/reference/c-api/dtype', '数据类型API'],
      ['/reference/c-api/array', '数组API'],
      ['/reference/c-api/iterator', '数组迭代API'],
      ['/reference/c-api/ufunc', 'UFunc API'],
      ['/reference/c-api/generalized-ufuncs', '一般的通函数API'],
      ['/reference/c-api/coremath', 'NumPy核心库'],
      ['/reference/c-api/deprecations', '弃用的 C API ']
    ]
  }, {
    title: 'NumPy 内部',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/internals/', '目录'],
      ['/reference/internals/code-explanations', 'NumPy C代码说明'],
      ['/reference/internals/alignment', '内存校准']
    ]
  }, {
    title: 'NumPy 和 SWIG',
    collapsable: true,
    sidebarDepth: 3,
    children: [
      ['/reference/swig/', '目录'],
      ['/reference/swig/interface-file', 'numpy.i：NumPy的SWIG接口文件'],
      ['/reference/swig/testing', '测试numpy.i Typemaps']
    ]
  }]
}