export default [{
  text: '数组对象',
  collapsible: true,
  children: [
    { text: '目录', link: '/reference/arrays/' },
    '/reference/arrays/ndarray', // { link: '/reference/arrays/ndarray', text: 'N维数组(ndarray)' },
    '/reference/arrays/scalars', // { link: '/reference/arrays/scalars', text: '标量' },
    '/reference/arrays/dtypes', // { link: '/reference/arrays/dtypes', text: '数据类型对象(dtype)' },
    '/reference/arrays/indexing', // { link: '/reference/arrays/indexing', text: '索引' },
    '/reference/arrays/nditer', // { link: '/reference/arrays/nditer', text: '迭代数组' },
    '/reference/arrays/classes', // { link: '/reference/arrays/classes', text: '标准数组子类' },
    '/reference/arrays/maskedarray', // { link: '/reference/arrays/maskedarray', text: '掩码数组' },
    '/reference/arrays/interface', // { link: '/reference/arrays/interface', text: '数组接口' },
    '/reference/arrays/datetime', // { link: '/reference/arrays/datetime', text: '日期时间和时间增量' },
  ]
}, {
  text: '常量',
  collapsible: true,
  children: [
    '/reference/constants', // { link: '/reference/constants', text: '常量' },
  ]
}, {
  text: '通函数(ufunc)',
  collapsible: true,
  children: [
    '/reference/ufuncs', // { link: '/reference/ufuncs', text: '通函数(ufunc)' },
  ]
}, {
  text: '常用API',
  collapsible: true,
  children: [
    { link: '/reference/routines/', text: '目录' },
    '/reference/routines/array-creation', // { link: '/reference/routines/array-creation', text: '创建数组' },
    '/reference/routines/array-manipulation', // { link: '/reference/routines/array-manipulation', text: '数组操作' },
    '/reference/routines/bitwise', // { link: '/reference/routines/bitwise', text: '二进制操作' },
    '/reference/routines/char', // { link: '/reference/routines/char', text: '字符串操作' },
    '/reference/routines/ctypeslib', // { link: '/reference/routines/ctypeslib', text: 'C-Types外部函数接口(numpy.ctypeslib)' },
    '/reference/routines/datetime', // { link: '/reference/routines/datetime', text: '时间日期相关' },
    '/reference/routines/dtype', // { link: '/reference/routines/dtype', text: '数据类型相关' },
    '/reference/routines/dual', // { link: '/reference/routines/dual', text: '可选的Scipy加速支持(numpy.dual)' },
    '/reference/routines/emath', // { link: '/reference/routines/emath', text: '具有自动域的数学函数(numpy.emath)' },
    '/reference/routines/err', // { link: '/reference/routines/err', text: '浮点错误处理' },
    '/reference/routines/fft', // { link: '/reference/routines/fft', text: '离散傅立叶变换(numpy.fft)' },
    '/reference/routines/financial', // { link: '/reference/routines/financial', text: '财金相关' },
    '/reference/routines/functional', // { link: '/reference/routines/functional', text: '实用的功能' },
    '/reference/routines/help', // { link: '/reference/routines/help', text: '特殊的NumPy帮助功能' },
    '/reference/routines/indexing', // { link: '/reference/routines/indexing', text: '索引相关' },
    '/reference/routines/io', // { link: '/reference/routines/io', text: '输入和输出' },
    '/reference/routines/linalg', // { link: '/reference/routines/linalg', text: '线性代数(numpy.linalg)' },
    '/reference/routines/logic', // { link: '/reference/routines/logic', text: '逻辑函数' },
    '/reference/routines/ma', // { link: '/reference/routines/ma', text: '操作掩码数组' },
    '/reference/routines/math', // { link: '/reference/routines/math', text: '数学函数' },
    '/reference/routines/matlib', // { link: '/reference/routines/matlib', text: '矩阵库(numpy.matlib)' },
    '/reference/routines/other', // { link: '/reference/routines/other', text: '杂项' },
    '/reference/routines/padding', // { link: '/reference/routines/padding', text: '填充数组' },
    '/reference/routines/polynomials', // { link: '/reference/routines/polynomials', text: '多项式' },
    '/reference/routines/random', // { link: '/reference/routines/random', text: '随机抽样(numpy.random)' },
    '/reference/routines/set', // { link: '/reference/routines/set', text: '集合操作' },
    '/reference/routines/sort', // { link: '/reference/routines/sort', text: '排序、搜索和计数' },
    '/reference/routines/statistics', // { link: '/reference/routines/statistics', text: '统计相关' },
    '/reference/routines/testing', // { link: '/reference/routines/testing', text: '测试支持(numpy.testing)' },
    '/reference/routines/window', // { link: '/reference/routines/window', text: '窗口函数' },
  ]
}, {
  text: '打包（numpy.distutils）',
  collapsible: true,
  children: [
    '/reference/distutils', // { link: '/reference/distutils', text: '打包（numpy.distutils）' },
  ]
}, {
  text: 'NumPy Distutils 的用户指南',
  collapsible: true,
  children: [
    '/reference/distutils_guide', // { link: '/reference/distutils_guide', text: 'NumPy Distutils 的用户指南' },
  ]
}, {
  text: 'NumPy C-API',
  collapsible: true,
  children: [
    { link: '/reference/c-api/', text: '目录' },
    '/reference/c-api/types-and-structures', // { link: '/reference/c-api/types-and-structures', text: 'Python类型和C结构' },
    '/reference/c-api/config', // { link: '/reference/c-api/config', text: '系统配置' },
    '/reference/c-api/dtype', // { link: '/reference/c-api/dtype', text: '数据类型API' },
    '/reference/c-api/array', // { link: '/reference/c-api/array', text: '数组API' },
    '/reference/c-api/iterator', // { link: '/reference/c-api/iterator', text: '数组迭代API' },
    '/reference/c-api/ufunc', // { link: '/reference/c-api/ufunc', text: 'UFunc API' },
    '/reference/c-api/generalized-ufuncs', // { link: '/reference/c-api/generalized-ufuncs', text: '一般的通函数API' },
    '/reference/c-api/coremath', // { link: '/reference/c-api/coremath', text: 'NumPy核心库' },
    '/reference/c-api/deprecations', // { link: '/reference/c-api/deprecations', text: '弃用的 C API ' },
  ]
}, {
  text: 'NumPy 内部',
  collapsible: true,
  children: [
    { link: '/reference/internals/', text: '目录' },
    '/reference/internals/code-explanations', // { link: '/reference/internals/code-explanations', text: 'NumPy C代码说明' },
    '/reference/internals/alignment', // { link: '/reference/internals/alignment', text: '内存校准' },
  ]
}, {
  text: 'NumPy 和 SWIG',
  collapsible: true,
  children: [
    { link: '/reference/swig/', text: '目录' },
    '/reference/swig/interface-file', // { link: '/reference/swig/interface-file', text: 'numpy.i：NumPy的SWIG接口文件' },
    '/reference/swig/testing', // { link: '/reference/swig/testing', text: '测试numpy.i Typemaps' },
  ]
}];
