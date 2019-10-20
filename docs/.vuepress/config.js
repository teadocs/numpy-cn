module.exports = ctx => ({
  dest: './dist',
  locales: {
    '/': {
      lang: 'zh-CN',
      title: 'NumPy 中文',
      description: '这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包。'
    },
    '/en/': {
      lang: 'en-US',
      title: 'NumPy',
      description: 'NumPy is the fundamental package for scientific computing with Python.'
    }
  },
  head: require('./configs/head'),
  theme: 'teadocs',
  themeConfig: require('./configs/themeConfig'),
  plugins: require('./configs/plugins'),
  extraWatchFiles: [
    '.vuepress/nav/en.js',
    '.vuepress/nav/zh.js',
    '.vuepress/sidebar/article_zh.js',
    '.vuepress/sidebar/bedocs_en.js',
    '.vuepress/sidebar/bedocs_zh.js',
    '.vuepress/sidebar/deep_zh.js',
    '.vuepress/sidebar/dev_en.js',
    '.vuepress/sidebar/dev_zh.js',
    '.vuepress/sidebar/f2py_zh.js',
    '.vuepress/sidebar/f2py_en.js',
    '.vuepress/sidebar/reference_en.js',
    '.vuepress/sidebar/reference_zh.js',
    '.vuepress/sidebar/user_en.js',
    '.vuepress/sidebar/user_zh.js',
    '.vuepress/configs/head.js',
    '.vuepress/configs/plugins.js',
    '.vuepress/configs/themeConfig.js'
  ]
});