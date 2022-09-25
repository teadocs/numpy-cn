import teadocThemeConfig from './configs/themeConfig'
import customPlugin from './configs/plugins'
import { head, navbarZh, navbarEn, sidebarZh, sidebarEn } from './configs'
import { DefaultThemeLocaleData, defineUserConfig } from 'vuepress'
import { defaultTheme } from 'vuepress-theme-teadocs'

export default defineUserConfig({

  dest: './dist',

  base: '/',

  head,

  locales: {
    '/': {
      lang: 'zh-CN',
      title: 'NumPy',
      subTitle: '中文网',
      description: '这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包。'
    },
    '/en/': {
      lang: 'en-US',
      title: 'NumPy',
      subTitle: '中文网',
      description: 'NumPy is the fundamental package for scientific computing with Python.'
    }
  },

  theme: defaultTheme({
    logo: 'https://static.numpy.thto.net/site/logo.png@w50h50',
    repo: 'teadocs/numpy-cn',
    docsDir: 'docs',
    editLink: true,
    colorModeSwitch: false,
    alert: [{
      id: '2019-7-29',
      title: '文档公告',
      content: `我们经常发布文档更新，部分页面的翻译可能仍在进行中。有关最新信息，请访问<a href="/en/">英文文档</a>。如果某个页面上的翻译有问题，请提issues<a href="https://github.com/teadocs/numpy-cn/issues" target="_blank">告诉我们</a>。`
    }],
    // 侧面板配置
    sidePanel: {
      enable: false,
      btnName: '快捷聊天室',
      title: 'NumPy 爱好者'
    },
    locales: {
      '/': {
        selectLanguageName: '简体中文',
        selectLanguageText: '选择语言',
        selectLanguageAriaLabel: '选择语言',
        editLinkText: '在 GitHub 上编辑此页',
        contributors: false,
        lastUpdated: true,
        lastUpdatedText: '上次更新',
        navbar: navbarZh,
        sidebarDepth: 3,
        sidebar: sidebarZh,
      },
      '/en/': {
        selectLanguageName: 'English',
        selectLanguageText: 'Languages',
        selectLanguageAriaLabel: 'Languages',
        editLinkText: 'Edit this page on GitHub',
        contributors: false,
        lastUpdated: true,
        lastUpdatedText: 'Last Updated',
        navbar: navbarEn,
        sidebarDepth: 3,
        sidebar: sidebarEn,
      },
    }
  }),
  plugins: customPlugin,
  // extraWatchFiles: [
  //   '.vuepress/nav/en.js',
  //   '.vuepress/nav/zh.js',
  //   '.vuepress/sidebar/article_zh.js',
  //   '.vuepress/sidebar/bedocs_en.js',
  //   '.vuepress/sidebar/bedocs_zh.js',
  //   '.vuepress/sidebar/deep_zh.js',
  //   '.vuepress/sidebar/dev_en.js',
  //   '.vuepress/sidebar/dev_zh.js',
  //   '.vuepress/sidebar/f2py_zh.js',
  //   '.vuepress/sidebar/f2py_en.js',
  //   '.vuepress/sidebar/reference_en.js',
  //   '.vuepress/sidebar/reference_zh.js',
  //   '.vuepress/sidebar/user_en.js',
  //   '.vuepress/sidebar/user_zh.js',
  //   '.vuepress/configs/head.js',
  //   '.vuepress/configs/plugins.js',
  //   '.vuepress/configs/themeConfig.js'
  // ]
});
