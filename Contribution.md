# 翻译贡献指南

想要参与翻译的小伙伴，请注意。

参与翻译的提前技能有：[GitHub](https://zh.wikipedia.org/wiki/Github)、[Git](https://zh.wikipedia.org/zh-hans/Git)、[Markdown](https://zh.wikipedia.org/wiki/Markdown)、[命令行操作](https://zh.wikipedia.org/wiki/%E5%91%BD%E4%BB%A4%E8%A1%8C)、[英语](https://zh.wikipedia.org/wiki/%E8%8B%B1%E8%AF%AD)（[Google翻译](https://translate.google.cn/)也可）

## Git教程

这里推荐一个很不错的开源Git教程：[https://github.com/geeeeeeeeek/git-recipes](https://github.com/geeeeeeeeek/git-recipes)

## Markdown教程

这里推荐一个不错的Markdown教程：[https://www.appinn.com/markdown/](https://www.appinn.com/markdown/)

在线编辑器推荐使用这个：[https://pandao.github.io/editor.md/](https://pandao.github.io/editor.md/)

## Github 贡献指南

了解github最好的方式是直接看[github官方的中文教程](https://help.github.com/cn)。

想要贡献翻译，务必请熟知github的贡献规则，可以参看[怎么在GitHub上为开源项目作贡献？](https://zhuanlan.zhihu.com/p/23457016)。

关于与原作者仓库同步的问题，强烈推荐一篇由Pandas中文文档翻译小组团队成员 [@Y-sir](https://github.com/Y-sir) 编写的文章：[Github上Fork别人的仓库，怎么保持自己的仓库内容和原始仓库同步](http://www.ysir308.com/archives/827)。

## 文档如何在本地跑起来？

Pandas 中文文档的最新版本使用的是 [VuePress](https://v1.vuepress.vuejs.org/zh/) 文档生成工具来驱动的。

由于 [VuePress](https://v1.vuepress.vuejs.org/zh/) 是基于 [Nodejs](https://zh.wikipedia.org/wiki/Node.js) 编写的工具，如果你想让文档在本地运行调试，你首先需要安装 [Nodejs](http://nodejs.cn/)  在你的电脑上，非Windows操作系统推荐使用 [nvm](https://github.com/nvm-sh/nvm/blob/master/README.md) 来安装 [Nodejs](http://nodejs.cn/) ，Windows操作系统的小伙伴可以直接下载最新版本的 Nodejs 的 [Windows 安装包](http://nodejs.cn/download/) 。

### 文档命令行说明

请先打开命令行或者终端工具，然后切换到文档所在的目录，然后运行以下功能命令。

#### 安装文档工具的依赖

这是拿到文档之后的第一步。

这个命令的作用是安装工具的依赖包，这是拿到文档之后**第一次**必须要运行的命令，且**只需要运行一次**，之后再翻译**无需**运行此命令。

``` bash
$ npm install
```

#### 运行本地环境

想要查看翻译的效果，可以运行下面这个命令。

``` bash
$ npm run dev
```

#### 生成静态html文件

``` bash
$ npm run build
```
