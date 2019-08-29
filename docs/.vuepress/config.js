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
  head: [
    ['link', {
      rel: 'dns-prefetch',
      href: `//cdn.bootcss.com`
    }],
    ['link', {
      rel: 'dns-prefetch',
      href: `//cdn.mathjax.org`
    }],
    // 使主题能够支持数学公式
    ['script', {
      type: 'text/x-mathjax-config'
    }, `
    MathJax.Hub.Config({
      showProcessingMessages: false, //关闭js加载过程信息
      messageStyle: "none", //不显示信息
      tex2jax: {
        "inlineMath": [["$", "$"], ["\\\\(", "\\\\)"]], 
        "processEscapes": true, 
        "ignoreClass": "document",
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'a', 'td'],
        "processClass": "math|output_area"
      },
      "HTML-CSS": {
        showMathMenu: false
      }
    })
    `],
    ['script', {
      src: '//cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
    }],
    // 监听路由重新渲染数学公式
    ['script', {}, `
      (function() {
        var url1 = window.location.href;
        var url2 = window.location.href;
        setInterval(function() {
          if (url1 === url2) {
            url2 = window.location.href;
          } else {
            url1 = url2;
            if (window.MathJax) window.MathJax.Hub.Typeset();
          }
        }, 200);
      })();
    `],
    ['link', {
      rel: 'icon',
      href: `/logo.png`
    }],
    ['link', {
      rel: 'manifest',
      href: '/manifest.json'
    }],
    ['meta', {
      name: 'theme-color',
      content: '#3eaf7c'
    }],
    ['meta', {
      name: 'apple-mobile-web-app-capable',
      content: 'yes'
    }],
    ['meta', {
      name: 'apple-mobile-web-app-status-bar-style',
      content: 'black'
    }],
    ['link', {
      rel: 'apple-touch-icon',
      href: `/icons/apple-touch-icon-152x152.png`
    }],
    ['link', {
      rel: 'mask-icon',
      href: '/icons/safari-pinned-tab.svg',
      color: '#3eaf7c'
    }],
    ['meta', {
      name: 'msapplication-TileImage',
      content: '/icons/msapplication-icon-144x144.png'
    }],
    ['meta', {
      name: 'msapplication-TileColor',
      content: '#000000'
    }],
    // 百度统计
    ['script', {}, `
      (function() {
        var onDocumentComplete = function() { 
          if (document.readyState =='complete') {
            var hm = document.createElement("script");
            hm.src = "https://hm.baidu.com/hm.js?a809b6f7e6517af8c15c6076273e80fe";
            var s = document.getElementsByTagName("script")[0]; 
            s.parentNode.insertBefore(hm, s);
          }
        }
        document.onreadystatechange = onDocumentComplete; //当页面加载状态改变的时候执行这个方法. 
      })();
    `],
  ],
  theme: 'teadocs',
  themeConfig: {
    logo: {
      text: 'NumPy',
      subText: '中文网',
      image: ''
    },
    alert: [{
      id: '2019-7-29',
      title: '文档公告',
      content: `我们经常发布文档更新，部分页面的翻译可能仍在进行中。有关最新信息，请访问<a href="/en/">英文文档</a>。如果某个页面上的翻译有问题，请提issues<a href="https://github.com/teadocs/numpy-cn/issues" target="_blank">告诉我们</a>。`
    }],
    repo: 'teadocs/numpy-cn',
    editLinks: true,
    docsDir: 'docs',
    locales: {
      '/': {
        label: '简体中文',
        selectText: '选择语言',
        editLinkText: '在 GitHub 上编辑此页',
        lastUpdated: '上次更新',
        nav: require('./nav/zh'),
        sidebar: {
          '/user/': require('./sidebar/user_zh')(),
          '/reference/': require('./sidebar/reference_zh')(),
          '/f2py': require('./sidebar/f2py_zh')(),
          '/dev': require('./sidebar/dev_zh')(),
          '/bedocs': require('./sidebar/bedocs_zh')()
        }
      },
      '/en/': {
        label: 'English',
        selectText: 'Languages',
        editLinkText: 'Edit this page on GitHub',
        lastUpdated: 'Last Updated',
        nav: require('./nav/en'),
        sidebar: {
          '/en/user/': require('./sidebar/user_en')(),
          '/en/reference/': require('./sidebar/reference_en')(),
          '/en/f2py': require('./sidebar/f2py_en')(),
          '/en/dev': require('./sidebar/dev_en')(),
          '/en/bedocs': require('./sidebar/bedocs_en')()
        }
      }
    }
  },
  plugins: [
    ['@vuepress/back-to-top', true],
    ['@vuepress/pwa', {
      serviceWorker: true,
      updatePopup: {
        '/': {
          message: "发现新内容可用",
          buttonText: "刷新"
        },
        '/en/': {
          message: "New content is available.",
          buttonText: "Refresh"
        }
      }
    }],
    ['@vuepress/medium-zoom', true],
    ['container', {
      type: 'vue',
      before: '<pre class="vue-container"><code>',
      after: '</code></pre>',
    }],
    ['container', {
      type: 'upgrade',
      before: info => `<UpgradePath title="${info}">`,
      after: '</UpgradePath>',
    }],
  ],
  extraWatchFiles: [
    '.vuepress/nav/en.js',
    '.vuepress/nav/zh.js',
    '.vuepress/sidebar/user_en.js',
    '.vuepress/sidebar/user_zh.js',
    '.vuepress/sidebar/reference_zh.js',
    '.vuepress/sidebar/reference_zh.js'
  ]
})