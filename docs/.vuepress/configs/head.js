module.exports = [
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
    content: '#489dc1'
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
  ['script', {
    src: 'https://hm.baidu.com/hm.js?a809b6f7e6517af8c15c6076273e80fe',
    defer: 'defer',
    async: 'true'
  }],
  // 谷歌统计
  ['script', {
    src: 'https://www.googletagmanager.com/gtag/js?id=UA-163860037-1',
    defer: 'defer',
    async: 'true'
  }],
  // 谷歌统计第二段代码
  ['script', {}, `
  (function() {
    document.addEventListener('readystatechange', function (e) {
      if (document.readyState == 'complete') {
        console.log('google init.');
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'UA-163860037-1');
      }
    });
  })();
`],
// 广告系统
['script', {}, `
  (function() {
    document.write("<s"+"cript defer='defer' type='text/javascript' src='https://analytics.numpy.org.cn/public/ad.js?"+Math.random()+"'></scr"+"ipt>"); 
  })();
`],
// 屏蔽评论
['script', {}, `
(function () {
  let styleContent = \`
  .forum-tips {
    position: absolute;
    left: 0px;
    top: 0px;
    width: 100%;
    height: 100%;
    background-color: rgba(255, 255, 255, 0.92);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 100;
  }
  \`
  let divContent = \`
    <a href="https://www.kuxai.com/f/numpy" target="_blank">
      评论系统已退役，点击进入<b>Numpy中文社区</b>，体验功能更强大的交流社区。
    </a>
  \`
  function asyncQuerySelector(selector, callback) {
    let el = '';
    let timer = window.setInterval(function() {
      el = document.querySelector(selector);
      if (el) {
        window.clearInterval(timer);
        callback(el);
      }
    });
  }
  function init() {
    asyncQuerySelector('#valine-vuepress-comment > .vwrap', function (parentNode) {
      if (!document.querySelector('.forum-tips')) {
        let newStyle = document.createElement('style');
        newStyle.innerHTML = styleContent;
        parentNode.appendChild(newStyle);
        let newDiv = document.createElement('div');
        newDiv.className = 'forum-tips';
        newDiv.innerHTML = divContent;
        parentNode.appendChild(newDiv);
      }
    });
  }
  document.addEventListener('readystatechange', function (e) {
    if (document.readyState == 'complete') {
      init();
    }
  });
  (function() {
    var url1 = window.location.href;
    var url2 = window.location.href;
    setInterval(function() {
      if (url1 === url2) {
        url2 = window.location.href;
      } else {
        url1 = url2;
        init();
      }
    }, 200);
  })();
})();
`]
] 
