'use strict';
const path = require('path');

module.exports = {
    doc: {
        name: "NumPy 中文文档",
        description: "这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包，本文档详细的提供了从用户指南到参考手册全方位的内容，用户可以通过本文档方便快速的查找NumPy的api说明以及用法。",
        version: "1.14.0",
        dir: "",
        outDir: "",
        staticDir: ""
    },
    theme: {
        dir: "", 
        title: "NumPy官方中文文档",
        headHtml: `
        <meta name="description" content="这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包，本文档详细的提供了从用户指南到参考手册全方位的内容，用户可以通过本文档方便快速的查找NumPy的api说明以及用法。" />
        <meta name="keywords" content="numpy中文文档,numpy中文api,numpy中文手册,numpy教程,numpy下载安装,numpy" />
        <link rel="shortcut icon" href="/static/favicon.ico"/>
        `,
        footHtml: `
        <script>
        var _hmt = _hmt || [];
        (function() {
          var hm = document.createElement("script");
          hm.src = "https://hm.baidu.com/hm.js?a809b6f7e6517af8c15c6076273e80fe";
          var s = document.getElementsByTagName("script")[0]; 
          setTimeout(function () {
            s.parentNode.insertBefore(hm, s);
          }, 100);
        })();
        </script>
        <script>
        (function() {
            var comments = window.document.createElement("div");
            comments.style.maxWidth = "900px";
            comments.style.backgroundColor = "#fff";
            comments.style.boxSizing = "border-box";
            comments.id = "comments";
            document.querySelector(".tea-container").appendChild(comments);
        })();
        </script>
        <script>
        (function() {
            var ipc = window.document.createElement("div");
            ipc.id = "ipcBox";
            ipc.style.fontSize = "12px";
            ipc.style.maxWidth = "900px";
            ipc.style.padding = "20px";  
            ipc.style.boxSizing = "border-box";
            ipc.style.margin = "0px";
            ipc.style.textAlign = "center";
            ipc.style.backgroundColor = "#fff";
            ipc.innerHTML = "<span style='color: #bdbdbd;'>@2018 numpy.org.cn </span><a style='color: #bdbdbd;' href='http://www.miitbeian.gov.cn/' target='_blank'>粤ICP备16025085号-3</a>"
            document.querySelector(".tea-container").appendChild(ipc);
        })();
        </script>
        <script>
        (function() {
            var script = document.createElement("script");
            script.src = "/gg.js";
            document.body.appendChild(script);
        })();
        </script>
        <script>
        (function() {
            var script = document.createElement("script");
            script.src = "https://code.tellto.cn/dist/js/init.min.js";
            script.setAttribute('data-el', '#comments');
            document.body.appendChild(script);
        })();
        </script>
        `,
        isMinify: true, 
        rootPath: "/"
    },
    nav: {
        tree: "./tree"
    }
}