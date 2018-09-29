'use strict';
const path = require('path');
const adJump = "window.open(\\\"http:\/\/gg.numpy.org.cn\/jump.php?from=\\\"+window.location.href+\\\"&url=http%3a%2f%2fwww.julyedu.com%2fweekend%2ftrain6%3ffrom%3dnumpy\\\")";

module.exports = {
    doc: {
        name: "NumPy 中文文档",
        description: "这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包。",
        version: "1.14.0",
        dir: "",
        outDir: "",
        staticDir: ""
    },
    theme: {
        dir: "", 
        title: "NumPy官方中文文档",
        headHtml: `
        <meta name="description" content="这是NumPy官方的中文文档，NumPy是用Python进行科学计算的基础软件包。" />
        <meta name="keywords" content="numpy中文文档,numpy中文api,numpy中文手册,numpy中文教程,numpy" />
        <link rel="shortcut icon" href="/static/favicon.ico"/>
        <style>
            #gg-box {
                height: 130px;
                width: 900px;
            }

            #gg-box img {
                opacity: 0.7;
                cursor: pointer;
                transition: opacity 0.4s;
            }

            #gg-box img:hover {
                opacity: 0.9;
            }

            @media screen and (max-width: 414px) {
                #gg-box {
                    display: none;
                }
            }

        </style>
        `,
        footHtml: `
        <script>
        var _hmt = _hmt || [];
        (function() {
          var hm = document.createElement("script");
          hm.src = "https://hm.baidu.com/hm.js?a809b6f7e6517af8c15c6076273e80fe";
          var s = document.getElementsByTagName("script")[0]; 
          s.parentNode.insertBefore(hm, s);
        })();
        </script>
        <script>
        (function() {
            var gg = window.document.createElement("div");
            gg.id = "gg-box";
            gg.style.boxSizing = "border-box";
            gg.style.textAlign = "center";
            gg.style.backgroundColor = "#fff";
            gg.innerHTML = "<img style='height:120px;width:800px;' src='/static/images/gg-qiyue-1.png' onclick='${adJump}' />"
            document.querySelector(".tea-container").appendChild(gg);
        })();
        </script>
        <script>
        (function() {
            var ipc = window.document.createElement("div");
            ipc.id = "ipcBox";
            ipc.style.fontSize = "14px";
            ipc.style.maxWidth = "900px";
            ipc.style.padding = "20px";  
            ipc.style.boxSizing = "border-box";
            ipc.style.margin = "0px";
            ipc.style.textAlign = "center";
            ipc.style.backgroundColor = "#fff";
            ipc.innerHTML = "<span>@2018 numpy.org.cn </span><a href='http://www.miitbeian.gov.cn/' target='_blank'>粤ICP备16025085号-3</a>"
            document.querySelector(".tea-container").appendChild(ipc);
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