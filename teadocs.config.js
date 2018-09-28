'use strict';
const path = require('path')

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
            var cy = window.document.createElement("div");
            cy.id = "SOHUCS";
            cy.style.maxWidth = "900px";
            cy.style.padding = "20px";  
            cy.style.boxSizing = "border-box";
            cy.style.margin = "0px";
            cy.style.backgroundColor = "#fff";
            document.querySelector(".tea-container").appendChild(cy);
        })()
        </script>
        <script type = "text/javascript" > (function() {
            var appid = 'cytG5UKih';
            var conf = 'prod_34fc698f6822858dad2f00c2bc25354e';
            var width = window.innerWidth || document.documentElement.clientWidth;
            if (width < 960) {
                window.document.write('<script id="changyan_mobile_js" charset="utf-8" type="text/javascript" src="http://changyan.sohu.com/upload/mobile/wap-js/changyan_mobile.js?client_id=' + appid + '&conf=' + conf + '"><\\\/script>');
            } else {
                var loadJs = function(d, a) {
                    var c = document.getElementsByTagName("head")[0] || document.head || document.documentElement;
                    var b = document.createElement("script");
                    b.setAttribute("type", "text/javascript");
                    b.setAttribute("charset", "UTF-8");
                    b.setAttribute("src", d);
                    if (typeof a === "function") {
                        if (window.attachEvent) {
                            b.onreadystatechange = function() {
                                var e = b.readyState;
                                if (e === "loaded" || e === "complete") {
                                    b.onreadystatechange = null;
                                    a();
                                }
                            }
                        } else {
                            b.onload = a;
                        }
                    }
                    c.appendChild(b);
                };
                loadJs("https://changyan.sohu.com/upload/changyan.js",
                function() {
                    window.changyan.api.config({
                        appid: appid,
                        conf: conf
                    });
                    var timer = setInterval(function() {
                         var adEle = document.getElementById('feedAv') 
                         if ( adEle ) {
                            document.getElementById('feedAv').id="feedAvBak";
                            document.getElementById('feedAvBak').style.display = "none";
                            window.clearInterval(timer);
                         }
                    });
                });
            }
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