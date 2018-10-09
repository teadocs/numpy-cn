(function () {
    var pathName = window.location.pathname;
    if (pathName !== '/' && pathName !== '/index.html') {

        function randomNum(minNum,maxNum) {
            switch(arguments.length){ 
                case 1: 
                    return parseInt(Math.random()*minNum+1,10); 
                break; 
                case 2: 
                    return parseInt(Math.random()*(maxNum-minNum+1)+minNum,10); 
                break; 
                    default: 
                        return 0; 
                    break; 
            } 
        }

        (function() {
            window._ggjump = function (ggType) {
                var ggUrls = [
                    escape("http://www.julyedu.com/weekend/train6?from=numpy"),
                    escape("https://ke.qq.com/course/326311?flowToken=1005515")
                ];
                var adJump = "http://gg.numpy.org.cn/jump.php?from=" + window.location.href + "&url=" + ggUrls[ggType];
                window.open(adJump);
            }
        })();
        //七月
        (function() {
            var ggStyle = window.document.createElement("style");
            ggStyle.innerHTML = "#gg-box1{max-width:900px;padding-left:60px;padding-bottom:20px;padding-right:60px;box-sizing:border-box;text-align:center;background-color:#fff}#gg-box1 img{height:100%;width:100%;opacity:.7;cursor:pointer;transition:opacity .4s}#gg-box1 img:hover{opacity:.9}"
            document.querySelector(".tea-container").appendChild(ggStyle);
            var gg = window.document.createElement("div");
            gg.id = "gg-box1";
            gg.innerHTML = "<img src='/static/images/gg-qiyue-1.png' onclick='_ggjump(0)' />";
            document.querySelector(".tea-container").appendChild(gg);
        })();
        //楚才国科
        (function() {
            if ( randomNum(0, 10) > 8 ) {
                return;
            }
            var articleChildren = $("article.markdown-body").children();
            var articleChildrenLen = articleChildren.length;
            if ( articleChildrenLen < 8 ) {
                return;
            }
            var lenHalf = Math.floor(articleChildrenLen / 2);
            var beEle = articleChildren.eq(lenHalf);
            var ggStyle = window.document.createElement("style");
            ggStyle.innerHTML = "#gg-box2{margin-bottom:10px;max-width:900px;box-sizing:border-box;text-align:center;background-color:#fff}#gg-box2 img{height:100%;width:100%;opacity:.7;cursor:pointer;transition:opacity .4s}#gg-box2 img:hover{opacity:.9}"
            document.querySelector(".tea-container").appendChild(ggStyle);
            var gg = window.document.createElement("div");
            gg.id = "gg-box2";
            gg.innerHTML = "<img src='/static/images/gg-cgqc.jpg' onclick='_ggjump(1)' />";
            $(beEle).after(gg);
        })();
    }
})();