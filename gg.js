(function () {
    var pathName = window.location.pathname;
    (function() {
        window._ggjump = function (ggType) {
            var ggUrls = [
                escape("http://www.julyedu.com/weekend/train6?from=numpy"),
                escape("https://ke.qq.com/course/326311?flowToken=1005515"),
                escape("https://1024dada.com/?channel=numpy&hmsr=numpy-cn&hmpl=&hmcu=&hmkw=&hmci=")
            ];
            var adJump = "http://gg.numpy.org.cn/jump.php?from=" + window.location.href + "&url=" + ggUrls[ggType];
            window.open(adJump);
        }
    })();

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

        //七月
        // (function() {
        //     var ggStyle = window.document.createElement("style");
        //     ggStyle.innerHTML = "#gg-box1{max-width:900px;padding-left:60px;padding-bottom:20px;padding-right:60px;box-sizing:border-box;text-align:center;background-color:#fff}#gg-box1 img{height:100%;width:100%;opacity:.7;cursor:pointer;transition:opacity .4s}#gg-box1 img:hover{opacity:.9}"
        //     document.querySelector(".tea-container").appendChild(ggStyle);
        //     var gg = window.document.createElement("div");
        //     gg.id = "gg-box1";
        //     gg.innerHTML = "<img src='/static/images/gg-qiyue-1.png' onclick='_ggjump(0)' />";
        //     document.querySelector(".tea-container").appendChild(gg);
        // })();
        //楚才国科
        // (function() {
        //     if ( randomNum(0, 10) > 8 ) {
        //         return;
        //     }
        //     var articleChildren = $("article.markdown-body").children();
        //     var articleChildrenLen = articleChildren.length;
        //     if ( articleChildrenLen < 8 ) {
        //         return;
        //     }
        //     var lenHalf = Math.floor(articleChildrenLen / 2);
        //     var beEle = articleChildren.eq(lenHalf);
        //     var ggStyle = window.document.createElement("style");
        //     ggStyle.innerHTML = "#gg-box2{margin-bottom:10px;max-width:900px;box-sizing:border-box;text-align:center;background-color:#fff}#gg-box2 img{height:100%;width:100%;opacity:.7;cursor:pointer;transition:opacity .4s}#gg-box2 img:hover{opacity:.9}"
        //     document.querySelector(".tea-container").appendChild(ggStyle);
        //     var gg = window.document.createElement("div");
        //     gg.id = "gg-box2";
        //     gg.innerHTML = "<img src='/static/images/gg-cgqc.jpg' onclick='_ggjump(1)' />";
        //     $(beEle).after(gg);
        // })();

    }

    //每日答答
    // (function() {
    //     var ggStyle = window.document.createElement("style");
    //     ggStyle.innerHTML = " #gg-1024dada-box{cursor: pointer;width: 120px;height: 500px;position: fixed;left: 1160px;top: 240px;background-color: #fff;z-index: 20;}#gg-1024dada-box img{opacity: 0.6;width: 100%;height: 100%;transition:opacity .4s;}#gg-1024dada-box img:hover{opacity: 1;} #gg-1024dada-box div.title{position: absolute; width: 100%; height: 30px; line-height: 30px; top: -30px; background-color: #fff; text-align: center;border: 1px solid #e0e0e0; box-sizing: border-box;}"
    //     document.querySelector(".tea-container").appendChild(ggStyle);
    //     var gg = window.document.createElement("div");
    //     gg.id = "gg-1024dada-box";
    //     gg.innerHTML = "<img src='/static/images/1024dada120x500.jpg' onclick='_ggjump(2)' /><div class='title'>赞 助 商</div>";
    //     document.querySelector(".tea-container").appendChild(gg);
    // })();
    
    //加载捐增浮窗
    var donationStyleHtml = '<style>' +
    '.donation-button {' +
        'position: fixed;' +
        'left: 1160px;' +
        'top: 70px;' +
        'background: #fff;' +
        'text-align: center;' +
        'padding: 8px;' +
        'font-weight: bold;' +
        'line-height: 1.4;' +
        'border-radius: 16px;' +
        'cursor: pointer;' +
        'transition: all 0.4s;' +
        'border: 1px solid #e0e0e0;' +
    '}' +
    '.donation-button:hover {' +
        'color: #b5a431;' +
        'box-shadow: 0px 0px 20px #b5a431;' +
    '}' +
    '.donation-mask {' +
        'position: fixed;' +
        'z-index: 1000;' +
        'top: 0;' +
        'left: 0;' +
        'bottom: 0;' +
        'right: 0;' +
        'background: rgba(0, 0, 0, 0.5);' +
        'transform: translateZ(1px);' +
    '}' +
    '.donation-mask .donation-pop {' +
        'opacity: 0;' +
        'transition: all 0.5s;' +
        'position: absolute;' +
        'top: 50%;' +
        'left: 50%;' +
        'padding: 30px;' +
        'background: #fff;' +
        'border-radius: 6px;' +
        'transform: translate(-50%, -50%);' +
    '}' +
    '.donation-mask .donation-pop .text {' +
        'font-size: 14px;' +
        'width: 310px;' +
        'padding-bottom: 15px;' +
    '}' +
    '.donation-mask .donation-pop.show {' +
        'opacity: 1;' +
    '}' +
    '.donation-mask .donation-close {' +
        'position: absolute;' +
        'top: 5px;' +
        'right: 10px;' +
        'font-size: 30px;' +
        'font-weight: 300;' +
        'cursor: pointer;' +
    '}' +
    '</style>';

    var donationButtonHtml = '<div class="donation-button">' + 
    '<i class="fa fa-slideshare"></i><br> ' + 
    '捐<br>' + 
    '赠<br>' + 
    '文<br>' + 
    '档<br>' + 
    '</div>';

    var showPop = function() {
        var closeHtml = '<span class="donation-close">×</span>';
        var maskHtml = '<div class="donation-mask"></div>';
        var popHtml = '<div class="donation-pop">' +
                    '<div class="donation-pop-body">' +
                        '<div class="text">希望所有学习NumPy的朋友都能够受益良多。NumPy中文文档的正常运转离不开广大NumPy友的支持！</div>' + 
                        '<table style="border-collapse: collapse;">' + 
                            '<thead>' +
                                '<tr>' +
                                    '<th style="text-align: center;border: 1px solid #404c58;background: #fff;">微信</th>' +
                                    '<th style="text-align: center;border: 1px solid #404c58;background: #fff;">支付宝</th>' +
                                '</tr>' +
                            '</thead>' +
                            '<tbody>' +
                                '<tr>' +
                                    '<td style="text-align: center;border: 1px solid #404c58;"><a target="_blank" rel="noopener noreferrer" href="/static/images/wechat-qr.jpg"><img src="/static/images/wechat-qr.jpg" width="150" style="max-width:100%;"></a></td>' + 
                                    '<td style="text-align: center;border: 1px solid #404c58;"><a target="_blank" rel="noopener noreferrer" href="/static/images/alipay-qr.jpg"><img src="/static/images/alipay-qr.jpg" width="150" style="max-width:100%;"></a></td>' +
                                '</tr>' + 
                            '</tbody>' +
                        '</table>'
                    '</div>' +
                    '</div>';
        var maskEl = $(maskHtml);
        var popEl = $(popHtml);
        var closeEl = $(closeHtml);
        popEl.append(closeEl);
        maskEl.append(popEl);
        $("body").append(maskEl);
        $(closeEl).on("click", function() {
            popEl.removeClass("show");
            setTimeout(function() {
                $(maskEl).remove();
            }, 400);
        });
        setTimeout(function() {
            popEl.addClass("show");
        }, 200);
    }

    var donationButtonEle = $(donationButtonHtml);
    $("body").append($(donationStyleHtml));
    $("body").append(donationButtonEle);
    donationButtonEle.on('click', function() {
        showPop();
    });
})();