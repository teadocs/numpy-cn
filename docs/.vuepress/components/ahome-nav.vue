<template>
  <div class="ahome-nav">
    <div class="nav-bar">
      <b class="name">我想要</b>
      <a class="btn-home-nav" :class="{'active': navId === 1}" @click="navId=1">关注公众号</a>
      <a class="btn-home-nav" :class="{'active': navId === 2}" @click="navId=2">加入群聊</a>
      <a class="btn-home-nav" :class="{'active': navId === 3}" @click="navId=3">贡献翻译</a>
      <a class="btn-home-nav donation" :class="{'active': navId === 4}" @click="navId=4">捐赠网站</a>
      <a class="btn-home-nav" :class="{'active': navId === 5}" @click="navId=5">其他事宜</a>
    </div>

    <!-- 关注公众号 -->
    <div class="block" :class="{'show': navId === 1}">
      <div class>
        <p>掌上中文数据科学社区，关注公众号获取各种Python的奇淫技巧、赚钱技巧，更有机会获得大厂内推。</p>
      </div>
      <div class="qr-item">
        <img src="https://extraimage.net/images/2019/09/23/ddf886b34735a450f51e5612681c1c18.png" alt="数量集公众号二维码" />
        <span class="text">或搜索“<b>数量集</b>”</span>
      </div> 
    </div>

    <!-- 加入群聊 -->
    <div class="block" :class="{'show': navId === 2}">
      <div class>
        <p>欢迎加入 <b>NumPy 中文社区</b> 群聊，群内有BAT公司大牛、常春藤校友、中国顶尖高校的各类学霸，以及众多大佬！</p>
      </div>
      <div class="qr-item">
        <img src="https://extraimage.net/images/2019/09/23/ab21cd448bc5613ef9dbba4e4286ac32.jpg" alt="站长微信二维码" />
        <span class="text">站长微信（可拉微信群，备注：numpy）</span>
      </div>
      <div class="qr-item">
        <img src="https://extraimage.net/images/2019/09/23/68ef79d67d2d47c3453edbeee47712b5.jpg" alt="qq群二维码" />
        <span class="text">QQ群二维码（群号：294639068）</span>
      </div>
    </div>

    <!-- 贡献翻译 -->
    <div class="block" :class="{'show': navId === 3}">
      <p>
        首先感谢
        <a href="https://github.com/teadocs/numpy-cn/graphs/contributors" target="_blank">这些小伙伴</a> 对数据科学汉化事业的做出的翻译贡献！🙏
      </p>
      <p>
        贡献翻译不需要太多门槛，想参与翻译的小伙伴，可以先查看
        <a
          href="https://github.com/teadocs/numpy-cn/blob/v0.25.0/Contribution.md"
          target="_blank"
        >《本文档翻译贡献指南》</a>。
      </p>
      <p>如果想加入汉化小组可以先添加站长的个人微信。</p>
    </div>

    <!-- 捐赠网站 -->
    <div class="block" :class="{'show': navId === 4}">
      <p>1、你的捐赠会帮助更多的国人看到优质的保持<b>免费</b>且<b>无广告</b>的内容！</p>
      <p>
        2、维护公益项目不易，你们的支持是我
        <b>坚持翻译</b>，不断优化
        <b>网站内容</b> 和
        <b>阅读体验</b> 的动力！
      </p>
      <p>捐赠数额不限，特大数额可以加入网站鸣谢列表或全站推荐。</p>
      <div class="qr-item">
        <img src="https://extraimage.net/images/2019/09/23/b23a24d371aeacdc1be5eb110436ecb9.jpg" alt="微信付款码" />
        <span class="text">
          <b>微信</b> 捐赠
        </span>
      </div>
      <div class="qr-item">
        <img src="https://extraimage.net/images/2019/09/23/4136d66b8cad62517838b702f57ac7da.jpg" alt="支付宝付款码" />
        <span class="text">
          <b>支付宝</b> 捐赠
        </span>
      </div>
    </div>

    <!-- 其他事宜 -->
    <div class="block" :class="{'show': navId === 5}">
      <p>如果您想要交换友情链接、广告赞助、转载内容、咨询网站主题、或者其他合作事宜，可以通过以下联系方式联系到站长。</p>
      <ul>
        <li>🍀微信个人号：iamnotahacker（备注：合作）</li>
        <li>🍁QQ号：17560235</li>
        <li>🌿邮箱：l@numpy.org.cn</li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      navId: 0
    };
  }
};
</script>

<style lang="stylus" scoped>
.ahome-nav {
  margin-top: 1.5rem;

  p {
    margin-top: 0.4rem;
    margin-bottom: 0.4rem;
  }

  a {
    &.btn-home-nav {
      position: relative;
      top: -1px;
      display: inline-block;
      margin-left: 2px;
      margin-right: 2px;
      border: 1px solid $accentColor;
      border-radius: 25px;
      font-size: 14px;
      padding-left: 12px;
      padding-right: 12px;
      cursor: pointer;

      &.active {
        background-color: $accentColor;
        color: #ffffff;
      }

      &:hover {
        background-color: $accentColor;
        color: #ffffff;
      }

      &.donation {
        border: 2px solid $accentColor;
      }
    }
  }

  .block {
    padding-top: 10px;
    box-sizing: border-box;
    transition: opacity 0.8s;
    opacity: 0;
    visibility: hidden;
    position: absolute;

    &.show {
      position: static;
      visibility: visible;
      opacity: 1;
    }

    .qr-item {
      padding-top: 10px;
      padding-right: 10px;
      text-align: center;
      width: 150px;
      float: left;

      img {
        height: 150px;
        width: 150px;
      }

      .text {
        display: inline-block;
        margin-top: 5px;
        font-size: 14px;
      }
    }

    &:after {
      content: '';
      display: block;
      height: 0;
      clear: both;
      visibility: hidden;
    }
  }
}

@media (max-width: 475px) {
  .ahome-nav {
    .nav-bar {
      .name {
        display: none;
      }
    }

    a {
      &.btn-home-nav {
      }
    }
  }
}

@media (max-width: 395px) {
  .ahome-nav {
    margin-top: 0.5rem;

    a {
      &.btn-home-nav {
        margin: 2px;
        margin-bottom 6px;
      }
    }
  }
}
</style>