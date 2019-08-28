<template>
  <div class="home-wxpub">
    <h3>值得一看</h3>
    <div class="recommend">
      <ul>
        <li v-for="(item, index) in recommend" :key="index">
          <a :href="item.url" :title="item.title" target="_blank">
            <div class="poster" :style="{'background-image': 'url(' + item.poster + ')'}"></div>
            <div class="title">{{item.title}}</div>
          </a>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      recommend: [
        {
          poster:
            "http://mmbiz.qpic.cn/mmbiz_jpg/NbtHzd5t0YCtcicW9XcHrp32oya9XdRKH4p5ZFCmSKfatgU7CHhRlvgelNXr5ze0shndR35bKzArQMiaGPspq3rA/0?wx_fmt=jpeg",
          title:
            "不要再去报班啦！纯新手入门机器/深度学习自学指南（附一个月速成方案）",
          url: "https://mp.weixin.qq.com/s/VYUGU2S-dWDCQfsl6C7oGQ"
        },
        {
          poster:
            "http://mmbiz.qpic.cn/mmbiz_jpg/NbtHzd5t0YDOnldyZKXZOpu5THWxcGEG2I1OnA94SnZ2Fw1RLPFPx4XhIF5eyp10zAM4A2lgcS19hJ7iaDJPCIw/0?wx_fmt=jpeg",
          title: "可能是最通俗易懂的Python入门资料整理和最优学习路线推荐。",
          url: "https://mp.weixin.qq.com/s/QUUYEimknwt5v3RUxIf8kQ"
        },
        {
          poster:
            "http://mmbiz.qpic.cn/mmbiz_jpg/NbtHzd5t0YD7yO1ibsjREqO5OpbqnTClnogFx3JCdGHmnV7uqzicDN6EsUe15LgWAvou4ASRl50nWo9Xg5M0v7tA/0?wx_fmt=jpeg",
          title: "程序员是如何通过开源项目月入 10 万金的？",
          url: "https://mp.weixin.qq.com/s/bgM6RkvEO95JE4SUDa4SrQ"
        },
        {
          poster:
            "http://mmbiz.qpic.cn/mmbiz_jpg/NbtHzd5t0YCJvjkIuLJrxq95dYn9eZpsHFXs9wY8kE6052oYfibzWGXuUzRKFpsRibk7HMs9hv651Nurd7By4RQw/0?wx_fmt=jpeg",
          title: "刚刚，我用Python做了个七夕礼物，差点被女朋友打死",
          url: "https://mp.weixin.qq.com/s/i2UQOqBr8zGqtgqTaHh3mg"
        }
      ]
    };
  }
};
</script>

<style lang="stylus" scoped>
.home-wxpub {
  margin-top: 2rem;
  margin-bottom: 1rem;

  h3 {
    border-left: 0.4rem solid $accentColor;
    padding-left: 0.6rem;
  }

  .recommend {
    ul {
      list-style: none;
      margin: 0px;
      padding-left: 0px;

      &::after {
        content: '.';
        display: block;
        height: 0;
        clear: both;
        visibility: hidden;
      }

      li {
        float: left;
        width: 25%;

        &:last-child {
          margin-right: 0px;
        }

        a {
          display: block;
          color: $textColor;
          font-size: 16px;
          font-weight: normal;
          padding: 8px;
          transition: all 0.5s;

          &:hover {
            color: #333;
            box-sizing: border-box;
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.12);
          }

          .poster {
            background-size: cover;
            width: 100%;
            height: 96px;
            margin-bottom: 10px;
          }

          .title {
            height: 60px;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
            overflow: hidden;
          }
        }
      }
    }
  }
}

@media (max-width: 750px) {
  .home-wxpub {
    ul {
      li {
        width: 50% !important;

        &:nth-child(odd) {
          a {
            margin-right: 5px;
          }
        }

        &:nth-child(even) {
          a {
            margin-left: 5px;
          }
        }

        a {
          padding: 0px !important;
        }
      }
    }
  }
}
</style>