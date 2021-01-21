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
const staticDomain = 'https://static.numpy.org.cn/';

export default {
  data() {
    return {
      recommend: [
        {
          poster: staticDomain + "home/li.jpg@w300h300",
          title: "你离开学只差这个视频：李宏毅机器学习2020版正式开放上线",
          url: "https://www.kuxai.com/article/2"
        },
        {
          poster: staticDomain + "home/article_poster_02.jpg@w300h300",
          title: "可能是最通俗易懂的Python入门资料整理和最优学习路线推荐。",
          url: "https://mp.weixin.qq.com/s/QUUYEimknwt5v3RUxIf8kQ"
        },
        {
          poster: staticDomain + "home/article_poster_03.jpg@w300h300",
          title: "几百个不同领域的数据集分享给你，别让巧妇难为无米之炊。",
          url: "https://mp.weixin.qq.com/s/lVKa2j2yXbjpjom4GD5Btw"
        },
        {
          poster: staticDomain + "home/article_poster_04.jpg@w300h300",
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