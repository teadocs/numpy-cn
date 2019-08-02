<template>
  <div class="home-footer">
    <div class="left-box item">
      <h4>🌐站点相关</h4>
      <ul>
        <li>
          <a href="/logo/">Pandas logo 捐赠名单 👍</a>
        </li>
        <li>
          <a href="http://www.beian.miit.gov.cn/" target="_blank">粤ICP备16025085号-8</a>
        </li>
      </ul>
    </div>
    <div class="left-box item">
      <h4>👥兄弟文档</h4>
      <ul>
        <li>
          <a href="https://www.numpy.org.cn/" target="_blank">NumPy 中文文档</a>
        </li>
        <li>
          <a href="https://www.matplotlib.org.cn/" target="_blank">Matplotlib 中文文档</a>
        </li>
      </ul>
    </div>
    <div class="right-box item">
      <h4>👬友情链接</h4>
      <ul>
        <li>
          <a href="https://www.leiphone.com/" target="_blank">雷锋网</a>
        </li>
        <li>
          <a href="https://ai.yanxishe.com/?from=pandas" target="_blank">AI研习社</a>
        </li>
      </ul>
    </div>
  </div>
</template>

<style lang="stylus" scoped>
.home-footer
  margin-top 30px
  padding-top 20px
  padding-bottom 20px
  border-top 1px solid #f8f8f8

  &:after
    content ""
    display block
    height 0
    clear both
    visibility hidden
  
  .item
    width 33.3%
    ul
      padding 0px
      list-style none
      li
        line-height 1.5
        a
          font-size 14px !important
          transition color 0.2s
          &:hover
            color #00502c

  .left-box
    float left

  .right-box
    float right

@media (max-width: 650px)
  .home-footer
    .item
      width 100%;
</style>