<template>
  <div class="home-footer">
    <div class="left-box item">
      <h4>🌐站点相关</h4>
      <ul>
        <li>
          <a href="http://www.beian.miit.gov.cn/" target="_blank">粤ICP备16025085号-3</a>
        </li>
      </ul>
    </div>

    <div class="left-box item">
      <h4>👥兄弟文档</h4>
      <ul>
        <li>
          <a href="https://www.pypandas.cn/" target="_blank">Pandas 中文网</a>
        </li>
        <li>
          <a href="https://www.matplotlib.org.cn/" target="_blank">Matplotlib 中文网</a>
        </li>
      </ul>
    </div>

    <div class="right-box item friends">
      <h4>👬友情链接</h4>
      <ul>
        <li>
          <a href="http://www.numpy.org/" target="_blank">NumPy 英文</a>
        </li>
        <li>
          <a href="http://www.kuxai.com/" target="_blank">AIHUB社区</a>
        </li>
        <li>
          <a href="http://tushare.org/" target="_blank">Tushare</a>
        </li>
        <li>
          <a href="https://www.angtk.com/" target="_blank">昂钛客AI</a>
        </li>
        <li>
          <a href="https://www.ziiai.com/" target="_blank">极智能</a>
        </li>
        <li>
          <a href="https://www.pytorchtutorial.com/" target="_blank">PyTorch 中文网</a>
        </li>
        <li>
          <a href="http://www.ezist.cn/" target="_blank">机器人</a>
        </li>
        <li>
          <a href="https://www.lizenghai.com/" target="_blank">Python量化投资</a>
        </li>
        <li>
          <a href="https://www.ricequant.com/" target="_blank">RiceQuant米筐量化交易平台</a>
        </li>
        <li>
          <a href="https://www.quantinfo.com/" target="_blank">宽客在线</a>
        </li>
        <li>
          <a href="https://www.leiphone.com/?from=numpy.org.cn" target="_blank">雷锋网</a>
        </li>
        <li>
          <a href="https://www.yanxishe.com/?from=numpy.org.cn" target="_blank">AI研习社</a>
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
            color darken($accentColor, 20%);
  
  .friends
    ul
      &:after
        content ""
        display block
        height 0
        clear both
        visibility hidden
      li
        float left
        margin-right 20px

  .left-box
    float left

  .right-box
    float right

@media (max-width: 650px)
  .home-footer
    .item
      width 100%;
</style>