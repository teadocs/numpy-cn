<template>
  <div class="awesome">
    <ul>
      <li v-for="(item, index) in data">
        <a :href="item.href" target="_blank"><span class="tag">{{item.tag}}</span>{{item.name}}</a>
      </li>
    </ul>
  </div>
</template>

<script>
  export default {
    props: {
      data: Array
    }
  }
</script>

<style lang="stylus" scoped>
ul
  list-style none
  padding 0px
  margin 0px
  li
    padding 0px
    margin 0px
    line-height 1.6
    a
      cursor pointer
      font-size 14px
      color #37a
      text-decoration none !important
      &:hover
        text-decoration none !important
        background-color #37a !important
        color #fff !important
      .tag
        background-color #3eaf7c
        color #fff
        margin-right 6px
        padding-left 2px
        padding-right 2px
</style>