import { pwaPlugin } from '@vuepress/plugin-pwa'
import { pwaPopupPlugin } from '@vuepress/plugin-pwa-popup'
import { getDirname, path } from '@vuepress/utils'
import { containerPlugin } from '@vuepress/plugin-container'
import { registerComponentsPlugin } from '@vuepress/plugin-register-components'

const __dirname = getDirname(import.meta.url)

export default [
  // backToTop(true),
  pwaPlugin({
    // serviceWorker: false,
  }),
  pwaPopupPlugin({
    locales: {
      '/': {
        message: "更新了新内容呢！",
        buttonText: "立即刷新"
      },
      '/en/': {
        message: "New content is available.",
        buttonText: "Refresh"
      }
    }
  }),
  // mediumZoom(true),
  containerPlugin({
    type: 'vue',
    before: () => '<pre class="vue-container"><code>',
    after: () => '</code></pre>',
  }),
  containerPlugin({
    type: 'upgrade',
    before: info => `<UpgradePath title="${info}">`,
    after: () => '</UpgradePath>',
  }),
  registerComponentsPlugin({
    componentsDir: path.resolve(__dirname, '../components'),
  }),
]
