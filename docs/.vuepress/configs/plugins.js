module.exports = [
  ['@vuepress/back-to-top', true],
  ['@vuepress/pwa', {
    serviceWorker: true,
    updatePopup: {
      '/': {
        message: "更新了新内容呢！",
        buttonText: "立即刷新"
      },
      '/en/': {
        message: "New content is available.",
        buttonText: "Refresh"
      }
    }
  }],
  ['@vuepress/medium-zoom', true],
  ['container', {
    type: 'vue',
    before: '<pre class="vue-container"><code>',
    after: '</code></pre>',
  }],
  ['container', {
    type: 'upgrade',
    before: info => `<UpgradePath title="${info}">`,
    after: '</UpgradePath>',
  }]
]
