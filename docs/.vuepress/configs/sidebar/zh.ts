import sidebarUserZh from './user_zh';
import sidebarReferenceZh from './reference_zh'
import sidebarF2pyZh from './f2py_zh'
import sidebarDevZh from './dev_zh'
import sidebarBedocsZh from './bedocs_zh'
import sidebarDeepZh from './deep_zh'
import sidebarArticleZh from './article_zh'
import { SidebarConfig } from 'vuepress-theme-teadocs'

export const sidebarZh: SidebarConfig = {
    '/user/': sidebarUserZh,
    '/reference/': sidebarReferenceZh,
    '/f2py/': sidebarF2pyZh,
    '/dev/': sidebarDevZh,
    '/bedocs/': sidebarBedocsZh,
    '/deep/': sidebarDeepZh,
    '/article/': sidebarArticleZh,
}
