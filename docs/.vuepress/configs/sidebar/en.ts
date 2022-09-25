import sidebarUserEn from './user_en';
import sidebarReferenceEn from './reference_en'
import sidebarF2pyEn from './f2py_en'
import sidebarDevEn from './dev_en'
import sidebarBedocsEn from './bedocs_en'
import { SidebarConfig } from 'vuepress-theme-teadocs'

export const sidebarEn: SidebarConfig = {
    '/user/': sidebarUserEn,
    '/reference/': sidebarReferenceEn,
    '/f2py/': sidebarF2pyEn,
    '/dev/': sidebarDevEn,
    '/bedocs/': sidebarBedocsEn,
}
