import { createApp } from "vue";
import ElementPlus from "element-plus";
import zhCn from "element-plus/es/locale/lang/zh-cn";
import "element-plus/dist/index.css";

import ContributorApp from "./ContributorApp.vue";

createApp(ContributorApp).use(ElementPlus, { locale: zhCn }).mount("#contributor-app");
