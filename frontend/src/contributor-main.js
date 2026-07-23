import { createApp } from "vue";
import ElementPlus from "element-plus";
import zhCn from "element-plus/es/locale/lang/zh-cn";

// 投稿页是面向外部作者的独立入口,不接 AppShell 壳层;
// 只引 token 与 Element Plus 映射,保证观感与站内一致(SPEC §3)。
import "@cr999ai/ui-vue/tokens.css";
import "element-plus/dist/index.css";
import "element-plus/theme-chalk/dark/css-vars.css";
import "@cr999ai/ui-vue/element-plus.css";

import { initColorScheme } from "@cr999ai/ui-vue";

import ContributorApp from "./ContributorApp.vue";
import { initDarkClassSync } from "./utils/darkClassSync";

initColorScheme();
initDarkClassSync();

createApp(ContributorApp).use(ElementPlus, { locale: zhCn }).mount("#contributor-app");
