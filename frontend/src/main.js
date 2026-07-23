import { createPinia } from "pinia";
import { createApp } from "vue";
import ElementPlus from "element-plus";
import zhCn from "element-plus/es/locale/lang/zh-cn";

// 样式顺序契约(design-system SPEC §3):
// tokens → Element Plus 原始样式(含深色变量) → token 到 --el-* 的映射 → 壳组件样式 → 应用样式
import "@cr999ai/ui-vue/tokens.css";
import "element-plus/dist/index.css";
import "element-plus/theme-chalk/dark/css-vars.css";
import "@cr999ai/ui-vue/element-plus.css";
import "@cr999ai/ui-vue/style.css";
import "./styles.css";

import { initBrandTheme, initColorScheme } from "@cr999ai/ui-vue";

import App from "./App.vue";
import router from "./router";
import { initDarkClassSync } from "./utils/darkClassSync";

initColorScheme();
initBrandTheme({ appId: "geo" });
initDarkClassSync();

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.use(ElementPlus, { locale: zhCn });
app.mount("#app");
