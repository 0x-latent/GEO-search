import { createRouter, createWebHashHistory } from "vue-router";

import { useSessionStore } from "@/stores/session";

// hash 路由：StaticFiles(html=True) 不支持 history 深层路径回退，
// hash 模式零后端改动，且与独立公开的 login.html 共存无冲突。
const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", redirect: "/overview" },
    {
      path: "/overview",
      name: "overview",
      component: () => import("@/views/BrandOverview.vue"),
      meta: { title: "品牌总览" },
    },
    {
      path: "/products/:code",
      name: "product-detail",
      component: () => import("@/views/ProductDetail.vue"),
      meta: { title: "产品详情" },
    },
    {
      path: "/sources",
      name: "sources",
      component: () => import("@/views/SourceAnalysis.vue"),
      meta: { title: "信源分析" },
    },
    {
      path: "/analysis",
      name: "analysis",
      component: () => import("@/views/MyAnalysis.vue"),
      meta: { title: "我的分析" },
    },
    {
      path: "/settings",
      name: "settings",
      component: () => import("@/views/Settings.vue"),
      meta: { title: "我的配置" },
    },
    {
      path: "/workbench",
      name: "workbench",
      component: () => import("@/views/Workbench.vue"),
      meta: { title: "工作台", adminOnly: true },
    },
    { path: "/:pathMatch(.*)*", redirect: "/overview" },
  ],
});

router.beforeEach(async (to) => {
  const session = useSessionStore();
  try {
    await session.ensureLoaded();
  } catch {
    return false; // client 已跳登录页
  }
  if (to.meta.adminOnly && !session.isAdmin) {
    return { path: "/overview" };
  }
  document.title = to.meta.title ? `${to.meta.title} - GEO 洞察平台` : "GEO 洞察平台";
  return true;
});

export default router;
