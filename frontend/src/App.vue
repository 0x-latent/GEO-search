<script setup>
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import { AppShell, SideNav } from "@cr999ai/ui-vue";

import { useSessionStore } from "@/stores/session";

const session = useSessionStore();
const route = useRoute();
const router = useRouter();

// SideNav item key 直接用路由 path,select 时 router.push 即可
const navItems = computed(() => {
  const items = [
    { key: "/overview", label: "品牌总览", iconName: "chart" },
    { key: "/sources", label: "信源分析", iconName: "link" },
    { key: "/analysis", label: "我的分析", iconName: "flask" },
    { key: "/settings", label: "我的配置", iconName: "settings" },
  ];
  if (session.isAdmin) {
    items.push({ key: "/workbench", label: "工作台", iconName: "briefcase" });
  }
  return items;
});

const activeKey = computed(() =>
  route.path.startsWith("/products") ? "/overview" : route.path
);

function onSelect(item) {
  router.push(item.key);
}

// 退出统一走壳层 UserMenu;session.logout() 同时处理应用自身会话与
// 门户 SSO 会话,直连(:8001)与门户两种入口都覆盖。
function onLogout() {
  session.logout();
}
</script>

<template>
  <AppShell app-id="geo" app-name="GEO 洞察" :logout-handler="onLogout">
    <template #sidenav>
      <SideNav :items="navItems" :active="activeKey" @select="onSelect" />
    </template>
    <router-view :key="route.fullPath" />
  </AppShell>
</template>
