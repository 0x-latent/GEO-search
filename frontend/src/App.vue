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
    { key: "/overview", label: "品牌总览", icon: "📊" },
    { key: "/sources", label: "信源分析", icon: "🔗" },
    { key: "/analysis", label: "我的分析", icon: "🧪" },
    { key: "/settings", label: "我的配置", icon: "⚙️" },
  ];
  if (session.isAdmin) {
    items.push({ key: "/workbench", label: "工作台", icon: "🗂️" });
  }
  return items;
});

const activeKey = computed(() =>
  route.path.startsWith("/products") ? "/overview" : route.path
);

function onSelect(item) {
  router.push(item.key);
}
</script>

<template>
  <AppShell app-id="geo" app-name="GEO 洞察">
    <template #sidenav>
      <SideNav :items="navItems" :active="activeKey" @select="onSelect" />
    </template>
    <!-- 原侧栏底部的退出按钮迁到顶栏 actions;返回门户由壳层 logo/用户菜单承担。
         保留应用自身 logout:直连(:8001)入口没有门户会话,壳层 UserMenu 的
         /portal/logout 覆盖不到,session.logout() 两种入口都处理。 -->
    <template #topbar-actions>
      <el-button v-if="session.user" link @click="session.logout()">
        退出({{ session.user.username }})
      </el-button>
    </template>
    <router-view :key="route.fullPath" />
  </AppShell>
</template>
