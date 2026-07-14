<script setup>
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";

import { useSessionStore } from "@/stores/session";

const session = useSessionStore();
const route = useRoute();
const router = useRouter();

const navItems = computed(() => {
  const items = [
    { path: "/overview", label: "品牌总览", icon: "📊" },
    { path: "/sources", label: "信源分析", icon: "🔗" },
    { path: "/analysis", label: "我的分析", icon: "🧪" },
    { path: "/settings", label: "我的配置", icon: "⚙️" },
  ];
  if (session.isAdmin) {
    items.push({ path: "/workbench", label: "工作台", icon: "🗂️" });
  }
  return items;
});

const activePath = computed(() =>
  route.path.startsWith("/products") ? "/overview" : route.path
);

// 返回门户：门户挂在同主机 80 端口的 /portal/。GEO 有两种入口（/geo/ 反代、:8001 直连），
// 都要能回门户，所以用去端口的绝对地址；本地开发（localhost）无门户，不显示。
const isPortalMode = !["localhost", "127.0.0.1"].includes(window.location.hostname);
function backToPortal() {
  window.location.href = `${window.location.protocol}//${window.location.hostname}/portal/`;
}
</script>

<template>
  <el-container style="min-height: 100vh">
    <el-aside width="216px" class="side">
      <div class="brand">
        <span class="brand-mark">G</span>
        <div>
          <strong>GEO 洞察平台</strong>
          <small>AI 搜索品牌表现</small>
        </div>
      </div>
      <el-menu
        :default-active="activePath"
        router
        class="side-menu"
        background-color="transparent"
      >
        <el-menu-item v-for="item in navItems" :key="item.path" :index="item.path">
          <span class="nav-icon">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </el-menu-item>
      </el-menu>
      <button v-if="isPortalMode" class="portal-back" @click="backToPortal">
        <span class="nav-icon">🏠</span>
        <span>返回门户</span>
      </button>
      <div class="side-user" v-if="session.user">
        <div>
          <strong>{{ session.user.username }}</strong>
          <small>{{ session.isAdmin ? "管理员" : "普通用户" }}</small>
        </div>
        <el-button link size="small" @click="session.logout()">退出</el-button>
      </div>
    </el-aside>
    <el-main style="padding: 0">
      <router-view :key="route.fullPath" />
    </el-main>
  </el-container>
</template>

<style scoped>
.side {
  background: #10322e;
  color: #e6f2f0;
  display: flex;
  flex-direction: column;
  position: sticky;
  top: 0;
  height: 100vh;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 18px 16px 14px;
}
.brand small {
  display: block;
  color: #9dbdb8;
  font-size: 12px;
}
.brand-mark {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: #0f766e;
  color: #fff;
  font-weight: 700;
}
.side-menu {
  border-right: none;
  flex: 1;
}
.side-menu :deep(.el-menu-item) {
  color: #cfe5e2;
}
.side-menu :deep(.el-menu-item.is-active) {
  color: #fff;
  background: #0f766e;
}
.side-menu :deep(.el-menu-item:hover) {
  background: rgba(15, 118, 110, 0.35);
}
.nav-icon {
  margin-right: 8px;
}
.portal-back {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 12px 20px;
  border: none;
  border-top: 1px solid rgba(255, 255, 255, 0.12);
  background: transparent;
  color: #cfe5e2;
  font-size: 14px;
  cursor: pointer;
  text-align: left;
}
.portal-back:hover {
  background: rgba(15, 118, 110, 0.35);
  color: #fff;
}
.side-user {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.12);
}
.side-user small {
  display: block;
  color: #9dbdb8;
  font-size: 12px;
}
</style>
