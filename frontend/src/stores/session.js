import { defineStore } from "pinia";

import { api } from "@/api/client";

export const useSessionStore = defineStore("session", {
  state: () => ({
    user: null,
    loaded: false,
  }),
  getters: {
    isAdmin: (state) => state.user?.role === "admin",
  },
  actions: {
    async ensureLoaded() {
      if (this.loaded) return this.user;
      this.user = await api("/api/auth/me");
      this.loaded = true;
      return this.user;
    },
    async logout() {
      try {
        await api("/api/auth/logout", { method: "POST" });
      } catch {
        /* 会话可能已过期 */
      }
      // 门户模式(经 /geo/ 子路径访问): 除应用自身会话外浏览器还持有门户
      // SSO 会话, 不退门户的话下一个请求会被网关重新注入身份头(退不掉)。
      // 门户退出端点是站点根路径, 不能带 /geo 前缀。
      if (window.location.pathname.startsWith("/geo")) {
        try {
          await fetch("/portal/logout", { method: "POST" });
        } catch {
          /* 门户不可达时仍回门户登录页 */
        }
        window.location.replace("/portal/login");
        return;
      }
      window.location.replace("/portal/login");
    },
  },
});
