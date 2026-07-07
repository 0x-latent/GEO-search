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
      window.location.href = "/login.html";
    },
  },
});
