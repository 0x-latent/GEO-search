// 统一 fetch 封装：同源 cookie 会话，401 一律回登录页。
export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// 应用根路径：/geo/ 反代下是 "/geo/"，:8001 直连下是 "/"。取当前页面所在
// 目录即可（hash 路由不改 pathname，模块加载时算一次），应用内的绝对路径
// （/api/...、/login.html）统一经 appUrl 换算，两种入口无需分别构建。
const APP_BASE = window.location.pathname.replace(/[^/]*$/, "");

export const appUrl = (path) => APP_BASE + String(path).replace(/^\//, "");

export async function api(path, options = {}) {
  const res = await fetch(appUrl(path), options);
  if (res.status === 401) {
    window.location.href = appUrl("/login.html");
    throw new ApiError("未登录", 401);
  }
  if (!res.ok) {
    let message = res.statusText;
    try {
      const data = await res.json();
      message = data.detail || JSON.stringify(data);
    } catch {
      /* 保留 statusText */
    }
    throw new ApiError(message, res.status);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const apiJson = (path, method, body) =>
  api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export function query(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== null && value !== undefined && value !== "") {
      search.set(key, value);
    }
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}
