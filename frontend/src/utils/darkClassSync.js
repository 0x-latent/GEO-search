// 设计系统用 <html data-theme="dark|light"> 控制明暗(未设置时跟随系统),
// 而 Element Plus 的深色样式挂在 html.dark 下。这里把两者桥接:
// 监听 data-theme 属性变化与系统偏好变化,同步 document.documentElement 的 'dark' class。
export function initDarkClassSync() {
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  const apply = () => {
    const theme = root.getAttribute("data-theme");
    const dark = theme === "dark" || (!theme && media.matches);
    root.classList.toggle("dark", dark);
  };

  new MutationObserver(apply).observe(root, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  // 跟随系统(data-theme 未设置)时,系统切换深浅也要同步
  if (typeof media.addEventListener === "function") {
    media.addEventListener("change", apply);
  } else if (typeof media.addListener === "function") {
    media.addListener(apply);
  }
  apply();
}
