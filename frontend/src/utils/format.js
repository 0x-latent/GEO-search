// 指标格式化：所有"率"均为按回答去重的占比（0-1），直接按百分比显示。
// 若出现 >100% 说明底层口径回归为条目计数，应修数据而不是在这里兜底。
export function fmtRate(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function fmtNumber(value) {
  if (value === null || value === undefined) return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: Number.isInteger(num) ? 0 : 2,
  }).format(num);
}

export function fmtDelta(value) {
  if (value === null || value === undefined) return null;
  const num = Number(value);
  const pct = `${(num * 100).toFixed(1)}pp`;
  return { text: `${num > 0 ? "+" : ""}${pct}`, direction: num > 0 ? "up" : num < 0 ? "down" : "flat" };
}

export const SEARCH_LABELS = { 1: "联网", 0: "非联网", "1": "联网", "0": "非联网", agg: "汇总" };

export const STAGE_META = {
  symptom: {
    title: "① 病症阶段",
    subtitle: "用户只有症状、泛式提问（如“感冒了吃什么药”）——看我们的品类是否被 AI 推荐",
    keyMetricLabel: "品类提及率",
  },
  category: {
    title: "② 品类阶段",
    subtitle: "用户已聚焦品类、比较品牌（如“感冒颗粒哪个牌子好”）——看品牌/通用名 vs 竞品",
    keyMetricLabel: "品牌提及率",
  },
  brand: {
    title: "③ 品牌阶段",
    subtitle: "用户直接询问产品（如“XX 怎么样”）——看 AI 说得准不准、有没有负面定性",
    keyMetricLabel: "知识准确率",
  },
};
