<script setup>
import { computed } from "vue";

import BaseChart from "@/components/charts/BaseChart.vue";
import { STAGE_META, fmtNumber, fmtRate } from "@/utils/format";

const props = defineProps({
  stage: { type: String, required: true },
  data: { type: Object, required: true },
  productName: { type: String, default: "" },
});
const emit = defineEmits(["open-evidence"]);

const meta = computed(() => STAGE_META[props.stage]);
const summary = computed(() => props.data.summary || {});

// 一句话结论：把核心指标翻译成业务语言
const conclusion = computed(() => {
  const s = summary.value;
  if (props.stage === "symptom") {
    if (s.category_mention_rate === null || s.category_mention_rate === undefined)
      return "本批次没有病症阶段的问题数据。";
    const rate = fmtRate(s.category_mention_rate);
    const verdict =
      s.category_mention_rate >= 0.6 ? "场景切入正确" : s.category_mention_rate >= 0.3 ? "品类有一定存在感，但不稳固" : "品类几乎不被推荐，场景选择可能有问题";
    return `用户泛式提问时，AI 推荐我方品类的比例为 ${rate}（${verdict}）。`;
  }
  if (props.stage === "category") {
    if (s.brand_mention_rate === null || s.brand_mention_rate === undefined)
      return "本批次没有品类阶段的问题数据。";
    return `聚焦品类后，品牌被提及 ${fmtRate(s.brand_mention_rate)}、被明确推荐 ${fmtRate(s.brand_rec_rate)}；竞品品牌提及 ${fmtRate(s.competitor_mention_rate)}。`;
  }
  const parts = [];
  if (s.accuracy_rate !== null && s.accuracy_rate !== undefined) {
    parts.push(`AI 对产品事实描述的准确率为 ${fmtRate(s.accuracy_rate)}（共 ${fmtNumber(s.total_claims)} 个知识点，错误 ${fmtNumber(s.wrong_claims)} 个）`);
  }
  if (s.negative_count) {
    parts.push(`检出对我方品牌的负面定性 ${s.negative_count} 条`);
  } else {
    parts.push("未检出对我方品牌的负面定性");
  }
  return parts.length ? parts.join("；") + "。" : "本批次没有品牌阶段数据。";
});

const metricCards = computed(() => {
  const s = summary.value;
  if (props.stage === "symptom") {
    return [
      { label: "品类提及率", value: fmtRate(s.category_mention_rate) },
      { label: "品牌提及率（参考）", value: fmtRate(s.brand_mention_rate) },
      { label: "回答样本", value: fmtNumber(s.total_answers) },
      { label: "我方负面提及", value: fmtNumber(s.negative_count) },
    ];
  }
  if (props.stage === "category") {
    return [
      { label: "品牌提及率", value: fmtRate(s.brand_mention_rate) },
      { label: "品牌推荐率", value: fmtRate(s.brand_rec_rate) },
      { label: "通用名提及率", value: fmtRate(s.generic_mention_rate) },
      { label: "竞品提及率", value: fmtRate(s.competitor_mention_rate) },
    ];
  }
  return [
    { label: "知识准确率", value: fmtRate(s.accuracy_rate) },
    { label: "知识点总数", value: fmtNumber(s.total_claims) },
    { label: "错误知识点", value: fmtNumber(s.wrong_claims) },
    { label: "我方负面提及", value: fmtNumber(s.negative_count) },
  ];
});

const trendOption = computed(() => {
  const trend = props.data.trend || {};
  const keyMetric = props.data.key_metric;
  const points = trend[keyMetric] || [];
  if (points.length < 2) return null;
  return {
    tooltip: { trigger: "axis", valueFormatter: (v) => fmtRate(v) },
    grid: { left: 48, right: 16, top: 24, bottom: 28 },
    xAxis: { type: "category", data: points.map((p) => p.batch_date) },
    yAxis: { type: "value", axisLabel: { formatter: (v) => fmtRate(v, 0) } },
    series: [
      {
        name: meta.value.keyMetricLabel,
        type: "line",
        data: points.map((p) => p.value),
        smooth: true,
        symbolSize: 8,
        color: "#0f766e",
      },
    ],
  };
});

const competitorOption = computed(() => {
  const rows = (props.data.competitors || []).slice(0, 10).reverse();
  if (!rows.length) return null;
  const isOurs = (row) =>
    row.name_type === "999品牌" ||
    (props.productName && row.rec_product.includes(props.productName.replace(/^999|^三九/, "")));
  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 130, right: 40, top: 8, bottom: 24 },
    xAxis: { type: "value" },
    yAxis: {
      type: "category",
      data: rows.map((r) => r.rec_product),
      axisLabel: { width: 118, overflow: "truncate" },
    },
    series: [
      {
        name: "提及次数",
        type: "bar",
        data: rows.map((r) => ({
          value: r.mention_count,
          itemStyle: { color: isOurs(r) ? "#0f766e" : "#b8c4c2" },
        })),
        barMaxWidth: 18,
        label: { show: true, position: "right", formatter: ({ value }) => fmtNumber(value) },
      },
    ],
  };
});

const evidenceButtons = computed(() => {
  const counts = props.data.evidence_counts || {};
  const defs =
    props.stage === "symptom"
      ? [
          ["category", "品类推荐明细"],
          ["recommendation", "推荐提及明细"],
          ["negative", "负面提及明细（含竞品）"],
        ]
      : props.stage === "category"
        ? [
            ["recommendation", "推荐提及明细"],
            ["negative", "负面提及明细（含竞品）"],
          ]
        : [
            ["accuracy", "校验明细"],
            ["negative", "负面提及明细（含竞品）"],
          ];
  return defs
    .map(([type, label]) => ({ type, label, count: counts[type] || 0 }))
    .filter((b) => b.count > 0);
});
</script>

<template>
  <el-card class="stage" shadow="never">
    <div class="stage-head">
      <div>
        <h2>{{ meta.title }}</h2>
        <p class="muted">{{ meta.subtitle }}</p>
      </div>
      <div class="evidence-buttons">
        <el-button
          v-for="button in evidenceButtons"
          :key="button.type"
          size="small"
          :type="button.type === 'negative' ? 'danger' : 'default'"
          plain
          @click="emit('open-evidence', { type: button.type, stage })"
        >
          {{ button.label }}（{{ button.count }}）
        </el-button>
      </div>
    </div>

    <el-alert :closable="false" type="info" class="conclusion">
      <template #title>
        <span style="font-size: 14px">{{ conclusion }}</span>
      </template>
    </el-alert>

    <div class="metric-row">
      <div v-for="card in metricCards" :key="card.label" class="metric-cell">
        <div class="metric-value">{{ card.value }}</div>
        <div class="metric-label">{{ card.label }}</div>
      </div>
    </div>

    <el-row :gutter="16">
      <el-col :span="competitorOption ? 12 : 24" v-if="trendOption">
        <h4 class="chart-title">{{ meta.keyMetricLabel }} · 跨批次趋势</h4>
        <BaseChart :option="trendOption" height="240px" />
      </el-col>
      <el-col :span="trendOption ? 12 : 24" v-if="competitorOption">
        <h4 class="chart-title">
          {{ stage === "symptom" ? "AI 推荐的品类结构（我方品类为深色）" : "推荐排行（我方为深色）" }}
        </h4>
        <BaseChart :option="competitorOption" height="240px" />
      </el-col>
    </el-row>
    <p v-if="!trendOption" class="muted" style="font-size: 12px">
      趋势需要 ≥2 个使用相同问题集的批次；定期用同一问题清单发起分析后，这里会自动连成趋势线。
    </p>
  </el-card>
</template>

<style scoped>
.stage {
  margin-bottom: 16px;
}
.stage-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}
.stage-head h2 {
  font-size: 16px;
  margin: 0 0 2px;
}
.stage-head p {
  margin: 0;
  font-size: 12px;
}
.conclusion {
  margin: 12px 0;
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.metric-cell {
  background: #f6f8f8;
  border-radius: 8px;
  padding: 12px 14px;
}
.chart-title {
  font-size: 13px;
  color: var(--geo-muted);
  margin: 4px 0 8px;
  font-weight: 500;
}
.evidence-buttons {
  flex-shrink: 0;
}
</style>
