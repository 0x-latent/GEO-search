<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import { ElMessage } from "element-plus";

import { api, query } from "@/api/client";
import BaseChart from "@/components/charts/BaseChart.vue";
import AnswerDialog from "@/components/product/AnswerDialog.vue";
import SourceAnswerDrawer from "@/components/source/SourceAnswerDrawer.vue";
import { fmtNumber, fmtRate } from "@/utils/format";

const props = defineProps({
  fixedProductCode: { type: String, default: "" },
  initialDatasetId: { type: String, default: "" },
  embedded: { type: Boolean, default: false },
});

const loading = ref(false);
const optionsLoading = ref(false);
const options = ref({ datasets: [], products: [], models: [], categories: [], domains: [], stages: [] });
const analysis = ref(null);
const activeTab = ref("overview");
const filters = reactive({
  datasetIds: [],
  productCodes: [],
  models: [],
  searchModes: [],
  categories: [],
  domains: [],
  stages: [],
});

const domainDrawer = reactive({ visible: false, domain: "", title: "信源回答" });
const answerDialog = reactive({
  visible: false,
  datasetId: "",
  questionId: "",
  model: "",
  searchEnabled: null,
  round: null,
});

function csv(values) {
  return values?.length ? values.join(",") : "";
}

const filterParams = computed(() => ({
  dataset_ids: csv(filters.datasetIds),
  product_codes: props.fixedProductCode || csv(filters.productCodes),
  models: csv(filters.models),
  search_modes: csv(filters.searchModes),
  categories: csv(filters.categories),
  domains: csv(filters.domains),
  stages: csv(filters.stages),
}));

async function loadOptions() {
  optionsLoading.value = true;
  try {
    options.value = await api("/api/insight/sources/options");
    if (props.initialDatasetId && !filters.datasetIds.length) {
      filters.datasetIds = [props.initialDatasetId];
    }
  } catch (error) {
    ElMessage.error(`加载信源筛选项失败：${error.message}`);
  } finally {
    optionsLoading.value = false;
  }
}

async function loadAnalysis() {
  loading.value = true;
  try {
    analysis.value = await api(`/api/insight/sources/analysis${query(filterParams.value)}`);
  } catch (error) {
    ElMessage.error(`加载信源分析失败：${error.message}`);
  } finally {
    loading.value = false;
  }
}

async function resetFilters() {
  filters.datasetIds = props.initialDatasetId ? [props.initialDatasetId] : [];
  filters.productCodes = [];
  filters.models = [];
  filters.searchModes = [];
  filters.categories = [];
  filters.domains = [];
  filters.stages = [];
  await loadAnalysis();
}

onMounted(async () => {
  await loadOptions();
  await loadAnalysis();
});

watch(() => props.fixedProductCode, loadAnalysis);

const summary = computed(() => analysis.value?.summary || {});
const categoryChart = computed(() => {
  const rows = analysis.value?.categories || [];
  return {
    grid: { left: 110, right: 30, top: 12, bottom: 32 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: { type: "value", name: "引用回答数" },
    yAxis: { type: "category", data: rows.map((row) => row.label).reverse() },
    series: [{
      type: "bar",
      data: rows.map((row) => row.answer_count).reverse(),
      itemStyle: { color: "#0f766e", borderRadius: [0, 4, 4, 0] },
      barMaxWidth: 24,
    }],
  };
});

function openDomain(row) {
  domainDrawer.domain = row.domain;
  domainDrawer.title = row.name || row.domain;
  domainDrawer.visible = true;
}

function openGap(row) {
  answerDialog.datasetId = row.dataset_id;
  answerDialog.questionId = row.question_id;
  answerDialog.model = row.model;
  answerDialog.searchEnabled = row.search_enabled;
  answerDialog.round = row.round;
  answerDialog.visible = true;
}

const severityTone = { high: "danger", medium: "warning", low: "info" };
</script>

<template>
  <section :class="['source-panel', { embedded }]" v-loading="loading || optionsLoading">
    <el-card shadow="never" class="filter-card">
      <div class="filter-grid">
        <el-select
          v-model="filters.datasetIds"
          multiple collapse-tags collapse-tags-tooltip clearable filterable
          placeholder="全部数据集/批次"
        >
          <el-option
            v-for="item in options.datasets"
            :key="item.dataset_id"
            :value="item.dataset_id"
            :label="`${item.batch_date || '未标日期'} · ${item.name}`"
          />
        </el-select>
        <el-select
          v-if="!fixedProductCode"
          v-model="filters.productCodes"
          multiple collapse-tags collapse-tags-tooltip clearable filterable
          placeholder="全部产品（可多选）"
        >
          <el-option
            v-for="item in options.products"
            :key="item.product_code"
            :value="item.product_code"
            :label="item.product_name"
          />
        </el-select>
        <el-select v-model="filters.models" multiple collapse-tags clearable filterable placeholder="全部模型">
          <el-option v-for="item in options.models" :key="item.model" :value="item.model" :label="item.model_name" />
        </el-select>
        <el-select v-model="filters.searchModes" multiple collapse-tags clearable placeholder="联网+非联网">
          <el-option :value="1" label="联网" />
          <el-option :value="0" label="非联网" />
        </el-select>
        <el-select v-model="filters.categories" multiple collapse-tags clearable filterable placeholder="全部信源分类">
          <el-option v-for="item in options.categories" :key="item.value" :value="item.value" :label="item.label" />
        </el-select>
        <el-select v-model="filters.domains" multiple collapse-tags collapse-tags-tooltip clearable filterable placeholder="全部信源域名">
          <el-option
            v-for="item in options.domains"
            :key="item.domain"
            :value="item.domain"
            :label="`${item.name || item.domain} · ${item.domain}`"
          />
        </el-select>
        <el-select v-model="filters.stages" multiple collapse-tags clearable placeholder="全部消费阶段">
          <el-option v-for="item in options.stages" :key="item.value" :value="item.value" :label="item.label" />
        </el-select>
        <el-space>
          <el-button type="primary" @click="loadAnalysis">应用筛选</el-button>
          <el-button @click="resetFilters">重置</el-button>
        </el-space>
      </div>
      <p class="filter-note">
        覆盖率默认以联网回答为分母；域名和分类筛选后，覆盖率表示所选信源在回答中的覆盖情况。
      </p>
    </el-card>

    <div class="source-cards">
      <el-card shadow="never"><small>联网回答</small><strong>{{ fmtNumber(summary.online_answers) }}</strong></el-card>
      <el-card shadow="never"><small>信源覆盖率</small><strong>{{ fmtRate(summary.coverage_rate) }}</strong><span>{{ fmtNumber(summary.cited_online_answers) }} 条带信源</span></el-card>
      <el-card shadow="never"><small>平均信源数</small><strong>{{ fmtNumber(summary.avg_sources_per_cited_answer) }}</strong><span>每条带信源回答</span></el-card>
      <el-card shadow="never"><small>规范域名</small><strong>{{ fmtNumber(summary.distinct_domains) }}</strong><span>{{ fmtNumber(summary.distinct_urls) }} 个 URL</span></el-card>
      <el-card shadow="never">
        <small>官方信源覆盖</small>
        <strong>{{ summary.official_coverage_rate == null ? "待配置" : fmtRate(summary.official_coverage_rate) }}</strong>
        <span v-if="summary.official_coverage_rate == null">需配置品牌官方域名</span>
      </el-card>
      <el-card shadow="never"><small>权威信源覆盖</small><strong>{{ fmtRate(summary.authority_coverage_rate) }}</strong></el-card>
      <el-card shadow="never"><small>联网无信源</small><strong class="danger">{{ fmtNumber(summary.online_without_sources) }}</strong></el-card>
    </div>

    <el-tabs v-model="activeTab" class="analysis-tabs">
      <el-tab-pane label="来源总览" name="overview">
        <div class="overview-grid">
          <el-card shadow="never">
            <template #header><strong>信源分类覆盖</strong></template>
            <BaseChart :option="categoryChart" height="320px" />
          </el-card>
          <el-card shadow="never">
            <template #header><strong>分类明细</strong></template>
            <el-table :data="analysis?.categories || []" size="small" max-height="320">
              <el-table-column prop="label" label="分类" min-width="130" />
              <el-table-column prop="answer_count" label="回答" width="75" sortable />
              <el-table-column label="覆盖率" width="90" sortable prop="coverage_rate">
                <template #default="{ row }">{{ fmtRate(row.coverage_rate) }}</template>
              </el-table-column>
              <el-table-column prop="refs" label="引用" width="75" sortable />
              <el-table-column prop="domain_count" label="域名" width="70" sortable />
            </el-table>
          </el-card>
        </div>

        <el-card shadow="never" class="domain-card">
          <template #header>
            <div class="card-head">
              <strong>信源域名排行</strong>
              <span>点击域名查看对应问题、回答和完整信源</span>
            </div>
          </template>
          <el-table :data="analysis?.domains || []" size="small" max-height="560">
            <el-table-column label="信源" min-width="220" fixed>
              <template #default="{ row }">
                <el-button link type="primary" @click="openDomain(row)">{{ row.name || row.domain }}</el-button>
                <div class="domain-text">{{ row.domain }}</div>
              </template>
            </el-table-column>
            <el-table-column prop="category_label" label="分类" width="130" />
            <el-table-column prop="authority" label="权威级" width="80" />
            <el-table-column prop="answer_count" label="引用回答" width="100" sortable />
            <el-table-column prop="coverage_rate" label="回答覆盖率" width="115" sortable>
              <template #default="{ row }">{{ fmtRate(row.coverage_rate) }}</template>
            </el-table-column>
            <el-table-column prop="refs" label="引用次数" width="95" sortable />
            <el-table-column prop="url_count" label="URL" width="75" sortable />
            <el-table-column label="涉及产品" min-width="180" show-overflow-tooltip>
              <template #default="{ row }">{{ row.products.join("、") }}</template>
            </el-table-column>
            <el-table-column label="涉及模型" min-width="160" show-overflow-tooltip>
              <template #default="{ row }">{{ row.models.join("、") }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="产品对比" name="products">
        <el-card shadow="never">
          <template #header>
            <div class="card-head"><strong>产品信源表现</strong><span>多产品组合时保留每个产品的独立分母</span></div>
          </template>
          <el-table :data="analysis?.products || []" size="small">
            <el-table-column prop="product_name" label="产品" min-width="150" />
            <el-table-column prop="online_answers" label="联网回答" width="100" sortable />
            <el-table-column prop="cited_answers" label="带信源回答" width="115" sortable />
            <el-table-column prop="coverage_rate" label="信源覆盖率" width="115" sortable>
              <template #default="{ row }"><b>{{ fmtRate(row.coverage_rate) }}</b></template>
            </el-table-column>
            <el-table-column prop="official_coverage_rate" label="官方信源覆盖" width="125" sortable>
              <template #default="{ row }">{{ fmtRate(row.official_coverage_rate) }}</template>
            </el-table-column>
            <el-table-column prop="authority_coverage_rate" label="权威信源覆盖" width="125" sortable>
              <template #default="{ row }">{{ fmtRate(row.authority_coverage_rate) }}</template>
            </el-table-column>
            <el-table-column prop="domain_count" label="信源域名" width="95" sortable />
          </el-table>
        </el-card>
      </el-tab-pane>

      <el-tab-pane :label="`信源缺口（${analysis?.gap_total || 0}）`" name="gaps">
        <el-card shadow="never">
          <template #header>
            <div class="card-head"><strong>信源缺口明细</strong><span>点击问题下钻到 AI 原始回答</span></div>
          </template>
          <el-table :data="analysis?.gaps || []" size="small" max-height="620" @row-click="openGap" :row-style="{ cursor: 'pointer' }">
            <el-table-column label="严重度" width="80">
              <template #default="{ row }"><el-tag :type="severityTone[row.severity]" size="small">{{ row.severity === "high" ? "高" : "中" }}</el-tag></template>
            </el-table-column>
            <el-table-column prop="gap_label" label="缺口" min-width="190" />
            <el-table-column prop="product_name" label="产品" width="130" />
            <el-table-column prop="question_text" label="问题" min-width="260" show-overflow-tooltip />
            <el-table-column prop="model_name" label="模型" width="110" />
            <el-table-column prop="scenario" label="场景" width="130" show-overflow-tooltip />
            <el-table-column prop="source_count" label="信源" width="70" />
            <el-table-column prop="round" label="轮次" width="60" />
          </el-table>
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <SourceAnswerDrawer
      v-model="domainDrawer.visible"
      :domain="domainDrawer.domain"
      :title="domainDrawer.title"
      :filter-params="filterParams"
    />
    <AnswerDialog
      v-model="answerDialog.visible"
      :dataset-id="answerDialog.datasetId"
      :question-id="answerDialog.questionId"
      :model="answerDialog.model"
      :search-enabled="answerDialog.searchEnabled"
      :round="answerDialog.round"
    />
  </section>
</template>

<style scoped>
.source-panel { display: grid; gap: 14px; }
.filter-card { margin-bottom: 0; }
.filter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px;
  align-items: center;
}
.filter-note { margin: 10px 0 0; color: #64748b; font-size: 12px; }
.source-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 12px; }
.source-cards :deep(.el-card__body) { display: grid; gap: 5px; }
.source-cards small, .source-cards span { color: #64748b; font-size: 12px; }
.source-cards strong { font-size: 24px; color: #0f3d38; }
.source-cards strong.danger { color: #b91c1c; }
.overview-grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(360px, .8fr); gap: 14px; }
.domain-card { margin-top: 14px; }
.card-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.card-head span { color: #64748b; font-size: 12px; font-weight: normal; }
.domain-text { color: #94a3b8; font-size: 11px; margin-left: 12px; }
@media (max-width: 1180px) {
  .source-cards { grid-template-columns: repeat(3, minmax(140px, 1fr)); }
  .overview-grid { grid-template-columns: 1fr; }
}
@media (max-width: 720px) {
  .source-cards { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
}
</style>
