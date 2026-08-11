<script setup>
import { computed, onMounted, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";

import { api, query } from "@/api/client";
import { fmtNumber } from "@/utils/format";

const activeTab = ref("datasets");

// ---------- 数据集 ----------
const datasets = ref([]);
const datasetFilter = ref("all");

async function loadDatasets() {
  try {
    datasets.value = await api("/api/sqlite/datasets");
  } catch (error) {
    ElMessage.error(`加载数据集失败：${error.message}`);
  }
}

async function removeDataset(row) {
  try {
    await ElMessageBox.confirm(
      `确认删除数据集「${row.name}」？将级联清除其全部问题、回答、信源、外部表和物化指标，不可恢复。`,
      "删除数据集",
      { type: "warning", confirmButtonText: "删除", confirmButtonClass: "el-button--danger" }
    );
  } catch {
    return;
  }
  try {
    await api(`/api/sqlite/datasets/${encodeURIComponent(row.dataset_id)}`, { method: "DELETE" });
    ElMessage.success("已删除");
    await loadDatasets();
  } catch (error) {
    ElMessage.error(`删除失败：${error.message}`);
  }
}

// ---------- 拆分明细 ----------
const splits = ref(null);
const splitsLoading = ref(false);
const splitTable = ref("mention_summary");
const SPLIT_TABS = [
  ["mention_summary", "提及/推荐率汇总"],
  ["rec_overview", "推荐产品排行"],
  ["type_summary", "名称类型汇总"],
  ["category_summary", "品类汇总"],
  ["question_details", "问题级明细"],
  ["yangweishu_brand_summary", "养胃舒品牌"],
];

async function loadSplits() {
  splitsLoading.value = true;
  try {
    splits.value = await api(`/api/sqlite/splits${query({ dataset_id: datasetFilter.value })}`);
  } catch (error) {
    ElMessage.error(`加载失败：${error.message}`);
  } finally {
    splitsLoading.value = false;
  }
}

const splitRows = computed(() => splits.value?.[splitTable.value] || []);
const splitColumns = computed(() =>
  splitRows.value.length ? Object.keys(splitRows.value[0]) : []
);

// ---------- 回答样本 ----------
const samples = ref([]);
const samplesLoading = ref(false);

async function loadSamples() {
  samplesLoading.value = true;
  try {
    samples.value = await api(
      `/api/sqlite/answers${query({ dataset_id: datasetFilter.value, limit: 200 })}`
    );
  } catch (error) {
    ElMessage.error(`加载失败：${error.message}`);
  } finally {
    samplesLoading.value = false;
  }
}

onMounted(async () => {
  await loadDatasets();
  await loadSplits();
});

function onTab(name) {
  if (name === "samples" && !samples.value.length) loadSamples();
}

function onDatasetFilter() {
  loadSplits();
  if (activeTab.value === "samples") loadSamples();
}
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>工作台</h1>
      <p>数据侧与管理员的明细视图：数据集、拆分表和回答样本。账户与权限由门户统一管理，业务结论请看「品牌总览」。</p>
    </div>

    <el-tabs v-model="activeTab" @tab-change="onTab">
      <el-tab-pane label="数据集" name="datasets">
        <el-table :data="datasets" size="small">
          <el-table-column prop="dataset_id" label="数据集ID" min-width="220" show-overflow-tooltip />
          <el-table-column prop="name" label="名称" min-width="150" show-overflow-tooltip />
          <el-table-column prop="batch_date" label="批次日期" width="100" />
          <el-table-column prop="product_code" label="产品" width="110" />
          <el-table-column label="归属" width="90">
            <template #default="{ row }">{{ row.owner_username || "系统" }}</template>
          </el-table-column>
          <el-table-column label="问题" width="80">
            <template #default="{ row }">{{ fmtNumber(row.questions) }}</template>
          </el-table-column>
          <el-table-column label="回答" width="90">
            <template #default="{ row }">{{ fmtNumber(row.answers) }}</template>
          </el-table-column>
          <el-table-column label="外部表" width="80">
            <template #default="{ row }">{{ fmtNumber(row.external_tables) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="90">
            <template #default="{ row }">
              <el-button link type="danger" size="small" @click="removeDataset(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="拆分明细" name="splits">
        <el-space style="margin-bottom: 12px" wrap>
          <el-select v-model="datasetFilter" style="width: 240px" @change="onDatasetFilter">
            <el-option value="all" label="全部数据集" />
            <el-option
              v-for="dataset in datasets"
              :key="dataset.dataset_id"
              :value="dataset.dataset_id"
              :label="dataset.name"
            />
          </el-select>
          <el-radio-group v-model="splitTable">
            <el-radio-button v-for="[key, label] in SPLIT_TABS" :key="key" :value="key">
              {{ label }}
            </el-radio-button>
          </el-radio-group>
        </el-space>
        <el-table :data="splitRows" v-loading="splitsLoading" size="small" max-height="620" border>
          <el-table-column
            v-for="column in splitColumns"
            :key="column"
            :prop="column"
            :label="column"
            :min-width="column === '推荐原因' || column === '提问词' ? 220 : 110"
            show-overflow-tooltip
            sortable
          />
        </el-table>
        <p class="muted" style="font-size: 12px">共 {{ splitRows.length }} 行（问题级明细仅预览部分，全量请用证据链接口）</p>
      </el-tab-pane>

      <el-tab-pane label="回答样本" name="samples">
        <el-space style="margin-bottom: 12px">
          <el-select v-model="datasetFilter" style="width: 240px" @change="onDatasetFilter">
            <el-option value="all" label="全部数据集" />
            <el-option
              v-for="dataset in datasets"
              :key="dataset.dataset_id"
              :value="dataset.dataset_id"
              :label="dataset.name"
            />
          </el-select>
        </el-space>
        <el-table :data="samples" v-loading="samplesLoading" size="small" max-height="620" border>
          <el-table-column prop="product_name" label="产品" width="90" />
          <el-table-column prop="model" label="模型" width="100" />
          <el-table-column prop="search_mode" label="模式" width="70" />
          <el-table-column prop="round" label="轮次" width="60" />
          <el-table-column prop="question_text" label="问题" min-width="200" show-overflow-tooltip />
          <el-table-column prop="answer_preview" label="回答摘录" min-width="300" show-overflow-tooltip />
          <el-table-column prop="answer_chars" label="字数" width="80" sortable />
          <el-table-column prop="source_count" label="信源" width="70" sortable />
        </el-table>
      </el-tab-pane>

    </el-tabs>
  </div>
</template>
