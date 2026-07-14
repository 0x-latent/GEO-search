<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";

import { api, query } from "@/api/client";
import EvidenceDrawer from "@/components/product/EvidenceDrawer.vue";
import StageSection from "@/components/product/StageSection.vue";
import SourceAnalysisPanel from "@/components/source/SourceAnalysisPanel.vue";
import { SEARCH_LABELS, fmtNumber, fmtRate } from "@/utils/format";

const route = useRoute();
const router = useRouter();
const productCode = computed(() => route.params.code);

const loading = ref(true);
const journey = ref(null);
const activeTab = ref(String(route.query.tab || "journey"));
const filters = reactive({
  datasetId: String(route.query.dataset_id || ""),
  model: "",
  search: "",
});

const drawer = reactive({ visible: false, type: "recommendation", stage: "", title: "", scenario: "" });

// 场景证据类型：标准数据集用推荐抽取明细，厂商预聚合数据集用专项指标
const scenarioEvidenceType = computed(() => {
  const stages = journey.value?.stages || {};
  const hasRec = Object.values(stages).some((s) => (s.evidence_counts || {}).recommendation > 0);
  return hasRec ? "recommendation" : "yang_metric";
});

function openScenarioEvidence(row) {
  drawer.type = scenarioEvidenceType.value;
  drawer.stage = "";
  drawer.scenario = row.scenario;
  drawer.title = journey.value?.product_name || "";
  drawer.visible = true;
}

async function load() {
  loading.value = true;
  try {
    const params = query({
      dataset_id: filters.datasetId,
      model: filters.model,
      search: filters.search,
    });
    journey.value = await api(
      `/api/insight/products/${encodeURIComponent(productCode.value)}/journey${params}`
    );
    if (!filters.datasetId && journey.value.selected) {
      filters.datasetId = journey.value.selected.dataset_id;
    }
  } catch (error) {
    ElMessage.error(`加载失败：${error.message}`);
  } finally {
    loading.value = false;
  }
}

onMounted(load);
watch(() => [filters.datasetId, filters.model, filters.search], load);

function openEvidence({ type, stage }) {
  drawer.type = type;
  drawer.stage = stage;
  drawer.scenario = "";
  drawer.title = journey.value?.product_name || "";
  drawer.visible = true;
}

function changeTab(name) {
  activeTab.value = name;
  router.replace({
    query: {
      ...route.query,
      tab: name === "sources" ? "sources" : undefined,
      dataset_id: filters.datasetId || undefined,
    },
  });
}
</script>

<template>
  <div class="page" v-loading="loading">
    <template v-if="journey">
      <div class="page-header detail-head">
        <div>
          <el-button link @click="router.push('/overview')">← 返回总览</el-button>
          <h1>
            {{ journey.product_name }}
            <el-tag v-if="journey.category" size="small" effect="plain">{{ journey.category }}</el-tag>
          </h1>
          <p>{{ activeTab === "sources" ? "查看该产品被 AI 引用的来源结构、域名排行和信源缺口。" : "消费者搜索链路三阶段分析 · 数据批次可切换，趋势基于相同问题集的批次序列。" }}</p>
        </div>
        <el-space v-if="activeTab === 'journey'" wrap>
          <el-select v-model="filters.datasetId" placeholder="批次" style="width: 230px">
            <el-option
              v-for="batch in journey.batches"
              :key="batch.dataset_id"
              :value="batch.dataset_id"
              :label="`${batch.batch_date} · ${batch.name}`"
            />
          </el-select>
          <el-select v-model="filters.model" placeholder="全部模型" clearable style="width: 140px">
            <el-option
              v-for="model in journey.filters?.models || []"
              :key="model"
              :value="model"
              :label="model"
            />
          </el-select>
          <el-select v-model="filters.search" placeholder="联网+非联网" clearable style="width: 130px">
            <el-option
              v-for="mode in journey.filters?.search_modes || []"
              :key="mode"
              :value="mode"
              :label="SEARCH_LABELS[mode] || mode"
            />
          </el-select>
        </el-space>
      </div>

      <el-tabs v-model="activeTab" class="product-tabs" @tab-change="changeTab">
        <el-tab-pane label="产品旅程" name="journey">
          <el-empty v-if="!journey.batches.length" description="该产品还没有分析数据">
            <el-button type="primary" @click="router.push('/analysis')">去发起分析</el-button>
          </el-empty>

          <template v-else>
            <StageSection
              v-for="stage in ['symptom', 'category', 'brand']"
              :key="stage"
              :stage="stage"
              :data="journey.stages[stage]"
              :product-name="journey.product_name"
              @open-evidence="openEvidence"
            />

            <el-card v-if="journey.scenarios?.length" class="scenario-card" shadow="never">
              <div class="scenario-head">
                <div>
                  <h2>场景拆解</h2>
                  <p class="muted">
                    同一消费场景（如"吃辣胃痛"）下多个问题的汇总表现——三阶段之外的横向视角，
                    回答"在这个具体场景里我们被推荐了吗"。
                  </p>
                </div>
              </div>
              <el-table :data="journey.scenarios" size="small" :default-sort="{ prop: 'brand_mention_rate', order: 'descending' }">
                <el-table-column prop="scenario" label="场景" min-width="150" />
                <el-table-column label="问题数" width="80">
                  <template #default="{ row }">{{ fmtNumber(row.question_count) }}</template>
                </el-table-column>
                <el-table-column label="回答样本" width="90">
                  <template #default="{ row }">{{ fmtNumber(row.total_answers) }}</template>
                </el-table-column>
                <el-table-column prop="brand_mention_rate" label="品牌提及/能见度" width="140" sortable>
                  <template #default="{ row }"><b>{{ fmtRate(row.brand_mention_rate) }}</b></template>
                </el-table-column>
                <el-table-column label="品牌推荐/前三率" width="140">
                  <template #default="{ row }">{{ fmtRate(row.brand_rec_rate) }}</template>
                </el-table-column>
                <el-table-column label="我方负面" width="90">
                  <template #default="{ row }">
                    <span :style="row.negative_count ? 'color:#b91c1c;font-weight:600' : ''">{{ row.negative_count ?? "—" }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="AI 推荐的品类" min-width="180">
                  <template #default="{ row }">
                    <el-tag v-for="cat in row.top_categories" :key="cat.category" size="small" effect="plain" style="margin-right: 4px">{{ cat.category }}</el-tag>
                    <span v-if="!row.top_categories?.length" class="muted">—</span>
                  </template>
                </el-table-column>
                <el-table-column label="" width="90">
                  <template #default="{ row }">
                    <el-button link type="primary" size="small" @click="openScenarioEvidence(row)">查看证据</el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>
          </template>
        </el-tab-pane>

        <el-tab-pane label="信源分析" name="sources" lazy>
          <SourceAnalysisPanel
            :fixed-product-code="productCode"
            :initial-dataset-id="filters.datasetId"
            embedded
          />
        </el-tab-pane>
      </el-tabs>

      <EvidenceDrawer
        v-model="drawer.visible"
        :dataset-id="filters.datasetId"
        :product-code="productCode"
        :stage="drawer.stage"
        :type="drawer.type"
        :title="drawer.title"
        :scenario="drawer.scenario"
      />
    </template>
  </div>
</template>

<style scoped>
.scenario-card {
  margin-bottom: 16px;
}
.scenario-head h2 {
  font-size: 16px;
  margin: 0 0 2px;
}
.scenario-head p {
  margin: 0 0 10px;
  font-size: 12px;
}
.detail-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  flex-wrap: wrap;
}
.detail-head h1 {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
