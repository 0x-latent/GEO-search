<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";

import { api, query } from "@/api/client";
import EvidenceDrawer from "@/components/product/EvidenceDrawer.vue";
import StageSection from "@/components/product/StageSection.vue";
import { SEARCH_LABELS } from "@/utils/format";

const route = useRoute();
const router = useRouter();
const productCode = computed(() => route.params.code);

const loading = ref(true);
const journey = ref(null);
const filters = reactive({
  datasetId: String(route.query.dataset_id || ""),
  model: "",
  search: "",
});

const drawer = reactive({ visible: false, type: "recommendation", stage: "", title: "" });

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
  drawer.title = journey.value?.product_name || "";
  drawer.visible = true;
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
          <p>消费者搜索链路三阶段分析 · 数据批次可切换，趋势基于相同问题集的批次序列。</p>
        </div>
        <el-space wrap>
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
      </template>

      <EvidenceDrawer
        v-model="drawer.visible"
        :dataset-id="filters.datasetId"
        :product-code="productCode"
        :stage="drawer.stage"
        :type="drawer.type"
        :title="drawer.title"
      />
    </template>
  </div>
</template>

<style scoped>
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
