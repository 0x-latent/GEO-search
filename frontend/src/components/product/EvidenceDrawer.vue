<script setup>
import { reactive, ref, watch } from "vue";
import { ElMessage } from "element-plus";

import { api, query } from "@/api/client";
import AnswerDialog from "@/components/product/AnswerDialog.vue";
import { SEARCH_LABELS, fmtRate } from "@/utils/format";

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  datasetId: { type: String, default: "" },
  productCode: { type: String, default: "" },
  stage: { type: String, default: "" },
  type: { type: String, default: "recommendation" },
  title: { type: String, default: "证据明细" },
  scenario: { type: String, default: "" },
});
const emit = defineEmits(["update:modelValue"]);

const loading = ref(false);
const items = ref([]);
const total = ref(0);
const page = ref(1);
const size = 30;

const answerDialog = reactive({
  visible: false,
  questionId: "",
  model: "",
  searchEnabled: null,
  round: null,
});

const TYPE_LABELS = {
  recommendation: "推荐提及",
  negative: "负面提及",
  accuracy: "准确率校验",
  category: "品类推荐",
  yang_metric: "专项指标",
};

async function load() {
  if (!props.datasetId) return;
  loading.value = true;
  try {
    const params = query({
      dataset_id: props.datasetId,
      type: props.type,
      product_code: props.productCode,
      stage: props.scenario ? "" : props.type === "category" ? "" : props.stage,
      scenario: props.scenario,
      page: page.value,
      size,
    });
    const result = await api(`/api/insight/evidence${params}`);
    items.value = result.items;
    total.value = result.total;
  } catch (error) {
    ElMessage.error(`加载证据失败：${error.message}`);
  } finally {
    loading.value = false;
  }
}

watch(
  () => [props.modelValue, props.type, props.stage, props.scenario],
  ([visible]) => {
    if (visible) {
      page.value = 1;
      load();
    }
  }
);

function openAnswer(row) {
  if (!row.question_id) return;
  answerDialog.questionId = row.question_id;
  answerDialog.model = row.model;
  answerDialog.searchEnabled = row.search_enabled;
  answerDialog.round = row.round;
  answerDialog.visible = true;
}

const STRENGTH_TONES = { strong: "success", moderate: "", mention: "info", caution: "warning" };
</script>

<template>
  <el-drawer
    :model-value="modelValue"
    :title="`${title}${scenario ? ` · 场景「${scenario}」` : ''} · ${TYPE_LABELS[type] || type}（共 ${total} 条）`"
    size="62%"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <div v-loading="loading">
      <p class="muted" style="margin-top: 0; font-size: 12px">
        点击任意一行查看该条 AI 回答的完整原文与信源。
      </p>
      <el-table :data="items" size="small" @row-click="openAnswer" :row-style="{ cursor: 'pointer' }">
        <el-table-column prop="question_text" label="问题" min-width="220" show-overflow-tooltip>
          <template #default="{ row }">{{ row.question_text || row.detail || "—" }}</template>
        </el-table-column>
        <el-table-column prop="model" label="模型" width="100" />
        <el-table-column label="联网" width="70">
          <template #default="{ row }">
            {{ row.search_enabled === null ? "汇总" : SEARCH_LABELS[row.search_enabled] }}
          </template>
        </el-table-column>
        <el-table-column prop="round" label="轮次" width="60" />
        <template v-if="type === 'accuracy'">
          <el-table-column label="判定" width="80">
            <template #default="{ row }">
              <el-tag :type="row.verdict === 'wrong' ? 'danger' : row.verdict === 'correct' ? 'success' : 'info'" size="small">
                {{ row.verdict === "wrong" ? "有错误" : row.verdict === "correct" ? "正确" : "无依据" }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="准确率" width="80">
            <template #default="{ row }">{{ fmtRate(row.payload?.准确率) }}</template>
          </el-table-column>
          <el-table-column prop="detail" label="错误摘要" min-width="260" show-overflow-tooltip />
        </template>
        <template v-else>
          <el-table-column prop="rec_product" label="推荐产品" width="150" show-overflow-tooltip />
          <el-table-column v-if="type !== 'category'" prop="name_type" label="名称类型" width="100" />
          <el-table-column v-else label="品类" width="110">
            <template #default="{ row }">{{ row.payload?.品类 || "—" }}</template>
          </el-table-column>
          <el-table-column prop="rank" label="排名" width="60" />
          <el-table-column label="强度" width="90">
            <template #default="{ row }">
              <el-tag v-if="row.strength" :type="STRENGTH_TONES[row.strength] ?? 'info'" size="small">
                {{ row.strength }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="detail" label="原因/摘录" min-width="200" show-overflow-tooltip />
        </template>
      </el-table>
      <el-pagination
        v-if="total > size"
        layout="prev, pager, next, total"
        :total="total"
        :page-size="size"
        :current-page="page"
        style="margin-top: 12px; justify-content: flex-end"
        @current-change="(p) => { page = p; load(); }"
      />
    </div>
    <AnswerDialog
      v-model="answerDialog.visible"
      :dataset-id="datasetId"
      :question-id="answerDialog.questionId"
      :model="answerDialog.model"
      :search-enabled="answerDialog.searchEnabled"
      :round="answerDialog.round"
    />
  </el-drawer>
</template>
