<script setup>
import { reactive, ref, watch } from "vue";
import { ElMessage } from "element-plus";

import { api, query } from "@/api/client";
import AnswerDialog from "@/components/product/AnswerDialog.vue";

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  domain: { type: String, default: "" },
  title: { type: String, default: "信源回答" },
  filterParams: { type: Object, default: () => ({}) },
});
const emit = defineEmits(["update:modelValue"]);

const loading = ref(false);
const rows = ref([]);
const answerDialog = reactive({
  visible: false,
  datasetId: "",
  questionId: "",
  model: "",
  searchEnabled: null,
  round: null,
});

async function load() {
  if (!props.modelValue || !props.domain) return;
  loading.value = true;
  try {
    rows.value = await api(`/api/insight/sources/answers${query({
      ...props.filterParams,
      domain: props.domain,
      limit: 200,
    })}`);
  } catch (error) {
    ElMessage.error(`加载信源回答失败：${error.message}`);
  } finally {
    loading.value = false;
  }
}

watch(() => [props.modelValue, props.domain], load);

function openAnswer(row) {
  answerDialog.datasetId = row.dataset_id;
  answerDialog.questionId = row.question_id;
  answerDialog.model = row.model;
  answerDialog.searchEnabled = row.search_enabled;
  answerDialog.round = row.round;
  answerDialog.visible = true;
}
</script>

<template>
  <el-drawer
    :model-value="modelValue"
    :title="`${title} · ${domain}（${rows.length} 条回答）`"
    size="68%"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <el-table
      :data="rows"
      v-loading="loading"
      size="small"
      max-height="720"
      @row-click="openAnswer"
      :row-style="{ cursor: 'pointer' }"
    >
      <el-table-column prop="product_name" label="产品" width="130" />
      <el-table-column prop="model_name" label="模型" width="110" />
      <el-table-column prop="question_text" label="问题" min-width="220" show-overflow-tooltip />
      <el-table-column prop="answer_preview" label="回答摘录" min-width="280" show-overflow-tooltip />
      <el-table-column label="匹配信源" min-width="260" show-overflow-tooltip>
        <template #default="{ row }">
          {{ row.sources?.map((source) => source.title || source.url).join("；") || "—" }}
        </template>
      </el-table-column>
      <el-table-column prop="round" label="轮次" width="60" />
    </el-table>

    <AnswerDialog
      v-model="answerDialog.visible"
      :dataset-id="answerDialog.datasetId"
      :question-id="answerDialog.questionId"
      :model="answerDialog.model"
      :search-enabled="answerDialog.searchEnabled"
      :round="answerDialog.round"
    />
  </el-drawer>
</template>
