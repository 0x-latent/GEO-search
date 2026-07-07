<script setup>
import { ref, watch } from "vue";
import { ElMessage } from "element-plus";

import { api, query } from "@/api/client";
import { SEARCH_LABELS } from "@/utils/format";

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  datasetId: { type: String, default: "" },
  questionId: { type: String, default: "" },
  model: { type: String, default: "" },
  searchEnabled: { type: [Number, null], default: null },
  round: { type: [Number, null], default: null },
});
const emit = defineEmits(["update:modelValue"]);

const loading = ref(false);
const answer = ref(null);
const candidates = ref([]);

watch(
  () => [props.modelValue, props.questionId],
  async ([visible]) => {
    if (!visible || !props.questionId) return;
    loading.value = true;
    answer.value = null;
    candidates.value = [];
    try {
      const params = query({
        model: props.model,
        search_enabled: props.searchEnabled,
        round: props.round,
      });
      const result = await api(
        `/api/insight/answers/${encodeURIComponent(props.datasetId)}/${encodeURIComponent(props.questionId)}${params}`
      );
      answer.value = result.answer;
      candidates.value = result.candidates || [];
      // 精确五元组未命中但只有一个候选时直接展示
      if (!answer.value && candidates.value.length === 1) {
        answer.value = candidates.value[0];
        candidates.value = [];
      }
    } catch (error) {
      ElMessage.error(`加载回答失败：${error.message}`);
    } finally {
      loading.value = false;
    }
  }
);
</script>

<template>
  <el-dialog
    :model-value="modelValue"
    width="720px"
    :title="answer ? `AI 原始回答 · ${answer.model_name || answer.model}` : 'AI 原始回答'"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <div v-loading="loading">
      <template v-if="answer">
        <el-descriptions :column="3" size="small" border style="margin-bottom: 12px">
          <el-descriptions-item label="问题" :span="3">{{ answer.question_text }}</el-descriptions-item>
          <el-descriptions-item label="模型">{{ answer.model_name || answer.model }}</el-descriptions-item>
          <el-descriptions-item label="联网">{{ SEARCH_LABELS[answer.search_enabled] }}</el-descriptions-item>
          <el-descriptions-item label="轮次">第 {{ answer.round }} 轮</el-descriptions-item>
        </el-descriptions>
        <div class="answer-text">{{ answer.answer_text }}</div>
        <template v-if="answer.sources?.length">
          <el-divider content-position="left">引用信源（{{ answer.sources.length }}）</el-divider>
          <ol class="sources">
            <li v-for="source in answer.sources" :key="source.source_index">
              <a :href="source.url" target="_blank" rel="noreferrer">
                {{ source.title || source.url }}
              </a>
              <span class="muted" v-if="source.domain">（{{ source.domain }}）</span>
            </li>
          </ol>
        </template>
      </template>
      <template v-else-if="candidates.length">
        <p class="muted">未精确定位到该轮回答，该问题共有 {{ candidates.length }} 条回答：</p>
        <el-table :data="candidates" size="small" @row-click="(row) => (answer = row)">
          <el-table-column prop="model" label="模型" width="120" />
          <el-table-column label="联网" width="80">
            <template #default="{ row }">{{ SEARCH_LABELS[row.search_enabled] }}</template>
          </el-table-column>
          <el-table-column prop="round" label="轮次" width="70" />
          <el-table-column prop="answer_chars" label="字数" width="80" />
        </el-table>
      </template>
      <el-empty v-else-if="!loading" description="未找到该回答" :image-size="60" />
    </div>
  </el-dialog>
</template>

<style scoped>
.sources {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  max-height: 160px;
  overflow-y: auto;
}
</style>
