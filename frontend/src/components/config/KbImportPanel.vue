<script setup>
// 知识库文档导入：上传 → 多模态识别进度 → 审核 diff（逐模块采纳，可改文本）→ 合并
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";

import { api } from "@/api/client";
import { useSessionStore } from "@/stores/session";

const props = defineProps({
  // 当前知识库数据（用于 diff 对比）与可选产品 key
  kbData: { type: Object, default: () => ({}) },
  scope: { type: String, default: "user" },
});
const emit = defineEmits(["applied"]);

const session = useSessionStore();
const fileInput = ref(null);
const productKey = ref("");
const submitting = ref(false);
const imports = ref([]);
let timer = null;

const productOptions = computed(() => Object.keys(props.kbData || {}));

const STATUS_META = {
  queued: { label: "排队中", tone: "info" },
  running: { label: "识别中", tone: "warning" },
  success: { label: "待审核", tone: "success" },
  failed: { label: "失败", tone: "danger" },
};
const STAGE_LABELS = {
  render: "解析文件",
  recognize: "多模态识别",
  extract: "结构化抽取",
  verify: "引用核验",
  done: "完成",
};

async function loadImports() {
  try {
    imports.value = await api("/api/kb-imports");
  } catch (error) {
    ElMessage.error(`加载导入任务失败：${error.message}`);
  }
}

function pollIfActive() {
  if (imports.value.some((i) => i.status === "queued" || i.status === "running")) {
    loadImports();
  }
}

onMounted(() => {
  loadImports();
  timer = setInterval(pollIfActive, 5000);
});
onBeforeUnmount(() => clearInterval(timer));

async function submit() {
  const file = fileInput.value?.files?.[0];
  if (!file) {
    ElMessage.warning("请先选择文件");
    return;
  }
  if (!productKey.value.trim()) {
    ElMessage.warning("请选择或填写产品");
    return;
  }
  submitting.value = true;
  try {
    const body = new FormData();
    body.append("file", file);
    body.append("product_key", productKey.value.trim());
    body.append("scope", props.scope);
    await api("/api/kb-imports", { method: "POST", body });
    ElMessage.success("已提交，后台识别中（可离开本页）");
    fileInput.value.value = "";
    await loadImports();
  } catch (error) {
    ElMessage.error(`提交失败：${error.message}`);
  } finally {
    submitting.value = false;
  }
}

async function retryImport(row) {
  try {
    await api(`/api/kb-imports/${encodeURIComponent(row.import_id)}/retry`, { method: "POST" });
    ElMessage.success("已重新入队（已识别的页面不会重复调用）");
    await loadImports();
  } catch (error) {
    ElMessage.error(`重试失败：${error.message}`);
  }
}

// ---------- 审核 ----------
const review = reactive({
  visible: false,
  importId: "",
  productKey: "",
  loading: false,
  modules: [], // {id, name, text(可编辑), quotes, verified, adopt, currentText}
});

async function openReview(row) {
  review.visible = true;
  review.loading = true;
  review.importId = row.import_id;
  review.productKey = row.product_key;
  try {
    const detail = await api(`/api/kb-imports/${encodeURIComponent(row.import_id)}`);
    const draftModules = detail.draft?.modules || {};
    const current = (props.kbData?.[row.product_key] || {}).modules || {};
    review.modules = Object.entries(draftModules)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([id, mod]) => {
        const currentText = current[id]?.text || "";
        return {
          id,
          name: mod.name,
          text: mod.text,
          quotes: mod.quotes || [],
          verified: mod.verified,
          quotesHit: `${mod.quotes_verified ?? 0}/${mod.quotes_total ?? 0}`,
          currentText,
          isNew: !currentText,
          adopt: mod.verified, // 引用全部可核验的默认勾选，存疑的默认不勾
        };
      });
  } catch (error) {
    ElMessage.error(`加载草稿失败：${error.message}`);
    review.visible = false;
  } finally {
    review.loading = false;
  }
}

async function applyReview() {
  const adopted = Object.fromEntries(
    review.modules.filter((m) => m.adopt && m.text.trim()).map((m) => [m.id, m.text])
  );
  if (!Object.keys(adopted).length) {
    ElMessage.warning("请至少勾选一个模块");
    return;
  }
  try {
    const result = await api(`/api/kb-imports/${encodeURIComponent(review.importId)}/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modules: adopted, scope: props.scope }),
    });
    ElMessage.success(`已合并 ${result.applied_modules.length} 个模块到「${result.product_key}」`);
    review.visible = false;
    emit("applied");
    await loadImports();
  } catch (error) {
    ElMessage.error(`合并失败：${error.message}`);
  }
}
</script>

<template>
  <el-card shadow="never" style="margin-bottom: 16px">
    <template #header>
      <div style="display: flex; justify-content: space-between; align-items: baseline">
        <strong>从文档导入知识库</strong>
        <span class="muted" style="font-size: 12px">
          支持 PDF / PPTX / TXT / 图片（≤20MB，≤40页）· 多模态识别 → 结构化草稿 → 人工审核后才会合并
        </span>
      </div>
    </template>
    <el-space wrap size="large">
      <input ref="fileInput" type="file" accept=".pdf,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp" />
      <el-select
        v-model="productKey"
        placeholder="产品（可输入新产品名）"
        filterable
        allow-create
        default-first-option
        style="width: 220px"
      >
        <el-option v-for="key in productOptions" :key="key" :value="key" :label="key" />
      </el-select>
      <el-button type="primary" :loading="submitting" @click="submit">上传并识别</el-button>
    </el-space>

    <el-table v-if="imports.length" :data="imports" size="small" style="margin-top: 14px">
      <el-table-column prop="filename" label="文件" min-width="180" show-overflow-tooltip />
      <el-table-column prop="product_key" label="产品" width="120" />
      <el-table-column label="状态" width="90">
        <template #default="{ row }">
          <el-tag :type="STATUS_META[row.status]?.tone || 'info'" size="small">
            {{ row.applied_at ? "已合并" : STATUS_META[row.status]?.label || row.status }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="进度" width="160">
        <template #default="{ row }">
          <span v-if="row.status === 'running'" style="font-size: 12px">
            {{ STAGE_LABELS[row.stage] || row.stage }}
            <template v-if="row.stage === 'recognize' && row.page_count">
              {{ row.pages_done || 0 }}/{{ row.page_count }} 页
            </template>
          </span>
          <span v-else-if="row.status === 'failed'" class="muted" style="font-size: 12px">
            {{ (row.error || "").slice(0, 40) }}
          </span>
          <span v-else class="muted" style="font-size: 12px">{{ row.page_count ? `${row.page_count} 页` : "—" }}</span>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="时间" width="150" />
      <el-table-column label="操作" width="110">
        <template #default="{ row }">
          <el-button
            v-if="row.status === 'success'"
            link
            type="primary"
            size="small"
            @click="openReview(row)"
          >
            {{ row.applied_at ? "再次审核" : "审核合并" }}
          </el-button>
          <el-button
            v-if="row.status === 'failed'"
            link
            type="primary"
            size="small"
            @click="retryImport(row)"
          >
            断点重试
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog
      v-model="review.visible"
      :title="`审核知识库草稿 · ${review.productKey}`"
      width="880px"
      top="4vh"
    >
      <div v-loading="review.loading">
        <el-alert type="warning" :closable="false" style="margin-bottom: 12px">
          <template #title>
            <span style="font-size: 13px">
              知识库是准确率校验的<b>标准答案</b>——请核对每个模块再采纳。
              「引用核验」通过表示内容有原文依据；<b>存疑模块默认不勾选</b>，需人工确认。文本可直接修改。
            </span>
          </template>
        </el-alert>
        <div v-for="mod in review.modules" :key="mod.id" class="module-block">
          <div class="module-head">
            <el-checkbox v-model="mod.adopt">
              <b>{{ mod.id }} {{ mod.name }}</b>
            </el-checkbox>
            <el-space :size="6">
              <el-tag v-if="mod.isNew" size="small" type="success" effect="plain">新增模块</el-tag>
              <el-tag v-else size="small" type="warning" effect="plain">覆盖已有内容</el-tag>
              <el-tooltip :content="`原文引用可核验 ${mod.quotesHit}`" placement="top">
                <el-tag size="small" :type="mod.verified ? 'success' : 'danger'">
                  {{ mod.verified ? "引用核验通过" : `引用存疑 ${mod.quotesHit}` }}
                </el-tag>
              </el-tooltip>
            </el-space>
          </div>
          <el-row :gutter="12" v-if="!mod.isNew">
            <el-col :span="12">
              <div class="pane-label">当前知识库</div>
              <div class="current-text">{{ mod.currentText }}</div>
            </el-col>
            <el-col :span="12">
              <div class="pane-label">文档抽取（可编辑）</div>
              <el-input v-model="mod.text" type="textarea" :rows="6" />
            </el-col>
          </el-row>
          <template v-else>
            <el-input v-model="mod.text" type="textarea" :rows="5" />
          </template>
          <details v-if="mod.quotes.length" class="quotes">
            <summary>原文引用（{{ mod.quotes.length }}）</summary>
            <ul>
              <li v-for="(quote, index) in mod.quotes" :key="index">{{ quote }}</li>
            </ul>
          </details>
        </div>
      </div>
      <template #footer>
        <el-button @click="review.visible = false">取消</el-button>
        <el-button type="primary" @click="applyReview">
          合并勾选的模块（{{ review.modules.filter((m) => m.adopt).length }}）
        </el-button>
      </template>
    </el-dialog>
  </el-card>
</template>

<style scoped>
.module-block {
  border: 1px solid #e4e9ec;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 12px;
}
.module-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.pane-label {
  font-size: 12px;
  color: var(--geo-muted);
  margin-bottom: 4px;
}
.current-text {
  font-size: 13px;
  line-height: 1.6;
  background: #f6f8f8;
  border-radius: 6px;
  padding: 8px 10px;
  max-height: 148px;
  overflow-y: auto;
  white-space: pre-wrap;
}
.quotes {
  margin-top: 8px;
  font-size: 12px;
  color: var(--geo-muted);
}
.quotes ul {
  margin: 6px 0 0;
  padding-left: 18px;
}
</style>
