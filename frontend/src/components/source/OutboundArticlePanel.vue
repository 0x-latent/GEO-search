<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";

import { api, query } from "@/api/client";
import AnswerDialog from "@/components/product/AnswerDialog.vue";
import { fmtNumber, fmtRate } from "@/utils/format";

const loading = ref(false);
const importing = ref(false);
const options = ref({ datasets: [], products: [], models: [] });
const dashboard = ref({ summary: {}, articles: [], platforms: [] });
const uploadRef = ref(null);
const selectedFile = ref(null);
const filters = reactive({ datasetIds: [], productCodes: [], models: [], searchModes: [] });
const form = reactive({
  title: "",
  platform: "",
  url: "",
  publishedAt: "",
  productCode: "",
  campaign: "",
});
const importVisible = ref(false);
const citationDrawer = reactive({ visible: false, loading: false, title: "", rows: [] });
const answerDialog = reactive({
  visible: false, datasetId: "", questionId: "", model: "",
  searchEnabled: null, round: null,
});

const summary = computed(() => dashboard.value?.summary || {});
const platformSuggestions = computed(() => dashboard.value?.platforms || []);

function csv(values) {
  return values?.length ? values.join(",") : "";
}

const filterParams = computed(() => ({
  dataset_ids: csv(filters.datasetIds),
  product_codes: csv(filters.productCodes),
  models: csv(filters.models),
  search_modes: csv(filters.searchModes),
}));

async function loadOptions() {
  options.value = await api("/api/insight/sources/options");
}

async function loadDashboard() {
  loading.value = true;
  try {
    dashboard.value = await api(`/api/outbound-articles${query(filterParams.value)}`);
  } catch (error) {
    ElMessage.error(`加载文章追踪失败：${error.message}`);
  } finally {
    loading.value = false;
  }
}

function resetFilters() {
  filters.datasetIds = [];
  filters.productCodes = [];
  filters.models = [];
  filters.searchModes = [];
  loadDashboard();
}

function handleFileChange(file) {
  selectedFile.value = file;
}

function handleFileRemove() {
  selectedFile.value = null;
}

function resetImportForm() {
  Object.assign(form, {
    title: "", platform: "", url: "", publishedAt: "",
    productCode: "", campaign: "",
  });
  selectedFile.value = null;
  uploadRef.value?.clearFiles();
}

async function submitImport() {
  if (!selectedFile.value?.raw) {
    ElMessage.warning("请选择 MD、TXT、DOCX 或 PDF 文件");
    return;
  }
  if (!form.platform.trim() || !form.url.trim()) {
    ElMessage.warning("请填写发布平台和发布链接");
    return;
  }
  const body = new FormData();
  body.append("file", selectedFile.value.raw);
  body.append("platform", form.platform.trim());
  body.append("url", form.url.trim());
  for (const [key, value] of Object.entries({
    published_at: form.publishedAt,
    title: form.title.trim(),
    product_code: form.productCode,
    campaign: form.campaign.trim(),
  })) {
    if (value) body.append(key, value);
  }
  importing.value = true;
  try {
    await api("/api/outbound-articles", { method: "POST", body });
    ElMessage.success("文章已导入，并已完成现有 AI 信源匹配");
    importVisible.value = false;
    resetImportForm();
    await loadDashboard();
  } catch (error) {
    ElMessage.error(`导入失败：${error.message}`);
  } finally {
    importing.value = false;
  }
}

async function openCitations(row) {
  citationDrawer.visible = true;
  citationDrawer.loading = true;
  citationDrawer.title = row.title;
  try {
    citationDrawer.rows = await api(
      `/api/outbound-articles/${encodeURIComponent(row.article_id)}/citations${query(filterParams.value)}`,
    );
  } catch (error) {
    ElMessage.error(`加载引用明细失败：${error.message}`);
  } finally {
    citationDrawer.loading = false;
  }
}

function openAnswer(row) {
  answerDialog.datasetId = row.dataset_id;
  answerDialog.questionId = row.question_id;
  answerDialog.model = row.model;
  answerDialog.searchEnabled = row.search_enabled;
  answerDialog.round = row.round;
  answerDialog.visible = true;
}

async function deleteArticle(row) {
  try {
    await ElMessageBox.confirm(
      `确认删除文章“${row.title}”及其引用匹配记录？`,
      "删除外发文章",
      { type: "warning", confirmButtonText: "删除", confirmButtonClass: "el-button--danger" },
    );
  } catch {
    return;
  }
  try {
    await api(`/api/outbound-articles/${encodeURIComponent(row.article_id)}`, { method: "DELETE" });
    ElMessage.success("已删除");
    await loadDashboard();
  } catch (error) {
    ElMessage.error(`删除失败：${error.message}`);
  }
}

function primaryPublication(row) {
  return row.publications?.[0] || {};
}

onMounted(async () => {
  try {
    await Promise.all([loadOptions(), loadDashboard()]);
  } catch (error) {
    ElMessage.error(`加载筛选项失败：${error.message}`);
  }
});
</script>

<template>
  <section class="article-panel" v-loading="loading">
    <el-card shadow="never" class="toolbar-card">
      <div class="toolbar">
        <div>
          <strong>外发文章资产库</strong>
          <p>导入已发布内容，按新一轮 AI 回答的信源 URL 追踪引用。</p>
        </div>
        <el-button type="primary" @click="importVisible = true">导入文章</el-button>
      </div>
      <div class="filter-grid">
        <el-select v-model="filters.datasetIds" multiple collapse-tags clearable filterable placeholder="全部查询批次">
          <el-option
            v-for="item in options.datasets" :key="item.dataset_id" :value="item.dataset_id"
            :label="`${item.batch_date || '未标日期'} · ${item.name}`"
          />
        </el-select>
        <el-select v-model="filters.productCodes" multiple collapse-tags clearable filterable placeholder="全部产品">
          <el-option v-for="item in options.products" :key="item.product_code" :value="item.product_code" :label="item.product_name" />
        </el-select>
        <el-select v-model="filters.models" multiple collapse-tags clearable filterable placeholder="全部模型">
          <el-option v-for="item in options.models" :key="item.model" :value="item.model" :label="item.model_name" />
        </el-select>
        <el-select v-model="filters.searchModes" multiple collapse-tags clearable placeholder="联网+非联网">
          <el-option :value="1" label="联网" />
          <el-option :value="0" label="非联网" />
        </el-select>
        <el-space>
          <el-button type="primary" @click="loadDashboard">应用筛选</el-button>
          <el-button @click="resetFilters">重置</el-button>
        </el-space>
      </div>
    </el-card>

    <div class="metric-cards">
      <el-card shadow="never"><small>导入文章</small><strong>{{ fmtNumber(summary.total_articles) }}</strong></el-card>
      <el-card shadow="never"><small>被引用文章</small><strong>{{ fmtNumber(summary.cited_articles) }}</strong></el-card>
      <el-card shadow="never"><small>文章引用率</small><strong>{{ fmtRate(summary.citation_rate) }}</strong></el-card>
      <el-card shadow="never"><small>引用回答</small><strong>{{ fmtNumber(summary.citation_answers) }}</strong></el-card>
      <el-card shadow="never"><small>信源引用次数</small><strong>{{ fmtNumber(summary.citation_refs) }}</strong></el-card>
      <el-card shadow="never"><small>发布平台</small><strong>{{ fmtNumber(summary.platforms) }}</strong></el-card>
    </div>

    <el-card shadow="never">
      <template #header>
        <div class="card-head">
          <strong>文章引用表现</strong>
          <span>当前只统计规范化 URL 完全一致的确定引用</span>
        </div>
      </template>
      <el-empty v-if="!dashboard.articles?.length" description="还没有导入外发文章" />
      <el-table v-else :data="dashboard.articles" size="small" max-height="620">
        <el-table-column label="文章" min-width="250" fixed>
          <template #default="{ row }">
            <el-button link type="primary" @click="openCitations(row)">{{ row.title }}</el-button>
            <div class="subtext">{{ row.source_filename }}</div>
          </template>
        </el-table-column>
        <el-table-column label="平台" min-width="130">
          <template #default="{ row }">
            <el-tag v-for="platform in row.platforms" :key="platform" size="small" effect="plain">{{ platform }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="发布链接" min-width="190" show-overflow-tooltip>
          <template #default="{ row }">
            <a :href="primaryPublication(row).url" target="_blank" rel="noopener">{{ primaryPublication(row).url }}</a>
          </template>
        </el-table-column>
        <el-table-column label="发布时间" width="165">
          <template #default="{ row }">{{ primaryPublication(row).published_at || "—" }}</template>
        </el-table-column>
        <el-table-column prop="product_name" label="产品" width="120">
          <template #default="{ row }">{{ row.product_name || "—" }}</template>
        </el-table-column>
        <el-table-column prop="campaign" label="活动" width="120" show-overflow-tooltip />
        <el-table-column prop="citation_answers" label="引用回答" width="100" sortable />
        <el-table-column prop="citation_questions" label="引用问题" width="100" sortable />
        <el-table-column label="状态" width="90">
          <template #default="{ row }">
            <el-tag :type="row.citation_refs ? 'success' : 'info'" size="small">
              {{ row.citation_refs ? "已引用" : "未引用" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="80" fixed="right">
          <template #default="{ row }"><el-button link type="danger" @click="deleteArticle(row)">删除</el-button></template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="importVisible" title="导入外发文章" width="620px" @closed="resetImportForm">
      <el-form label-position="top">
        <el-form-item label="文章文件" required>
          <el-upload
            ref="uploadRef" drag :auto-upload="false" :limit="1"
            accept=".md,.markdown,.txt,.docx,.pdf"
            :on-change="handleFileChange" :on-remove="handleFileRemove"
          >
            <div class="upload-copy">拖入文件或点击选择</div>
            <template #tip><span>支持 Markdown、TXT、Word（.docx）和 PDF，最大 20MB；扫描 PDF 可导入但不抽取正文</span></template>
          </el-upload>
        </el-form-item>
        <el-row :gutter="14">
          <el-col :span="12">
            <el-form-item label="发布平台" required>
              <el-select v-model="form.platform" filterable allow-create default-first-option placeholder="如：微信公众号">
                <el-option v-for="item in platformSuggestions" :key="item" :value="item" :label="item" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="发布时间">
              <el-date-picker
                v-model="form.publishedAt" type="datetime" value-format="YYYY-MM-DDTHH:mm:ss"
                placeholder="选择发布时间" style="width: 100%"
              />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="发布链接" required>
          <el-input v-model="form.url" placeholder="https://..." />
        </el-form-item>
        <el-form-item label="文章标题">
          <el-input v-model="form.title" placeholder="留空时从文档首行自动识别" />
        </el-form-item>
        <el-row :gutter="14">
          <el-col :span="12">
            <el-form-item label="关联产品">
              <el-select v-model="form.productCode" clearable filterable placeholder="可选">
                <el-option v-for="item in options.products" :key="item.product_code" :value="item.product_code" :label="item.product_name" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="12"><el-form-item label="活动/项目"><el-input v-model="form.campaign" placeholder="可选" /></el-form-item></el-col>
        </el-row>
      </el-form>
      <template #footer>
        <el-button @click="importVisible = false">取消</el-button>
        <el-button type="primary" :loading="importing" @click="submitImport">导入并匹配</el-button>
      </template>
    </el-dialog>

    <el-drawer v-model="citationDrawer.visible" :title="`引用明细 · ${citationDrawer.title}`" size="70%">
      <el-table v-loading="citationDrawer.loading" :data="citationDrawer.rows" size="small" @row-click="openAnswer" :row-style="{ cursor: 'pointer' }">
        <el-table-column prop="question_text" label="查询问题" min-width="260" show-overflow-tooltip />
        <el-table-column prop="model_name" label="模型" width="120" />
        <el-table-column prop="dataset_name" label="查询批次" min-width="150" show-overflow-tooltip />
        <el-table-column prop="platform" label="平台" width="110" />
        <el-table-column prop="source_title" label="信源标题" min-width="180" show-overflow-tooltip />
        <el-table-column prop="cited_at" label="回答时间" width="165" />
        <el-table-column label="匹配" width="90"><template #default><el-tag size="small" type="success">精确 URL</el-tag></template></el-table-column>
      </el-table>
      <el-empty v-if="!citationDrawer.loading && !citationDrawer.rows.length" description="当前筛选范围内还未发现引用" />
    </el-drawer>

    <AnswerDialog
      v-model="answerDialog.visible" :dataset-id="answerDialog.datasetId"
      :question-id="answerDialog.questionId" :model="answerDialog.model"
      :search-enabled="answerDialog.searchEnabled" :round="answerDialog.round"
    />
  </section>
</template>

<style scoped>
.article-panel { display: grid; gap: 14px; }
.toolbar { display: flex; justify-content: space-between; align-items: center; gap: 20px; margin-bottom: 14px; }
.toolbar p { margin: 5px 0 0; color: #64748b; font-size: 13px; }
.filter-grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)) auto; gap: 10px; align-items: center; }
.metric-cards { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 12px; }
.metric-cards :deep(.el-card__body) { display: grid; gap: 5px; }
.metric-cards small { color: #64748b; font-size: 12px; }
.metric-cards strong { color: #0f3d38; font-size: 24px; }
.card-head { display: flex; justify-content: space-between; gap: 12px; }
.card-head span, .subtext { color: #94a3b8; font-size: 12px; }
.subtext { margin-left: 12px; }
.el-tag + .el-tag { margin-left: 4px; }
a { color: #0f766e; text-decoration: none; }
.upload-copy { padding: 12px; color: #475569; }
:deep(.el-upload), :deep(.el-upload-dragger) { width: 100%; }
@media (max-width: 1200px) {
  .filter-grid { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
  .metric-cards { grid-template-columns: repeat(3, minmax(130px, 1fr)); }
}
</style>
