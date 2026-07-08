<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";

import { api, apiJson } from "@/api/client";

const router = useRouter();

// ---------- 选项与上传 ----------
const options = ref(null);
const fileInput = ref(null);
const defaultProduct = ref("");
const parsed = ref(null);
const parsing = ref(false);
const inputMode = ref("upload");

// 在线填报：直接在页面里编辑问题清单
const LEVEL_OPTIONS = ["病症", "品类", "品牌"];
const formRows = ref([
  { question: "", product: "", level: "病症", scenario: "" },
]);

const STAGE_LABELS_CN = { symptom: "病症", category: "品类", brand: "品牌" };

const reportWarnings = computed(() => {
  const report = parsed.value?.report;
  if (!report) return [];
  const warnings = [];
  if (report.duplicates_removed) warnings.push(`去除了 ${report.duplicates_removed} 个重复问题`);
  if (report.missing_product) warnings.push(`${report.missing_product} 个问题缺产品（请填默认产品名或补产品列）`);
  if (report.missing_level) warnings.push(`${report.missing_level} 个问题未填层级，默认按「病症」阶段处理`);
  if (report.unknown_levels?.length)
    warnings.push(`无法识别的层级值：${report.unknown_levels.join("、")}（已按「病症」处理；建议使用 病症/品类/品牌）`);
  return warnings;
});

const form = reactive({
  datasetName: "",
  productCode: "",
  batchDate: new Date().toISOString().slice(0, 10),
  searchMode: "both",
  rounds: 1,
  route: "relay",
  selectedModels: [],
  variantByModel: {},
  concurrency: 20,
  concurrencyByModel: {},
});

onMounted(async () => {
  try {
    options.value = await api("/api/jobs/options");
    form.rounds = options.value.default_rounds || 1;
    form.route = options.value.default_route || "relay";
    form.concurrency = options.value.default_concurrency || 20;
    form.selectedModels = options.value.models.map((m) => m.key);
    for (const model of options.value.models) {
      form.variantByModel[model.key] = model.default_model;
      form.concurrencyByModel[model.key] = options.value.default_concurrency || 20;
    }
  } catch (error) {
    ElMessage.error(`加载选项失败：${error.message}`);
  }
  await loadJobs();
  timer = setInterval(pollIfActive, 8000);
});

function afterParse() {
  ElMessage.success(`校验通过：共 ${parsed.value.total} 个问题`);
  // 依据解析出的产品自动预选主数据产品
  const firstCode = parsed.value.questions[0]?.product_code;
  if (firstCode && options.value?.products?.some((p) => p.product_code === firstCode)) {
    form.productCode = firstCode;
  }
}

async function parseFile() {
  const file = fileInput.value?.files?.[0];
  if (!file) {
    ElMessage.warning("请先选择问题文件");
    return;
  }
  parsing.value = true;
  try {
    const body = new FormData();
    body.append("file", file);
    body.append("default_product", defaultProduct.value.trim());
    parsed.value = await api("/api/jobs/parse", { method: "POST", body });
    afterParse();
  } catch (error) {
    parsed.value = null;
    ElMessage.error(`解析失败：${error.message}`);
  } finally {
    parsing.value = false;
  }
}

async function composeRows() {
  const rows = formRows.value.filter((row) => row.question.trim());
  if (!rows.length) {
    ElMessage.warning("请至少填写一个问题");
    return;
  }
  parsing.value = true;
  try {
    parsed.value = await apiJson("/api/jobs/compose", "POST", {
      rows,
      default_product: defaultProduct.value.trim(),
    });
    afterParse();
  } catch (error) {
    parsed.value = null;
    ElMessage.error(`校验失败：${error.message}`);
  } finally {
    parsing.value = false;
  }
}

const estimatedCalls = computed(() => {
  if (!parsed.value || !options.value) return 0;
  const models = options.value.models.filter((m) => form.selectedModels.includes(m.key));
  const searchable = models.filter((m) => m.supports_search).length;
  const plain = models.length - searchable;
  const perQuestion =
    form.searchMode === "both"
      ? searchable * 2 + plain
      : form.searchMode === "search"
        ? searchable
        : models.length;
  return parsed.value.total * perQuestion * form.rounds;
});

async function submitJob() {
  if (!parsed.value) return;
  if (!form.datasetName.trim()) {
    ElMessage.warning("请填写数据集名称");
    return;
  }
  try {
    await ElMessageBox.confirm(
      `将发起约 ${estimatedCalls.value} 次模型调用，任务完成后会自动生成分析数据。确认提交？`,
      "提交分析任务",
      { confirmButtonText: "提交", cancelButtonText: "再想想" }
    );
  } catch {
    return;
  }
  const overrides = {};
  for (const model of options.value.models) {
    if (
      form.selectedModels.includes(model.key) &&
      form.variantByModel[model.key] !== model.default_model
    ) {
      overrides[model.key] = form.variantByModel[model.key];
    }
  }
  const concurrencyOverrides = {};
  if (options.value.can_tune_concurrency) {
    for (const key of form.selectedModels) {
      if (form.concurrencyByModel[key] !== form.concurrency) {
        concurrencyOverrides[key] = form.concurrencyByModel[key];
      }
    }
  }
  try {
    await apiJson("/api/jobs", "POST", {
      dataset_name: form.datasetName.trim(),
      questions: parsed.value.questions,
      models: form.selectedModels,
      model_overrides: overrides,
      search_mode: form.searchMode,
      rounds: form.rounds,
      route: form.route,
      product_code: form.productCode || null,
      batch_date: form.batchDate,
      concurrency: form.concurrency,
      model_concurrency: concurrencyOverrides,
    });
    ElMessage.success("任务已提交，正在排队执行");
    parsed.value = null;
    form.datasetName = "";
    if (fileInput.value) fileInput.value.value = "";
    await loadJobs();
  } catch (error) {
    ElMessage.error(`提交失败：${error.message}`);
  }
}

// ---------- 任务列表（自动轮询） ----------
const jobs = ref([]);
const jobsLoading = ref(false);
let timer = null;

const STATUS_META = {
  queued: { label: "排队中", tone: "info" },
  running: { label: "执行中", tone: "warning" },
  success: { label: "已完成", tone: "success" },
  failed: { label: "失败", tone: "danger" },
  cancelled: { label: "已取消", tone: "info" },
};
const STAGE_LABELS = {
  collect: "采集回答",
  analyze: "统计报表",
  extract: "推荐抽取",
  verify: "准确率校验",
  import: "入库",
  materialize: "指标物化",
  done: "完成",
};
const STAGE_ORDER = ["collect", "analyze", "extract", "verify", "import", "materialize", "done"];

async function loadJobs() {
  jobsLoading.value = true;
  try {
    jobs.value = await api("/api/jobs");
  } catch (error) {
    ElMessage.error(`加载任务失败：${error.message}`);
  } finally {
    jobsLoading.value = false;
  }
}

function pollIfActive() {
  if (jobs.value.some((job) => job.status === "queued" || job.status === "running")) {
    loadJobs();
  }
}

onBeforeUnmount(() => clearInterval(timer));

function stageProgress(job) {
  if (job.status === "success") return 100;
  // 采集阶段占大头：有细粒度进度时按 5%~70% 区间映射
  if (job.stage === "collect" && job.collect_progress?.total) {
    const frac = job.collect_progress.done / job.collect_progress.total;
    return Math.round(5 + frac * 65);
  }
  const index = STAGE_ORDER.indexOf(job.stage);
  if (index < 0) return 5;
  return Math.round(((index + 1) / STAGE_ORDER.length) * 100);
}

function stageText(job) {
  const label = STAGE_LABELS[job.stage] || job.stage || "—";
  if (job.stage === "collect" && job.collect_progress?.total) {
    return `${label} ${job.collect_progress.done}/${job.collect_progress.total}`;
  }
  return label;
}

async function retryJob(job) {
  try {
    await api(`/api/jobs/${encodeURIComponent(job.job_id)}/retry`, { method: "POST" });
    ElMessage.success("已重新入队，将从断点继续");
    await loadJobs();
  } catch (error) {
    ElMessage.error(`重试失败：${error.message}`);
  }
}

async function cancelJob(job) {
  try {
    await ElMessageBox.confirm("确认取消该任务？已完成的调用会保留，之后可用「重试」从断点继续。", "取消任务", {
      type: "warning",
    });
  } catch {
    return;
  }
  try {
    await api(`/api/jobs/${encodeURIComponent(job.job_id)}/cancel`, { method: "POST" });
    ElMessage.success("已取消");
    await loadJobs();
  } catch (error) {
    ElMessage.error(`取消失败：${error.message}`);
  }
}

// ---------- 日志 ----------
const logDrawer = reactive({ visible: false, jobId: "", text: "" });

async function showLog(job) {
  logDrawer.jobId = job.job_id;
  logDrawer.visible = true;
  logDrawer.text = "加载中...";
  try {
    const result = await api(`/api/jobs/${encodeURIComponent(job.job_id)}/log`);
    logDrawer.text = result.log || "（暂无日志）";
  } catch (error) {
    logDrawer.text = `读取日志失败：${error.message}`;
  }
}

function viewDataset(job) {
  if (job.product_code) {
    router.push({ path: `/products/${job.product_code}`, query: { dataset_id: job.dataset_id } });
  } else {
    ElMessage.info("该任务未关联产品，请在工作台查看数据集");
  }
}
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>我的分析</h1>
      <p>上传问题清单 → 选择产品与模型 → 系统自动采集 AI 回答并生成三阶段分析。同一产品定期用相同问题集提交，即可积累趋势。</p>
    </div>

    <el-card shadow="never" class="block">
      <template #header>
        <div class="block-head">
          <strong>① 准备问题</strong>
          <span class="muted">
            层级说明：<b>病症</b>=泛式提问看品类推荐，<b>品类</b>=比较品牌，<b>品牌</b>=问具体产品（测准确率/负面）。
            轮次、模型、联网在第②步按整个批次统一设置，保证跨批次可比。
          </span>
        </div>
      </template>
      <el-tabs v-model="inputMode">
        <el-tab-pane label="上传文件" name="upload">
          <el-space wrap size="large">
            <input ref="fileInput" type="file" accept=".xlsx,.csv,.json" />
            <el-input
              v-model="defaultProduct"
              placeholder="默认产品名（文件里没有产品列时使用）"
              style="width: 260px"
            />
            <el-button type="primary" :loading="parsing" @click="parseFile">校验并预览</el-button>
          </el-space>
          <p class="muted" style="font-size: 12px; margin-bottom: 0">
            支持 Excel / CSV / JSON，单次上限 {{ options?.max_questions ?? 500 }} 题，列：问题 / 产品 / 层级 / 场景 · 模板：
            <a href="/api/templates/questions.xlsx">Excel</a> ·
            <a href="/api/templates/questions.csv">CSV</a> ·
            <a href="/api/templates/questions.json">JSON</a>
          </p>
        </el-tab-pane>
        <el-tab-pane label="在线填报" name="form">
          <el-table :data="formRows" size="small" border>
            <el-table-column label="问题" min-width="320">
              <template #default="{ row }">
                <el-input v-model="row.question" placeholder="如：感冒发烧吃什么药好得快？" size="small" />
              </template>
            </el-table-column>
            <el-table-column label="产品" width="170">
              <template #default="{ row }">
                <el-input v-model="row.product" placeholder="如 999感冒灵" size="small" />
              </template>
            </el-table-column>
            <el-table-column label="层级" width="110">
              <template #default="{ row }">
                <el-select v-model="row.level" size="small">
                  <el-option v-for="lv in LEVEL_OPTIONS" :key="lv" :value="lv" :label="lv" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="场景" width="140">
              <template #default="{ row }">
                <el-input v-model="row.scenario" placeholder="可选" size="small" />
              </template>
            </el-table-column>
            <el-table-column width="60">
              <template #default="{ $index }">
                <el-button link type="danger" size="small" @click="formRows.splice($index, 1)">删</el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-space style="margin-top: 10px">
            <el-button
              size="small"
              @click="formRows.push({ question: '', product: formRows.at(-1)?.product || '', level: formRows.at(-1)?.level || '病症', scenario: '' })"
            >
              + 添加问题
            </el-button>
            <el-button type="primary" size="small" :loading="parsing" @click="composeRows">
              校验并预览
            </el-button>
          </el-space>
        </el-tab-pane>
      </el-tabs>

      <template v-if="parsed">
        <el-divider content-position="left">校验结果</el-divider>
        <el-space wrap style="margin-bottom: 8px">
          <el-tag type="success">共 {{ parsed.total }} 题</el-tag>
          <el-tag v-for="(count, stage) in parsed.report?.stage_counts || {}" :key="stage" effect="plain">
            {{ STAGE_LABELS_CN[stage] }}阶段 {{ count }} 题
          </el-tag>
        </el-space>
        <el-alert
          v-for="warning in reportWarnings"
          :key="warning"
          type="warning"
          :title="warning"
          :closable="false"
          style="margin-bottom: 6px"
        />
        <el-table :data="parsed.preview" size="small" style="margin-top: 8px" max-height="260">
          <el-table-column prop="id" label="编号" width="180" />
          <el-table-column prop="product" label="产品" width="120" />
          <el-table-column prop="level" label="层级" width="90" />
          <el-table-column prop="question" label="问题" min-width="300" show-overflow-tooltip />
        </el-table>
      </template>
    </el-card>

    <el-card v-if="parsed && options" shadow="never" class="block">
      <template #header>
        <div class="block-head">
          <strong>② 配置分析任务</strong>
          <span class="muted">关联产品后，结果会出现在该产品的详情页和趋势里</span>
        </div>
      </template>
      <el-form label-width="110px" label-position="left" style="max-width: 860px">
        <el-row :gutter="24">
          <el-col :span="12">
            <el-form-item label="数据集名称" required>
              <el-input v-model="form.datasetName" placeholder="如：感冒灵摸底0707" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="关联产品">
              <el-select v-model="form.productCode" placeholder="选择产品（趋势锚点）" clearable style="width: 100%">
                <el-option
                  v-for="product in options.products"
                  :key="product.product_code"
                  :value="product.product_code"
                  :label="product.product_name"
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="批次日期">
              <el-date-picker
                v-model="form.batchDate"
                type="date"
                value-format="YYYY-MM-DD"
                style="width: 100%"
              />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="联网模式">
              <el-select v-model="form.searchMode" style="width: 100%">
                <el-option value="both" label="联网 + 非联网" />
                <el-option value="search" label="仅联网" />
                <el-option value="nosearch" label="仅非联网" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="每题轮次">
              <el-select v-model="form.rounds" style="width: 100%">
                <el-option v-for="n in [1, 2, 3, 5]" :key="n" :value="n" :label="`${n} 轮`" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8" v-if="options.can_choose_route">
            <el-form-item label="调用链路">
              <el-select v-model="form.route" style="width: 100%">
                <el-option value="relay" label="new-api 中继（默认）" />
                <el-option value="direct" label="厂商直连" />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8" v-if="options.can_tune_concurrency">
            <el-form-item label="并发数">
              <el-input-number
                v-model="form.concurrency"
                :min="1"
                :max="options.max_concurrency || 50"
                style="width: 100%"
                @change="(v) => { for (const k of Object.keys(form.concurrencyByModel)) form.concurrencyByModel[k] = v; }"
              />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="模型">
          <div class="model-picker">
            <div v-for="model in options.models" :key="model.key" class="model-option">
              <el-checkbox
                :model-value="form.selectedModels.includes(model.key)"
                @change="(checked) => {
                  form.selectedModels = checked
                    ? [...form.selectedModels, model.key]
                    : form.selectedModels.filter((k) => k !== model.key);
                }"
              >
                {{ model.name }}
                <el-tag size="small" :type="model.supports_search ? 'success' : 'info'" effect="plain">
                  {{ model.supports_search ? "支持联网" : "不联网" }}
                </el-tag>
              </el-checkbox>
              <el-space :size="6">
                <el-select v-model="form.variantByModel[model.key]" size="small" style="width: 190px">
                  <el-option
                    v-for="variant in model.variants"
                    :key="variant.id"
                    :value="variant.id"
                    :label="variant.label"
                  />
                </el-select>
                <el-tooltip v-if="options.can_tune_concurrency" content="该模型并发数" placement="top">
                  <el-input-number
                    v-model="form.concurrencyByModel[model.key]"
                    :min="1"
                    :max="options.max_concurrency || 50"
                    size="small"
                    :controls="false"
                    style="width: 58px"
                  />
                </el-tooltip>
              </el-space>
            </div>
          </div>
          <p class="muted" style="font-size: 12px; margin: 6px 0 0; width: 100%">
            所有平台同时并发采集{{ options.can_tune_concurrency ? "，可全局或按模型调整（遇限流会自适应降速）" : "，默认每模型并发 " + (options.default_concurrency || 20) }}。
          </p>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="submitJob">
            提交分析任务（约 {{ estimatedCalls }} 次调用）
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="block">
      <template #header>
        <div class="block-head">
          <strong>任务列表</strong>
          <span class="muted">采集 → 报表 → 抽取 → 校验 → 入库 → 物化；执行中自动刷新</span>
        </div>
      </template>
      <el-table :data="jobs" v-loading="jobsLoading" size="small">
        <el-table-column prop="dataset_name" label="数据集" min-width="150" show-overflow-tooltip />
        <el-table-column prop="username" label="提交人" width="90" />
        <el-table-column label="状态" width="90">
          <template #default="{ row }">
            <el-tag :type="STATUS_META[row.status]?.tone || 'info'" size="small">
              {{ STATUS_META[row.status]?.label || row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" min-width="190">
          <template #default="{ row }">
            <el-progress
              :percentage="stageProgress(row)"
              :status="row.status === 'failed' ? 'exception' : row.status === 'success' ? 'success' : undefined"
              :stroke-width="8"
            >
              <span style="font-size: 12px">{{ stageText(row) }}</span>
            </el-progress>
          </template>
        </el-table-column>
        <el-table-column prop="question_count" label="题数" width="60" />
        <el-table-column prop="batch_date" label="批次" width="100" />
        <el-table-column prop="created_at" label="创建时间" width="150" />
        <el-table-column label="操作" width="230" fixed="right">
          <template #default="{ row }">
            <el-button link size="small" @click="showLog(row)">日志</el-button>
            <el-button
              v-if="row.status === 'success'"
              link
              type="primary"
              size="small"
              @click="viewDataset(row)"
            >
              查看分析 →
            </el-button>
            <el-button
              v-if="row.status === 'failed' || row.status === 'cancelled'"
              link
              type="primary"
              size="small"
              @click="retryJob(row)"
            >
              断点重试
            </el-button>
            <el-button
              v-if="row.status === 'queued' || row.status === 'running'"
              link
              type="danger"
              size="small"
              @click="cancelJob(row)"
            >
              取消
            </el-button>
            <el-tooltip v-if="row.error && row.status === 'failed'" :content="row.error" placement="top">
              <el-button link type="danger" size="small">错误</el-button>
            </el-tooltip>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-drawer v-model="logDrawer.visible" :title="`任务日志 · ${logDrawer.jobId}`" size="55%">
      <pre class="log-view">{{ logDrawer.text }}</pre>
    </el-drawer>
  </div>
</template>

<style scoped>
.block {
  margin-bottom: 16px;
}
.block-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
}
.block-head .muted {
  font-size: 12px;
}
.model-picker {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  gap: 10px;
  width: 100%;
}
.model-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  border: 1px solid #e4e9ec;
  border-radius: 8px;
  padding: 8px 12px;
}
.log-view {
  background: #101418;
  color: #d5e0e6;
  padding: 14px;
  border-radius: 8px;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  max-height: 78vh;
  overflow-y: auto;
}
</style>
