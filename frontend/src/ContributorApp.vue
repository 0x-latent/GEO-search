<script setup>
import { onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";

const base = window.location.pathname.replace(/[^/]*$/, "");
const url = (path) => base + path.replace(/^\//, "");
const workspace = ref(null);
const submissions = ref([]);
const loading = ref(true);
const error = ref("");
const selected = ref(null);
const file = ref(null);
const revisionFile = ref(null);
const form = reactive({
  product_code: "", title: "", campaign: "", submitter_name: "",
  submitter_email: "", published_platform: "", published_url: "", published_at: "", planned_platform: "", planned_at: "",
});
const publication = reactive({ platform: "", url: "", published_at: "", evidence: "" });
let refreshTimer;

const statusText = {
  queued: "已排队", reviewing: "AI 审稿中", awaiting_admin: "待指定人员审批",
  revision_requested: "请修改", approved_waiting_publication: "审核通过，待发布",
  rejected: "已拒绝", tracked: "已通过并进入引用追踪", review_failed: "审稿失败",
  blocked_missing_kb: "产品知识库待补充",
  publication_pending: "待发布验收", publication_accepted: "验收通过，正在登记",
};

async function request(path, options = {}) {
  const response = await fetch(url(path), options);
  if (!response.ok) {
    let message = response.statusText;
    try { message = (await response.json()).detail || message; } catch { /* noop */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

async function load() {
  workspace.value = await request("api/contributor/context");
  submissions.value = await request("api/contributor/submissions");
  if (!form.product_code && workspace.value.products.length) {
    form.product_code = workspace.value.products[0].product_code;
  }
}

async function initialize() {
  try {
    const fragment = new URLSearchParams(window.location.hash.slice(1));
    const inviteId = fragment.get("invite");
    const token = fragment.get("token");
    if (inviteId && token) {
      await request("api/contributor/session", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ invite_id: inviteId, token }),
      });
      history.replaceState(null, "", window.location.pathname + window.location.search);
    }
    await load();
  } catch (e) {
    error.value = e.message || "邀请链接无效或已过期";
  } finally { loading.value = false; }
}

async function submit() {
  if (!file.value) return ElMessage.warning("请选择文章文件");
  const data = new FormData();
  data.append("file", file.value);
  Object.entries(form).forEach(([key, value]) => { if (value) data.append(key, value); });
  try {
    await request("api/contributor/submissions", { method: "POST", body: data });
    ElMessage.success("投稿成功，AI 审稿已进入队列");
    form.title = ""; form.campaign = ""; form.published_platform = ""; form.published_url = "";
    file.value = null;
    await load();
  } catch (e) { ElMessage.error(e.message); }
}

async function openDetail(row) {
  selected.value = await request(`api/contributor/submissions/${row.submission_id}`);
  publication.platform = row.published_platform || "";
  publication.url = row.published_url || "";
  publication.published_at = row.published_at || "";
  publication.evidence = row.publication_evidence || "";
}

async function uploadRevision() {
  if (!revisionFile.value) return ElMessage.warning("请选择修订文件");
  const data = new FormData(); data.append("file", revisionFile.value);
  try {
    selected.value = await request(`api/contributor/submissions/${selected.value.submission_id}/revision`, { method: "POST", body: data });
    ElMessage.success("修订版已上传并重新进入审稿队列"); await load();
  } catch (e) { ElMessage.error(e.message); }
}

async function savePublication() {
  try {
    selected.value = await request(`api/contributor/submissions/${selected.value.submission_id}/publication`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...publication, expected_version: selected.value.current_version }),
    });
    ElMessage.success("发布信息已保存"); await load();
  } catch (e) { ElMessage.error(e.message); }
}

onMounted(async () => {
  await initialize();
  refreshTimer = window.setInterval(() => { if (workspace.value && !selected.value) load().catch(() => {}); }, 10000);
});
onBeforeUnmount(() => window.clearInterval(refreshTimer));
</script>

<template>
  <div class="portal-shell">
    <header><div><strong>外部文章投稿</strong><span v-if="workspace"> · {{ workspace.company_name }}</span></div><small>安全邀请工作区</small></header>
    <main v-loading="loading">
      <el-result v-if="error" icon="error" title="无法进入投稿工作区" :sub-title="error" />
      <template v-else-if="workspace">
        <el-alert type="info" :closable="false" show-icon title="上传后将启动 Agent 核验，由指定人员审批。收到可发布通知后，请按获批版本发布。支持 MD、TXT、DOCX、PDF，最大 20MB。" />
        <el-card v-if="workspace.project" shadow="never" class="spaced"><h2>{{ workspace.project.name }}</h2><p style="white-space:pre-wrap">{{ workspace.project.brief }}</p><p>发布渠道：{{ workspace.project.channels.join('、') || '按项目要求' }}</p><p v-if="workspace.project.target_questions">目标问题：{{ workspace.project.target_questions }}</p><p v-if="workspace.project.submission_deadline">投稿截止：{{ workspace.project.submission_deadline }}</p><el-tag v-if="workspace.project.status==='archived'">项目已归档，仅可查看历史</el-tag></el-card>
        <el-row :gutter="20" class="content-row">
          <el-col :xs="24" :lg="10">
            <el-card shadow="never">
              <template #header><strong>提交文章</strong></template>
              <el-form label-position="top">
                <el-form-item label="主产品"><el-select v-model="form.product_code" style="width:100%"><el-option v-for="p in workspace.products" :key="p.product_code" :label="p.product_name" :value="p.product_code" /></el-select></el-form-item>
                <el-form-item label="文章标题"><el-input v-model="form.title" maxlength="300" show-word-limit /></el-form-item>
                <el-form-item label="文章文件"><input type="file" accept=".md,.markdown,.txt,.docx,.pdf" @change="file=$event.target.files[0]" /></el-form-item>
                <el-row :gutter="12"><el-col :span="12"><el-form-item label="投稿人姓名"><el-input v-model="form.submitter_name" /></el-form-item></el-col><el-col :span="12"><el-form-item label="投稿人邮箱"><el-input v-model="form.submitter_email" /></el-form-item></el-col></el-row>
                <template v-if="workspace.project"><el-form-item label="计划发布渠道"><el-select v-if="workspace.project.channels.length" v-model="form.planned_platform"><el-option v-for="c in workspace.project.channels" :key="c" :label="c" :value="c" /></el-select><el-input v-else v-model="form.planned_platform" /></el-form-item><el-form-item label="计划发布时间"><el-date-picker v-model="form.planned_at" type="datetime" value-format="YYYY-MM-DDTHH:mm:ssZ" /></el-form-item></template>
                <template v-else><el-form-item label="项目/活动（可选）"><el-input v-model="form.campaign" /></el-form-item>
                <el-divider content-position="left">已发布时填写</el-divider>
                <el-row :gutter="12"><el-col :span="10"><el-form-item label="发布平台"><el-input v-model="form.published_platform" /></el-form-item></el-col><el-col :span="14"><el-form-item label="发布 URL"><el-input v-model="form.published_url" /></el-form-item></el-col></el-row>
                </template>
                <el-button type="primary" size="large" style="width:100%" :disabled="workspace.project?.status==='archived' || !workspace.remaining_submissions" @click="submit">提交并启动 AI 审稿</el-button>
              </el-form>
            </el-card>
          </el-col>
          <el-col :xs="24" :lg="14">
            <el-card shadow="never"><template #header><strong>投稿记录</strong><span class="remaining">剩余 {{ workspace.remaining_submissions }} 次</span></template>
              <el-table :data="submissions" @row-click="openDetail" style="cursor:pointer">
                <el-table-column prop="title" label="标题" min-width="180" show-overflow-tooltip />
                <el-table-column label="状态" width="170"><template #default="{row}"><el-tag>{{ statusText[row.status] || row.status }}</el-tag></template></el-table-column>
                <el-table-column prop="current_version" label="版本" width="65"><template #default="{row}">v{{ row.current_version }}</template></el-table-column>
                <el-table-column prop="updated_at" label="更新时间" width="180" />
              </el-table>
            </el-card>
          </el-col>
        </el-row>
      </template>
    </main>

    <el-drawer v-model="selected" size="min(680px, 92vw)" :title="selected?.title">
      <template v-if="selected">
        <el-descriptions :column="1" border><el-descriptions-item label="状态">{{ statusText[selected.status] }}</el-descriptions-item><el-descriptions-item label="版本">v{{ selected.current_version }}</el-descriptions-item><el-descriptions-item v-if="selected.admin_feedback" label="管理员意见">{{ selected.admin_feedback }}</el-descriptions-item></el-descriptions>
        <section v-if="selected.findings?.length"><h3>管理员确认的问题</h3><el-card v-for="item in selected.findings" :key="item.finding_id" shadow="never" class="finding"><el-tag :type="item.severity==='high'?'danger':'warning'">{{ item.severity }}</el-tag><p>{{ item.excerpt }}</p><p class="muted">{{ item.suggestion }}</p></el-card></section>
        <p><a :href="url(`api/contributor/submissions/${selected.submission_id}/file`)">下载当前版本原稿（v{{ selected.current_version }}）</a></p>
        <section v-if="workspace?.project?.status!=='archived' && (['revision_requested','review_failed','blocked_missing_kb'].includes(selected.status) || (selected.project_id && ['approved_waiting_publication','publication_pending'].includes(selected.status)))"><h3>上传修订版</h3><p v-if="selected.approved_version">修改正文会使当前批准失效，需重新核验和审批。</p><input type="file" accept=".md,.markdown,.txt,.docx,.pdf" @change="revisionFile=$event.target.files[0]" /> <el-button type="primary" @click="uploadRevision">上传并重审</el-button></section>
        <section v-if="selected.status==='approved_waiting_publication' && workspace?.project?.status!=='archived'"><h3>补充发布信息</h3><el-input v-model="publication.platform" placeholder="发布平台" /><el-input v-model="publication.url" placeholder="https://..." class="spaced" /><el-date-picker v-model="publication.published_at" type="datetime" value-format="YYYY-MM-DDTHH:mm:ssZ" placeholder="实际发布时间" /><el-input v-if="selected.project_id" v-model="publication.evidence" type="textarea" placeholder="发布凭证说明或凭证链接" class="spaced" /><el-button type="primary" @click="savePublication">{{ selected.project_id ? '提交发布验收' : '保存并进入引用追踪' }}</el-button></section>
      </template>
    </el-drawer>
  </div>
</template>

<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#25324a;font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif}.portal-shell header{height:64px;background:#152b4f;color:white;padding:0 max(24px,calc((100vw - 1280px)/2));display:flex;align-items:center;justify-content:space-between;font-size:20px}.portal-shell header small{font-size:12px;opacity:.75}.portal-shell main{max-width:1280px;margin:24px auto;padding:0 20px}.content-row{margin-top:18px}.remaining{float:right;color:#7a8499;font-size:13px}.finding{margin:10px 0}.muted{color:#778197}.spaced{margin:10px 0}@media(max-width:1199px){.el-col{margin-bottom:18px}}
</style>
