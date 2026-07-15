<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { api, apiJson } from "@/api/client";

const submissions = ref([]);
const companies = ref([]);
const invites = ref([]);
const products = ref([]);
const dashboard = ref(null);
const detail = ref(null);
const companyName = ref("");
const invite = reactive({ company_id: "", allowed_product_codes: [], expires_at: "", max_submissions: 20 });
const feedback = ref("");
const selectedFindings = ref([]);
const status = ref("");
const inviteLink = ref("");
const loading = ref(false);
const statusOptions = ["queued","reviewing","awaiting_admin","revision_requested","approved_waiting_publication","rejected","tracked","review_failed","blocked_missing_kb"];
const portalBase = window.location.pathname.replace(/[^/]*$/, "");

async function load() {
  loading.value = true;
  try {
    [submissions.value, companies.value, invites.value, dashboard.value] = await Promise.all([
      api(`/api/admin/article-submissions${status.value ? `?status=${status.value}` : ""}`),
      api("/api/admin/contributor-companies"), api("/api/admin/contributor-invites"),
      api("/api/admin/article-review/dashboard"),
    ]);
    const options = await api("/api/jobs/options"); products.value = options.products;
  } catch (e) { ElMessage.error(e.message); }
  finally { loading.value = false; }
}

async function open(row) {
  detail.value = await api(`/api/admin/article-submissions/${row.submission_id}`);
  feedback.value = ""; selectedFindings.value = [];
}

async function createCompany() {
  if (!companyName.value.trim()) return;
  try { const company = await apiJson("/api/admin/contributor-companies", "POST", { name: companyName.value }); companyName.value = ""; invite.company_id = company.company_id; await load(); ElMessage.success("公司已创建"); }
  catch (e) { ElMessage.error(e.message); }
}

async function createInvite() {
  try {
    const result = await apiJson("/api/admin/contributor-invites", "POST", invite);
    inviteLink.value = `${window.location.origin}${portalBase}article-submit.html#invite=${encodeURIComponent(result.invite_id)}&token=${encodeURIComponent(result.token)}`;
    await navigator.clipboard?.writeText(inviteLink.value);
    ElMessage.success("安全邀请链接已生成并复制；密钥仅显示这一次");
    invites.value = await api("/api/admin/contributor-invites");
  } catch (e) { ElMessage.error(e.message); }
}

async function revokeInvite(inviteId) {
  try {
    await ElMessageBox.confirm("撤销后该链接及其所有外部会话会立即失效。", "撤销邀请", { type: "warning" });
    await api(`/api/admin/contributor-invites/${inviteId}`, { method: "DELETE" });
    ElMessage.success("邀请已撤销"); invites.value = await api("/api/admin/contributor-invites");
  } catch (e) { if (e !== "cancel") ElMessage.error(e.message || e); }
}

async function action(name) {
  try {
    if (name === "reject") await ElMessageBox.confirm("确认拒绝这篇投稿？", "拒绝投稿", { type: "warning" });
    detail.value = await apiJson(`/api/admin/article-submissions/${detail.value.submission_id}/${name === "request_revision" ? "request-revision" : name}`, "POST", { feedback: feedback.value, finding_ids: selectedFindings.value });
    ElMessage.success("审核决定已保存"); await load();
  } catch (e) { if (e !== "cancel") ElMessage.error(e.message || e); }
}

async function simpleAction(name) {
  try { await api(`/api/admin/article-submissions/${detail.value.submission_id}/${name}`, { method: "POST" }); ElMessage.success("操作成功"); detail.value = null; await load(); }
  catch (e) { ElMessage.error(e.message); }
}

const counts = computed(() => dashboard.value?.counts || {});
onMounted(load);
</script>

<template>
  <div v-loading="loading">
    <el-row :gutter="12" class="metrics">
      <el-col :span="4"><el-statistic title="排队" :value="counts.queued || 0" /></el-col>
      <el-col :span="4"><el-statistic title="运行" :value="counts.running || 0" /></el-col>
      <el-col :span="4"><el-statistic title="成功" :value="counts.success || 0" /></el-col>
      <el-col :span="4"><el-statistic title="失败" :value="counts.failed || 0" /></el-col>
      <el-col :span="4"><el-statistic title="实际并发" :value="dashboard?.settings?.effective_concurrency || 0" /></el-col>
      <el-col :span="4"><el-statistic title="平均耗时(秒)" :value="Math.round((dashboard?.average_duration_ms || 0)/1000)" /></el-col>
    </el-row>
    <el-card shadow="never" class="block">
      <template #header><strong>外部公司与邀请链接</strong></template>
      <el-row :gutter="12">
        <el-col :span="6"><el-input v-model="companyName" placeholder="新公司名称"><template #append><el-button @click="createCompany">创建</el-button></template></el-input></el-col>
        <el-col :span="5"><el-select v-model="invite.company_id" placeholder="选择公司" style="width:100%"><el-option v-for="c in companies" :key="c.company_id" :label="c.name" :value="c.company_id" /></el-select></el-col>
        <el-col :span="6"><el-select v-model="invite.allowed_product_codes" multiple collapse-tags placeholder="允许产品（空=全部）" style="width:100%"><el-option v-for="p in products" :key="p.product_code" :label="p.product_name" :value="p.product_code" /></el-select></el-col>
        <el-col :span="4"><el-date-picker v-model="invite.expires_at" value-format="YYYY-MM-DDTHH:mm:ssZ" type="datetime" placeholder="有效期" style="width:100%" /></el-col>
        <el-col :span="3"><el-button type="primary" :disabled="!invite.company_id || !invite.expires_at" @click="createInvite">生成链接</el-button></el-col>
      </el-row>
      <div class="quota">最大投稿数 <el-input-number v-model="invite.max_submissions" :min="1" :max="10000" size="small" /></div>
      <el-input v-if="inviteLink" v-model="inviteLink" readonly class="invite-link" />
      <el-table :data="invites" size="small" class="invite-table">
        <el-table-column prop="company_name" label="公司" width="150" /><el-table-column prop="expires_at" label="到期时间" width="190" /><el-table-column label="使用次数" width="110"><template #default="{row}">{{ row.submission_count }}/{{ row.max_submissions }}</template></el-table-column><el-table-column label="状态" width="100"><template #default="{row}"><el-tag :type="row.active?'success':'info'">{{ row.revoked_at?'已撤销':row.active?'有效':'已过期' }}</el-tag></template></el-table-column><el-table-column><template #default="{row}"><el-button v-if="row.active" type="danger" link @click.stop="revokeInvite(row.invite_id)">撤销</el-button></template></el-table-column>
      </el-table>
    </el-card>
    <el-card shadow="never">
      <template #header><div class="header"><strong>投稿审核队列</strong><el-space><el-select v-model="status" clearable placeholder="全部状态" @change="load"><el-option v-for="s in statusOptions" :key="s" :label="s" :value="s" /></el-select><el-button @click="load">刷新</el-button></el-space></div></template>
      <el-table :data="submissions" @row-click="open" style="cursor:pointer">
        <el-table-column prop="company_name" label="公司" width="150" /><el-table-column prop="title" label="标题" min-width="240" show-overflow-tooltip /><el-table-column prop="submitter_name" label="投稿人" width="100" /><el-table-column prop="status" label="状态" width="190"><template #default="{row}"><el-tag>{{ row.status }}</el-tag></template></el-table-column><el-table-column prop="review_stage" label="审稿阶段" width="130" /><el-table-column prop="updated_at" label="更新时间" width="180" />
      </el-table>
    </el-card>

    <el-drawer v-model="detail" size="min(900px,94vw)" :title="detail?.title">
      <template v-if="detail">
        <el-descriptions :column="2" border><el-descriptions-item label="公司">{{ detail.company_name }}</el-descriptions-item><el-descriptions-item label="状态">{{ detail.status }}</el-descriptions-item><el-descriptions-item label="投稿人">{{ detail.submitter_name }} / {{ detail.submitter_email }}</el-descriptions-item><el-descriptions-item label="版本">v{{ detail.current_version }}</el-descriptions-item><el-descriptions-item label="AI 摘要" :span="2">{{ detail.report?.summary || detail.review_error || "尚未生成" }}</el-descriptions-item></el-descriptions>
        <el-collapse class="article-body"><el-collapse-item title="查看当前投稿全文"><pre>{{ detail.content_text || '正文尚未解析' }}</pre></el-collapse-item></el-collapse>
        <h3>AI 问题项</h3>
        <el-checkbox-group v-model="selectedFindings">
          <el-card v-for="item in detail.findings" :key="item.finding_id" shadow="never" class="finding"><el-checkbox :value="item.finding_id">反馈给投稿方</el-checkbox> <el-tag :type="item.severity==='high'?'danger':'warning'">{{ item.severity }}</el-tag><strong> {{ item.verdict }}</strong><p>{{ item.excerpt }}</p><p class="muted">依据：{{ item.evidence }}</p><p>建议：{{ item.suggestion }}</p></el-card>
        </el-checkbox-group>
        <h3 v-if="detail.similarities?.filter(x=>x.similarity_level!=='hidden').length">相似历史文章</h3>
        <el-card v-for="item in detail.similarities?.filter(x=>x.similarity_level!=='hidden')" :key="item.match_id" shadow="never" class="finding"><el-tag :type="item.similarity_level==='high'?'danger':'warning'">{{ item.exact_hash ? '精确重复' : item.similarity_level }}</el-tag> {{ item.matched_title || item.matched_id }} · {{ Math.round((item.semantic_score||item.lexical_score)*100) }}%<p>{{ item.overlap_summary }}</p><el-collapse><el-collapse-item title="全文对照"><h4>当前投稿相关段落</h4><pre>{{ item.source_excerpt }}</pre><el-divider /><h4>历史文章全文</h4><pre>{{ item.matched_content_text || item.matched_excerpt }}</pre></el-collapse-item></el-collapse></el-card>
        <template v-if="detail.status==='awaiting_admin'"><el-input v-model="feedback" type="textarea" :rows="3" placeholder="给投稿方的人工意见" class="feedback" /><el-space><el-button type="success" @click="action('approve')">审核通过</el-button><el-button type="warning" @click="action('request_revision')">退回修改</el-button><el-button type="danger" @click="action('reject')">拒绝</el-button></el-space></template>
        <el-space v-if="['review_failed','blocked_missing_kb'].includes(detail.status)"><el-button type="primary" @click="simpleAction('retry')">重试审稿</el-button></el-space>
        <el-button v-if="['queued','reviewing'].includes(detail.status)" type="danger" @click="simpleAction('cancel')">取消任务</el-button>
      </template>
    </el-drawer>
  </div>
</template>

<style scoped>.metrics,.block{margin-bottom:16px}.metrics :deep(.el-col){background:#fff;padding:14px;border:1px solid #edf0f5}.header{display:flex;align-items:center;justify-content:space-between}.invite-link,.feedback,.article-body,.invite-table{margin-top:14px}.quota{margin-top:10px;color:#667085;font-size:13px}.finding{margin:10px 0}.muted{color:#7b8497;font-size:13px}pre{white-space:pre-wrap;word-break:break-word;max-height:520px;overflow:auto;font-family:inherit;line-height:1.7}</style>
