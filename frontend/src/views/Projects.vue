<script setup>
import { computed, onMounted, reactive, ref } from 'vue';
import { useRoute } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { api, apiJson, appUrl } from '@/api/client';
import { useSessionStore } from '@/stores/session';

const session = useSessionStore();
const route = useRoute();
const projects = ref([]), options = ref({ products: [], companies: [] });
const selected = ref(null), detail = ref(null), creating = ref(false), busy = ref(false);
const tab = ref('submissions'), inviteLink = ref(''), feedback = ref(''), notes = ref(''), visibleFindings = ref([]);
const form = reactive({ name: '', brief: '', product_codes: [], company_ids: [], channels: [], target_questions: '',
  kb_products: {}, approval_steps: [{ name: '专家审核', username: '' }], monitor_interval_hours: 24,
  submission_deadline: null, monitor_until: null });
const invite = reactive({ company_id: '', expires_at: '', max_submissions: 20 });
const labels = { queued: '待核验', reviewing: 'Agent 核验中', awaiting_admin: '待指定人员审批', revision_requested: '待代理修改',
  approved_waiting_publication: '可发布', publication_pending: '待发布验收', publication_accepted: '验收通过，正在登记',
  tracked: '已发布 · 追踪中', rejected: '不予采用', review_failed: '核验异常', blocked_missing_kb: '知识待核实' };
const currentApprover = computed(() => detail.value?.approval_steps?.[detail.value?.approval_step]);
const canDecide = computed(() => selected.value?.status === 'active' && detail.value?.status === 'awaiting_admin' && currentApprover.value?.username === session.user?.username);
const myPending = computed(() => selected.value?.submissions.filter(s => s.status === 'awaiting_admin' && selected.value.approval_steps[s.approval_step]?.username === session.user?.username).length || 0);
const currentMetrics = computed(() => selected.value?.snapshots?.[0]?.result);

async function run(fn) {
  if (busy.value) return;
  busy.value = true;
  try { await fn(); } catch (e) { if (e !== 'cancel' && e !== 'close') ElMessage.error(e.message || String(e)); }
  finally { busy.value = false; }
}
async function load() {
  [projects.value, options.value] = await Promise.all([api('/api/projects'), api('/api/projects/options')]);
}
async function openProject(pid) {
  selected.value = await api(`/api/projects/${encodeURIComponent(pid)}`);
  notes.value = selected.value.review_notes;
  invite.company_id = selected.value.company_ids[0] || '';
  inviteLink.value = '';
}
async function refreshProject() { await openProject(selected.value.project_id); }
async function openSubmission(row) {
  detail.value = await api(`/api/projects/${selected.value.project_id}/submissions/${row.submission_id}`);
  feedback.value = ''; visibleFindings.value = detail.value.findings?.filter(f => f.external_visible).map(f => f.finding_id) || [];
}
function endpoint(action) { return `/api/projects/${selected.value.project_id}/submissions/${detail.value.submission_id}/${action}`; }
async function create() {
  await run(async () => {
    const result = await apiJson('/api/projects', 'POST', { ...form, kb_products: Object.fromEntries(Object.entries(form.kb_products).filter(([k,v]) => form.product_codes.includes(k) && v?.trim())) });
    creating.value = false; await load(); await openProject(result.project_id); ElMessage.success('项目已创建');
  });
}
async function decide(action) {
  await run(async () => {
    detail.value = await apiJson(endpoint('decisions'), 'POST', { action, version: detail.value.current_version, step: detail.value.approval_step, feedback: feedback.value, finding_ids: visibleFindings.value });
    feedback.value = ''; await refreshProject(); ElMessage.success('审批决定已记录');
  });
}
async function resolveFinding(f) {
  await run(async () => {
    const { value } = await ElMessageBox.prompt('填写核实依据，解释为何该问题属于误报。真实问题请退回修改。', '记录误报依据', { inputType: 'textarea', inputValidator: v => !!v?.trim() || '请填写依据' });
    detail.value = await apiJson(endpoint(`findings/${f.finding_id}`), 'POST', { version: detail.value.current_version, step: detail.value.approval_step, note: value });
  });
}
async function accept(action) {
  await run(async () => {
    detail.value = await apiJson(endpoint('acceptance'), 'POST', { action, version: detail.value.current_version, publication_revision: detail.value.publication_revision, feedback: feedback.value });
    await refreshProject(); ElMessage.success('验收结论已记录');
  });
}
async function createInvite() {
  await run(async () => {
    const result = await apiJson(`/api/projects/${selected.value.project_id}/invites`, 'POST', invite);
    await refreshProject();
    inviteLink.value = `${window.location.origin}${appUrl('article-submit.html')}#invite=${encodeURIComponent(result.invite_id)}&token=${encodeURIComponent(result.token)}`;
    ElMessage.success('邀请已生成，请复制链接交给对应代理');
  });
}
async function revoke(iid) {
  await run(async () => {
    await ElMessageBox.confirm('撤销后该邀请及会话立即失效。', '撤销邀请');
    await api(`/api/projects/${selected.value.project_id}/invites/${iid}`, { method: 'DELETE' }); await refreshProject();
  });
}
async function retrospective(archive) {
  await run(async () => {
    if (archive) await ElMessageBox.confirm('归档会结束项目投稿、审批及定期监测，历史记录保留。', '归档项目');
    await apiJson(`/api/projects/${selected.value.project_id}/retrospective`, 'POST', { notes: notes.value, archive });
    await refreshProject(); await load(); ElMessage.success(archive ? '项目已归档' : '复盘已保存');
  });
}
onMounted(() => run(async () => { await load(); if (route.query.project) await openProject(String(route.query.project)); }));
</script>

<template>
  <main class="projects-page" v-loading="busy">
    <div class="project-heading"><div><h1>项目协作</h1><p>代理投稿、知识核验、审批发布与效果追踪</p></div><el-space><el-button @click="run(load)">刷新</el-button><el-button type="primary" @click="creating=true">建立项目</el-button></el-space></div>
    <el-alert v-if="!options.knowledge_connected" title="项目审稿的知识库连接尚未配置。可以先建立项目和邀请，稿件提交后将等待知识库就绪。" type="warning" :closable="false" show-icon />
    <el-table :data="projects" @row-click="p => run(() => openProject(p.project_id))" class="project-list">
      <el-table-column prop="name" label="项目" min-width="200" /><el-table-column prop="owner_username" label="负责人" width="140" />
      <el-table-column label="审批流程" min-width="240"><template #default="{row}">{{ row.approval_steps.map(s=>s.name).join(' → ') }}</template></el-table-column>
      <el-table-column label="状态" width="100"><template #default="{row}"><el-tag>{{ row.status==='active'?'进行中':'已归档' }}</el-tag></template></el-table-column>
      <el-table-column prop="last_monitor_at" label="最近监测" min-width="180" />
    </el-table>
    <el-empty v-if="!projects.length" description="建立第一个项目，配置代理、知识范围与审批人员" />

    <el-dialog v-model="creating" title="建立 GEO 项目" width="min(820px,95vw)">
      <el-form label-position="top">
        <el-form-item label="项目名称"><el-input v-model="form.name" maxlength="200" /></el-form-item>
        <el-form-item label="项目 Brief"><el-input v-model="form.brief" type="textarea" :rows="3" placeholder="传播目标、内容要求、需避免的表达" /></el-form-item>
        <el-row :gutter="16"><el-col :span="12"><el-form-item label="产品 / 知识范围"><el-select v-model="form.product_codes" multiple><el-option v-for="p in options.products" :key="p.product_code" :label="p.product_name" :value="p.product_code" /></el-select></el-form-item></el-col>
        <el-col :span="12"><el-form-item label="合作代理公司"><el-select v-model="form.company_ids" multiple><el-option v-for="c in options.companies" :key="c.company_id" :label="c.name" :value="c.company_id" /></el-select><small v-if="!options.companies.length">请先由管理员在信源分析的投稿审核中创建公司。</small></el-form-item></el-col></el-row>
        <el-form-item label="发布渠道"><el-select v-model="form.channels" multiple filterable allow-create default-first-option placeholder="输入渠道名称后回车" /></el-form-item>
        <el-form-item label="目标问题"><el-input v-model="form.target_questions" type="textarea" placeholder="目标消费者搜索问题，作为审稿及复盘参考" /></el-form-item>
        <el-collapse><el-collapse-item title="知识库产品映射（编号不同时填写）"><el-form-item v-for="code in form.product_codes" :key="code" :label="code"><el-input v-model="form.kb_products[code]" :placeholder="`默认使用 ${code}`" /></el-form-item></el-collapse-item></el-collapse>
        <h3>指定审批人 · 按顺序审批</h3>
        <div v-for="(step,index) in form.approval_steps" :key="index" class="step-row"><span>{{ index+1 }}</span><el-input v-model="step.name" placeholder="节点名称，如专家审核" /><el-input v-model="step.username" placeholder="审批人的门户用户名" /><el-button :disabled="form.approval_steps.length===1" @click="form.approval_steps.splice(index,1)">移除</el-button></div>
        <el-button :disabled="form.approval_steps.length>=10" @click="form.approval_steps.push({name:'负责人终审',username:''})">追加审批节点</el-button>
        <p class="muted">使用已有门户账户的准确用户名。项目建立后固定此审批顺序，所有节点通过才能发布。</p>
        <el-row :gutter="16"><el-col :span="12"><el-form-item label="投稿截止（可选）"><el-date-picker v-model="form.submission_deadline" type="datetime" value-format="YYYY-MM-DDTHH:mm:ssZ" /></el-form-item></el-col><el-col :span="12"><el-form-item label="监测结束（可选）"><el-date-picker v-model="form.monitor_until" type="datetime" value-format="YYYY-MM-DDTHH:mm:ssZ" /></el-form-item></el-col></el-row>
        <el-form-item label="监测间隔（小时）"><el-input-number v-model="form.monitor_interval_hours" :min="1" :max="720" /></el-form-item>
      </el-form><template #footer><el-button @click="creating=false">取消</el-button><el-button type="primary" :loading="busy" @click="create">创建项目</el-button></template>
    </el-dialog>

    <el-drawer :model-value="!!selected" @close="selected=null; detail=null" size="min(1200px,96vw)" :title="selected?.name">
      <template v-if="selected">
        <el-descriptions :column="2" border><el-descriptions-item label="负责人">{{ selected.owner_username }}</el-descriptions-item><el-descriptions-item label="我的审批待办">{{ myPending }}</el-descriptions-item><el-descriptions-item label="Brief" :span="2">{{ selected.brief }}</el-descriptions-item><el-descriptions-item label="审批流程" :span="2">{{ selected.approval_steps.map(s=>`${s.name}（${s.username}）`).join(' → ') }}</el-descriptions-item></el-descriptions>
        <el-tabs v-model="tab" class="project-tabs">
          <el-tab-pane label="稿件与审批" name="submissions"><el-button @click="run(refreshProject)">刷新状态</el-button>
            <el-table :data="selected.submissions" @row-click="row => run(() => openSubmission(row))"><el-table-column prop="title" label="稿件" min-width="200" /><el-table-column prop="company_name" label="代理" /><el-table-column label="状态" min-width="160"><template #default="{row}">{{ labels[row.status] || row.status }}</template></el-table-column><el-table-column label="当前审批人"><template #default="{row}">{{ row.status==='awaiting_admin' ? selected.approval_steps[row.approval_step]?.username : '—' }}</template></el-table-column><el-table-column prop="current_version" label="版本" width="65" /></el-table>
          </el-tab-pane>
          <el-tab-pane v-if="selected.can_manage" label="代理邀请" name="invites">
            <el-form inline v-if="selected.status==='active'"><el-form-item label="代理"><el-select v-model="invite.company_id" style="width:180px"><el-option v-for="c in options.companies.filter(c=>selected.company_ids.includes(c.company_id))" :key="c.company_id" :label="c.name" :value="c.company_id" /></el-select></el-form-item><el-form-item label="到期"><el-date-picker v-model="invite.expires_at" type="datetime" value-format="YYYY-MM-DDTHH:mm:ssZ" /></el-form-item><el-form-item label="稿件额度"><el-input-number v-model="invite.max_submissions" :min="1" :max="10000" /></el-form-item><el-button type="primary" @click="createInvite">生成项目入口</el-button></el-form>
            <el-input v-if="inviteLink" :model-value="inviteLink" readonly type="textarea" :rows="3" aria-label="项目邀请链接" />
            <el-table :data="selected.invites"><el-table-column prop="company_name" label="公司" /><el-table-column prop="expires_at" label="到期时间" /><el-table-column label="额度"><template #default="{row}">{{ row.submission_count }}/{{ row.max_submissions }}</template></el-table-column><el-table-column label="状态"><template #default="{row}">{{ row.active?'有效':'已失效' }} <el-button v-if="row.active" link type="danger" @click="revoke(row.invite_id)">撤销</el-button></template></el-table-column></el-table>
          </el-tab-pane>
          <el-tab-pane label="监测与复盘" name="monitor">
            <el-alert title="定期分析项目负责人已导入的相关产品 GEO 数据。此处不发起新的模型采集；请在“我的分析”持续采集和导入。没有样本时不记为零引用。" type="info" :closable="false" />
            <p>间隔 {{ selected.monitor_interval_hours }} 小时 · 最近更新 {{ selected.last_monitor_at || '尚未监测' }}</p>
            <el-alert v-if="selected.monitor_error" :title="selected.monitor_error" type="warning" :closable="false" />
            <el-button v-if="selected.can_manage && selected.status==='active'" @click="run(async()=>{await apiJson(`/api/projects/${selected.project_id}/monitor`,'POST',{});await refreshProject()})">立即更新观测</el-button>
            <el-descriptions v-if="currentMetrics" :column="3" border class="project-tabs"><el-descriptions-item label="回答样本">{{ currentMetrics.sample_answers }}</el-descriptions-item><el-descriptions-item label="追踪文章">{{ currentMetrics.tracked_articles }}</el-descriptions-item><el-descriptions-item label="被引用文章">{{ currentMetrics.data_status==='no_samples'?'暂无样本':currentMetrics.cited_articles }}</el-descriptions-item></el-descriptions>
            <el-table :data="selected.snapshots"><el-table-column prop="captured_at" label="观测时间" min-width="190" /><el-table-column label="回答样本"><template #default="{row}">{{ row.result.sample_answers }}</template></el-table-column><el-table-column label="累计引用"><template #default="{row}">{{ row.result.data_status==='no_samples'?'暂无样本':row.result.citation_refs }}</template></el-table-column><el-table-column label="覆盖问题"><template #default="{row}">{{ row.result.data_status==='no_samples'?'—':row.result.covered_questions }}</template></el-table-column><el-table-column label="模型 / 引用回答数" min-width="180"><template #default="{row}">{{ Object.entries(row.result.models).map(([k,v])=>`${k}: ${v}`).join('；') || '—' }}</template></el-table-column></el-table>
            <el-collapse v-if="currentMetrics?.evidence?.length"><el-collapse-item title="最近观测的引用依据（最多 200 条）"><el-table :data="currentMetrics.evidence"><el-table-column prop="title" label="文章" /><el-table-column prop="model" label="模型" /><el-table-column prop="question_id" label="问题" /><el-table-column prop="dataset_id" label="数据批次" /><el-table-column prop="url" label="引用链接" /></el-table></el-collapse-item></el-collapse>
            <h3>复盘与下一轮优化</h3><el-input v-model="notes" type="textarea" :rows="5" :readonly="!selected.can_manage || selected.status!=='active'" placeholder="记录效果、问题及下一轮内容需求" />
            <el-space v-if="selected.can_manage && selected.status==='active'" class="project-tabs"><el-button type="primary" @click="retrospective(false)">保存复盘</el-button><el-button @click="retrospective(true)">完成并归档</el-button></el-space>
          </el-tab-pane>
        </el-tabs>
      </template>
    </el-drawer>

    <el-drawer :model-value="!!detail" @close="detail=null" append-to-body size="min(850px,94vw)" :title="detail?.title">
      <template v-if="detail">
        <el-descriptions :column="2" border><el-descriptions-item label="状态">{{ labels[detail.status] }}</el-descriptions-item><el-descriptions-item label="版本">v{{ detail.current_version }}</el-descriptions-item><el-descriptions-item label="AI 核验报告" :span="2">{{ detail.report?.summary || detail.review_error || '等待核验' }}</el-descriptions-item><el-descriptions-item v-if="detail.admin_feedback" label="反馈" :span="2">{{ detail.admin_feedback }}</el-descriptions-item></el-descriptions>
        <el-collapse><el-collapse-item title="当前稿件正文"><pre>{{ detail.content_text || '等待解析' }}</pre></el-collapse-item><el-collapse-item title="核验依据与历史版本"><section v-for="r in detail.report_history" :key="r.version"><h4>版本 {{ r.version }} · {{ r.created_at }}</h4><p>{{ r.summary }}</p><pre>{{ JSON.stringify(JSON.parse(r.knowledge_snapshot_json),null,2) }}</pre></section></el-collapse-item></el-collapse>
        <h3>核验问题</h3><el-checkbox-group v-model="visibleFindings"><el-card v-for="f in detail.findings" :key="f.finding_id" shadow="never" class="finding"><el-checkbox :value="f.finding_id" :disabled="!canDecide">反馈给代理</el-checkbox> <el-tag v-if="f.blocks_publication" type="danger">{{ f.reviewer_note?'已记录误报依据':'阻断发布' }}</el-tag><p>{{ f.excerpt }}</p><p>依据：{{ f.evidence }}</p><p>建议：{{ f.suggestion }}</p><p v-if="f.reviewer_note">核实记录：{{ f.reviewer_note }}</p><el-button v-if="canDecide && f.blocks_publication && !f.reviewer_note" link @click="resolveFinding(f)">记录误报依据</el-button></el-card></el-checkbox-group>
        <section v-if="canDecide"><h3>{{ currentApprover.name }} · {{ currentApprover.username }}</h3><el-input v-model="feedback" type="textarea" placeholder="审批意见（退回或不予采用时必填）" /><el-space class="project-tabs"><el-button type="success" :disabled="busy" @click="decide('approve')">通过当前节点</el-button><el-button type="warning" :disabled="busy" @click="decide('request_revision')">退回修改</el-button><el-button type="danger" :disabled="busy" @click="decide('reject')">不予采用</el-button></el-space></section>
        <section v-if="detail.published_url"><h3>发布记录</h3><p>{{ detail.published_platform }} · {{ detail.published_at }}</p><p>{{ detail.published_url }}</p><p>凭证：{{ detail.publication_evidence }}</p></section>
        <section v-if="detail.status==='publication_pending' && selected?.can_manage && selected.status==='active'"><el-alert title="请人工打开发布页，核对链接、正文和当前批准版本，再记录验收结论。" type="info" :closable="false" /><el-input v-model="feedback" type="textarea" placeholder="核对依据或整改要求（必填）" /><el-space class="project-tabs"><el-button type="success" @click="accept('accept')">验收通过</el-button><el-button @click="accept('correct_metadata')">退回补正发布信息</el-button><el-button type="warning" @click="accept('revise_content')">正文不一致，退回重审</el-button></el-space></section>
        <el-button v-if="['review_failed','blocked_missing_kb'].includes(detail.status) && selected?.can_manage && selected.status==='active'" @click="run(async()=>{await apiJson(endpoint('retry'),'POST',{});await openSubmission(detail);await refreshProject()})">问题解决后重新核验</el-button>
        <h3>审批历史</h3><el-table :data="detail.decisions"><el-table-column prop="version" label="版本" width="70" /><el-table-column prop="reviewer" label="审批人" /><el-table-column label="结论"><template #default="{row}">{{ {approve:'通过',request_revision:'退回修改',reject:'不予采用'}[row.action] }}</template></el-table-column><el-table-column prop="feedback" label="意见" /></el-table>
      </template>
    </el-drawer>
  </main>
</template>

<style scoped>
.projects-page{padding:24px;max-width:1500px;margin:auto}.project-heading{display:flex;justify-content:space-between;align-items:center;gap:16px}.project-heading h1{margin:0;font-size:24px}.project-heading p,.muted{color:var(--el-text-color-secondary)}.project-list,.project-tabs{margin-top:18px}.step-row{display:flex;gap:10px;align-items:center;margin:10px 0}.el-select{width:100%}.finding{margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:500px;overflow:auto}h3{margin-top:24px}small{color:var(--el-text-color-secondary)}
</style>
