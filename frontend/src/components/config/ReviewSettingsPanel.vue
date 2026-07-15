<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import { api, apiJson } from "@/api/client";

const models = ref([]); const settings = reactive({}); const dashboard = ref(null); const loading = ref(false);
const variants = (key) => models.value.find((m) => m.key === key)?.variants || [];
const selectedPrimary = computed({ get: () => settings.primary_model_key && settings.primary_model_id ? `${settings.primary_model_key}|${settings.primary_model_id}` : "", set: (value) => { const [key,id] = value.split("|"); settings.primary_model_key=key; settings.primary_model_id=id; } });
const selectedFallback = computed({ get: () => settings.fallback_model_key && settings.fallback_model_id ? `${settings.fallback_model_key}|${settings.fallback_model_id}` : "", set: (value) => { const [key,id] = (value||"|").split("|"); settings.fallback_model_key=key||null; settings.fallback_model_id=id||null; } });
const options = computed(() => models.value.flatMap(m => m.variants.map(v => ({ value:`${m.key}|${v.id}`, label:`${m.name} / ${v.label}` }))));
async function load(){loading.value=true;try{const result=await api("/api/admin/article-review/settings");models.value=result.models;Object.assign(settings,result.settings);dashboard.value=await api("/api/admin/article-review/dashboard");}catch(e){ElMessage.error(e.message)}finally{loading.value=false}}
async function save(){try{Object.assign(settings,await apiJson("/api/admin/article-review/settings","PUT",settings));ElMessage.success("AI 审稿设置已保存，Worker 最多 5 秒内生效");await load()}catch(e){ElMessage.error(e.message)}}
onMounted(load);
</script>
<template><el-card shadow="never" v-loading="loading"><template #header><strong>AI 审稿设置</strong></template>
  <el-alert type="info" :closable="false" title="并发表示同时进行的 AI 请求数。实际值受服务器环境上限保护，修改后台值无需重启；修改环境上限需要重建 Worker。" />
  <el-form label-width="170px" class="form">
    <el-form-item label="自动启动"><el-switch v-model="settings.auto_start" /></el-form-item><el-form-item label="暂停队列"><el-switch v-model="settings.queue_paused" /></el-form-item>
    <el-form-item label="主审模型"><el-select v-model="selectedPrimary" style="width:420px"><el-option v-for="o in options" :key="o.value" :label="o.label" :value="o.value" /></el-select></el-form-item>
    <el-form-item label="备用模型"><el-select v-model="selectedFallback" clearable style="width:420px"><el-option v-for="o in options" :key="o.value" :label="o.label" :value="o.value" /></el-select></el-form-item>
    <el-form-item label="AI 请求并发"><el-input-number v-model="settings.ai_concurrency" :min="1" :max="100" /><span class="hint">配置 {{ settings.ai_concurrency }} / 环境上限 {{ settings.environment_max }} / 实际 {{ settings.effective_concurrency }}</span></el-form-item>
    <el-form-item label="请求超时（秒）"><el-input-number v-model="settings.request_timeout_seconds" :min="10" :max="900" /></el-form-item><el-form-item label="重试次数"><el-input-number v-model="settings.retry_count" :min="0" :max="10" /></el-form-item>
    <el-form-item label="相似度阈值"><el-slider v-model="settings.similarity_threshold" :min="0.4" :max="0.95" :step="0.01" style="width:360px" show-input /></el-form-item><el-form-item label="相似候选 Top K"><el-input-number v-model="settings.similarity_top_k" :min="1" :max="50" /></el-form-item>
    <el-form-item><el-button type="primary" @click="save">保存设置</el-button><el-button @click="load">刷新状态</el-button></el-form-item>
  </el-form>
  <el-descriptions v-if="dashboard" title="Worker 状态" :column="2" border><el-descriptions-item label="心跳">{{ dashboard.worker?.heartbeat_at || '尚未启动' }}</el-descriptions-item><el-descriptions-item label="在途请求">{{ dashboard.worker?.active_requests || 0 }}</el-descriptions-item><el-descriptions-item label="最近错误" :span="2">{{ dashboard.latest_error?.error_message || dashboard.worker?.last_error || '无' }}</el-descriptions-item></el-descriptions>
</el-card></template>
<style scoped>.form{margin-top:20px;max-width:760px}.hint{margin-left:14px;color:#6f7890}</style>
