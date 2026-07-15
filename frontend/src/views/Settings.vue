<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";

import { api, apiJson, appUrl } from "@/api/client";
import BrandsEditor from "@/components/config/BrandsEditor.vue";
import KbEditor from "@/components/config/KbEditor.vue";
import KbImportPanel from "@/components/config/KbImportPanel.vue";
import ReviewSettingsPanel from "@/components/config/ReviewSettingsPanel.vue";
import { useSessionStore } from "@/stores/session";

const session = useSessionStore();
const activeTab = ref("brands");

// scope: mine（我的配置，分析优先使用）| global（全局默认，仅管理员，产品主数据来源）
const scope = ref("mine");
const brandsData = ref(null);
const kbData = ref(null);
const mineSource = reactive({ brands: "", kb: "" });
const brandsEditor = ref(null);
const kbEditor = ref(null);
const rawBrands = ref("");
const rawKb = ref("");
const password = ref("");

const PATHS = computed(() => ({
  brands: scope.value === "mine" ? "/api/config/my/brands" : "/api/config/brands",
  kb: scope.value === "mine" ? "/api/config/my/knowledge-base" : "/api/config/knowledge-base",
}));

async function load() {
  try {
    if (scope.value === "mine") {
      const [brands, kb] = await Promise.all([api(PATHS.value.brands), api(PATHS.value.kb)]);
      brandsData.value = brands.data;
      kbData.value = kb.data;
      mineSource.brands = brands.source;
      mineSource.kb = kb.source;
    } else {
      const [brands, kb] = await Promise.all([api(PATHS.value.brands), api(PATHS.value.kb)]);
      brandsData.value = brands;
      kbData.value = kb;
    }
    rawBrands.value = JSON.stringify(brandsData.value, null, 2);
    rawKb.value = JSON.stringify(kbData.value, null, 2);
  } catch (error) {
    ElMessage.error(`加载配置失败：${error.message}`);
  }
}

onMounted(load);

async function saveStructured(kind) {
  const editor = kind === "brands" ? brandsEditor.value : kbEditor.value;
  if (!editor) return;
  const { errors } = editor.issues;
  if (errors.length) {
    ElMessage.error(`请先修正：${errors[0]}`);
    return;
  }
  const data = editor.toJson();
  try {
    await apiJson(PATHS.value[kind], "PUT", { data });
    ElMessage.success(
      scope.value === "global"
        ? "全局配置已保存（品牌配置已同步产品主数据）"
        : "已保存自定义配置，下次分析生效"
    );
    await load();
  } catch (error) {
    ElMessage.error(`保存失败：${error.message}`);
  }
}

async function saveRaw(kind) {
  try {
    const text = kind === "brands" ? rawBrands.value : rawKb.value;
    let data;
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new Error(`JSON 格式错误：${error.message}`);
    }
    await apiJson(PATHS.value[kind], "PUT", { data });
    ElMessage.success("已保存");
    await load();
  } catch (error) {
    ElMessage.error(`保存失败：${error.message}`);
  }
}

async function resetMine(kind) {
  if (scope.value !== "mine") return;
  try {
    await ElMessageBox.confirm("确认删除自定义配置并恢复全局默认？", "恢复默认", { type: "warning" });
  } catch {
    return;
  }
  try {
    await api(PATHS.value[kind], { method: "DELETE" });
    ElMessage.success("已恢复全局默认");
    await load();
  } catch (error) {
    ElMessage.error(`操作失败：${error.message}`);
  }
}

async function changePassword() {
  if (password.value.length < 6) {
    ElMessage.warning("密码至少 6 位");
    return;
  }
  try {
    await apiJson("/api/auth/me/password", "PUT", { password: password.value });
    ElMessage.success("密码已修改，请重新登录");
    window.location.href = appUrl("/login.html");
  } catch (error) {
    ElMessage.error(`修改失败：${error.message}`);
  }
}

function switchScope(value) {
  scope.value = value;
  load();
}
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>我的配置</h1>
      <p>品牌/竞品/品类词决定提及率口径，知识库决定准确率校验依据。分析任务优先使用你的自定义配置，未自定义时用全局默认。</p>
    </div>

    <el-space style="margin-bottom: 14px" v-if="session.isAdmin">
      <el-radio-group :model-value="scope" @change="switchScope">
        <el-radio-button value="mine">我的配置</el-radio-button>
        <el-radio-button value="global">全局默认（管理员）</el-radio-button>
      </el-radio-group>
      <span class="muted" style="font-size: 12px">
        全局品牌配置同时是「产品主数据」的来源（总览卡显示名、上传产品下拉）。
      </span>
    </el-space>
    <el-space v-else style="margin-bottom: 14px">
      <el-tag :type="mineSource.brands === 'user' ? 'success' : 'info'" size="small">
        品牌配置：{{ mineSource.brands === "user" ? "自定义" : "全局默认" }}
      </el-tag>
      <el-tag :type="mineSource.kb === 'user' ? 'success' : 'info'" size="small">
        知识库：{{ mineSource.kb === "user" ? "自定义" : "全局默认" }}
      </el-tag>
    </el-space>

    <el-tabs v-model="activeTab">
      <el-tab-pane v-if="session.isAdmin" label="AI 审稿设置" name="article-review" lazy>
        <ReviewSettingsPanel />
      </el-tab-pane>
      <el-tab-pane label="品牌与竞品" name="brands">
        <el-card shadow="never">
          <BrandsEditor v-if="brandsData" ref="brandsEditor" :data="brandsData" />
          <div class="actions">
            <el-button type="primary" @click="saveStructured('brands')">保存品牌配置</el-button>
            <el-button v-if="scope === 'mine'" @click="resetMine('brands')">恢复全局默认</el-button>
          </div>
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="知识库" name="kb">
        <KbImportPanel
          v-if="kbData"
          :kb-data="kbData"
          :scope="scope === 'mine' ? 'user' : 'global'"
          @applied="load"
        />
        <el-card shadow="never">
          <KbEditor v-if="kbData" ref="kbEditor" :data="kbData" />
          <div class="actions">
            <el-button type="primary" @click="saveStructured('kb')">保存知识库</el-button>
            <el-button v-if="scope === 'mine'" @click="resetMine('kb')">恢复全局默认</el-button>
          </div>
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="高级（JSON）" name="advanced">
        <el-alert
          type="warning"
          :closable="false"
          title="直接编辑底层 JSON，请确认格式正确后保存。日常维护建议用前两个结构化页签。"
          style="margin-bottom: 12px"
        />
        <el-row :gutter="16">
          <el-col :span="12">
            <el-card shadow="never">
              <template #header><strong>brands</strong></template>
              <el-input v-model="rawBrands" type="textarea" :rows="18" spellcheck="false" class="editor" />
              <div class="actions">
                <el-button type="primary" @click="saveRaw('brands')">保存</el-button>
              </div>
            </el-card>
          </el-col>
          <el-col :span="12">
            <el-card shadow="never">
              <template #header><strong>knowledge_base</strong></template>
              <el-input v-model="rawKb" type="textarea" :rows="18" spellcheck="false" class="editor" />
              <div class="actions">
                <el-button type="primary" @click="saveRaw('kb')">保存</el-button>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-tab-pane>

      <el-tab-pane label="账号" name="account">
        <el-card shadow="never" style="max-width: 480px">
          <template #header><strong>修改我的密码</strong></template>
          <el-space>
            <el-input
              v-model="password"
              type="password"
              show-password
              placeholder="新密码（至少 6 位）"
              style="width: 260px"
            />
            <el-button type="primary" @click="changePassword">修改密码</el-button>
          </el-space>
          <p class="muted" style="font-size: 12px">修改后需要重新登录。</p>
        </el-card>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<style scoped>
.editor :deep(textarea) {
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
}
.actions {
  margin-top: 14px;
}
</style>
