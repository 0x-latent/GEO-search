<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";

import { api, apiJson, query } from "@/api/client";
import { useSessionStore } from "@/stores/session";
import { fmtNumber } from "@/utils/format";

const session = useSessionStore();
const activeTab = ref("datasets");

// ---------- 数据集 ----------
const datasets = ref([]);
const datasetFilter = ref("all");

async function loadDatasets() {
  try {
    datasets.value = await api("/api/sqlite/datasets");
  } catch (error) {
    ElMessage.error(`加载数据集失败：${error.message}`);
  }
}

async function removeDataset(row) {
  try {
    await ElMessageBox.confirm(
      `确认删除数据集「${row.name}」？将级联清除其全部问题、回答、信源、外部表和物化指标，不可恢复。`,
      "删除数据集",
      { type: "warning", confirmButtonText: "删除", confirmButtonClass: "el-button--danger" }
    );
  } catch {
    return;
  }
  try {
    await api(`/api/sqlite/datasets/${encodeURIComponent(row.dataset_id)}`, { method: "DELETE" });
    ElMessage.success("已删除");
    await loadDatasets();
  } catch (error) {
    ElMessage.error(`删除失败：${error.message}`);
  }
}

// ---------- 拆分明细 ----------
const splits = ref(null);
const splitsLoading = ref(false);
const splitTable = ref("mention_summary");
const SPLIT_TABS = [
  ["mention_summary", "提及/推荐率汇总"],
  ["rec_overview", "推荐产品排行"],
  ["type_summary", "名称类型汇总"],
  ["category_summary", "品类汇总"],
  ["question_details", "问题级明细"],
  ["yangweishu_brand_summary", "养胃舒品牌"],
];

async function loadSplits() {
  splitsLoading.value = true;
  try {
    splits.value = await api(`/api/sqlite/splits${query({ dataset_id: datasetFilter.value })}`);
  } catch (error) {
    ElMessage.error(`加载失败：${error.message}`);
  } finally {
    splitsLoading.value = false;
  }
}

const splitRows = computed(() => splits.value?.[splitTable.value] || []);
const splitColumns = computed(() =>
  splitRows.value.length ? Object.keys(splitRows.value[0]) : []
);

// ---------- 回答样本 ----------
const samples = ref([]);
const samplesLoading = ref(false);

async function loadSamples() {
  samplesLoading.value = true;
  try {
    samples.value = await api(
      `/api/sqlite/answers${query({ dataset_id: datasetFilter.value, limit: 200 })}`
    );
  } catch (error) {
    ElMessage.error(`加载失败：${error.message}`);
  } finally {
    samplesLoading.value = false;
  }
}

// ---------- 用户管理 ----------
const users = ref([]);
const newUser = reactive({ username: "", password: "", role: "user" });

async function loadUsers() {
  try {
    users.value = await api("/api/auth/users");
  } catch (error) {
    ElMessage.error(`加载用户失败：${error.message}`);
  }
}

async function createUser() {
  try {
    await apiJson("/api/auth/users", "POST", { ...newUser });
    ElMessage.success(`已创建用户 ${newUser.username}`);
    newUser.username = "";
    newUser.password = "";
    await loadUsers();
  } catch (error) {
    ElMessage.error(`创建失败：${error.message}`);
  }
}

async function resetPassword(user) {
  try {
    const { value } = await ElMessageBox.prompt(`为 ${user.username} 设置新密码（至少 6 位）`, "重置密码", {
      inputType: "password",
    });
    if (!value) return;
    await apiJson(`/api/auth/users/${encodeURIComponent(user.username)}/password`, "PUT", {
      password: value,
    });
    ElMessage.success("已重置密码");
  } catch (error) {
    if (error !== "cancel" && error?.message) ElMessage.error(`操作失败：${error.message}`);
  }
}

async function toggleRole(user) {
  const role = user.role === "admin" ? "user" : "admin";
  try {
    await apiJson(`/api/auth/users/${encodeURIComponent(user.username)}/role`, "PUT", { role });
    ElMessage.success("已调整角色");
    await loadUsers();
  } catch (error) {
    ElMessage.error(`操作失败：${error.message}`);
  }
}

async function removeUser(user) {
  try {
    await ElMessageBox.confirm(`确认删除用户 ${user.username}？`, "删除用户", { type: "warning" });
  } catch {
    return;
  }
  try {
    await api(`/api/auth/users/${encodeURIComponent(user.username)}`, { method: "DELETE" });
    ElMessage.success("已删除");
    await loadUsers();
  } catch (error) {
    ElMessage.error(`删除失败：${error.message}`);
  }
}

onMounted(async () => {
  await loadDatasets();
  await Promise.all([loadSplits(), loadUsers()]);
});

function onTab(name) {
  if (name === "samples" && !samples.value.length) loadSamples();
}

function onDatasetFilter() {
  loadSplits();
  if (activeTab.value === "samples") loadSamples();
}
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>工作台</h1>
      <p>数据侧与管理员的明细视图：数据集、拆分表、回答样本和用户管理。业务结论请看「品牌总览」。</p>
    </div>

    <el-tabs v-model="activeTab" @tab-change="onTab">
      <el-tab-pane label="数据集" name="datasets">
        <el-table :data="datasets" size="small">
          <el-table-column prop="dataset_id" label="数据集ID" min-width="220" show-overflow-tooltip />
          <el-table-column prop="name" label="名称" min-width="150" show-overflow-tooltip />
          <el-table-column prop="batch_date" label="批次日期" width="100" />
          <el-table-column prop="product_code" label="产品" width="110" />
          <el-table-column label="归属" width="90">
            <template #default="{ row }">{{ row.owner_username || "系统" }}</template>
          </el-table-column>
          <el-table-column label="问题" width="80">
            <template #default="{ row }">{{ fmtNumber(row.questions) }}</template>
          </el-table-column>
          <el-table-column label="回答" width="90">
            <template #default="{ row }">{{ fmtNumber(row.answers) }}</template>
          </el-table-column>
          <el-table-column label="外部表" width="80">
            <template #default="{ row }">{{ fmtNumber(row.external_tables) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="90">
            <template #default="{ row }">
              <el-button link type="danger" size="small" @click="removeDataset(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="拆分明细" name="splits">
        <el-space style="margin-bottom: 12px" wrap>
          <el-select v-model="datasetFilter" style="width: 240px" @change="onDatasetFilter">
            <el-option value="all" label="全部数据集" />
            <el-option
              v-for="dataset in datasets"
              :key="dataset.dataset_id"
              :value="dataset.dataset_id"
              :label="dataset.name"
            />
          </el-select>
          <el-radio-group v-model="splitTable">
            <el-radio-button v-for="[key, label] in SPLIT_TABS" :key="key" :value="key">
              {{ label }}
            </el-radio-button>
          </el-radio-group>
        </el-space>
        <el-table :data="splitRows" v-loading="splitsLoading" size="small" max-height="620" border>
          <el-table-column
            v-for="column in splitColumns"
            :key="column"
            :prop="column"
            :label="column"
            :min-width="column === '推荐原因' || column === '提问词' ? 220 : 110"
            show-overflow-tooltip
            sortable
          />
        </el-table>
        <p class="muted" style="font-size: 12px">共 {{ splitRows.length }} 行（问题级明细仅预览部分，全量请用证据链接口）</p>
      </el-tab-pane>

      <el-tab-pane label="回答样本" name="samples">
        <el-space style="margin-bottom: 12px">
          <el-select v-model="datasetFilter" style="width: 240px" @change="onDatasetFilter">
            <el-option value="all" label="全部数据集" />
            <el-option
              v-for="dataset in datasets"
              :key="dataset.dataset_id"
              :value="dataset.dataset_id"
              :label="dataset.name"
            />
          </el-select>
        </el-space>
        <el-table :data="samples" v-loading="samplesLoading" size="small" max-height="620" border>
          <el-table-column prop="product_name" label="产品" width="90" />
          <el-table-column prop="model" label="模型" width="100" />
          <el-table-column prop="search_mode" label="模式" width="70" />
          <el-table-column prop="round" label="轮次" width="60" />
          <el-table-column prop="question_text" label="问题" min-width="200" show-overflow-tooltip />
          <el-table-column prop="answer_preview" label="回答摘录" min-width="300" show-overflow-tooltip />
          <el-table-column prop="answer_chars" label="字数" width="80" sortable />
          <el-table-column prop="source_count" label="信源" width="70" sortable />
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="用户管理" name="users">
        <el-card shadow="never" style="max-width: 620px; margin-bottom: 16px">
          <template #header><strong>新建用户</strong></template>
          <el-space wrap>
            <el-input v-model="newUser.username" placeholder="用户名" style="width: 160px" />
            <el-input
              v-model="newUser.password"
              type="password"
              placeholder="密码（至少 6 位）"
              style="width: 180px"
            />
            <el-select v-model="newUser.role" style="width: 120px">
              <el-option value="user" label="普通用户" />
              <el-option value="admin" label="管理员" />
            </el-select>
            <el-button type="primary" @click="createUser">创建</el-button>
          </el-space>
        </el-card>
        <el-table :data="users" size="small" style="max-width: 760px">
          <el-table-column prop="username" label="用户名" min-width="140">
            <template #default="{ row }">
              {{ row.username }}
              <el-tag v-if="row.username === session.user?.username" size="small">我</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="角色" width="100">
            <template #default="{ row }">
              {{ row.role === "admin" ? "管理员" : "普通用户" }}
            </template>
          </el-table-column>
          <el-table-column label="创建时间" width="170">
            <template #default="{ row }">
              {{ new Date(row.created_at * 1000).toLocaleString("zh-CN") }}
            </template>
          </el-table-column>
          <el-table-column label="操作" width="240">
            <template #default="{ row }">
              <el-button link size="small" @click="resetPassword(row)">重置密码</el-button>
              <el-button
                link
                size="small"
                :disabled="row.username === session.user?.username"
                @click="toggleRole(row)"
              >
                设为{{ row.role === "admin" ? "普通用户" : "管理员" }}
              </el-button>
              <el-button
                link
                type="danger"
                size="small"
                :disabled="row.username === session.user?.username"
                @click="removeUser(row)"
              >
                删除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>
