<script setup>
// 知识库结构化编辑器：产品 → 模块列表（07 准确率校验的比对依据）。
// JSON 结构：{ 产品key: { product_name, modules: { "01": {name, text}, ... } } }
import { computed, reactive, ref, watch } from "vue";
import { ElMessageBox } from "element-plus";

const props = defineProps({
  data: { type: Object, required: true },
});

const state = reactive({ products: [] });
const activeProduct = ref("");

function fromJson(data) {
  state.products = Object.entries(data || {}).map(([key, entry]) => ({
    key,
    productName: (entry || {}).product_name || key,
    modules: Object.entries((entry || {}).modules || {}).map(([id, mod]) => ({
      id,
      name: (mod || {}).name || "",
      text: (mod || {}).text || "",
    })),
  }));
  if (!state.products.some((p) => p.key === activeProduct.value)) {
    activeProduct.value = state.products[0]?.key || "";
  }
}

watch(() => props.data, fromJson, { immediate: true });

const current = computed(() => state.products.find((p) => p.key === activeProduct.value));

function toJson() {
  return Object.fromEntries(
    state.products
      .filter((p) => p.key.trim())
      .map((p) => [
        p.key.trim(),
        {
          product_name: p.productName.trim() || p.key.trim(),
          modules: Object.fromEntries(
            p.modules
              .filter((m) => m.id.trim())
              .map((m) => [m.id.trim(), { name: m.name.trim(), text: m.text }])
          ),
        },
      ])
  );
}

const issues = computed(() => {
  const errors = [];
  const keys = state.products.map((p) => p.key.trim()).filter(Boolean);
  if (new Set(keys).size !== keys.length) errors.push("产品 key 有重复");
  for (const product of state.products) {
    const ids = product.modules.map((m) => m.id.trim()).filter(Boolean);
    if (new Set(ids).size !== ids.length) errors.push(`「${product.key}」的模块编号有重复`);
  }
  return { errors, warnings: [] };
});

defineExpose({ toJson, issues });

async function addProduct() {
  try {
    const { value } = await ElMessageBox.prompt(
      "产品 key 需与问题里的产品可对应（支持名称/别名模糊匹配）",
      "新增产品知识库",
      { inputPlaceholder: "如 感冒灵" }
    );
    if (!value?.trim()) return;
    state.products.push({ key: value.trim(), productName: value.trim(), modules: [] });
    activeProduct.value = value.trim();
  } catch {
    /* 取消 */
  }
}

function nextModuleId(product) {
  const nums = product.modules.map((m) => parseInt(m.id, 10)).filter((n) => !Number.isNaN(n));
  return String(Math.max(0, ...nums) + 1).padStart(2, "0");
}
</script>

<template>
  <div>
    <el-alert type="info" :closable="false" style="margin-bottom: 14px">
      <template #title>
        <span style="font-size: 13px">
          知识库是"品牌阶段·知识准确率"的比对标准：AI 回答里与这些模块矛盾的陈述会被判为错误。
          按产品维护，产品 key 与问题里的产品名自动模糊匹配（名称/别名）。
        </span>
      </template>
    </el-alert>
    <el-alert
      v-for="error in issues.errors"
      :key="error"
      type="error"
      :title="error"
      :closable="false"
      style="margin-bottom: 8px"
    />

    <el-space style="margin-bottom: 12px">
      <el-select v-model="activeProduct" placeholder="选择产品" style="width: 220px" filterable>
        <el-option v-for="p in state.products" :key="p.key" :value="p.key" :label="p.key" />
      </el-select>
      <el-button size="small" @click="addProduct">+ 新增产品</el-button>
      <el-button
        v-if="current"
        size="small"
        type="danger"
        plain
        @click="state.products = state.products.filter((p) => p.key !== activeProduct)"
      >
        删除当前产品
      </el-button>
    </el-space>

    <template v-if="current">
      <el-form label-width="90px" style="max-width: 480px; margin-bottom: 8px">
        <el-form-item label="产品全名">
          <el-input v-model="current.productName" size="small" />
        </el-form-item>
      </el-form>
      <el-collapse>
        <el-collapse-item v-for="(mod, index) in current.modules" :key="index">
          <template #title>
            <span style="font-weight: 600">{{ mod.id }} {{ mod.name || "（未命名模块）" }}</span>
          </template>
          <el-space style="margin-bottom: 8px">
            <el-input v-model="mod.id" size="small" style="width: 80px" placeholder="编号" />
            <el-input v-model="mod.name" size="small" style="width: 240px" placeholder="模块名，如 功效主治" />
            <el-button link type="danger" size="small" @click="current.modules.splice(index, 1)">
              删除模块
            </el-button>
          </el-space>
          <el-input v-model="mod.text" type="textarea" :rows="6" placeholder="该模块的标准知识文本" />
        </el-collapse-item>
      </el-collapse>
      <el-button
        size="small"
        style="margin-top: 10px"
        @click="current.modules.push({ id: nextModuleId(current), name: '', text: '' })"
      >
        + 添加模块
      </el-button>
    </template>
    <el-empty v-else description="还没有产品知识库，点击上方新增" :image-size="60" />
  </div>
</template>
