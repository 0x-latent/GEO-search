<script setup>
// 品牌配置结构化编辑器：表格/标签编辑，序列化回 brands.yaml 的 JSON 结构。
// 数据链路（帮助用户理解"配置如何关联到问题"）：
//   上传问题的"产品"列 → product_name_map → category_keywords → 品类提及率
//   brand_999（名称+别名）→ 999品牌识别；known_brand_competitors → 竞品识别
import { computed, reactive, watch } from "vue";

const props = defineProps({
  data: { type: Object, required: true },
});
const emit = defineEmits(["change"]);

const state = reactive({
  products: [],
  competitors: [],
  genericNames: [],
  componentKeywords: [],
  categoryKeywords: [],
  productNameMap: [],
});

function fromJson(data) {
  state.products = Object.entries(data.brand_999 || {}).map(([name, spec]) => ({
    name,
    aliases: [...((spec || {}).aliases || [])],
    category: (spec || {}).category || "",
  }));
  state.competitors = [...(data.known_brand_competitors || [])];
  state.genericNames = [...(data.generic_names || [])];
  state.componentKeywords = [...(data.component_keywords || [])];
  state.categoryKeywords = Object.entries(data.category_keywords || {}).map(([key, words]) => ({
    key,
    keywords: [...(words || [])],
  }));
  state.productNameMap = Object.entries(data.product_name_map || {}).map(([dataName, categoryKey]) => ({
    dataName,
    categoryKey,
  }));
}

watch(() => props.data, fromJson, { immediate: true });

function toJson() {
  return {
    ...props.data,
    brand_999: Object.fromEntries(
      state.products
        .filter((p) => p.name.trim())
        .map((p) => [p.name.trim(), { aliases: p.aliases.filter(Boolean), category: p.category.trim() }])
    ),
    known_brand_competitors: state.competitors.filter(Boolean),
    generic_names: state.genericNames.filter(Boolean),
    component_keywords: state.componentKeywords.filter(Boolean),
    category_keywords: Object.fromEntries(
      state.categoryKeywords
        .filter((c) => c.key.trim())
        .map((c) => [c.key.trim(), c.keywords.filter(Boolean)])
    ),
    product_name_map: Object.fromEntries(
      state.productNameMap
        .filter((m) => m.dataName.trim() && m.categoryKey.trim())
        .map((m) => [m.dataName.trim(), m.categoryKey.trim()])
    ),
  };
}

// 校验：保存前给出明确错误/警告
const issues = computed(() => {
  const errors = [];
  const warnings = [];
  const names = state.products.map((p) => p.name.trim()).filter(Boolean);
  if (new Set(names).size !== names.length) errors.push("品牌名有重复");
  state.products.forEach((p) => {
    if (!p.name.trim()) warnings.push("存在未命名的品牌行（保存时会被丢弃）");
  });
  const catKeys = new Set(state.categoryKeywords.map((c) => c.key.trim()).filter(Boolean));
  state.productNameMap.forEach((m) => {
    if (m.categoryKey && !catKeys.has(m.categoryKey)) {
      errors.push(`产品关联「${m.dataName}」指向的品类「${m.categoryKey}」在品类关键词里不存在`);
    }
  });
  return { errors, warnings: [...new Set(warnings)] };
});

defineExpose({ toJson, issues });
const categoryOptions = computed(() =>
  state.categoryKeywords.map((c) => c.key).filter(Boolean)
);
</script>

<template>
  <div>
    <el-alert type="info" :closable="false" style="margin-bottom: 14px">
      <template #title>
        <span style="font-size: 13px">
          配置如何生效：上传问题里的「产品」列 → <b>产品关联表</b> → <b>品类关键词</b> 决定品类提及率；
          <b>品牌与别名</b> 决定 999 品牌识别；<b>竞品清单</b> 决定竞品识别。改完保存即对下一次分析生效。
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

    <h4>品牌与别名（brand_999）</h4>
    <el-table :data="state.products" size="small" border>
      <el-table-column label="品牌名" width="180">
        <template #default="{ row }">
          <el-input v-model="row.name" placeholder="如 999感冒灵" size="small" />
        </template>
      </el-table-column>
      <el-table-column label="别名（回车添加）" min-width="320">
        <template #default="{ row }">
          <el-select
            v-model="row.aliases"
            multiple
            filterable
            allow-create
            default-first-option
            :reserve-keyword="false"
            placeholder="输入别名后回车"
            size="small"
            style="width: 100%"
          />
        </template>
      </el-table-column>
      <el-table-column label="品类" width="140">
        <template #default="{ row }">
          <el-input v-model="row.category" placeholder="如 感冒药" size="small" />
        </template>
      </el-table-column>
      <el-table-column width="60">
        <template #default="{ $index }">
          <el-button link type="danger" size="small" @click="state.products.splice($index, 1)">删</el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-button
      size="small"
      style="margin: 8px 0 18px"
      @click="state.products.push({ name: '', aliases: [], category: '' })"
    >
      + 添加品牌
    </el-button>

    <h4>产品关联表（product_name_map）— 数据中的产品名 ↔ 品类</h4>
    <p class="muted hint">上传问题里"产品"列的写法（含短名）都要在这里挂到一个品类，品类提及率才有依据。</p>
    <el-table :data="state.productNameMap" size="small" border style="max-width: 560px">
      <el-table-column label="数据中的产品名" width="200">
        <template #default="{ row }">
          <el-input v-model="row.dataName" placeholder="如 感冒灵 / 999感冒灵" size="small" />
        </template>
      </el-table-column>
      <el-table-column label="所属品类（引用下方品类关键词）">
        <template #default="{ row }">
          <el-select v-model="row.categoryKey" filterable allow-create size="small" style="width: 100%">
            <el-option v-for="key in categoryOptions" :key="key" :value="key" :label="key" />
          </el-select>
        </template>
      </el-table-column>
      <el-table-column width="60">
        <template #default="{ $index }">
          <el-button link type="danger" size="small" @click="state.productNameMap.splice($index, 1)">删</el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-button
      size="small"
      style="margin: 8px 0 18px"
      @click="state.productNameMap.push({ dataName: '', categoryKey: '' })"
    >
      + 添加关联
    </el-button>

    <h4>品类关键词（category_keywords）— AI 回答中出现即算"品类被提及"</h4>
    <el-table :data="state.categoryKeywords" size="small" border>
      <el-table-column label="品类" width="180">
        <template #default="{ row }">
          <el-input v-model="row.key" placeholder="如 感冒药" size="small" />
        </template>
      </el-table-column>
      <el-table-column label="关键词（回车添加）">
        <template #default="{ row }">
          <el-select
            v-model="row.keywords"
            multiple
            filterable
            allow-create
            default-first-option
            :reserve-keyword="false"
            size="small"
            style="width: 100%"
          />
        </template>
      </el-table-column>
      <el-table-column width="60">
        <template #default="{ $index }">
          <el-button link type="danger" size="small" @click="state.categoryKeywords.splice($index, 1)">删</el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-button
      size="small"
      style="margin: 8px 0 18px"
      @click="state.categoryKeywords.push({ key: '', keywords: [] })"
    >
      + 添加品类
    </el-button>

    <el-row :gutter="16">
      <el-col :span="8">
        <h4>竞品品牌（{{ state.competitors.length }}）</h4>
        <el-select
          v-model="state.competitors"
          multiple
          filterable
          allow-create
          default-first-option
          :reserve-keyword="false"
          placeholder="输入竞品名后回车"
          style="width: 100%"
        />
      </el-col>
      <el-col :span="8">
        <h4>通用名清单（{{ state.genericNames.length }}）</h4>
        <el-select
          v-model="state.genericNames"
          multiple
          filterable
          allow-create
          default-first-option
          :reserve-keyword="false"
          placeholder="法定通用名，如 感冒灵颗粒"
          style="width: 100%"
        />
      </el-col>
      <el-col :span="8">
        <h4>成分词（{{ state.componentKeywords.length }}）</h4>
        <el-select
          v-model="state.componentKeywords"
          multiple
          filterable
          allow-create
          default-first-option
          :reserve-keyword="false"
          placeholder="如 对乙酰氨基酚"
          style="width: 100%"
        />
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
h4 {
  margin: 6px 0 8px;
  font-size: 14px;
}
.hint {
  font-size: 12px;
  margin: 0 0 8px;
}
</style>
