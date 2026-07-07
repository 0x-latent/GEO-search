<script setup>
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage } from "element-plus";

import { api } from "@/api/client";
import DeltaBadge from "@/components/DeltaBadge.vue";
import { fmtRate } from "@/utils/format";

const router = useRouter();
const loading = ref(true);
const cards = ref([]);

onMounted(async () => {
  try {
    cards.value = await api("/api/insight/products");
  } catch (error) {
    ElMessage.error(`加载失败：${error.message}`);
  } finally {
    loading.value = false;
  }
});

function open(card) {
  if (!card.batch_count) return;
  router.push(`/products/${card.product_code}`);
}
</script>

<template>
  <div class="page" v-loading="loading">
    <div class="page-header">
      <h1>品牌总览</h1>
      <p>每个产品在 AI 搜索中的三阶段表现（消费者链路：病症 → 品类 → 品牌），涨跌为与上一可比批次的差值。</p>
    </div>

    <el-empty v-if="!loading && !cards.length" description="暂无产品数据" />

    <div class="card-grid">
      <el-card
        v-for="card in cards"
        :key="card.product_code"
        :class="['health-card', { disabled: !card.batch_count }]"
        shadow="hover"
        @click="open(card)"
      >
        <div class="card-head">
          <div>
            <strong class="name">{{ card.product_name }}</strong>
            <el-tag v-if="card.category" size="small" effect="plain" style="margin-left: 8px">
              {{ card.category }}
            </el-tag>
          </div>
          <span class="muted batch" v-if="card.latest_batch">
            {{ card.latest_batch.batch_date }} · 共 {{ card.batch_count }} 批
          </span>
        </div>

        <template v-if="card.metrics">
          <div class="rows">
            <div class="row">
              <span class="metric-label">病症阶段 · 品类提及率</span>
              <span class="metric-value">{{ fmtRate(card.metrics.symptom_category_rate) }}</span>
              <DeltaBadge :value="card.delta?.symptom_category_rate ?? null" />
            </div>
            <div class="row">
              <span class="metric-label">品类阶段 · 品牌提及率</span>
              <span class="metric-value">{{ fmtRate(card.metrics.category_brand_rate) }}</span>
              <DeltaBadge :value="card.delta?.category_brand_rate ?? null" />
            </div>
            <div class="row">
              <span class="metric-label">品牌阶段 · 知识准确率</span>
              <span class="metric-value">{{ fmtRate(card.metrics.accuracy_rate) }}</span>
              <DeltaBadge :value="card.delta?.accuracy_rate ?? null" />
            </div>
            <div class="row">
              <span class="metric-label">品牌负面提及</span>
              <span class="metric-value" :class="{ danger: card.metrics.negative_count > 0 }">
                {{ card.metrics.negative_count ?? "—" }} 条
              </span>
              <DeltaBadge :value="card.delta?.negative_rate ?? null" inverse />
            </div>
          </div>
          <div class="card-foot">
            <span class="muted" v-if="!card.delta">尚无可比历史批次（问题集需一致才计算涨跌）</span>
            <el-button link type="primary">查看详情 →</el-button>
          </div>
        </template>
        <el-empty v-else description="尚未采集数据" :image-size="48">
          <el-button size="small" @click.stop="router.push('/analysis')">去发起分析</el-button>
        </el-empty>
      </el-card>
    </div>
  </div>
</template>

<style scoped>
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 16px;
}
.health-card {
  cursor: pointer;
}
.health-card.disabled {
  cursor: default;
  opacity: 0.75;
}
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 12px;
}
.name {
  font-size: 16px;
}
.batch {
  font-size: 12px;
}
.rows {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  align-items: center;
  gap: 10px;
}
.row .metric-value {
  font-size: 18px;
}
.danger {
  color: #b91c1c;
}
.card-foot {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 14px;
  font-size: 12px;
}
</style>
