<script setup>
import { computed } from "vue";

import { fmtDelta } from "@/utils/format";

const props = defineProps({
  value: { type: [Number, null], default: null },
  // 对负面类指标，涨=坏（红），跌=好（绿）
  inverse: { type: Boolean, default: false },
});

const delta = computed(() => fmtDelta(props.value));
const tone = computed(() => {
  if (!delta.value || delta.value.direction === "flat") return "info";
  const good = props.inverse
    ? delta.value.direction === "down"
    : delta.value.direction === "up";
  return good ? "success" : "danger";
});
</script>

<template>
  <el-tag v-if="delta" :type="tone" size="small" effect="light" round>
    <span v-if="delta.direction === 'up'">▲</span>
    <span v-else-if="delta.direction === 'down'">▼</span>
    {{ delta.text }}
  </el-tag>
  <span v-else class="muted" style="font-size: 12px">—</span>
</template>
