<script setup>
import { BarChart, LineChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from "echarts/components";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import VChart from "vue-echarts";

use([LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

defineProps({
  option: { type: Object, required: true },
  height: { type: String, default: "260px" },
});
</script>

<template>
  <v-chart :option="option" :style="{ height, width: '100%' }" autoresize />
</template>
