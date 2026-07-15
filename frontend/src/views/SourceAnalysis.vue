<script setup>
import { ref } from "vue";

import OutboundArticlePanel from "@/components/source/OutboundArticlePanel.vue";
import ContributorReviewPanel from "@/components/source/ContributorReviewPanel.vue";
import SourceAnalysisPanel from "@/components/source/SourceAnalysisPanel.vue";
import { useSessionStore } from "@/stores/session";

const activeTab = ref("sources");
const session = useSessionStore();
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>信源分析</h1>
      <p>分析 AI 回答的来源结构，并追踪对外发布文章是否进入新一轮 AI 信源。</p>
    </div>
    <el-tabs v-model="activeTab" class="source-workspace-tabs">
      <el-tab-pane label="信源结构分析" name="sources"><SourceAnalysisPanel /></el-tab-pane>
      <el-tab-pane label="外发文章追踪" name="articles" lazy><OutboundArticlePanel /></el-tab-pane>
      <el-tab-pane v-if="session.isAdmin" label="外部投稿审核" name="submissions" lazy><ContributorReviewPanel /></el-tab-pane>
    </el-tabs>
  </div>
</template>

<style scoped>
.source-workspace-tabs :deep(> .el-tabs__header) { margin-bottom: 16px; }
</style>
