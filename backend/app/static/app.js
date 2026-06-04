const state = {
  datasetId: "all",
  data: null,
  splits: null,
  answers: [],
  activeTab: "overview",
};

const titles = {
  overview: ["总览", "查看两套数据在同一 SQLite 标准下的样本规模、覆盖范围和信源结构。"],
  coverage: ["产品覆盖", "对比不同数据集、产品和采集模式的样本覆盖。"],
  models: ["模型表现", "按模型查看问题覆盖、回答量、联网状态和信源引用。"],
  splits: ["拆分表现", "查看已有分析结果里的品牌、通用名、竞品、品类和问题级推荐明细。"],
  sources: ["信源", "定位不同数据集里被 AI 引用最多的域名。"],
  assets: ["数据资产", "查看已入库的 Excel/CSV 工作表，摆脱运行时 Excel 依赖。"],
  samples: ["回答样本", "快速抽查标准回答表中的问题、模型、回答和信源数量。"],
};

const api = async (path) => {
  const res = await fetch(path);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
};

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".nav").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  document.getElementById("dataset-select").addEventListener("change", async (event) => {
    state.datasetId = event.target.value;
    await loadDashboard();
  });
  document.getElementById("refresh").addEventListener("click", loadDashboard);
  boot();
});

async function boot() {
  try {
    const datasets = await api("/api/sqlite/datasets");
    const select = document.getElementById("dataset-select");
    select.innerHTML = [
      `<option value="all">全部数据集</option>`,
      ...datasets.map((item) => `<option value="${escapeHtml(item.dataset_id)}">${escapeHtml(item.name)}</option>`),
    ].join("");
    await loadDashboard();
  } catch (error) {
    showStatus(`无法加载 SQLite 数据库：${error.message}`, true);
  }
}

async function loadDashboard() {
  try {
    showStatus("正在加载数据...", false);
    const query = encodeURIComponent(state.datasetId);
    const [overview, answers, splits] = await Promise.all([
      api(`/api/sqlite/overview?dataset_id=${query}`),
      api(`/api/sqlite/answers?dataset_id=${query}&limit=160`),
      api(`/api/sqlite/splits?dataset_id=${query}`),
    ]);
    state.data = overview;
    state.answers = answers;
    state.splits = splits;
    renderAll();
    showStatus("", false);
  } catch (error) {
    showStatus(`加载失败：${error.message}`, true);
  }
}

function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".nav").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === tab);
  });
  document.getElementById("view-title").textContent = titles[tab][0];
  document.getElementById("view-subtitle").textContent = titles[tab][1];
}

function renderAll() {
  if (!state.data) return;
  renderKpis();
  renderDatasets();
  renderSearchModes();
  renderProducts();
  renderModels();
  renderLevels();
  renderSplits();
  renderSources();
  renderAssets();
  renderScenarios();
  renderAnswerSamples();
}

function renderSplits() {
  const splits = state.splits || {};
  renderTable("mention-split-table", splits.mention_summary || [], [
    ["dataset_id", "数据集"],
    ["产品", "产品"],
    ["问题层级", "层级"],
    ["模型", "模型"],
    ["联网", "联网"],
    ["总回答数", "回答"],
    ["目标品牌", "目标品牌"],
    ["目标品牌提及率", "目标品牌提及率"],
    ["目标品牌前三率", "目标品牌前三率"],
    ["目标品牌首位率", "目标品牌首位率"],
    ["平均位次", "平均位次"],
    ["品类提及率", "品类提及率"],
    ["999品牌提及率", "999品牌提及率"],
    ["999品牌推荐率", "999品牌推荐率"],
    ["通用名提及率", "通用名提及率"],
    ["通用名推荐率", "通用名推荐率"],
    ["竞品品牌提及率", "竞品提及率"],
    ["竞品品牌推荐率", "竞品推荐率"],
  ]);

  renderTable("rec-overview-table", splits.rec_overview || [], [
    ["dataset_id", "数据集"],
    ["产品", "产品"],
    ["问题层级", "层级"],
    ["模型", "模型"],
    ["联网", "联网"],
    ["排名", "排名"],
    ["被推荐产品", "被推荐产品"],
    ["名称类型", "名称类型"],
    ["提及次数", "提及次数"],
    ["提及率", "提及率"],
    ["强推荐次数", "强推荐次数"],
    ["强推荐率", "强推荐率"],
    ["平均首位率", "平均首位率"],
    ["平均位次", "平均位次"],
  ]);

  renderTable("type-summary-table", splits.type_summary || [], [
    ["dataset_id", "数据集"],
    ["产品", "产品"],
    ["问题层级", "层级"],
    ["模型", "模型"],
    ["联网", "联网"],
    ["名称类型", "名称类型"],
    ["推荐条目数", "推荐条目数"],
    ["强推荐数", "强推荐数"],
    ["强推荐占比", "强推荐占比"],
    ["涉及推荐产品数", "涉及产品数"],
  ]);

  renderTable("category-summary-table", splits.category_summary || [], [
    ["dataset_id", "数据集"],
    ["产品", "产品"],
    ["模型", "模型"],
    ["联网", "联网"],
    ["品类", "品类"],
    ["推荐次数", "推荐次数"],
    ["强推荐数", "强推荐数"],
  ]);

  renderTable("yang-brand-table", splits.yangweishu_brand_summary || [], [
    ["dataset_id", "数据集"],
    ["层级", "层级"],
    ["模型", "模型"],
    ["目标品牌", "目标品牌"],
    ["样本数", "样本数"],
    ["平均能见度", "平均能见度"],
    ["平均前三率", "平均前三率"],
    ["平均首位率", "平均首位率"],
    ["平均位次", "平均位次"],
  ]);

  renderTable("question-detail-table", splits.question_details || [], [
    ["dataset_id", "数据集"],
    ["问题ID", "问题ID"],
    ["产品", "产品"],
    ["问题层级", "层级"],
    ["模型", "模型"],
    ["联网", "联网"],
    ["轮次", "轮次"],
    ["推荐排名", "排名"],
    ["推荐产品", "推荐产品"],
    ["名称类型", "名称类型"],
    ["推荐强度", "强度"],
    ["推荐原因", "推荐原因"],
  ]);
}

function renderKpis() {
  const cards = [
    ["数据集", state.data.cards.datasets],
    ["产品", state.data.cards.products],
    ["问题", state.data.cards.questions],
    ["回答", state.data.cards.answers],
    ["模型", state.data.cards.models],
    ["引用 URL", state.data.cards.source_urls],
    ["外部表", state.data.cards.external_tables],
    ["外部行", state.data.cards.external_rows],
  ];
  document.getElementById("kpis").innerHTML = cards
    .map(([label, value]) => `<div class="kpi"><strong>${formatNumber(value)}</strong><span>${label}</span></div>`)
    .join("");
}

function renderDatasets() {
  renderTable("dataset-table", state.data.datasets, [
    ["name", "数据集"],
    ["questions", "问题"],
    ["answers", "回答"],
    ["products", "产品"],
    ["models", "模型"],
    ["source_urls", "引用 URL"],
    ["external_tables", "外部表"],
    ["external_rows", "外部行"],
  ]);
}

function renderSearchModes() {
  renderBars("search-bars", state.data.search_modes, {
    label: (row) => `${datasetShort(row.dataset_id)} / ${row.search_mode}`,
    value: (row) => row.answers,
    detail: (row) => `${formatNumber(row.answers)} 条，均 ${formatNumber(row.avg_answer_chars)} 字，信源 ${formatNumber(row.source_refs)}`,
  });
}

function renderProducts() {
  renderTable("product-table", state.data.products, [
    ["dataset_id", "数据集"],
    ["product_name", "产品"],
    ["questions", "问题"],
    ["answers", "回答"],
    ["models", "模型"],
    ["search_answers", "联网回答"],
    ["nosearch_answers", "离线回答"],
    ["avg_answer_chars", "平均字数"],
  ]);
}

function renderModels() {
  renderBars("model-bars", state.data.models, {
    label: (row) => `${datasetShort(row.dataset_id)} / ${row.model}`,
    value: (row) => row.answers,
    detail: (row) => `${formatNumber(row.answers)} 条，问题 ${formatNumber(row.questions)}，信源 ${formatNumber(row.source_refs)}`,
  });
  renderTable("model-table", state.data.models, [
    ["dataset_id", "数据集"],
    ["model", "模型"],
    ["answers", "回答"],
    ["questions", "问题"],
    ["search_answers", "联网回答"],
    ["nosearch_answers", "离线回答"],
    ["avg_answer_chars", "平均字数"],
    ["source_refs", "信源引用"],
  ]);
}

function renderLevels() {
  renderTable("level-table", state.data.levels, [
    ["dataset_id", "数据集"],
    ["source_level", "原始层级"],
    ["level", "标准层级"],
    ["questions", "问题数"],
  ]);
}

function renderSources() {
  renderBars("source-bars", state.data.sources, {
    label: (row) => row.domain,
    value: (row) => row.refs,
    detail: (row) => `${datasetShort(row.dataset_id)} / ${formatNumber(row.refs)} 次`,
  });
  renderTable("source-table", state.data.sources, [
    ["dataset_id", "数据集"],
    ["domain", "域名"],
    ["refs", "引用次数"],
  ]);
}

function renderAssets() {
  renderTable("asset-table", state.data.external_tables, [
    ["dataset_id", "数据集"],
    ["file_name", "文件"],
    ["sheet_name", "工作表"],
    ["row_count", "行数"],
  ]);
}

function renderScenarios() {
  renderTable("scenario-table", state.data.scenarios, [
    ["dataset_id", "数据集"],
    ["source_level", "层级"],
    ["scenario", "场景"],
    ["questions", "问题数"],
  ]);
}

function renderAnswerSamples() {
  renderTable("answer-table", state.answers, [
    ["dataset_id", "数据集"],
    ["product_name", "产品"],
    ["source_level", "层级"],
    ["scenario", "场景"],
    ["model", "模型"],
    ["search_mode", "模式"],
    ["round", "轮次"],
    ["answer_chars", "字数"],
    ["source_count", "信源"],
    ["question_text", "问题"],
    ["answer_preview", "回答摘录"],
  ]);
}

function renderTable(id, rows, columns) {
  const target = document.getElementById(id);
  if (!rows || rows.length === 0) {
    target.innerHTML = `<p class="empty">暂无数据</p>`;
    return;
  }
  target.innerHTML = `
    <table>
      <thead>
        <tr>${columns.map(([, label]) => `<th>${escapeHtml(label)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
            <tr>${columns.map(([key]) => `<td>${formatCell(row[key])}</td>`).join("")}</tr>
          `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderBars(id, rows, config) {
  const target = document.getElementById(id);
  if (!rows || rows.length === 0) {
    target.innerHTML = `<p class="empty">暂无数据</p>`;
    return;
  }
  const values = rows.map((row) => Number(config.value(row)) || 0);
  const max = Math.max(...values, 1);
  target.innerHTML = rows
    .map((row) => {
      const value = Number(config.value(row)) || 0;
      const width = Math.max(2, (value / max) * 100);
      return `
        <div class="bar-row">
          <div class="bar-label" title="${escapeHtml(config.label(row))}">${escapeHtml(config.label(row))}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <div class="bar-value">${escapeHtml(config.detail(row))}</div>
        </div>
      `;
    })
    .join("");
}

function showStatus(text, isError) {
  const el = document.getElementById("status");
  if (!text) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = text;
  el.classList.toggle("error", Boolean(isError));
}

function datasetShort(id) {
  if (!id) return "";
  if (id === "baseline_8products_20260423") return "8产品";
  if (id === "weitai_yangweishu_20260602") return "养胃舒";
  return id;
}

function formatCell(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "number") return formatNumber(value);
  return escapeHtml(String(value));
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0";
  const fractionDigits = Number.isInteger(number) ? 0 : 4;
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: fractionDigits }).format(number);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
