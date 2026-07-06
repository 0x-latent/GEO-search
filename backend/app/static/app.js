const state = {
  datasetId: "all",
  data: null,
  splits: null,
  answers: [],
  activeTab: "overview",
  user: null,
};

const titles = {
  overview: ["总览", "查看两套数据在同一 SQLite 标准下的样本规模、覆盖范围和信源结构。"],
  coverage: ["产品覆盖", "对比不同数据集、产品和采集模式的样本覆盖。"],
  models: ["模型表现", "按模型查看问题覆盖、回答量、联网状态和信源引用。"],
  splits: ["拆分表现", "查看已有分析结果里的品牌、通用名、竞品、品类和问题级推荐明细。"],
  sources: ["信源", "定位不同数据集里被 AI 引用最多的域名。"],
  assets: ["数据资产", "查看已入库的 Excel/CSV 工作表，摆脱运行时 Excel 依赖。"],
  samples: ["回答样本", "快速抽查标准回答表中的问题、模型、回答和信源数量。"],
  analysis: ["我的分析", "上传问题，选择模型，系统自动采集回答并生成分析数据集。"],
  myconfig: ["我的配置", "维护自己的品牌配置和知识库，分析任务会优先使用你的配置。"],
  datasets: ["数据集管理", "查看全部数据集的归属和规模，删除不再需要的数据集。"],
  users: ["用户管理", "创建、删除用户，重置密码和调整角色。"],
};

const api = async (path, options) => {
  const res = await fetch(path, options);
  if (res.status === 401) {
    window.location.replace("/login.html");
    throw new Error("未登录");
  }
  if (!res.ok) {
    let message = res.statusText;
    try {
      const data = await res.json();
      message = data.detail || JSON.stringify(data);
    } catch (err) {
      // 保留 statusText
    }
    throw new Error(message);
  }
  return res.json();
};

const apiJson = (path, method, body) =>
  api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".nav").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  document.getElementById("dataset-select").addEventListener("change", async (event) => {
    state.datasetId = event.target.value;
    await loadDashboard();
  });
  document.getElementById("refresh").addEventListener("click", loadDashboard);
  document.getElementById("logout").addEventListener("click", logout);
  document.getElementById("create-user-form").addEventListener("submit", createUser);
  document.getElementById("change-password-form").addEventListener("submit", changeOwnPassword);
  document.getElementById("job-parse").addEventListener("click", parseJobFile);
  document.getElementById("job-submit").addEventListener("click", submitJob);
  document.getElementById("job-refresh").addEventListener("click", loadJobs);
  document.getElementById("my-brands-save").addEventListener("click", () => saveMyConfig("brands"));
  document.getElementById("my-brands-reset").addEventListener("click", () => resetMyConfig("brands"));
  document.getElementById("my-kb-save").addEventListener("click", () => saveMyConfig("kb"));
  document.getElementById("my-kb-reset").addEventListener("click", () => resetMyConfig("kb"));
  document.getElementById("global-brands-save").addEventListener("click", saveGlobalBrands);
  document.getElementById("global-kb-save").addEventListener("click", saveGlobalKb);
  boot();
});

async function boot() {
  try {
    state.user = await api("/api/auth/me");
    document.getElementById("current-username").textContent = state.user.username;
    document.getElementById("current-role").textContent =
      state.user.role === "admin" ? "管理员" : "普通用户";
    if (state.user.role === "admin") {
      document.getElementById("nav-users").hidden = false;
      document.getElementById("nav-datasets").hidden = false;
      document.getElementById("global-config-panels").hidden = false;
      document.getElementById("job-route-wrap").hidden = false;
    }
  } catch (error) {
    return;
  }
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
  if (tab === "users") {
    loadUsers();
  } else if (tab === "analysis") {
    loadJobOptions();
    loadJobs();
  } else if (tab === "myconfig") {
    loadMyConfigs();
  } else if (tab === "datasets") {
    loadDatasetAdmin();
  }
}

// ---------- 我的分析 ----------

async function loadJobOptions() {
  if (state.jobOptions) return;
  try {
    state.jobOptions = await api("/api/jobs/options");
    renderModelPicker();
  } catch (error) {
    showStatus(`加载模型选项失败：${error.message}`, true);
  }
}

function renderModelPicker() {
  const target = document.getElementById("job-models");
  target.innerHTML = state.jobOptions.models
    .map(
      (model) => `
      <div class="model-option">
        <label class="model-check">
          <input type="checkbox" data-model-key="${escapeHtml(model.key)}" checked />
          <strong>${escapeHtml(model.name)}</strong>
          ${model.supports_search ? '<span class="tag">支持联网</span>' : '<span class="tag muted">不联网</span>'}
        </label>
        <select data-model-variant="${escapeHtml(model.key)}">
          ${model.variants
            .map(
              (v, i) =>
                `<option value="${escapeHtml(v.id)}" ${i === 0 ? "selected" : ""}>${escapeHtml(v.label)}</option>`,
            )
            .join("")}
        </select>
      </div>
    `,
    )
    .join("");
}

async function parseJobFile() {
  const fileInput = document.getElementById("job-file");
  if (!fileInput.files.length) {
    showStatus("请先选择问题文件", true);
    return;
  }
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  form.append("default_product", document.getElementById("job-default-product").value.trim());
  try {
    const result = await api("/api/jobs/parse", { method: "POST", body: form });
    state.parsedQuestions = result.questions;
    showStatus(`解析成功：共 ${result.total} 个问题`, false);
    renderTable(
      "job-preview",
      result.preview.map((q) => ({ 编号: q.id, 产品: q.product, 层级: q.level, 问题: q.question })),
      [
        ["编号", "编号"],
        ["产品", "产品"],
        ["层级", "层级"],
        ["问题", "问题"],
      ],
    );
    document.getElementById("job-config-panel").hidden = false;
  } catch (error) {
    state.parsedQuestions = null;
    document.getElementById("job-config-panel").hidden = true;
    showStatus(`解析失败：${error.message}`, true);
  }
}

async function submitJob() {
  if (!state.parsedQuestions) {
    showStatus("请先解析问题文件", true);
    return;
  }
  const models = [];
  const overrides = {};
  document.querySelectorAll("#job-models input[data-model-key]").forEach((box) => {
    if (!box.checked) return;
    const key = box.dataset.modelKey;
    models.push(key);
    const select = document.querySelector(`select[data-model-variant="${key}"]`);
    const option = state.jobOptions.models.find((m) => m.key === key);
    if (select && option && select.value !== option.default_model) {
      overrides[key] = select.value;
    }
  });
  const payload = {
    dataset_name: document.getElementById("job-dataset-name").value.trim(),
    questions: state.parsedQuestions,
    models,
    model_overrides: overrides,
    search_mode: document.getElementById("job-search-mode").value,
    rounds: Number(document.getElementById("job-rounds").value),
  };
  if (state.user.role === "admin") {
    payload.route = document.getElementById("job-route").value;
  }
  const estimated = estimateCalls(payload);
  if (!window.confirm(`将发起约 ${estimated} 次模型调用，确认提交？`)) return;
  try {
    await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showStatus("任务已提交，正在排队执行", false);
    state.parsedQuestions = null;
    document.getElementById("job-config-panel").hidden = true;
    document.getElementById("job-preview").innerHTML = "";
    document.getElementById("job-file").value = "";
    await loadJobs();
  } catch (error) {
    showStatus(`提交失败：${error.message}`, true);
  }
}

function estimateCalls(payload) {
  let modes = payload.search_mode === "both" ? 2 : 1;
  const searchable = payload.models.filter((key) => {
    const m = state.jobOptions.models.find((item) => item.key === key);
    return m && m.supports_search;
  }).length;
  const plain = payload.models.length - searchable;
  const perQuestion =
    payload.search_mode === "both" ? searchable * 2 + plain : payload.search_mode === "search" ? searchable : payload.models.length;
  return payload.questions.length * perQuestion * payload.rounds;
}

const JOB_STATUS_LABELS = {
  queued: "排队中",
  running: "执行中",
  success: "已完成",
  failed: "失败",
};
const JOB_STAGE_LABELS = {
  collect: "采集回答",
  analyze: "统计报表",
  extract: "推荐抽取",
  import: "入库",
  done: "完成",
};

async function loadJobs() {
  try {
    const jobs = await api("/api/jobs");
    renderTable(
      "job-table",
      jobs.map((job) => ({
        任务: job.job_id,
        名称: job.dataset_name,
        提交人: job.username,
        状态: JOB_STATUS_LABELS[job.status] || job.status,
        阶段: JOB_STAGE_LABELS[job.stage] || job.stage || "-",
        题数: job.question_count,
        模型: JSON.parse(job.models_json).join("、"),
        创建时间: job.created_at,
        错误: job.error || "",
      })),
      [
        ["任务", "任务ID"],
        ["名称", "数据集"],
        ["提交人", "提交人"],
        ["状态", "状态"],
        ["阶段", "阶段"],
        ["题数", "题数"],
        ["模型", "模型"],
        ["创建时间", "创建时间"],
        ["错误", "错误"],
      ],
    );
    // 表格行点击查看日志
    document.querySelectorAll("#job-table tbody tr").forEach((tr, index) => {
      tr.style.cursor = "pointer";
      tr.addEventListener("click", () => showJobLog(jobs[index].job_id));
    });
  } catch (error) {
    showStatus(`加载任务失败：${error.message}`, true);
  }
}

async function showJobLog(jobId) {
  try {
    const result = await api(`/api/jobs/${encodeURIComponent(jobId)}/log`);
    const view = document.getElementById("job-log");
    view.hidden = false;
    view.textContent = result.log || "（暂无日志）";
    view.scrollTop = view.scrollHeight;
  } catch (error) {
    showStatus(`读取日志失败：${error.message}`, true);
  }
}

// ---------- 我的配置 ----------

async function loadMyConfigs() {
  try {
    const [brands, kb] = await Promise.all([
      api("/api/config/my/brands"),
      api("/api/config/my/knowledge-base"),
    ]);
    document.getElementById("my-brands-editor").value = JSON.stringify(brands.data, null, 2);
    document.getElementById("my-brands-source").textContent =
      brands.source === "user" ? "当前：自定义配置" : "当前：全局默认（保存后生成你的副本）";
    document.getElementById("my-kb-editor").value = JSON.stringify(kb.data, null, 2);
    document.getElementById("my-kb-source").textContent =
      kb.source === "user" ? "当前：自定义配置" : "当前：全局默认（保存后生成你的副本）";
    if (state.user.role === "admin") {
      const [globalBrands, globalKb] = await Promise.all([
        api("/api/config/brands"),
        api("/api/config/knowledge-base"),
      ]);
      document.getElementById("global-brands-editor").value = JSON.stringify(globalBrands, null, 2);
      document.getElementById("global-kb-editor").value = JSON.stringify(globalKb, null, 2);
    }
  } catch (error) {
    showStatus(`加载配置失败：${error.message}`, true);
  }
}

function readEditorJson(id) {
  const text = document.getElementById(id).value;
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error("JSON 格式错误：" + error.message);
  }
}

async function saveMyConfig(kind) {
  const editorId = kind === "brands" ? "my-brands-editor" : "my-kb-editor";
  const path = kind === "brands" ? "/api/config/my/brands" : "/api/config/my/knowledge-base";
  try {
    const data = readEditorJson(editorId);
    await api(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data }),
    });
    showStatus("已保存自定义配置", false);
    await loadMyConfigs();
  } catch (error) {
    showStatus(`保存失败：${error.message}`, true);
  }
}

async function resetMyConfig(kind) {
  const path = kind === "brands" ? "/api/config/my/brands" : "/api/config/my/knowledge-base";
  if (!window.confirm("确认删除自定义配置并恢复全局默认？")) return;
  try {
    await api(path, { method: "DELETE" });
    showStatus("已恢复全局默认", false);
    await loadMyConfigs();
  } catch (error) {
    showStatus(`操作失败：${error.message}`, true);
  }
}

async function saveGlobalBrands() {
  try {
    const data = readEditorJson("global-brands-editor");
    await api("/api/config/brands", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data }),
    });
    showStatus("全局品牌配置已保存（自动生成版本快照）", false);
  } catch (error) {
    showStatus(`保存失败：${error.message}`, true);
  }
}

async function saveGlobalKb() {
  try {
    const data = readEditorJson("global-kb-editor");
    await api("/api/config/knowledge-base", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data }),
    });
    showStatus("全局知识库已保存", false);
  } catch (error) {
    showStatus(`保存失败：${error.message}`, true);
  }
}

// ---------- 数据集管理（admin） ----------

async function loadDatasetAdmin() {
  try {
    const datasets = await api("/api/sqlite/datasets");
    const target = document.getElementById("dataset-admin-table");
    if (!datasets.length) {
      target.innerHTML = `<p class="empty">暂无数据集</p>`;
      return;
    }
    target.innerHTML = `
      <table>
        <thead>
          <tr><th>数据集ID</th><th>名称</th><th>归属</th><th>问题</th><th>回答</th><th>外部表</th><th>操作</th></tr>
        </thead>
        <tbody>
          ${datasets
            .map(
              (d) => `
            <tr>
              <td>${escapeHtml(d.dataset_id)}</td>
              <td>${escapeHtml(d.name)}</td>
              <td>${escapeHtml(d.owner_username || "系统")}</td>
              <td>${formatNumber(d.questions)}</td>
              <td>${formatNumber(d.answers)}</td>
              <td>${formatNumber(d.external_tables)}</td>
              <td><button class="danger" data-dataset="${escapeHtml(d.dataset_id)}">删除</button></td>
            </tr>
          `,
            )
            .join("")}
        </tbody>
      </table>
    `;
    target.querySelectorAll("button[data-dataset]").forEach((button) => {
      button.addEventListener("click", async () => {
        const id = button.dataset.dataset;
        if (!window.confirm(`确认删除数据集 ${id}？该操作不可恢复。`)) return;
        try {
          await api(`/api/sqlite/datasets/${encodeURIComponent(id)}`, { method: "DELETE" });
          showStatus(`已删除数据集 ${id}`, false);
          await loadDatasetAdmin();
        } catch (error) {
          showStatus(`删除失败：${error.message}`, true);
        }
      });
    });
  } catch (error) {
    showStatus(`加载数据集失败：${error.message}`, true);
  }
}

async function logout() {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch (err) {
    // 会话可能已过期，直接跳转
  }
  window.location.replace("/login.html");
}

async function loadUsers() {
  try {
    const users = await api("/api/auth/users");
    renderUsers(users);
  } catch (error) {
    showStatus(`加载用户失败：${error.message}`, true);
  }
}

function renderUsers(users) {
  const target = document.getElementById("user-table");
  if (!users || users.length === 0) {
    target.innerHTML = `<p class="empty">暂无用户</p>`;
    return;
  }
  target.innerHTML = `
    <table>
      <thead>
        <tr><th>用户名</th><th>角色</th><th>创建时间</th><th>操作</th></tr>
      </thead>
      <tbody>
        ${users
          .map((user) => {
            const name = escapeHtml(user.username);
            const isSelf = user.username === state.user.username;
            const nextRole = user.role === "admin" ? "user" : "admin";
            const roleLabel = user.role === "admin" ? "管理员" : "普通用户";
            const created = new Date(user.created_at * 1000).toLocaleString("zh-CN");
            return `
              <tr>
                <td>${name}${isSelf ? "（我）" : ""}</td>
                <td>${roleLabel}</td>
                <td>${created}</td>
                <td class="user-actions">
                  <button data-action="reset" data-username="${name}">重置密码</button>
                  <button data-action="role" data-username="${name}" data-role="${nextRole}" ${isSelf ? "disabled" : ""}>
                    设为${nextRole === "admin" ? "管理员" : "普通用户"}
                  </button>
                  <button data-action="delete" data-username="${name}" class="danger" ${isSelf ? "disabled" : ""}>删除</button>
                </td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
  target.querySelectorAll("button[data-action]").forEach((button) => {
    button.addEventListener("click", () => handleUserAction(button.dataset));
  });
}

async function handleUserAction({ action, username, role }) {
  try {
    if (action === "delete") {
      if (!window.confirm(`确认删除用户 ${username}？`)) return;
      await api(`/api/auth/users/${encodeURIComponent(username)}`, { method: "DELETE" });
      showStatus(`已删除用户 ${username}`, false);
    } else if (action === "reset") {
      const password = window.prompt(`为 ${username} 设置新密码（至少 6 位）：`);
      if (!password) return;
      await apiJson(`/api/auth/users/${encodeURIComponent(username)}/password`, "PUT", { password });
      showStatus(`已重置 ${username} 的密码`, false);
    } else if (action === "role") {
      await apiJson(`/api/auth/users/${encodeURIComponent(username)}/role`, "PUT", { role });
      showStatus(`已调整 ${username} 的角色`, false);
    }
    await loadUsers();
  } catch (error) {
    showStatus(`操作失败：${error.message}`, true);
  }
}

async function createUser(event) {
  event.preventDefault();
  const username = document.getElementById("new-username").value.trim();
  const password = document.getElementById("new-password").value;
  const role = document.getElementById("new-role").value;
  try {
    await apiJson("/api/auth/users", "POST", { username, password, role });
    event.target.reset();
    showStatus(`已创建用户 ${username}`, false);
    await loadUsers();
  } catch (error) {
    showStatus(`创建失败：${error.message}`, true);
  }
}

async function changeOwnPassword(event) {
  event.preventDefault();
  const password = document.getElementById("own-password").value;
  try {
    await apiJson("/api/auth/me/password", "PUT", { password });
    window.alert("密码已修改，请重新登录");
    window.location.replace("/login.html");
  } catch (error) {
    showStatus(`修改失败：${error.message}`, true);
  }
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
