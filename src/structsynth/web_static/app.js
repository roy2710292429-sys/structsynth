const state = {
  provider: "mock",
  jobId: null,
  pollTimer: null,
  result: null,
  activeTab: "overview",
  activeArm: "spectral",
  hasChatEnvKey: false,
  hasEmbeddingEnvKey: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const ARM_LABELS = {
  baseline: "普通仿写",
  matrix: "仅 M 条件",
  spectral: "谱模板",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("visible");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => node.classList.remove("visible"), 3000);
}

function seedLines() {
  return $("#seeds-input")
    .value.split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function updateSeedCount() {
  const count = seedLines().length;
  $("#seed-count").textContent = `${count} 题`;
  $("#seed-hint").textContent =
    count < 3
      ? "至少需要 3 道种子题。"
      : count < 18
        ? "当前样本较小，谱结构可能不稳定。"
        : "建议保留一部分同类题作留出集，用于独立评测。";
  $$("#seeds-input").forEach((input) => {
    input.style.borderColor =
      count > 0 && count < 3 ? "rgba(220, 106, 77, .55)" : "";
  });
}

function syncRange(input) {
  const output = document.querySelector(`output[for="${input.id}"]`);
  if (!output) return;
  const min = Number(input.min);
  const max = Number(input.max);
  const value = Number(input.value);
  const ratio = ((value - min) / (max - min)) * 100;
  input.style.setProperty("--range", `${ratio}%`);
  if (input.id === "activation" || input.id === "neighbor-fraction") {
    output.textContent = `${value}%`;
  } else if (input.id === "temperature") {
    output.textContent = value.toFixed(2);
  } else {
    output.textContent = value;
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    if (Number(config.api_schema || 0) < 3) {
      const latest = `${window.location.protocol}//${window.location.hostname}:8769/`;
      document.body.innerHTML = `<div class="error-state" style="margin:15vh auto">
        <span>OUTDATED BACKGROUND</span>
        <h2>正在切换到新版后台</h2>
        <p>检测到旧服务进程。如果没有自动跳转，请点击：
          <a href="${latest}">${latest}</a>
        </p>
      </div>`;
      window.setTimeout(() => window.location.replace(latest), 500);
      return;
    }
    $("#chat-base-url").value =
      config.base_url || "https://api.deepseek.com";
    $("#chat-model").value = config.chat_model || "";
    $("#embedding-base-url").value =
      config.embedding_base_url || "https://api.openai.com/v1";
    $("#embedding-model").value = config.embedding_model || "";
    state.hasChatEnvKey = Boolean(config.has_env_chat_key);
    state.hasEmbeddingEnvKey = Boolean(config.has_env_embedding_key);
    $("#chat-env-status").textContent = config.has_env_chat_key
      ? "检测到聊天 API Key 环境变量，可留空。"
      : "未检测到聊天 API Key 环境变量；需要在下方填写。";
    $("#embedding-env-status").textContent = config.has_env_embedding_key
      ? "检测到嵌入 API Key 环境变量，可留空。"
      : "未检测到嵌入 API Key 环境变量；需要在下方填写。";
  } catch {
    $("#chat-env-status").textContent = "无法读取本机聊天环境配置。";
    $("#embedding-env-status").textContent = "无法读取本机嵌入环境配置。";
  }
}

async function loadExample(silent = false) {
  try {
    const response = await fetch("/api/examples");
    if (!response.ok) throw new Error("示例题读取失败。");
    const payload = await response.json();
    const rows = Array.isArray(payload) ? payload : payload.questions || [];
    $("#seeds-input").value = rows
      .map((row) => (typeof row === "string" ? row : row.text))
      .filter(Boolean)
      .join("\n");
    updateSeedCount();
    if (!silent) toast("已载入演示种子题。");
  } catch (error) {
    if (!silent) toast(error.message);
  }
}

async function importJson(file) {
  try {
    const payload = JSON.parse(await file.text());
    const rows = Array.isArray(payload) ? payload : payload.questions || payload.seeds;
    if (!Array.isArray(rows)) throw new Error("JSON 中没有 questions 或 seeds 列表。");
    $("#seeds-input").value = rows
      .map((row) => (typeof row === "string" ? row : row.text))
      .filter(Boolean)
      .join("\n");
    updateSeedCount();
    toast(`已导入 ${rows.length} 道题。`);
  } catch (error) {
    toast(`导入失败：${error.message}`);
  } finally {
    $("#json-file").value = "";
  }
}

function buildPayload() {
  const lines = seedLines();
  if (lines.length < 3) throw new Error("请至少输入 3 道种子题。");
  if (state.provider === "openai" && !$("#chat-base-url").value.trim()) {
    throw new Error("真实 API 模式需要填写 Chat Base URL。");
  }
  const chatKey = $("#chat-api-key").value.trim();
  if (state.provider === "openai" && !chatKey && !state.hasChatEnvKey) {
    throw new Error("请填写 Chat API Key（DeepSeek Key），或先配置环境变量。");
  }
  if (state.provider === "openai" && !$("#chat-model").value.trim()) {
    throw new Error("真实 API 模式需要填写 Chat model。");
  }
  if (state.provider === "openai" && !$("#embedding-base-url").value.trim()) {
    throw new Error("真实 API 模式需要填写 Embedding Base URL。");
  }
  if (state.provider === "openai" && !$("#embedding-model").value.trim()) {
    throw new Error("真实 API 模式需要填写 Embedding model。");
  }
  const embeddingKey = $("#embedding-api-key").value.trim();
  const sameApiOrigin =
    $("#chat-base-url").value.trim().replace(/\/+$/, "") ===
    $("#embedding-base-url").value.trim().replace(/\/+$/, "");
  if (
    state.provider === "openai" &&
    !embeddingKey &&
    !state.hasEmbeddingEnvKey &&
    !(sameApiOrigin && (chatKey || state.hasChatEnvKey))
  ) {
    throw new Error("聊天与嵌入接口不同，请填写独立的 Embedding API Key。");
  }
  return {
    seeds: lines.map((text, index) => ({
      id: `q${String(index + 1).padStart(3, "0")}`,
      text,
    })),
    provider: state.provider,
    count: Number($("#count").value),
    components: Number($("#components").value),
    clusters: Number($("#clusters").value),
    activation_quantile: Number($("#activation").value) / 100,
    neighbor_fraction: Number($("#neighbor-fraction").value) / 100,
    temperature: Number($("#temperature").value),
    api: {
      chat_base_url: $("#chat-base-url").value,
      chat_api_key: $("#chat-api-key").value,
      chat_model: $("#chat-model").value,
      embedding_base_url: $("#embedding-base-url").value,
      embedding_api_key: $("#embedding-api-key").value,
      embedding_model: $("#embedding-model").value,
    },
  };
}

async function startRun() {
  if (state.jobId) return;
  let payload;
  try {
    payload = buildPayload();
  } catch (error) {
    toast(error.message);
    return;
  }
  $("#run-button").disabled = true;
  $("#run-button span:first-child").textContent = "正在启动...";
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "无法启动实验。");
    state.jobId = data.id;
    showRunning();
    pollJob();
  } catch (error) {
    resetRunButton();
    toast(error.message);
  }
}

function resetRunButton() {
  $("#run-button").disabled = false;
  $("#run-button span:first-child").textContent = "开始生成同类题";
}

function showRunning() {
  $("#empty-state").hidden = true;
  $("#results").hidden = true;
  $("#running-state").hidden = false;
  $("#progress-value").textContent = "0%";
  $("#progress-bar-fill").style.width = "0%";
  $("#progress-circle").style.strokeDashoffset = "327";
  $("#running-message").textContent = "任务已提交，正在连接...";
  $("#log-lines").innerHTML = "";
  renderTimeline([]);
}

function stageIndex(stage) {
  if (["segment"].includes(stage)) return 0;
  if (["spectral"].includes(stage)) return 1;
  if (["matrix_baseline"].includes(stage)) return 2;
  if (["template"].includes(stage)) return 3;
  if (stage.startsWith("generate")) return 4;
  if (stage === "evaluate") return 5;
  if (stage === "done") return 6;
  return -1;
}

function renderTimeline(logs) {
  const names = ["语段切分", "谱矩阵", "候选聚类", "槽位归纳", "条件生成", "评测"];
  const current = logs.length ? stageIndex(logs.at(-1).stage) : -1;
  $("#run-timeline").innerHTML = names
    .map((name, index) => {
      const cls =
        index < current ? "done" : index === current ? "active" : "";
      return `<div class="timeline-step ${cls}"><span>0${index + 1}</span><br>${name}</div>`;
    })
    .join("");
}

function renderProgress(job) {
  const value = Number(job.progress || 0);
  const circumference = 327;
  $("#progress-value").textContent = `${value}%`;
  $("#progress-bar-fill").style.width = `${value}%`;
  $("#progress-circle").style.strokeDashoffset =
    circumference - (circumference * value) / 100;
  $("#running-message").textContent = job.message || "处理中...";
  const logs = job.logs || [];
  renderTimeline(logs);
  $("#log-lines").innerHTML = logs
    .slice(-20)
    .map(
      (log) => `<div class="log-line">
        <span>${escapeHtml(log.time)}</span>
        <b>${escapeHtml(log.percent)}%</b>
        <span>${escapeHtml(log.detail)}</span>
      </div>`,
    )
    .join("");
  $("#log-lines").scrollTop = $("#log-lines").scrollHeight;
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const response = await fetch(`/api/runs/${state.jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "任务读取失败。");
    renderProgress(job);
    if (job.status === "complete") {
      state.result = job.result;
      window.clearTimeout(state.pollTimer);
      state.jobId = null;
      resetRunButton();
      renderResults(job);
      return;
    }
    if (job.status === "failed") {
      window.clearTimeout(state.pollTimer);
      state.jobId = null;
      resetRunButton();
      showFailure(job.error || "未知错误");
      return;
    }
    state.pollTimer = window.setTimeout(pollJob, 850);
  } catch (error) {
    state.pollTimer = window.setTimeout(pollJob, 1600);
    toast(error.message);
  }
}

function showFailure(error) {
  $("#running-state").hidden = true;
  $("#empty-state").hidden = true;
  $("#results").hidden = false;
  $(".results-head").hidden = true;
  $("#result-tabs").hidden = true;
  $("#tab-content").innerHTML = `<div class="error-state">
    <span>RUN FAILED</span>
    <h2>实验没有完成</h2>
    <p>${escapeHtml(error)}</p>
  </div>`;
}

function renderResults(job) {
  $("#running-state").hidden = true;
  $("#empty-state").hidden = true;
  $("#results").hidden = false;
  $(".results-head").hidden = false;
  $("#result-tabs").hidden = false;
  const result = state.result;
  const seedCount = seedLines().length;
  $("#result-meta").textContent =
    `${seedCount} 道种子题 · ${result.segments.length} 个语段 · ` +
    `${result.settings.count} 道/组 · ${result.provider === "mock" ? "离线演示" : "真实 API"}`;
  $("#download-report").href = `/api/runs/${job.id}/files/report.md`;
  $("#download-archive").href = `/api/runs/${job.id}/archive`;
  state.activeTab = "overview";
  setActiveTab("overview");
}

function setActiveTab(tab) {
  state.activeTab = tab;
  $$("#result-tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  renderTab();
}

function renderTab() {
  if (!state.result) return;
  const renderers = {
    overview: renderOverview,
    segments: renderSegments,
    spectral: renderSpectral,
    templates: renderTemplates,
    generations: renderGenerations,
    evaluation: renderEvaluation,
  };
  $("#tab-content").innerHTML = renderers[state.activeTab]();
  if (state.activeTab === "generations") {
    bindArmFilter();
  }
}

function metricBars(metric) {
  const aggregate = state.result.evaluation.aggregate;
  return ["baseline", "matrix", "spectral"]
    .map((arm) => {
      const value = Number(aggregate[arm]?.[metric] || 0);
      return `<div class="arm-row">
        <label>${ARM_LABELS[arm]}</label>
        <span class="mini-bar"><i style="width:${Math.max(2, value * 100)}%"></i></span>
        <strong>${percent(value)}</strong>
      </div>`;
    })
    .join("");
}

function renderOverview() {
  const result = state.result;
  const aggregate = result.evaluation.aggregate;
  const spectral = aggregate.spectral || {};
  const diagnostics = result.spectral.diagnostics;
  return `<div class="overview-grid">
    <article class="chart-card">
      <div class="card-kicker">
        <span>结构族命中率</span>
        <small>FAMILY MATCH, NOT VERBATIM</small>
      </div>
      ${metricBars("family_match_rate")}
      <div class="card-kicker" style="margin-top:28px">
        <span>连续结构相似度</span>
        <small>DEPENDENCY + OPERATIONS</small>
      </div>
      ${metricBars("structural_similarity")}
    </article>
    <div class="stat-stack">
      <article class="summary-card">
        <span>结构族命中率</span>
        <strong>${percent(spectral.family_match_rate ?? spectral.isomorphism_rate)}</strong>
        <small>${spectral.n || 0} 道生成题</small>
      </article>
      <article class="summary-card">
        <span>思维深度对齐</span>
        <strong>${percent(spectral.reasoning_depth_alignment)}</strong>
        <small>步骤、瓶颈与验证负担</small>
      </article>
      <article class="summary-card">
        <span>连续结构相似</span>
        <strong>${percent(spectral.structural_similarity)}</strong>
        <small>允许灵活表面对应</small>
      </article>
      <article class="summary-card">
        <span>种子思维深度</span>
        <strong>${result.reasoning_profile?.depth_level || "—"} / 5</strong>
        <small>${escapeHtml(result.reasoning_profile?.depth_label || "未标注")}</small>
      </article>
    </div>
    <article class="chart-card">
      <div class="card-kicker">
        <span>思维深度对齐</span>
        <small>DEPTH ALIGNMENT</small>
      </div>
      ${metricBars("reasoning_depth_alignment")}
      <div class="card-kicker" style="margin-top:28px">
        <span>认知操作覆盖</span>
        <small>OPERATION COVERAGE</small>
      </div>
      ${metricBars("cognitive_coverage")}
    </article>
    <article class="chart-card">
      <div class="card-kicker">
        <span>实验规模</span>
        <small>THIS RUN</small>
      </div>
      <div class="matrix-stats">
        <div class="matrix-stat"><span>种子语段</span><strong>${result.segments.length}</strong></div>
        <div class="matrix-stat"><span>特征向量</span><strong>${diagnostics.n_components}</strong></div>
        <div class="matrix-stat"><span>高激活语段</span><strong>${diagnostics.n_high_activation_segments}</strong></div>
        <div class="matrix-stat"><span>每组题目</span><strong>${result.settings.count}</strong></div>
      </div>
    </article>
    <article class="notice-card">
      <b>NOTE</b>
      <span>${escapeHtml(result.evaluation.warning)}</span>
    </article>
  </div>`;
}

function renderSegments() {
  const grouped = new Map();
  state.result.segments.forEach((segment) => {
    if (!grouped.has(segment.question_id)) grouped.set(segment.question_id, []);
    grouped.get(segment.question_id).push(segment);
  });
  return `<div class="section-list">${[...grouped.entries()]
    .map(
      ([questionId, segments]) => `<article class="list-card">
        <div class="list-card-head">
          <strong>${escapeHtml(questionId)}</strong>
          <span>${segments.length} SEGMENTS</span>
        </div>
        <div class="segment-flow">
          ${segments
            .map(
              (segment) => `<span class="segment-chip">
                <i>S${String(segment.order).padStart(2, "0")}</i>
                <span>
                  ${segment.role ? `<b>${escapeHtml(segment.role)}${segment.operation ? ` · ${escapeHtml(segment.operation)}` : ""}</b>` : ""}
                  ${escapeHtml(segment.text)}
                </span>
              </span>`,
            )
            .join("")}
        </div>
      </article>`,
    )
    .join("")}</div>`;
}

function renderSpectral() {
  const spectral = state.result.spectral;
  const diagnostics = spectral.diagnostics;
  const maxEigen = Math.max(
    ...diagnostics.top_by_component.map((row) => Math.abs(row.eigenvalue)),
    1e-9,
  );
  return `<div class="spectral-layout">
    <article class="chart-card">
      <div class="card-kicker">
        <span>领先特征谱</span>
        <small>SEMANTIC + SOFT ROLE / LOCAL NEIGHBORS</small>
      </div>
      <div class="eigen-list">
        ${diagnostics.top_by_component
          .map(
            (row, index) => `<div class="eigen-row">
              <span>v${index + 1}</span>
              <span class="mini-bar"><i style="width:${Math.max(
                2,
                (Math.abs(row.eigenvalue) / maxEigen) * 100,
              )}%"></i></span>
              <strong>${formatNumber(row.eigenvalue, 3)}</strong>
            </div>`,
          )
          .join("")}
      </div>
      <div class="matrix-stats">
        <div class="matrix-stat"><span>M 平均非对角</span><strong>${formatNumber(
          diagnostics.matrix_stats.M_mean_offdiag,
          4,
        )}</strong></div>
        <div class="matrix-stat"><span>N 平均非对角</span><strong>${formatNumber(
          diagnostics.matrix_stats.N_mean_offdiag,
          4,
        )}</strong></div>
        <div class="matrix-stat"><span>A 最小值</span><strong>${formatNumber(
          diagnostics.matrix_stats.A_min,
          3,
        )}</strong></div>
        <div class="matrix-stat"><span>A 最大值</span><strong>${formatNumber(
          diagnostics.matrix_stats.A_max,
          3,
        )}</strong></div>
      </div>
    </article>
    <div class="candidate-list">
      ${spectral.candidates
        .map(
          (candidate) => `<article class="candidate-card">
            <div class="candidate-head">
              <strong>${escapeHtml(candidate.id)}</strong>
              <span>order ${formatNumber(candidate.order_hint, 2)} · activation ${formatNumber(candidate.activation, 3)}</span>
            </div>
            <p>${escapeHtml(candidate.exemplars[0] || "无代表语段")}</p>
            <div class="exemplar">${candidate.segment_ids.length} 条证据语段：${candidate.segment_ids
              .map(escapeHtml)
              .join(" · ")}</div>
          </article>`,
        )
        .join("")}
    </div>
  </div>`;
}

function reasoningProfileCard(profile) {
  const depth = Number(profile?.depth_level || 0);
  const operations = profile?.operations || [];
  const chain = profile?.dependency_chain || [];
  return `<article class="profile-card">
    <div class="profile-head">
      <div>
        <span class="template-label">REASONING PROFILE</span>
        <h3>思维难度画像</h3>
        <p>${escapeHtml(profile?.bottleneck || "尚未归纳主要认知瓶颈")}</p>
      </div>
      <div class="depth-gauge">
        <span>DEPTH</span>
        <strong>${depth || "—"}<small>/5</small></strong>
        <div class="depth-dots">${[1, 2, 3, 4, 5]
          .map((level) => `<i class="${level <= depth ? "filled" : ""}"></i>`)
          .join("")}</div>
      </div>
    </div>
    <div class="profile-columns">
      <div>
        <span class="profile-label">COGNITIVE OPERATIONS</span>
        <div class="profile-chips">${operations.map((item) => `<span>${escapeHtml(item)}</span>`).join("") || "—"}</div>
      </div>
      <div>
        <span class="profile-label">DEPENDENCY CHAIN</span>
        <ol class="reasoning-chain">${chain.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>—</li>"}</ol>
      </div>
    </div>
    <div class="profile-invariants">
      <span>${escapeHtml(profile?.abstraction_level || "未标注抽象层级")}</span>
      ${(profile?.depth_invariants || []).map((item) => `<p>${escapeHtml(item)}</p>`).join("")}
    </div>
  </article>`;
}

function templateCard(template, primary = false) {
  return `<article class="template-card ${primary ? "primary" : ""}">
    <div class="template-head">
      <span class="template-label">${primary ? "SPECTRAL CONDITION" : "M-ONLY CONTROL"}</span>
      <h3>${escapeHtml(template.name)}</h3>
      <p>${escapeHtml(template.description)}</p>
    </div>
    <div class="slot-stack">
      ${template.slots
        .map(
          (slot, index) => `<div class="slot-row">
            <span class="slot-index">${String(index + 1).padStart(2, "0")}</span>
            <div class="slot-copy">
              <strong>${escapeHtml(slot.name)}</strong>
              <p>${escapeHtml(slot.definition)}</p>
            </div>
            <span class="required-badge">${slot.required ? "REQ" : "OPT"}</span>
          </div>`,
        )
        .join("")}
    </div>
    <div class="invariants">
      <span>STRUCTURAL INVARIANTS</span>
      <p>${template.invariants.map(escapeHtml).join("<br>") || "未归纳出显式不变量"}</p>
    </div>
  </article>`;
}

function renderTemplates() {
  const profile =
    state.result.reasoning_profile ||
    state.result.templates.spectral.reasoning_profile ||
    {};
  return `${reasoningProfileCard(profile)}
    <div class="template-grid" style="margin-top:18px">
      ${templateCard(state.result.templates.spectral, true)}
      ${templateCard(state.result.templates.matrix, false)}
    </div>`;
}

function renderQuestionCards(arm) {
  return (state.result.generations[arm] || [])
    .map(
      (question, index) => `<article class="question-card">
        <div class="question-meta">
          <span>#${String(index + 1).padStart(2, "0")}</span>
          <em>${escapeHtml(question.question_type || "unknown")}${question.depth_level ? ` · depth ${question.depth_level}/5` : ""}</em>
        </div>
        <p>${escapeHtml(question.text)}</p>
        ${
          (question.reasoning_outline || []).length
            ? `<ol class="reasoning-outline">${question.reasoning_outline
                .map((step) => `<li>${escapeHtml(step)}</li>`)
                .join("")}</ol>`
            : ""
        }
        ${
          Object.keys(question.slot_texts || {}).length
            ? `<div class="slot-text-list">${Object.entries(question.slot_texts)
                .map(
                  ([slot, text]) =>
                    `<span>${escapeHtml(slot)} · ${escapeHtml(text)}</span>`,
                )
                .join("")}</div>`
            : ""
        }
      </article>`,
    )
    .join("");
}

function renderGenerations() {
  return `<div class="arm-filter" id="arm-filter">
      ${["baseline", "matrix", "spectral"]
        .map(
          (arm) =>
            `<button class="${state.activeArm === arm ? "active" : ""}" type="button" data-arm="${arm}">${ARM_LABELS[arm]}</button>`,
        )
        .join("")}
    </div>
    <div class="question-grid" id="question-grid">${renderQuestionCards(state.activeArm)}</div>`;
}

function bindArmFilter() {
  $$("#arm-filter button").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeArm = button.dataset.arm;
      renderTab();
    });
  });
}

function renderEvaluation() {
  const details = state.result.evaluation.details;
  return `<div class="table-scroll">
    <table class="evaluation-table">
      <thead><tr>
        <th>题目</th><th>条件</th><th>结构族</th><th>结构相似</th><th>深度对齐</th><th>槽位覆盖</th><th>判定依据</th>
      </tr></thead>
      <tbody>
        ${details
          .map(
            (row) => `<tr>
              <td>${escapeHtml(row.id)}</td>
              <td>${ARM_LABELS[row.arm] || escapeHtml(row.arm)}</td>
              <td><span class="${row.same_family ? "pass" : "fail"}">${row.same_family ? "SAME" : "DRIFT"}</span></td>
              <td>${percent(row.structural_similarity)}</td>
              <td>${percent(row.depth_alignment)}</td>
              <td>${percent(row.coverage)}</td>
              <td>${escapeHtml(row.judge_reason)}</td>
            </tr>`,
          )
          .join("")}
      </tbody>
    </table>
  </div>
  <div class="evaluation-note">${escapeHtml(state.result.evaluation.warning)}</div>`;
}

function bindEvents() {
  $("#seeds-input").addEventListener("input", updateSeedCount);
  $("#load-example").addEventListener("click", () => loadExample(false));
  $("#clear-seeds").addEventListener("click", () => {
    $("#seeds-input").value = "";
    updateSeedCount();
    $("#seeds-input").focus();
  });
  $("#import-json").addEventListener("click", () => $("#json-file").click());
  $("#json-file").addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) importJson(file);
  });
  $$("#provider-switch button").forEach((button) => {
    button.addEventListener("click", () => {
      state.provider = button.dataset.provider;
      $$("#provider-switch button").forEach((item) =>
        item.classList.toggle("active", item === button),
      );
      $("#api-fields").hidden = state.provider !== "openai";
    });
  });
  $("#deepseek-preset").addEventListener("click", () => {
    $("#chat-base-url").value = "https://api.deepseek.com";
    $("#chat-model").value = "deepseek-flash";
    toast("已填入 DeepSeek Base URL 与聊天模型。");
  });
  $$(".range-field input").forEach((input) => {
    syncRange(input);
    input.addEventListener("input", () => syncRange(input));
  });
  $("#run-button").addEventListener("click", startRun);
  $("#result-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-tab]");
    if (button) setActiveTab(button.dataset.tab);
  });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      startRun();
    }
  });
}

async function init() {
  bindEvents();
  updateSeedCount();
  loadConfig();
  await loadExample(true);
}

init();
