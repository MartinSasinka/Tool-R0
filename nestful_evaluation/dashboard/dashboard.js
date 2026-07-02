/* global Plotly */
let DATA = null;
let filteredTasks = [];
let sortState = { col: "pass_rate", asc: true };
let selectedTaskId = null;

const MODE_COLORS = {
  TOOL_SCHEMA_ERROR: "#a855f7",
  TOOL_SELECTION_ERROR: "#c084fc",
  TOOL_ARGUMENT_ERROR: "#f97316",
  RUNTIME_TYPE_ERROR: "#06b6d4",
  FINAL_ANSWER_ERROR: "#ef4444",
  STRUCTURED_OUTPUT_ERROR: "#3b82f6",
  CONTROL_FLOW_ERROR: "#eab308",
  REASONING_PLANNING_ERROR: "#f59e0b",
  UNKNOWN_ERROR: "#64748b",
};

const BUCKET_CLASS = {
  easy: "green", mostly_pass: "green", unstable: "orange",
  mostly_fail: "red", hard_fail: "red",
};

async function init() {
  setupNav();
  if (window.DASHBOARD_DATA) {
    DATA = window.DASHBOARD_DATA;
    onDataLoaded();
    return;
  }
  try {
    const resp = await fetch("dashboard_data.json");
    if (!resp.ok) throw new Error(resp.statusText);
    DATA = await resp.json();
    onDataLoaded();
  } catch (e) {
    document.getElementById("load-banner").classList.remove("hidden");
    document.getElementById("load-status").textContent =
      "Could not auto-load dashboard_data.json (" + e.message + "). Use file picker or run: python -m http.server";
  }
  document.getElementById("file-input").addEventListener("change", loadFile);
}

function loadFile(ev) {
  const f = ev.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    DATA = JSON.parse(reader.result);
    document.getElementById("load-banner").classList.add("hidden");
    onDataLoaded();
  };
  reader.readAsText(f);
}

function onDataLoaded() {
  filteredTasks = [...DATA.tasks];
  renderOverview();
  renderDifficulty();
  renderDomains();
  renderFailureModes();
  renderHeatmap();
  renderToolAnalysis();
  renderGoldTypes();
  renderExplorer();
  renderSubsets();
  document.getElementById("generated-at").textContent =
    "Generated: " + (DATA.overview.generated_at || "");
}

function setupNav() {
  document.querySelectorAll(".topnav a[data-section]").forEach(a => {
    a.addEventListener("click", ev => {
      ev.preventDefault();
      document.querySelector(a.getAttribute("href"))?.scrollIntoView({ behavior: "smooth" });
    });
  });
}

function renderOverview() {
  const o = DATA.overview;
  const cards = [
    ["Total tasks", o.total_tasks, "blue"],
    ["Total rollouts", o.total_rollouts, "gray"],
    ["Pass rate", o.overall_pass_rate_percent + "%", o.overall_pass_rate >= 0.7 ? "green" : o.overall_pass_rate >= 0.4 ? "orange" : "red"],
    ["Hard-fail tasks", o.hard_fail_tasks, "red"],
    ["Mostly-fail", o.mostly_fail_tasks, "red"],
    ["Unstable", o.unstable_tasks, "orange"],
    ["Easy tasks", o.easy_tasks, "green"],
    ["Top failure mode", o.most_common_failure_mode, "purple"],
    ["Hardest domain", o.most_problematic_domain, "orange"],
    ["Malformed calls", o.malformed_tool_call_percent + "%", "purple"],
  ];
  document.getElementById("overview-cards").innerHTML = cards.map(([l, v, c]) =>
    `<div class="card ${c}"><div class="label">${l}</div><div class="value">${esc(v)}</div></div>`
  ).join("");
}

function renderDifficulty() {
  const buckets = DATA.difficulty_bucket_summary;
  Plotly.newPlot("chart-difficulty", [{
    type: "bar",
    x: buckets.map(b => b.bucket),
    y: buckets.map(b => b.task_count),
    marker: { color: ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"] },
  }], {
    paper_bgcolor: "#1e293b", plot_bgcolor: "#1e293b",
    font: { color: "#e2e8f0" }, margin: { t: 30, b: 60 },
    title: { text: "Tasks by Difficulty Bucket", font: { size: 14 } },
  }, { responsive: true, displayModeBar: false });

  const hist = DATA.pass_count_histogram;
  Plotly.newPlot("chart-pass-hist", [{
    type: "bar",
    x: hist.map(h => h.pass_count),
    y: hist.map(h => h.task_count),
    marker: { color: "#3b82f6" },
  }], {
    paper_bgcolor: "#1e293b", plot_bgcolor: "#1e293b",
    font: { color: "#e2e8f0" }, margin: { t: 30 },
    xaxis: { title: "Passed rollouts per task" },
    yaxis: { title: "Number of tasks" },
    title: { text: "Pass Count Histogram", font: { size: 14 } },
  }, { responsive: true, displayModeBar: false });
}

function renderDomains() {
  const rows = [...DATA.problem_domain_summary].sort((a, b) => a.pass_rate - b.pass_rate);
  document.getElementById("domain-table").innerHTML = tableHtml(
    ["problem_domain", "task_count", "rollout_count", "pass_rate", "fail_rate", "most_common_failure_modes"],
    rows.map(r => ({
      ...r,
      pass_rate: pct(r.pass_rate),
      fail_rate: pct(r.fail_rate),
      most_common_failure_modes: (r.most_common_failure_modes || []).join(", "),
    }))
  );
  Plotly.newPlot("chart-domains", [{
    type: "bar", orientation: "h",
    y: rows.map(r => r.problem_domain),
    x: rows.map(r => 100 * r.pass_rate),
    marker: { color: rows.map(r => r.pass_rate < 0.4 ? "#ef4444" : r.pass_rate < 0.6 ? "#f97316" : "#22c55e") },
  }], {
    paper_bgcolor: "#1e293b", plot_bgcolor: "#1e293b",
    font: { color: "#e2e8f0" }, margin: { l: 160, t: 30 },
    xaxis: { title: "Pass rate (%)" },
    title: { text: "Pass Rate by Problem Domain", font: { size: 14 } },
  }, { responsive: true, displayModeBar: false });
}

function renderFailureModes() {
  const rows = DATA.failure_mode_summary;
  document.getElementById("failure-table").innerHTML = rows.map(r => `
    <tr>
      <td><span class="badge purple">${r.failure_mode}</span></td>
      <td>${r.failed_rollout_count}</td>
      <td>${r.affected_task_count}</td>
      <td>${r.percentage_of_failed_rollouts}%</td>
      <td>${pct(r.average_task_pass_rate)}</td>
      <td style="max-width:320px;font-size:12px;color:#94a3b8">${esc(r.interpretation)}</td>
      <td style="max-width:220px;font-size:12px">${esc(r.recommended_training_strategy)}</td>
    </tr>`).join("");

  Plotly.newPlot("chart-failure-modes", [{
    type: "bar", orientation: "h",
    y: rows.map(r => r.failure_mode),
    x: rows.map(r => r.failed_rollout_count),
    marker: { color: rows.map(r => MODE_COLORS[r.failure_mode] || "#64748b") },
  }], {
    paper_bgcolor: "#1e293b", plot_bgcolor: "#1e293b",
    font: { color: "#e2e8f0" }, margin: { l: 180, t: 30 },
    title: { text: "Failed Rollouts by Failure Mode", font: { size: 14 } },
  }, { responsive: true, displayModeBar: false });
}

function renderHeatmap() {
  const heat = DATA.domain_failure_heatmap;
  const domains = [...new Set(heat.map(h => h.problem_domain))].sort();
  const modes = Object.keys(MODE_COLORS);
  const z = domains.map(d => modes.map(m => {
    const hit = heat.find(h => h.problem_domain === d && h.failure_mode === m);
    return hit ? hit.failed_rollout_count : 0;
  }));
  Plotly.newPlot("chart-heatmap", [{
    type: "heatmap", x: modes, y: domains, z,
    colorscale: [[0, "#1e293b"], [0.5, "#f97316"], [1, "#ef4444"]],
  }], {
    paper_bgcolor: "#1e293b", plot_bgcolor: "#1e293b",
    font: { color: "#e2e8f0", size: 10 },
    margin: { l: 140, b: 120, t: 30 },
    xaxis: { tickangle: -45 },
    title: { text: "Domain × Failure Mode", font: { size: 14 } },
  }, { responsive: true, displayModeBar: false });
}

function renderToolAnalysis() {
  const te = DATA.tool_error_summary;
  document.getElementById("tool-stats").innerHTML = `
    <div class="cards">
      <div class="card blue"><div class="label">Avg tools (passed)</div><div class="value">${te.avg_tool_calls_passed}</div></div>
      <div class="card red"><div class="label">Avg tools (failed)</div><div class="value">${te.avg_tool_calls_failed}</div></div>
    </div>`;
  const mal = Object.entries(te.malformed_by_tool || {}).slice(0, 20);
  const fail = Object.entries(te.failed_rollout_by_tool || {}).slice(0, 20);
  if (mal.length) {
    Plotly.newPlot("chart-malformed-tools", [{
      type: "bar", orientation: "h",
      y: mal.map(([k]) => k).reverse(),
      x: mal.map(([, v]) => v).reverse(),
      marker: { color: "#ec4899" },
    }], {
      paper_bgcolor: "#1e293b", plot_bgcolor: "#1e293b",
      font: { color: "#e2e8f0" }, margin: { l: 120, t: 30 },
      title: { text: "Malformed Tool Calls by Tool", font: { size: 14 } },
    }, { responsive: true, displayModeBar: false });
  }
  if (fail.length) {
    Plotly.newPlot("chart-failed-tools", [{
      type: "bar", orientation: "h",
      y: fail.map(([k]) => k).reverse(),
      x: fail.map(([, v]) => v).reverse(),
      marker: { color: "#f97316" },
    }], {
      paper_bgcolor: "#1e293b", plot_bgcolor: "#1e293b",
      font: { color: "#e2e8f0" }, margin: { l: 120, t: 30 },
      title: { text: "Failed Rollouts by Tool Used", font: { size: 14 } },
    }, { responsive: true, displayModeBar: false });
  }
  const examples = te.malformed_examples || [];
  const exEl = document.getElementById("malformed-examples");
  if (exEl && examples.length) {
    exEl.innerHTML = "<h4 style=\"margin:12px 0 8px\">Representative malformed tool calls</h4>" +
      tableHtml(
        ["tool_name", "error_message", "task_id", "rollout_id"],
        examples
      );
  }
}

function renderGoldTypes() {
  const rows = DATA.gold_answer_type_summary;
  document.getElementById("gold-table").innerHTML = tableHtml(
    ["gold_answer_type", "rollout_count", "pass_rate"],
    rows.map(r => ({ ...r, pass_rate: pct(r.pass_rate) }))
  );
  Plotly.newPlot("chart-gold-types", [{
    type: "bar",
    x: rows.map(r => r.gold_answer_type),
    y: rows.map(r => 100 * r.pass_rate),
    marker: { color: "#06b6d4" },
  }], {
    paper_bgcolor: "#1e293b", plot_bgcolor: "#1e293b",
    font: { color: "#e2e8f0" }, margin: { t: 30 },
    yaxis: { title: "Pass rate (%)" },
    title: { text: "Pass Rate by Gold Answer Type", font: { size: 14 } },
  }, { responsive: true, displayModeBar: false });
}

function renderExplorer() {
  populateFilterOptions();
  applyFilters();
  document.getElementById("filter-search").addEventListener("input", applyFilters);
  ["filter-bucket", "filter-mode", "filter-domain", "filter-gtype", "filter-malformed"].forEach(id => {
    document.getElementById(id).addEventListener("change", applyFilters);
  });
  document.getElementById("filter-pass-min").addEventListener("change", applyFilters);
  document.getElementById("filter-pass-max").addEventListener("change", applyFilters);
}

function populateFilterOptions() {
  const modes = new Set(), domains = new Set(), gtypes = new Set(), buckets = new Set();
  DATA.tasks.forEach(t => {
    buckets.add(t.difficulty_bucket);
    gtypes.add(t.gold_answer_type);
    t.problem_domains.forEach(d => domains.add(d));
    t.all_failure_modes.forEach(m => modes.add(m));
  });
  fillSelect("filter-bucket", [...buckets].sort());
  fillSelect("filter-mode", [...modes].sort());
  fillSelect("filter-domain", [...domains].sort());
  fillSelect("filter-gtype", [...gtypes].sort());
}

function fillSelect(id, opts) {
  const el = document.getElementById(id);
  el.innerHTML = '<option value="">All</option>' +
    opts.map(o => `<option value="${esc(o)}">${esc(o)}</option>`).join("");
}

function applyFilters() {
  const q = document.getElementById("filter-search").value.toLowerCase();
  const bucket = document.getElementById("filter-bucket").value;
  const mode = document.getElementById("filter-mode").value;
  const domain = document.getElementById("filter-domain").value;
  const gtype = document.getElementById("filter-gtype").value;
  const mal = document.getElementById("filter-malformed").value;
  const pmin = parseInt(document.getElementById("filter-pass-min").value, 10);
  const pmax = parseInt(document.getElementById("filter-pass-max").value, 10);

  filteredTasks = DATA.tasks.filter(t => {
    if (bucket && t.difficulty_bucket !== bucket) return false;
    if (mode && !t.all_failure_modes.includes(mode)) return false;
    if (domain && !t.problem_domains.includes(domain)) return false;
    if (gtype && t.gold_answer_type !== gtype) return false;
    if (mal === "yes" && t.malformed_tool_call_count === 0) return false;
    if (mal === "no" && t.malformed_tool_call_count > 0) return false;
    if (!isNaN(pmin) && t.pass_count < pmin) return false;
    if (!isNaN(pmax) && t.pass_count > pmax) return false;
    if (q && !(t.task_prompt || "").toLowerCase().includes(q) &&
        !(t.task_id || "").includes(q)) return false;
    return true;
  });
  sortTasks();
  renderTaskTable();
  document.getElementById("explorer-count").textContent =
    filteredTasks.length + " / " + DATA.tasks.length + " tasks";
}

function sortTasks() {
  const col = sortState.col;
  const dir = sortState.asc ? 1 : -1;
  filteredTasks.sort((a, b) => {
    let va = a[col], vb = b[col];
    if (typeof va === "string") return va.localeCompare(vb) * dir;
    return ((va ?? 0) - (vb ?? 0)) * dir;
  });
}

function renderTaskTable() {
  const thead = document.getElementById("explorer-thead");
  const cols = [
    ["task_id", "Task ID"], ["pass_count", "Pass"], ["pass_rate", "Rate"],
    ["difficulty_bucket", "Bucket"], ["dominant_failure_mode", "Failure"],
    ["problem_domains", "Domains"], ["gold_answer_type", "Gold type"],
    ["task_prompt_short", "Prompt"],
  ];
  thead.innerHTML = "<tr>" + cols.map(([k, l]) =>
    `<th data-col="${k}">${l}${sortState.col === k ? (sortState.asc ? " ▲" : " ▼") : ""}</th>`
  ).join("") + "</tr>";
  thead.querySelectorAll("th").forEach(th => {
    th.onclick = () => {
      const c = th.dataset.col;
      if (sortState.col === c) sortState.asc = !sortState.asc;
      else { sortState.col = c; sortState.asc = true; }
      sortTasks();
      renderTaskTable();
    };
  });

  const tbody = document.getElementById("explorer-tbody");
  tbody.innerHTML = filteredTasks.slice(0, 500).map(t => `
    <tr data-id="${esc(t.task_id)}" class="${t.task_id === selectedTaskId ? "selected" : ""}">
      <td style="font-family:monospace;font-size:11px">${esc(t.task_id.slice(0, 8))}…</td>
      <td>${t.pass_count}/${t.num_rollouts}</td>
      <td>${pct(t.pass_rate)}</td>
      <td><span class="badge ${BUCKET_CLASS[t.difficulty_bucket] || "gray"}">${t.difficulty_bucket}</span></td>
      <td>${t.dominant_failure_mode ? `<span class="badge purple">${t.dominant_failure_mode}</span>` : "—"}</td>
      <td>${(t.problem_domains || []).map(d => `<span class="badge blue">${d}</span>`).join("")}</td>
      <td>${t.gold_answer_type}</td>
      <td style="max-width:280px;font-size:12px">${esc(t.task_prompt_short)}</td>
    </tr>`).join("");

  tbody.querySelectorAll("tr").forEach(tr => {
    tr.onclick = () => showTaskDetail(tr.dataset.id);
  });
  if (filteredTasks.length > 500) {
    document.getElementById("explorer-truncated").textContent =
      "Showing first 500 tasks — narrow filters to see more.";
  } else {
    document.getElementById("explorer-truncated").textContent = "";
  }
}

function showTaskDetail(taskId) {
  selectedTaskId = taskId;
  renderTaskTable();
  const t = DATA.tasks.find(x => x.task_id === taskId);
  if (!t) return;
  const rollouts = DATA.rollouts.filter(r => r.task_id === taskId);
  const panel = document.getElementById("task-detail");
  panel.classList.add("show");
  panel.innerHTML = `
    <h3>Task ${esc(taskId)}</h3>
    <p><span class="badge ${BUCKET_CLASS[t.difficulty_bucket]}">${t.difficulty_bucket}</span>
       pass ${t.pass_count}/${t.num_rollouts} (${pct(t.pass_rate)})</p>
    <p style="color:#94a3b8;font-size:13px">${esc(t.explanation)}</p>
    <h4>Full prompt</h4><pre>${esc(t.task_prompt)}</pre>
    <h4>Gold answer</h4><pre>${esc(JSON.stringify(t.gold_answer, null, 2))}</pre>
    <h4>Rollouts (${rollouts.length})</h4>
    ${rollouts.map(r => `
      <div class="rollout-card ${r.passed ? "pass" : "fail"}">
        <strong>Rollout ${r.rollout_id}</strong> — ${r.passed ? "PASS" : "FAIL"}
        ${r.dominant_failure_mode ? `<span class="badge purple">${r.dominant_failure_mode}</span>` : ""}
        <div style="font-size:12px;margin-top:6px">
          Predicted: <code>${esc(JSON.stringify(r.predicted_answer))}</code><br>
          Tools: ${(r.tool_names_used || []).join(", ") || "none"}<br>
          Error: ${esc(r.error_message || "—")}<br>
          Modes: ${(r.detected_failure_modes || []).map(m => `<span class="badge purple">${m}</span>`).join(" ")}
        </div>
        <pre style="margin-top:8px;max-height:120px">${esc(r.trace_snippet || "")}</pre>
      </div>`).join("")}
    <button class="btn secondary" onclick="document.getElementById('task-detail').classList.remove('show')">Close</button>`;
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderSubsets() {
  const grid = document.getElementById("subset-grid");
  grid.innerHTML = (DATA.training_subset_recommendations || []).map(s => `
    <div class="subset-card">
      <h4>${esc(s.subset_name)}</h4>
      <p><strong>${s.task_count}</strong> tasks, <strong>${s.rollout_count}</strong> rollouts</p>
      <p>Avg pass: ${pct(s.average_pass_rate)}</p>
      <p style="font-size:12px;color:#94a3b8">${esc(s.recommended_use)}</p>
      <p style="font-size:11px">Domains: ${(s.top_problem_domains || []).join(", ")}</p>
      <p style="font-size:11px">Modes: ${(s.top_failure_modes || []).join(", ")}</p>
      <button class="btn small" onclick="exportSubset('${s.subset_name}')">Export JSONL subset</button>
    </div>`).join("");
}

function exportSubset(name) {
  const sub = DATA.training_subset_recommendations.find(s => s.subset_name === name);
  if (!sub) return;
  const tasks = DATA.tasks.filter(t => sub.task_ids.includes(t.task_id));
  downloadJson(tasks, name + ".json");
}

function exportVisibleTasks(fmt) {
  const rows = filteredTasks.length ? filteredTasks : DATA.tasks;
  if (fmt === "csv") downloadCsv(rows, "visible_tasks.csv");
  else downloadJson(rows, "visible_tasks.json");
}

function exportPreset(preset) {
  let rows;
  switch (preset) {
    case "hard": rows = DATA.tasks.filter(t => t.difficulty_bucket === "hard_fail"); break;
    case "unstable": rows = DATA.tasks.filter(t => t.difficulty_bucket === "unstable"); break;
    case "schema": rows = DATA.tasks.filter(t => t.all_failure_modes.includes("TOOL_SCHEMA_ERROR")); break;
    case "typed": rows = DATA.tasks.filter(t =>
      t.all_failure_modes.includes("RUNTIME_TYPE_ERROR") ||
      t.all_failure_modes.includes("STRUCTURED_OUTPUT_ERROR")); break;
    case "reasoning": rows = DATA.tasks.filter(t =>
      t.all_failure_modes.includes("FINAL_ANSWER_ERROR") ||
      t.all_failure_modes.includes("REASONING_PLANNING_ERROR")); break;
    default: return;
  }
  downloadJson(rows, preset + "_tasks.json");
}

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  triggerDownload(blob, filename);
}

function downloadCsv(rows, filename) {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const lines = [keys.join(",")];
  rows.forEach(r => {
    lines.push(keys.map(k => {
      let v = r[k];
      if (typeof v === "object") v = JSON.stringify(v);
      v = String(v ?? "").replace(/"/g, '""');
      return `"${v}"`;
    }).join(","));
  });
  triggerDownload(new Blob([lines.join("\n")], { type: "text/csv" }), filename);
}

function triggerDownload(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function tableHtml(cols, rows) {
  return `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
    rows.map(r => `<tr>${cols.map(c => `<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("") +
    "</tbody></table>";
}

function pct(v) { return (100 * (v || 0)).toFixed(1) + "%"; }
function esc(s) {
  if (s == null) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

document.addEventListener("DOMContentLoaded", init);
