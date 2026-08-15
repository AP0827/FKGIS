"use strict";

/* ------------------------------------------------------------------ */
/* FKGIS frontend controller                                           */
/* ------------------------------------------------------------------ */

const state = {
  cases: [],
  currentCaseId: null,
  graphPayload: null,
  network: null,
  allVisNodes: [],
  allVisEdges: [],
  variant: "raw",
  nodeTypes: new Set(["Entity", "Event", "Location", "Time"]),
  significantOnly: false,
  edgeLabels: true,
};

const $ = (sel) => document.querySelector(sel);
const el = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
async function api(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res;
}

function setStatus(status) {
  const elt = el("case-status");
  elt.textContent = status || "idle";
  elt.className = "status " + (status || "idle");
}

function notify(message, isError = false) {
  const elt = el("run-progress");
  elt.hidden = false;
  elt.textContent = message;
  elt.style.color = isError ? "var(--danger)" : "";
}

/* ------------------------------------------------------------------ */
/* Health + cases                                                      */
/* ------------------------------------------------------------------ */
async function loadHealth() {
  try {
    const h = await api("/api/health");
    const llm = h.llm_configured ? `${h.llm_model}` : "no LLM key configured";
    el("health").textContent = `LLM: ${llm} • Sessions: ${h.sessions_dir}`;
  } catch (_) { /* ignore */ }
}

async function loadCases() {
  state.cases = await api("/api/cases");
  const list = el("case-list");
  list.innerHTML = "";
  el("case-list-empty").hidden = state.cases.length > 0;
  for (const c of state.cases) {
    const li = document.createElement("li");
    li.textContent = c.name;
    li.dataset.caseId = c.id;
    if (c.id === state.currentCaseId) li.classList.add("active");
    li.addEventListener("click", () => selectCase(c.id));
    list.appendChild(li);
  }
}

async function selectCase(caseId) {
  state.currentCaseId = caseId;
  document.querySelectorAll("#case-list li").forEach((li) => {
    li.classList.toggle("active", li.dataset.caseId === caseId);
  });
  const c = await api(`/api/cases/${caseId}`);
  el("case-panel").hidden = false;
  el("case-title").textContent = c.name;
  el("case-id").textContent = c.id;
  setStatus(c.status);
  el("run-pipeline").disabled = c.documents.length === 0;
  renderDocuments(c.documents);
  el("graph-panel").hidden = false;
  el("welcome").hidden = true;
  await refreshResults();
  await loadGraph();
}

function renderDocuments(docs) {
  const list = el("doc-list");
  list.innerHTML = "";
  for (const d of docs) {
    const li = document.createElement("li");
    const name = document.createElement("div");
    name.className = "doc-name";
    const nm = document.createElement("span");
    nm.textContent = d.name;
    const tp = document.createElement("span");
    tp.className = "type";
    tp.textContent = d.doc_type;
    name.append(nm, tp);
    const rm = document.createElement("button");
    rm.className = "remove-btn";
    rm.textContent = "×";
    rm.title = "Remove document";
    rm.addEventListener("click", async () => {
      await api(`/api/cases/${state.currentCaseId}/documents/${encodeURIComponent(d.name)}`, { method: "DELETE" });
      await loadCases();
      await selectCase(state.currentCaseId);
    });
    li.append(name, rm);
    list.appendChild(li);
  }
}

async function refreshResults() {
  if (!state.currentCaseId) return;
  const data = await api(`/api/cases/${state.currentCaseId}/results`);
  el("results-panel").hidden = false;
  el("results-content").innerHTML = renderResults(data);
}

function fmtTime(seconds) {
  if (seconds == null) return "—";
  return `${seconds.toFixed(2)}s`;
}

function renderResults(data) {
  const timing = data.timing || {};
  const verify = data.outputs?.case_output?.verification?.checks || null;
  let html = "";

  if (Object.keys(timing).length) {
    html += "<h4>Pipeline Timing</h4><table><tbody>" +
      Object.entries(timing)
        .map(([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td><td>${fmtTime(v)}</td></tr>`)
        .join("") +
      "</tbody></table>";
  }

  if (verify) {
    html += "<h4>Verification</h4><table><tbody>" +
      Object.entries(verify)
        .filter(([k]) => k !== "issues")
        .map(([k, v]) => `<tr><td>${k.replace(/_/g, " ")}</td><td>${String(v)}</td></tr>`)
        .join("") +
      "</tbody></table>";
  }

  const stats = data.graph_stats || {};
  if (Object.keys(stats).length) {
    html += "<h4>Graph Statistics</h4>";
    for (const [artifact, s] of Object.entries(stats)) {
      html += `<h5 style="margin:10px 0 6px;color:var(--muted)">${artifact}</h5>`;
      html += "<table><tbody>" +
        [
          ["Nodes", s.nodes],
          ["Edges", s.edges],
          ["Density", s.density?.toFixed(5)],
          ["Connected components", s.components],
          ["Largest component", s.largest_component],
          ["Average degree", s.avg_degree?.toFixed(3)],
          ["Average clustering", s.avg_clustering?.toFixed(4)],
          ["Nullish entities", s.nullish_entities],
          ["Unique relation types", s.unique_relation_types],
        ].map(([k, v]) => `<tr><td>${k}</td><td>${v ?? "—"}</td></tr>`).join("") +
        "</tbody></table>";
    }
  }
  if (!html) html = '<p class="muted">Run the pipeline to see results.</p>';
  return html;
}

/* ------------------------------------------------------------------ */
/* Graph                                                               */
/* ------------------------------------------------------------------ */
async function loadGraph() {
  if (!state.currentCaseId) return;
  const payload = await api(`/api/cases/${state.currentCaseId}/graph?variant=${state.variant}`);
  state.graphPayload = payload;
  state.allVisNodes = payload.nodes;
  state.allVisEdges = payload.edges;

  // Node type filters
  const types = [...new Set(payload.nodes.map((n) => n.group))];
  const container = el("node-type-filters");
  container.innerHTML = '<label>Node types</label>';
  const row = document.createElement("div");
  row.className = "segmented";
  for (const t of types) {
    const btn = document.createElement("button");
    btn.textContent = t;
    btn.dataset.type = t;
    if (state.nodeTypes.has(t)) btn.classList.add("active");
    btn.addEventListener("click", () => {
      const on = btn.classList.toggle("active");
      if (on) state.nodeTypes.add(t);
      else state.nodeTypes.delete(t);
      rebuildNetwork();
    });
    row.appendChild(btn);
  }
  container.appendChild(row);

  renderStats(payload);
  rebuildNetwork();
  updateExportLinks();
}

function renderStats(payload) {
  const st = payload.stats || {};
  const pr = payload.pagerank || {};
  const typeCounts = (st.node_types || {});
  const top = (st.pagerank_top || []).slice(0, 5).map(([score, label]) => label).join(", ");
  el("stats-bar").innerHTML = `
    <span><b>${st.nodes ?? 0}</b> nodes</span>
    <span><b>${st.edges ?? 0}</b> edges</span>
    <span><b>${pr.significant_nodes_count ?? 0}</b> significant (PageRank)</span>
    <span>types: ${Object.entries(typeCounts).map(([t, n]) => `${t} ${n}`).join(", ")}</span>
    ${top ? `<span>top: ${top}</span>` : ""}`;
}

function rebuildNetwork() {
  if (!state.currentCaseId) return;
  const nodes = state.allVisNodes.filter(
    (n) => state.nodeTypes.has(n.group) && (!state.significantOnly || n.significant)
  );
  const ids = new Set(nodes.map((n) => n.id));
  const edges = state.allVisEdges.filter((e) => ids.has(e.from) && ids.has(e.to));

  const elt = el("graph-container");
  elt.style.display = "block";
  el("graph-empty").hidden = nodes.length > 0;

  const groupStyles = {
    Entity: { color: { background: "#4C72B0", border: "#2c4a7c" } },
    Event: { color: { background: "#DD8452", border: "#a85c2c" } },
    Location: { color: { background: "#55A868", border: "#357a46" } },
    Time: { color: { background: "#C44E52", border: "#8f2f34" } },
  };

  const options = {
    groups: groupStyles,
    nodes: {
      shape: "dot",
      font: { color: "#e6ebf5", size: 12, strokeWidth: 3, strokeColor: "#0b1019" },
      borderWidth: 1,
      shadow: { enabled: true, size: 6 },
    },
    edges: {
      arrows: { to: { enabled: false } },
      color: { color: "#93a0b8", opacity: 0.5 },
      width: 1,
      smooth: { type: "continuous", roundness: 0.35 },
      font: { color: "#93a0b8", size: 9, strokeWidth: 2, strokeColor: "#0b1019", background: "#161d2e" },
    },
    interaction: { hover: true, tooltipDelay: 150, dragView: true, zoomView: true },
    layout: {
      improvedLayout: false,
    },
    physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      stabilization: { iterations: 250 },
      forceAtlas2Based: { gravitationalConstant: -60, centralGravity: 0.01, springLength: 110, damping: 0.6 },
      minVelocity: 0.5,
    },
  };

  if (!state.edgeLabels) {
    edges.forEach((e) => (e.label = ""));
  }

  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  if (state.network) {
    state.network.destroy();
  }
  state.network = new vis.Network(elt, data, options);
}

function updateExportLinks() {
  if (!state.currentCaseId) return;
  const base = `/api/cases/${state.currentCaseId}/graph.png?variant=${state.variant}`;
  el("export-png").href = `${base}&fmt=png`;
  el("export-svg").href = `${base}&fmt=svg`;
  el("export-pdf").href = `${base}&fmt=pdf`;
  el("export-json").href = `/api/cases/${state.currentCaseId}/graph.json?variant=${state.variant}`;
}

function focusEntity(query) {
  if (!state.network || !query.trim()) return;
  const needle = query.trim().toLowerCase();
  const node = state.allVisNodes.find(
    (n) => n.label.toLowerCase().includes(needle) || String(n.id).toLowerCase().includes(needle)
  );
  if (!node) {
    notify(`No entity matching "${query.trim()}"`, true);
    return;
  }
  state.network.selectNodes([node.id], true);
  state.network.focus(node.id, { scale: 2.2, animation: { duration: 400, easingFunction: "easeInOutQuad" } });
}

/* ------------------------------------------------------------------ */
/* Pipeline run                                                        */
/* ------------------------------------------------------------------ */
async function runPipeline() {
  if (!state.currentCaseId) return;
  el("run-pipeline").disabled = true;
  notify("Starting pipeline...");
  setStatus("running");
  await api(`/api/cases/${state.currentCaseId}/run`, { method: "POST" });
  pollRun();
}

async function pollRun() {
  const poll = async () => {
    let status;
    try {
      status = await api(`/api/cases/${state.currentCaseId}/run/status`);
    } catch (_) {
      return;
    }
    notify(status.message || status.state);
    if (status.state === "running") {
      setTimeout(poll, 1200);
      return;
    }
    if (status.state === "error") {
      setStatus("error");
      el("run-pipeline").disabled = false;
      return;
    }
    setStatus("done");
    el("run-pipeline").disabled = false;
    await loadCases();
    await selectCase(state.currentCaseId);
  };
  poll();
}

/* ------------------------------------------------------------------ */
/* Wire-up                                                             */
/* ------------------------------------------------------------------ */
async function refreshCase() {
  if (!state.currentCaseId) return;
  await loadCases();
  await selectCase(state.currentCaseId);
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadHealth();
  await loadCases();

  el("new-case-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = el("case-name").value.trim();
    if (!name) return;
    await api("/api/cases", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
    el("case-name").value = "";
    await loadCases();
  });

  const loadSample = async () => {
    const c = await api("/api/cases/sample", { method: "POST" });
    await loadCases();
    await selectCase(c.id);
  };
  el("load-sample").addEventListener("click", loadSample);
  el("welcome-sample").addEventListener("click", loadSample);

  el("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = el("doc-file");
    if (!fileInput.files.length) return;
    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("doc_type", el("doc-type").value);
    try {
      await api(`/api/cases/${state.currentCaseId}/documents`, { method: "POST", body: fd });
      fileInput.value = "";
      await refreshCase();
    } catch (err) {
      notify(err.message, true);
    }
  });

  el("run-pipeline").addEventListener("click", runPipeline);

  el("variant-toggle").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-variant]");
    if (!btn) return;
    state.variant = btn.dataset.variant;
    document.querySelectorAll("#variant-toggle button").forEach((b) => b.classList.toggle("active", b === btn));
    loadGraph();
  });

  el("significant-only").addEventListener("change", (e) => {
    state.significantOnly = e.target.checked;
    rebuildNetwork();
  });
  el("edge-labels").addEventListener("change", (e) => {
    state.edgeLabels = e.target.checked;
    rebuildNetwork();
  });

  el("focus-btn").addEventListener("click", () => focusEntity(el("entity-search").value));
  el("entity-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") focusEntity(el("entity-search").value);
  });
  el("reset-view").addEventListener("click", () => {
    if (state.network) {
      state.network.fit({ animation: true });
      state.network.unselectAll();
    }
  });
});