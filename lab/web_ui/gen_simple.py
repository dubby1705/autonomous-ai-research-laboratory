import pathlib

HERE = pathlib.Path(__file__).resolve().parent
parts = []
P4 = pathlib.Path

/* ------------------------------------------------------------ header */
function header(app, opts) {
  const o = opts || {};
  const hd = el("div", { class: "hd" });
  hd.appendChild(el("div", {
    class: "brand",
    html: 'AARL<small>AUTONOMOUS AI RESEARCH LABORATORY</small>' }));
  const on = el("div", { class: "online" });
  on.appendChild(el("span", {
    class: "dot" + (o.dot ? " " + o.dot : "") }));
  on.appendChild(el("span", { text: o.label || "ENGINE ONLINE" }));
  hd.appendChild(on);
  app.appendChild(hd);
}

function bar(app, value, disabled, onSubmit) {
  const form = el("form", { class: "bar" });
  const input = el("input", {
    type: "text", autocomplete: "off", spellcheck: "false",
    placeholder: 'Research: Design a new advanced GPU architecture',
    "aria-label": "Research problem", value: value || "" });
  const btn = el("button", { type: "submit", text: "ENGAGE" });
  if (disabled) btn.disabled = true;
  form.appendChild(input);
  form.appendChild(btn);
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const problem = input.value.trim().replace(
      /^(?:research|investigate)\s*[:\-]?\s*/i, "");
    if (problem.length < 4) { input.focus(); return; }
    onSubmit(problem, input, btn);
  });
  app.appendChild(form);
  const statusline = el("div", { class: "statusline", id: "statusline" });
  app.appendChild(statusline);
  return { input, btn, statusline };
}

function section(app, title) {
  const sec = el("div", { class: "sec" });
  sec.appendChild(el("div", { class: "sec-title", text: title }));
  app.appendChild(sec);
  return sec;
}
function kv(sec, k, v, mono) {
  const row = el("div", { class: "kv" });
  row.appendChild(el("span", { class: "k", text: k }));
  row.appendChild(el("span", {
    class: "v" + (mono ? " mono" : ""),
    text: v == null || v === "" ? "\u2014" : String(v) }));
  sec.appendChild(row);
  return row;
}
function statBox(label, value) {
  const b = el("div", { class: "stat" });
  b.appendChild(el("b", { text: value }));
  b.appendChild(el("span", { text: label }));
  return b;
}
function statusLine(msg, tag) {
  const sl = $("#statusline");
  if (!sl) return;
  sl.innerHTML = "";
  if (tag) sl.appendChild(el("span", {
    class: "tag", text: tag + "  " }));
  sl.appendChild(el("span", { text: msg }));
}

function renderIdle() {
  const app = $("#app");
  app.innerHTML = "";
  header(app, { label: "ENGINE ONLINE", dot: "" });
  const b = bar(app, "", false, engage);
  statusLine("AARL ready \u2014 describe a research problem and press ENGAGE.");
  S.phase = "idle";
  S.runId = null;
}

function renderRunning(problem) {
  const app = $("#app");
  app.innerHTML = "";
  header(app, { label: "ENGINE ONLINE", dot: "busy" });
  const b = bar(app, problem, true, () => {});
  b.input.disabled = true;
  b.btn.disabled = true;
  statusLine("Research in progress\u2026", "RUNNING");
  S.phase = "running";

  const psec = section(app, "CURRENT RESEARCH");
  kv(psec, "Problem", problem);
  kv(psec, "Status", "INITIALIZING");
  const counts = el("div", { class: "stat-row" });
  counts.appendChild(statBox("HYPOTHESES", "0"));
  counts.appendChild(statBox("EXPERIMENTS", "0"));
  counts.appendChild(statBox("EVIDENCE", "0"));
  psec.appendChild(counts);

  const asec = section(app, "LIVE ACTIVITY");
  asec.appendChild(el("ul", { id: "act-list", class: "act" }));
}

function updateRunning(run, events) {
  if (S.phase !== "running") return;
  const app = $("#app");
  const psec = app.querySelector(".sec");
  if (!psec) return;
  const status = (run && run.status) || "running";
  const stages = (run && run.stages) || [];
  let stageLabel = "";
  for (const st of stages) {
    if (st.state === "running" && st.label) { stageLabel = st.label; break; }
    if (st.state === "done" && !stageLabel && st.label) { stageLabel = st.label; }
  }
  const problemRow = psec.querySelector(".kv:first-child");
  if (problemRow) {
    const v = problemRow.querySelector(".v");
    if (v) v.textContent = (run && run.problem) || "\u2014";
  }
  const rows = psec.querySelectorAll(".kv");
  if (rows.length >= 2) {
    const vv = rows[1].querySelector(".v");
    if (vv) {
      vv.textContent = status.toUpperCase();
      vv.style.color = status === "done" ? "var(--good)"
        : status === "failed" ? "var(--bad)" : "var(--acc2)";
    }
  }
  const space = (run && run.space) || {};
  const stats = (run && run.stats) || {};
  const dirCount = (space && space.nodes) ? space.nodes.length
    : (stats && stats.directions) ? stats.directions : 0;
  const hypCount = (run && run.directions) ? run.directions.length : 0;
  const countsEl = psec.querySelector(".stat-row");
  if (countsEl) {
    const boxes = countsEl.querySelectorAll(".stat b");
    if (boxes.length >= 3) {
      boxes[0].textContent = hypCount;
      boxes[1].textContent = dirCount;
      boxes[2].textContent = Math.max(1, (run && run.literature_overlap)
        ? run.literature_overlap.length : 0);
    }
  }
  paintActivity(events || []);
}

function paintActivity(events) {
  const ul = $("#act-list");
  if (!ul) return;
  ul.innerHTML = "";
  const items = (events || []).slice(0, 8);
  if (!items.length) {
    const li = el("li");
    li.appendChild(el("span", { class: "g run", text: "\u2192" }));
    li.appendChild(el("span", { class: "t",
      text: "Initializing research engine\u2026" }));
    ul.appendChild(li);
    return;
  }
  items.forEach((ev) => {
    const [glyph, cls] = GLYPH[ev.severity] || GLYPH.info;
    const li = el("li");
    li.appendChild(el("span", { class: "g " + cls, text: glyph }));
    li.appendChild(el("span", { class: "t", text: ev.text || "" }));
    if (ev.at) {
      li.appendChild(el("span", {
        class: "at", text: String(ev.at).slice(11, 19) || "" }));
    }
    ul.appendChild(li);
  });
}
"""

parts.append(r"""
/* ============================================================
   AARL · simple Research Command Center
   Idle → command bar · Running → live panel · Done → report
   All data comes from the real AARL backend (lab.web_api).
   ============================================================ */
(function () {
"use strict";
const $ = (s, r) => (r || document).querySelector(s);
function el(tag, attrs, html) {
  const n = document.createElement(tag);
  if (attrs) for (const k of Object.keys(attrs)) {
    if (k === "class") n.className = attrs[k];
    else if (k === "text") n.textContent = attrs[k];
    else if (k === "html") n.innerHTML = attrs[k];
    else n.setAttribute(k, attrs[k]);
  }
  if (html != null) n.innerHTML = html;
  return n;
}
const esc = (s) => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");

async function api(url, opts) {
  const res = await fetch(url, opts);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error((body && body.error) || ("HTTP " + res.status));
  return body;
}
const get = (u) => api(u);
const post = (u, p) => api(u, { method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(p || {}) });

const STAGE_LABELS = {
  understanding: "ANALYZING PROBLEM", domains: "DETECTING DOMAINS",
  literature: "SEARCHING KNOWLEDGE", graph: "BUILDING KNOWLEDGE GRAPH",
  exploring: "EXPLORING POSSIBILITY SPACE", hypotheses: "GENERATING HYPOTHESES",
  mechanisms: "DESIGNING EXPERIMENTS", simulations: "SIMULATING",
  evaluating: "VALIDATING RESULTS",
};
const S = { runId: null, phase: "idle", timer: 0, busy: false };

const GLYPH = {
  info:    ["\u2022", "g-dim"],
  notice:  ["\u2022", "g-dim"],
  success: ["\u2713", "g-ok"],
  alert:   ["!", "g-bad"],
  trace:   ["#", "g-dim"],
};

/* -------------------------
