"use strict";

const $ = (id) => document.getElementById(id);
const show = (id, text, cls = "") => { const e = $(id); e.textContent = text; e.className = "out " + cls; };
const pretty = (o) => JSON.stringify(o, null, 2);

// short activity log above the chart (newest on top, capped) ----------------
function logLine(message, cls = "") {
  const box = $("log");
  const line = document.createElement("div");
  line.className = "logline " + cls;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  box.prepend(line);
  while (box.childElementCount > 50) box.lastElementChild.remove();
}

async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(typeof data === "object" ? data.detail || pretty(data) : data);
  return data;
}

// per-section run-dir selectors --------------------------------------------
const runValue = (id) => { const v = $(id).value; return v.startsWith("(latest)") ? null : v; };

async function loadRuns(preferred) {
  const data = await api("GET", "/runs");
  const latest = data.latest ? ` → ${data.latest}` : "";
  const opts = [`(latest)${latest}`, ...data.runs.map((r) => r.run)];
  // populate every run selector, preserving its current choice (or `preferred`)
  document.querySelectorAll("select.runsel").forEach((sel) => {
    const keep = preferred && opts.includes(preferred) ? preferred : sel.value;
    sel.innerHTML = opts.map((o) => `<option>${o}</option>`).join("");
    if (opts.includes(keep)) sel.value = keep;
  });
  // populate granularity dropdowns once
  document.querySelectorAll("select.gran").forEach((g) => {
    if (!g.dataset.filled) {
      g.innerHTML = data.granularities.map((x) => `<option>${x}</option>`).join("");
      g.dataset.filled = "1";
    }
  });
}

// poll a background job until it finishes ----------------------------------
function progressText(job) {
  const p = job.progress;
  if (!p || !p.total) return `(${job.status})`;
  const pct = Math.round((100 * p.done) / p.total);
  const bar = "█".repeat(Math.round(pct / 5)).padEnd(20, "·");
  return `${p.done}/${p.total} ${pct}%\n[${bar}]`;
}

async function pollJob(jobId, outId, label) {
  let lastPct = -1;
  for (;;) {
    const job = await api("GET", `/jobs/${jobId}`);
    if (job.status === "succeeded") { show(outId, `${label} ✓\n` + pretty(job.result), "ok"); return job; }
    if (job.status === "failed") { show(outId, `${label} ✗\n` + job.error, "err"); throw new Error(job.error); }
    if (job.status === "cancelled") { show(outId, `${label} ⊘ cancelled`, "err"); return job; }
    show(outId, `${label}… ${progressText(job)}`);
    // log every ~20% so the activity log shows progress without spamming
    const p = job.progress;
    if (p && p.total) {
      const pct = Math.round((100 * p.done) / p.total);
      if (pct >= lastPct + 20 || pct === 100) { logLine(`${label} ${p.done}/${p.total} (${pct}%)`); lastPct = pct; }
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

// actions -------------------------------------------------------------------
const actions = {
  async calendar() {
    const r = await api("POST", "/calendar",
      { start_year: +$("cal_start").value, end_year: +$("cal_end").value });
    show("out_calendar", `wrote ${r.rows} rows → ${r.path}`, "ok");
    logLine(`calendar: ${r.rows} rows`, "ok");
  },

  async preprocess() {
    const body = {
      start: $("pp_start").value, end: $("pp_end").value,
      tz: $("pp_tz").value || null, raw_log_uri: $("pp_uri").value || null,
      workers: $("pp_workers").value ? +$("pp_workers").value : null,
      refresh: $("pp_refresh").checked,
    };
    const { job_id } = await api("POST", "/preprocess", body);
    logLine(`preprocess ${body.start}..${body.end} started (job ${job_id})`);
    ppJobId = job_id;
    setCancel(true);
    try {
      const job = await pollJob(job_id, "out_preprocess", "preprocess");
      if (job.status === "cancelled") { logLine("preprocess cancelled", "err"); return; }
      logLine(`preprocess ✓ run ${job.result.run_dir}`, "ok");
      await loadRuns(job.result.run_dir); // select the new run everywhere
    } finally {
      ppJobId = null;
      setCancel(false);
    }
  },

  async train() {
    const run = runValue("tr_run"), gran = $("tr_gran").value;
    const { job_id } = await api("POST", "/train", { granularity: gran, run_dir: run });
    logLine(`train ${gran} on ${run || "latest"} started (job ${job_id})`);
    const job = await pollJob(job_id, "out_train", "train");
    logLine(`train ${gran} ✓ mae=${Math.round(job.result.metrics.mae).toLocaleString()}`, "ok");
    await loadRuns();
  },

  async predict() {
    const run = runValue("pr_run"), gran = $("pr_gran").value, mode = $("pr_mode").value;
    const num = (v) => Math.round(v).toLocaleString();
    if (mode === "backtest") {
      const r = await api("POST", "/backtest", {
        granularity: gran, run_dir: run,
        from_date: $("pr_from").value, to_date: $("pr_to").value || $("pr_from").value,
      });
      const rows = r.backtest.map((p) =>
        `<tr><td>${p.bucket_start}</td><td>${num(p.actual)}</td><td>${num(p.p50)}</td><td>${num(p.p90)}</td><td>${num(p.p95)}</td></tr>`).join("");
      show("out_predict", `backtest saved → ${r.saved}\nMAE ${num(r.metrics.mae)} · MAPE ${r.metrics.mape.toFixed(1)}%`, "ok");
      $("out_predict").insertAdjacentHTML("beforeend",
        `<table><tr><th>bucket</th><th>actual</th><th>P50</th><th>P90</th><th>P95</th></tr>${rows}</table>`);
      logLine(`backtest ${gran} on ${r.run_dir} → MAPE ${r.metrics.mape.toFixed(1)}%`, "ok");
      return;
    }
    const body = { granularity: gran, run_dir: run };
    if (mode === "window") {
      body.from_date = $("pr_from").value; body.to_date = $("pr_to").value || $("pr_from").value;
    } else { body.horizon = +$("pr_horizon").value; }
    const r = await api("POST", "/predict", body);
    const rows = r.forecast.map((p) =>
      `<tr><td>${p.bucket_start}</td><td>${num(p.p50)}</td><td>${num(p.p90)}</td><td>${num(p.p95)}</td></tr>`).join("");
    const note = r.gap_steps > 0 ? `\n⚠ ${r.gap_steps} step lead-in before window` : "";
    show("out_predict", `saved → ${r.saved}${note}`, "ok");
    $("out_predict").insertAdjacentHTML("beforeend",
      `<table><tr><th>bucket</th><th>P50</th><th>P90</th><th>P95</th></tr>${rows}</table>`);
    logLine(`predict ${gran} on ${r.run_dir} → ${r.forecast.length} buckets`, "ok");
  },

  async visualize() {
    const g = $("vz_gran").value, h = +$("vz_height").value, run = runValue("vz_run");
    const kind = $("vz_kind").value;
    const q = new URLSearchParams({ granularity: g, height: h, kind });
    if (run) q.set("run_dir", run);
    // probe first so errors show as text rather than a broken iframe
    const res = await fetch(`/visualize?${q}`);
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
    $("chart").style.height = (h + 60) + "px"; // fit the chart + range slider, no clipping
    $("chart").src = `/visualize?${q}&_=${Date.now()}`;
    show("out_visualize", "rendered ✓", "ok");
    logLine(`chart: ${kind} ${g} on ${run || "latest"}`, "ok");
  },
};

// wiring --------------------------------------------------------------------
document.querySelectorAll("button[data-act]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const act = btn.dataset.act, outId = "out_" + act;
    btn.disabled = true;
    try { show(outId, "working…"); await actions[act](); }
    catch (e) { show(outId, "error: " + e.message, "err"); logLine(`${act} error: ${e.message}`, "err"); }
    finally { btn.disabled = false; }
  });
});
// cancel the in-flight preprocess job ---------------------------------------
let ppJobId = null;
const setCancel = (on) => { const b = $("pp_cancel"); b.style.display = on ? "" : "none"; b.disabled = false; };
$("pp_cancel").addEventListener("click", async () => {
  if (!ppJobId) return;
  $("pp_cancel").disabled = true;
  try { await api("POST", `/jobs/${ppJobId}/cancel`); logLine("preprocess cancel requested"); }
  catch (e) { logLine("cancel failed: " + e.message, "err"); $("pp_cancel").disabled = false; }
});

$("pr_mode").addEventListener("change", (e) => {
  const horizon = e.target.value === "horizon";  // window + backtest both use from/to dates
  $("pr_window_box").style.display = horizon ? "none" : "";
  $("pr_horizon_box").style.display = horizon ? "" : "none";
});
$("refreshRuns").addEventListener("click", () => loadRuns());
// per-section ↻ buttons beside each run-dir selector (preserve current selections)
document.querySelectorAll(".refreshrun").forEach((b) =>
  b.addEventListener("click", () => loadRuns().then(() => logLine("run list refreshed"))));
loadRuns()
  .then(() => logLine("ready"))
  .catch((e) => logLine("failed to load runs: " + e.message, "err"));
