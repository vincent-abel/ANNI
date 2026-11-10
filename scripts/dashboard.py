#!/usr/bin/env python3
"""Live page for the 555M run: progress, loss, GPU, tmux pane and log tail.

    scripts/dashboard.py [--session run555m] [--log logs/run555m/pretrain.log]

Every 10 steps a row goes to logs/run555m/curve.csv (time, tokens, loss, speed,
TFLOPS, VRAM), so the curve survives the log. Rows read from an existing log
have no time. A restart appends after the last row.

With --export FILE, write the page as one self-contained file for an artifact
panel: it polls this server when it can reach it, else shows the embedded
snapshot.

Read-only. Host and port come from dashboard.local.json, else dashboard.json;
ANNI_DASH_HOST and ANNI_DASH_PORT override them. No authentication, so keep
the host on a private interface.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_POINTS = 3000
MAX_RECENT = 4500	# every step of the last 10 hours, for the short timeframes
CURVE_EVERY = 10 * 131072	# tokens between rows, 10 steps of the run
# A resume seeds the trainer's loss average with the loss of the first tokens
# of the dataset, far above the real one; 5 time constants of that average.
RESUME_SETTLE = 13_000_000
D_MODEL, D_FF, N_BLOCKS, VOCAB = 1536, 6272, 14, 100000
N_HEADS, MAX_CTX = 24, 2048
# Rough estimates, in tokens so they follow the real speed, set from the samples
# of this run. The last value is the expected training loss, extended from its
# curve.
MILESTONES = [
    ["Recognizable words", "Mostly noise, with words and bits of code", 300_000_000, 3.7],
    ["Readable text", "Rough sentences; continues what you write, does not answer",
     600_000_000, 3.4],
    ["First chat try", "Clumsy conversation, after about 45 min of chat fine-tune on a copy",
     1_200_000_000, 3.1],
    ["Simple answers", "Short answers that fit the question; facts often wrong",
     2_500_000_000, 2.9],
    ["Steady conversation", "Follows a request and stays on topic; still invents facts",
     5_000_000_000, 2.75],
    ["Useful answers", "Most easy questions right; small code that may work",
     10_000_000_000, 2.55],
    ["Final model", "The full result; the chat fine-tune follows on its own", None, 2.4],
]
RESUME = re.compile(rb"\[([\d-]+ [\d:]+)\] resuming epoch \d+ at window (\d+)/")
TESTS = ROOT / "logs/run555m/tests"	# samples of scripts/run555m_checks.py
HELLASWAG = ROOT / "logs/run555m/hellaswag.jsonl"	# one bench result per line
TRAIN_ARGS = re.compile(rb"batch=(\d+) grad_accum=(\d+) lr=(\S+) window=(\d+)")


def step_slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def flops_per_token(window: int) -> float:
    """Train FLOPs of one token: 6 per weight of the blocks and head, plus
    causal attention. The trainer's own figure counts the FFN twice and drops
    the head, so it is not used here."""
    weights = N_BLOCKS * (4 * D_MODEL ** 2 + 2 * D_MODEL * D_FF) + D_MODEL * VOCAB
    return 6.0 * weights + 6.0 * N_BLOCKS * window * D_MODEL
PROGRESS = re.compile(
    rb"ep \d+/\d+ .*?(\d+)/(\d+) NLL ([\d.]+) win ([\d.]+) ([\d.]+)ms/t "
    rb"(\d+)t/s ~([\d.]+)TF (\d+)MB")


def run_params(args: tuple | None, total_tokens: int) -> list:
    """Groups of (label, value) for the top of the page."""
    d, f, v = D_MODEL, D_FF, VOCAB
    block = 4 * d * d + 2 * d * f + f + d + 4 * d
    total = N_BLOCKS * block + d * v + v
    n = lambda x: f"{x:,}"
    groups = [
        ["Model", [
            ["Parameters", n(total)],
            ["Vocabulary", n(v)],
            ["D model", n(d)],
            ["Blocks", n(N_BLOCKS)],
            ["Heads", f"{N_HEADS} of {d // N_HEADS}"],
            ["FFN width", n(f)],
            ["Max context", n(MAX_CTX)],
        ]],
        ["Matrices", [
            ["Shared embed and head", f"{n(d)} x {n(v)} ({n(d * v)})"],
            ["Wq, Wk, Wv, Wo", f"{n(d)} x {n(d)}, 4 per block"],
            ["FFN W1", f"{n(d)} x {n(f)}"],
            ["FFN W2", f"{n(f)} x {n(d)}"],
            ["One block", n(block)],
            ["All blocks", n(N_BLOCKS * block)],
        ]],
    ]
    train = [["Tokens", f"{total_tokens / 1e9:.2f} G"] if total_tokens else ["Tokens", "-"]]
    if args:
        batch, accum, lr, window = args
        step = batch * accum * window
        train += [["Window", n(window)], ["Batch", f"{batch} x {accum} accumulation"],
                  ["Tokens per step", n(step)]]
        if total_tokens:
            train.append(["Steps", n(total_tokens // step)])
        train.append(["Learning rate", f"{lr}, warmup 1,000, cosine"])
    train += [["Adam", "beta2 0.95, weight decay 0.1"], ["Gradient clip", "1.0"],
              ["Precision", "BF16, flash attention"]]
    groups.append(["Training", train])
    return groups


def load_config() -> dict:
    local = ROOT / "dashboard.local.json"
    path = local if local.exists() else ROOT / "dashboard.json"
    if path != local:
        print(f"dashboard: no {local.name}, using the mockup {path.name}",
              file=sys.stderr, flush=True)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {"host": os.environ.get("ANNI_DASH_HOST", cfg["host"]),
            "port": int(os.environ.get("ANNI_DASH_PORT", cfg["port"]))}


class Log:
    """Follow the trainer log; progress lines end in \\r, the rest in \\n."""

    def __init__(self, path: Path, window: int, curve: Path):
        self.path = path
        self.curve = curve
        self.curve_last = 0
        self.curve_times: list[int] = []
        if curve.exists():
            rows = curve.read_text(encoding="utf-8").splitlines()
            if len(rows) > 1:
                self.curve_last = int(rows[-1].split(",")[1])
            self.curve_times = [int(r.split(",")[0]) for r in rows[1:] if r[0] != ","]
        else:
            curve.write_text("time,tokens,nll,tok_s,tflops,vram_mb\n", encoding="utf-8")
        self.live = False
        self.window = window
        self.pos = 0
        self.partial = b""
        self.points: list[list[float]] = []
        self.recent: list[list[float]] = []
        self.resumes: list[dict] = []
        self.settled = 0	# tokens from which the loss can be trusted again
        self.every = 1
        self.seen = 0
        self.last: dict = {}
        self.speed: list[float] = []
        self.lines: list[str] = []
        self.current = ""
        self.args: tuple | None = None
        self.lock = threading.Lock()

    def line(self, raw: bytes) -> None:
        m = PROGRESS.search(raw)
        t = TRAIN_ARGS.search(raw)
        if t:
            self.args = (int(t[1]), int(t[2]), t[3].decode(), int(t[4]))
        r = RESUME.search(raw)
        if r:
            self.resume(r[1].decode(), int(r[2]) * self.window)
        if not m:
            text = raw.decode("utf-8", "replace").strip()
            if text:
                self.lines = (self.lines + [text])[-40:]
            return
        self.current = raw.decode("utf-8", "replace").strip()
        done, total = int(m[1]), int(m[2])
        tok_s = int(m[6])
        tflops = round(tok_s * flops_per_token(self.window) / 1e12, 2)
        self.speed = (self.speed + [tok_s])[-200:]
        self.last = {"windows": done, "total": total, "nll": float(m[3]),
                     "tok_s": tok_s, "tflops": tflops, "mb": int(m[8]),
                     "tokens": done * self.window,
                     "total_tokens": total * self.window}
        tokens = done * self.window
        if tokens < self.settled:
            return
        if tokens >= self.curve_last + CURVE_EVERY:
            self.curve_last = tokens
            stamp = int(time.time()) if self.live else ""
            with open(self.curve, "a", encoding="utf-8") as f:
                f.write(f"{stamp},{tokens},{float(m[3])},{tok_s},{tflops},{int(m[8])}\n")
        self.recent = (self.recent + [[tokens / 1e9, float(m[3])]])[-MAX_RECENT:]
        self.seen += 1
        if self.seen % self.every == 0:
            self.points.append([done * self.window / 1e9, float(m[3])])
            if len(self.points) > MAX_POINTS:
                self.points = self.points[::2]
                self.every *= 2

    def resume(self, stamp: str, tokens: int) -> None:
        """Mark a restart and forget the points of the work it threw away."""
        resumed = time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
        stopped = [t for t in self.curve_times if t < resumed]
        if self.resumes and self.resumes[-1]["tokens"] == tokens / 1e9:
            self.resumes[-1]["resumed"] = resumed	# restarted before the next save
        else:
            self.resumes.append({"tokens": tokens / 1e9, "resumed": resumed,
                                 "stopped": stopped[-1] if stopped else None})
        self.points = [p for p in self.points if p[0] * 1e9 <= tokens]
        self.recent = [p for p in self.recent if p[0] * 1e9 <= tokens]
        self.settled = tokens + RESUME_SETTLE

    def update(self) -> None:
        with self.lock:
            if not self.path.exists():
                return
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
            parts = re.split(rb"[\r\n]", self.partial + data)
            self.partial = parts.pop()
            for raw in parts:
                self.line(raw)
            self.live = True

    def state(self) -> dict:
        with self.lock:
            last = dict(self.last)
            if self.speed:
                avg = sum(self.speed) / len(self.speed)
                last["tok_s_avg"] = avg
                left = last["total_tokens"] - last["tokens"]
                last["eta_s"] = left / avg if avg > 0 else None
            return {"progress": last, "points": list(self.points),
                    "recent": list(self.recent),
                    "resumes": list(self.resumes),
                    "run": run_params(self.args, last.get("total_tokens", 0)),
                    "log": self.lines + [self.current]}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                              errors="replace").stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpu() -> dict:
    out = run(["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu="
               "memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"])
    try:
        used, total, util, temp, power = [float(x) for x in out.split("\n")[0].split(",")]
    except ValueError:
        return {}
    return {"used": used, "total": total, "util": util, "temp": temp, "power": power}


def levels() -> list[Path]:
    """Folders of the test levels done, the newest first."""
    return sorted((d for d in TESTS.glob("[0-9]*") if d.is_dir()), reverse=True)


def tests() -> tuple[list, dict]:
    """The last test of each kind as (kind, title, output), the newest kind
    first, and for each kind all its runs as (title, output), the newest first."""
    history = {m[0]: [] for m in MILESTONES}
    for level in levels():
        for name, runs in history.items():
            path = level / f"{step_slug(name)}.txt"
            if path.exists():
                runs.append(path.read_text(encoding="utf-8").split("\n\n", 1))
    latest = [[name, *runs[0]] for name, runs in reversed(history.items()) if runs]
    return latest, history


def hellaswag_rows() -> list[dict]:
    """Bench results, the oldest first."""
    if not HELLASWAG.exists():
        return []
    return [json.loads(line) for line in HELLASWAG.read_text(encoding="utf-8").splitlines()]


def snapshot(log: Log, session: str) -> dict:
    log.update()
    state = log.state()
    state["gpu"] = gpu()
    state["milestones"] = MILESTONES
    state["tests"], state["history"] = tests()
    state["hellaswag"] = hellaswag_rows()
    state["session"] = session
    state["pane"] = run(["tmux", "capture-pane", "-p", "-t",
                         f"={session}:", "-S", "-200"]).rstrip()
    state["now"] = time.time()
    return state


def make_handler(log: Log, session: str, page: bytes):
    class Handler(http.server.BaseHTTPRequestHandler):
        def send(self, body: bytes, kind: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if kind == "application/json":	# the exported page has another origin
                self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self.send(page, "text/html; charset=utf-8")
            elif self.path == "/state":
                self.send(json.dumps(snapshot(log, session)).encode(),
                          "application/json")
            else:
                self.send_error(404)

        def log_message(self, *args) -> None:
            pass

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ANNI 555M</title>
<style>
:root { --bg:#f4f4f2; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e2e1dc;
  --series:#2a78d6; --trend:#d03b3b; --average:#2e9a55; --term:#1a1a19; --termink:#f0efec; }
@media (prefers-color-scheme: dark) { :root { --bg:#111110; --surface:#1a1a19; --ink:#f0efec;
  --ink2:#c3c2b7; --line:#33332f; --series:#3987e5; --trend:#e66767; --average:#52c47e; --term:#0d0d0c; } }
* { box-sizing:border-box; }
body { margin:0; padding:20px; background:var(--bg); color:var(--ink);
  font:15px/1.4 system-ui,sans-serif; }
main { max-width:1200px; margin:auto; display:grid; gap:14px; }
header { display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; }
h1 { font-size:20px; margin:0; } h2 { font-size:14px; margin:0 0 8px; color:var(--ink2); font-weight:600; }
#status { color:var(--ink2); font-size:13px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }
.tile b { display:block; font-size:24px; font-weight:600; font-variant-numeric:tabular-nums; }
.tile span { color:var(--ink2); font-size:13px; }
.bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; margin-top:10px; }
.bar i { display:block; height:100%; width:0; background:var(--series); border-radius:4px; }
.chart { position:relative; } svg { display:block; width:100%; height:280px; }
.legend { font-size:13px; color:var(--ink2); margin:-4px 0 6px; }
.legend i { display:inline-block; width:16px; height:2px; vertical-align:middle; margin:0 5px 0 12px; }
.legend i:first-child { margin-left:0; }
.head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.range button { background:none; border:0; border-bottom:2px solid transparent; color:var(--ink2); font:inherit;
  font-size:13px; padding:0 6px 2px; cursor:pointer; } .range button.on { color:var(--ink); font-weight:600; border-bottom-color:var(--series); }
.tip { position:absolute; display:none; pointer-events:none; background:var(--term); color:var(--termink);
  padding:4px 8px; border-radius:4px; font-size:13px; white-space:nowrap; }
pre { margin:0; padding:12px; background:var(--term); color:var(--termink); border-radius:6px;
  font:12.5px/1.4 ui-monospace,monospace; height:260px; overflow:auto; white-space:pre-wrap; word-break:break-all; }
pre.out { height:auto; }
#tests, #hist, .view { display:grid; gap:14px; min-width:0; } #tests:empty { display:none; }
.view[hidden] { display:none; }
nav { display:flex; gap:6px; flex-wrap:wrap; border-bottom:1px solid var(--line); }
nav a { padding:6px 12px; color:var(--ink2); text-decoration:none; font-size:14px;
  border-bottom:2px solid transparent; margin-bottom:-1px; }
nav a.on { color:var(--ink); font-weight:600; border-bottom-color:var(--series); }
.params { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:6px 28px; }
.params h3 { font-size:13px; margin:0 0 6px; color:var(--ink2); font-weight:600; }
.params div div { display:flex; justify-content:space-between; gap:12px; font-size:13px; padding:2px 0;
  border-bottom:1px solid var(--line); }
.params span { color:var(--ink2); } .params b { font-weight:500; font-variant-numeric:tabular-nums; text-align:right; }
.steps { width:100%; border-collapse:collapse; font-size:13px; }
.steps th { text-align:left; color:var(--ink2); font-weight:600; padding:0 12px 6px 0; }
.steps td { padding:5px 12px 5px 0; border-top:1px solid var(--line); vertical-align:top; }
.steps td:first-child { font-weight:500; white-space:nowrap; } .steps .say { color:var(--ink2); }
.steps .num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.steps th:last-child, .steps td:last-child { padding-right:0; }
.two { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:14px; }
</style></head><body><main>
<header><h1>ANNI 555M training</h1><div id="status">connecting</div></header>
<nav id="tabs"><a href="#">Live</a><a href="#history">History</a><a href="#hellaswag">HellaSwag</a></nav>
<div id="live" class="view">
<section class="card"><div id="params" class="params"></div></section>
<section class="card"><h2 id="pct">Progress</h2><div class="bar"><i id="fill"></i></div></section>
<section class="tiles">
<div class="card tile"><b id="nll">-</b><span>training loss (NLL)</span></div>
<div class="card tile"><b id="speed">-</b><span>tokens per second</span></div>
<div class="card tile"><b id="tflops">-</b><span>TFLOPS</span></div>
<div class="card tile"><b id="tokens">-</b><span>tokens read</span></div>
<div class="card tile"><b id="eta">-</b><span>time left</span></div>
<div class="card tile"><b id="vram">-</b><span>VRAM used</span></div>
<div class="card tile"><b id="gpu">-</b><span>GPU load, temperature</span></div>
</section>
<section class="card"><div class="head"><h2>Training loss by billions of tokens</h2>
<div id="range" class="range"><button class="on">All</button><button data-h="10">10 h</button><button data-h="3">3 h</button><button data-h="1">1 h</button><button data-h="0.5">30 min</button><button data-h="0.3333">20 min</button><button data-h="0.1667">10 min</button><button data-h="0.0833">5 min</button></div></div>
<div class="legend"><i style="background:var(--series)"></i>Loss<i style="background:var(--trend)"></i>Mean trend<span id="slope"></span><span id="avglabel" style="margin-left:12px" hidden><i style="background:var(--average)"></i>Moving average</span></div>
<div id="chart" class="chart"><svg id="svg" role="img" aria-label="Training loss over tokens"></svg><div id="tip" class="tip"></div></div></section>
<section class="card"><h2>When to test the model (rough estimates)</h2><div style="overflow-x:auto"><table class="steps">
<thead><tr><th>Step</th><th>What to expect</th><th class="num">Tokens</th><th class="num">Expected NLL</th><th class="num">When</th><th class="num">Check</th></tr></thead>
<tbody id="steps"></tbody></table></div></section>
<section class="two">
<div class="card"><h2 id="panetitle">tmux</h2><pre id="pane"></pre></div>
<div class="card"><h2>Trainer log</h2><pre id="log"></pre></div>
</section>
<section id="tests"></section></div>
<div id="history" class="view" hidden><nav id="kinds"></nav><section id="hist"></section></div>
<div id="bench" class="view" hidden><section class="card"><h2>HellaSwag at each step</h2>
<p class="say">Choosing the right ending of a sentence among four, on 10,042 questions, by log-likelihood, without examples. Chance is 25 %. The score is the accuracy normalized by length, the usual headline number; raw accuracy is next to it.</p>
<div style="overflow-x:auto"><table class="steps"><thead><tr><th>Step</th><th>Tested</th><th class="num">Tokens</th><th class="num">Training NLL</th><th class="num">Score</th><th class="num">Raw</th><th class="num">NLL/token of right endings</th><th class="num">Questions</th></tr></thead>
<tbody id="benchrows"></tbody></table></div><p id="benchnone" class="say">No result yet: the first comes with the next step.</p></section></div></main>
<script>
const BASE = "", SNAPSHOT = null;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
let pts = [], recent = [], resumes = [], testsSeen = "", history = {}, speed = 17000, hours = 0;
const geos = {};
const when = (t, opt) => new Date(t * 1000).toLocaleString("en-GB", opt || {day: "numeric", month: "short",
  hour: "2-digit", minute: "2-digit"});
function dur(s) { if (s == null) return "-"; const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600),
  m = Math.floor(s % 3600 / 60); return d ? d + " d " + h + " h" : h + " h " + m + " min"; }
function el(tag, at, text) { const e = document.createElementNS(NS, tag);
  for (const k in at) e.setAttribute(k, at[k]); if (text != null) e.textContent = text; return e; }
function ticks(lo, hi, n) { const raw = (hi - lo) / n || 1, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(k => k * p).find(s => s >= raw); const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out; }
const short = () => hours > 0 && hours < 10;
function draw() {
  $("avglabel").hidden = !short();
  if (!hours) { plot("", pts, false); return; }
  const first = recent.length ? recent[0][0] : Infinity, all = pts.filter(p => p[0] < first).concat(recent);
  const end = all.length ? all[all.length - 1][0] : 0, from = end - speed * hours * 3600 / 1e9;
  plot("", all.filter(p => p[0] >= from), true);
}
// Straight line across what is shown, from its first value to its last one;
// on the short views each end is a mean, so noise does not tilt the line.
function trend(data) {
  const n = short() ? Math.max(5, Math.ceil(data.length / 100)) : 1, avg = a => a.reduce((t, p) => t + p[1], 0) / a.length;
  return [[data[0][0], avg(data.slice(0, n)), true], [data[data.length - 1][0], avg(data.slice(-n))]];
}
// Mean of the last sixth of the tokens shown; it restarts after a stop.
function average(data) {
  const span = data[data.length - 1][0] - data[0][0], sum = [0], out = [];
  for (const p of data) sum.push(sum[sum.length - 1] + p[1]);
  const at = x => { let lo = 0, hi = data.length; while (lo < hi) { const m = lo + hi >> 1;
    if (data[m][0] < x) lo = m + 1; else hi = m; } return lo; };
  for (let s = 0, e; s < data.length; s = e) {
    for (e = s + 1; e < data.length && data[e][0] - data[e - 1][0] <= span / 48; e++);
    for (let i = s; i < e; i++) { const a = Math.max(s, at(data[i][0] - span / 6));
      out.push([data[i][0], (sum[i + 1] - sum[a]) / (i + 1 - a), i === s]); }
  }
  return out;
}
function plot(id, data, zoom) {
  const svg = $(id + "svg"), W = svg.clientWidth, H = svg.clientHeight, L = 44, R = 12, T = 10, B = 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.replaceChildren(); geos[id] = null;
  const css = getComputedStyle(document.documentElement), ink = css.getPropertyValue("--ink2"),
    line = css.getPropertyValue("--line"), ser = css.getPropertyValue("--series");
  if (data.length < 2) { svg.append(el("text", {x: W / 2, y: H / 2, "text-anchor": "middle", fill: ink, "font-size": 13},
    "Waiting for training steps")); return; }
  const last = data[data.length - 1][0], x0 = data[0][0],
    x1 = (zoom ? last : Math.max(last, ...resumes.map(r => r.tokens))) || 1; let y0 = Infinity, y1 = -Infinity;
  for (const p of data) { y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  if (zoom) { const pad = (y1 - y0) * 0.1 || 0.01; y0 -= pad; y1 += pad; } else { y0 = Math.floor(y0); y1 = Math.ceil(y1); }
  const X = v => L + (v - x0) / (x1 - x0 || 1) * (W - L - R), Y = v => T + (y1 - v) / (y1 - y0 || 1) * (H - T - B);
  for (const v of ticks(y0, y1, 5)) { svg.append(el("line", {x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: line}));
    svg.append(el("text", {x: L - 6, y: Y(v) + 4, "text-anchor": "end", fill: ink, "font-size": 12}, v)); }
  for (const v of ticks(x0, x1, 6)) svg.append(el("text", {x: X(v), y: H - 8, "text-anchor": "middle", fill: ink, "font-size": 12}, v + " G"));
  resumes.filter(r => r.tokens >= x0 && r.tokens <= x1).forEach((r, i) => { const x = X(r.tokens), left = x > W / 2, clock = {hour: "2-digit", minute: "2-digit"};
    svg.append(el("line", {x1: x, x2: x, y1: T, y2: H - B, stroke: ink, "stroke-dasharray": "4 4"}));
    svg.append(el("text", {x: x + (left ? -6 : 6), y: T + 12 + i % 2 * 16, "text-anchor": left ? "end" : "start", fill: ink, "font-size": 12},
      r.stopped ? `stopped ${when(r.stopped)}, resumed ${when(r.resumed, clock)}` : `resumed ${when(r.resumed)}`)); });
  svg.append(el("path", {d: data.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(""),
    fill: "none", stroke: ser, "stroke-width": 2, "stroke-linejoin": "round"}));
  const fit = trend(data), slope = (fit[1][1] - fit[0][1]) / (fit[1][0] - fit[0][0]);
  $("slope").textContent = isFinite(slope) ? `: ${slope.toFixed(2).replace("-", "\u2212")} NLL per G tokens` : "";
  for (const [path, color] of [[fit, "--trend"], ...(short() ? [[average(data), "--average"]] : [])])
    svg.append(el("path", {d: path.map(p => (p[2] ? "M" : "L") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(""),
      fill: "none", stroke: css.getPropertyValue(color), "stroke-width": 2, "stroke-linejoin": "round"}));
  const cross = el("line", {y1: T, y2: H - B, stroke: ink, visibility: "hidden"}),
    dot = el("circle", {r: 5, fill: ser, stroke: css.getPropertyValue("--surface"), "stroke-width": 2, visibility: "hidden"});
  svg.append(cross, dot); geos[id] = {X, Y, cross, dot, W, data};
}
function hover(id) {
  const box = $(id + "chart"), tip = $(id + "tip");
  box.addEventListener("mousemove", e => {
    const geo = geos[id]; if (!geo) return; const r = $(id + "svg").getBoundingClientRect(), mx = (e.clientX - r.left) * geo.W / r.width;
    let best = geo.data[0], bd = Infinity; for (const p of geo.data) { const d = Math.abs(geo.X(p[0]) - mx); if (d < bd) { bd = d; best = p; } }
    const x = geo.X(best[0]), y = geo.Y(best[1]);
    geo.cross.setAttribute("x1", x); geo.cross.setAttribute("x2", x); geo.cross.setAttribute("visibility", "visible");
    geo.dot.setAttribute("cx", x); geo.dot.setAttribute("cy", y); geo.dot.setAttribute("visibility", "visible");
    tip.textContent = best[0].toFixed(3) + " G tokens, loss " + best[1].toFixed(3); tip.style.display = "block";
    tip.style.left = Math.max(0, Math.min(x + 10, r.width - tip.offsetWidth - 4)) + "px"; tip.style.top = Math.max(y - 34, 0) + "px";
  });
  box.addEventListener("mouseleave", () => { tip.style.display = "none"; const geo = geos[id];
    if (geo) { geo.cross.setAttribute("visibility", "hidden"); geo.dot.setAttribute("visibility", "hidden"); } });
}
hover("");
$("range").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return;
  hours = +b.dataset.h || 0;
  [...$("range").children].forEach(x => x.classList.toggle("on", x === b)); draw(); });
function follow(pre, text) { const end = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
  pre.textContent = text; if (end) pre.scrollTop = pre.scrollHeight; }
function params(groups) {
  $("params").replaceChildren(...groups.map(([title, rows]) => { const g = document.createElement("div"),
    h = document.createElement("h3"); h.textContent = title; g.append(h);
    for (const [k, v] of rows) { const r = document.createElement("div"), a = document.createElement("span"),
      b = document.createElement("b"); a.textContent = k; b.textContent = v; r.append(a, b); g.append(r); }
    return g; }));
}
function result(title, text) {
  const c = document.createElement("div"), h = document.createElement("h2"), o = document.createElement("pre");
  c.className = "card"; h.textContent = title; o.className = "out"; o.textContent = text; c.append(h, o); return c;
}
const slug = name => name.toLowerCase().replaceAll(" ", "-");
function route() {
  const [tab, sub] = location.hash.slice(1).split("/"), old = tab === "history", bench = tab === "hellaswag";
  $("live").hidden = old || bench; $("history").hidden = !old; $("bench").hidden = !bench;
  [...$("tabs").children].forEach((a, i) => a.classList.toggle("on", i === (old ? 1 : bench ? 2 : 0)));
  if (!old) { if (!bench) draw(); return; }
  const names = Object.keys(history), name = names.find(n => slug(n) === sub) || names[0];
  $("kinds").replaceChildren(...names.map(n => { const a = document.createElement("a");
    a.href = "#history/" + slug(n); a.textContent = `${n} (${history[n].length})`;
    a.classList.toggle("on", n === name); return a; }));
  const runs = name ? history[name] : [];
  $("hist").replaceChildren(...(runs.length ? runs.map(([title, text]) => result(title, text))
    : [result("No test of this kind yet", "")]));
}
const pct = x => x == null ? "-" : (100 * x).toFixed(1) + " %";
function bench(rows) {
  $("benchrows").replaceChildren(...[...rows].reverse().map(b => { const r = document.createElement("tr");
    const cells = [b.step, b.tested, b.tokens == null ? "-" : b.tokens.toFixed(3) + " G",
      b.nll == null ? "-" : b.nll.toFixed(2), b.error ? "failed" : pct(b.acc_norm), pct(b.acc),
      b.nll_token == null ? "-" : b.nll_token.toFixed(2), b.questions == null ? "-" : b.questions.toLocaleString("en")];
    cells.forEach((text, i) => { const c = document.createElement("td"); c.textContent = text;
      if (i > 1) c.className = "num"; if (i === 4 && b.error) c.title = b.error; r.append(c); });
    return r; }));
  $("benchnone").hidden = rows.length > 0;
}
function steps(s) {
  const p = s.progress;
  bench(s.hellaswag);
  $("steps").replaceChildren(...s.milestones.map(([name, say, at, nll]) => {
    const goal = at || p.total_tokens, left = goal - p.tokens, r = document.createElement("tr");
    const secs = p.tok_s_avg ? left / p.tok_s_avg : null;
    const cells = [name, say, goal ? (goal / 1e9).toFixed(2) + " G" : "-", "~" + nll.toFixed(1),
      !goal ? "-" : left <= 0 ? "✓ Reached" : secs == null ? "-" : "in " + dur(secs) + ", " + when(s.now + secs)];
    cells.forEach((text, i) => { const c = document.createElement("td"); c.textContent = text;
      c.className = ["", "say", "num", "num", "num"][i]; r.append(c); });
    const c = document.createElement("td"); c.className = "num";
    c.textContent = s.history[name].length ? "✓ Tested" : left <= 0 && at ? "at next save" : "-";
    r.append(c); return r; }));
  const seen = JSON.stringify([s.tests, s.history]);
  if (seen !== testsSeen) { testsSeen = seen; history = s.history;
    $("tests").replaceChildren(...s.tests.map(([name, title, text]) => result(`${name}: ${title}`, text))); route(); }
}
function show(s, label) {
    params(s.run);
    const p = s.progress, g = s.gpu; if (p.tok_s_avg) speed = p.tok_s_avg;
    $("status").textContent = label + " " + new Date(s.now * 1000).toLocaleString();
    $("panetitle").textContent = "tmux " + s.session;
    if (p.total) { const f = p.windows / p.total * 100;
      $("pct").textContent = `Progress: ${f.toFixed(2)}%, ${(p.tokens / 1e9).toFixed(3)} of ${(p.total_tokens / 1e9).toFixed(2)} G tokens`;
      $("fill").style.width = f + "%"; $("nll").textContent = p.nll.toFixed(3);
      $("speed").textContent = Math.round(p.tok_s_avg).toLocaleString("en-US"); $("eta").textContent = dur(p.eta_s);
      $("tflops").textContent = p.tflops.toFixed(1); $("tokens").textContent = p.tokens.toLocaleString("en-US"); }
    if (g.total) { $("vram").textContent = (g.used / 1024).toFixed(1) + " / " + (g.total / 1024).toFixed(1) + " GiB";
      $("gpu").textContent = Math.round(g.util) + "%, " + Math.round(g.temp) + " °C"; }
    steps(s);
    follow($("pane"), s.pane || "(no tmux session)"); follow($("log"), s.log.join("\n"));
    if (s.points.length !== pts.length || s.resumes.length !== resumes.length ||
        s.recent.at(-1)?.[0] !== recent.at(-1)?.[0]) { pts = s.points; recent = s.recent; resumes = s.resumes; draw(); }
}
async function tick() {
  try { show(await (await fetch(BASE + "/state")).json(), "live, updated"); }
  catch (e) { if (SNAPSHOT) show(SNAPSHOT, "live server unreachable, snapshot of");
    else $("status").textContent = "disconnected"; }
  setTimeout(tick, 2000);
}
addEventListener("resize", draw); addEventListener("hashchange", route); draw(); tick();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="run555m")
    ap.add_argument("--log", default=str(ROOT / "logs/run555m/pretrain.log"))
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--export", help="write a self-contained page and exit")
    ap.add_argument("--print-url", action="store_true", help="print the page URL and exit")
    args = ap.parse_args()
    cfg = load_config()
    if args.print_url:
        print(f"http://{cfg['host']}:{cfg['port']}/")
        return 0
    log = Log(Path(args.log), args.window, Path(args.log).parent / "curve.csv")
    if args.export:
        data = json.dumps(snapshot(log, args.session)).replace("</", "<\\/")
        base = json.dumps(f"http://{cfg['host']}:{cfg['port']}")
        Path(args.export).write_text(PAGE.replace(
            'const BASE = "", SNAPSHOT = null;',
            f"const BASE = {base}, SNAPSHOT = {data};"), encoding="utf-8")
        return 0
    handler = make_handler(log, args.session, PAGE.encode())
    server = http.server.ThreadingHTTPServer((cfg["host"], cfg["port"]), handler)
    print(f"dashboard: http://{cfg['host']}:{cfg['port']}/", flush=True)

    def follow() -> None:	# the curve must not depend on an open page
        while True:
            log.update()
            time.sleep(20)

    threading.Thread(target=follow, daemon=True).start()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
