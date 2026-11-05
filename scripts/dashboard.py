#!/usr/bin/env python3
"""Live page for the 555M run: progress, loss, GPU, tmux pane and log tail.

    scripts/dashboard.py [--session run555m] [--log logs/run555m/pretrain.log]

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
PROGRESS = re.compile(
    rb"ep \d+/\d+ .*?(\d+)/(\d+) NLL ([\d.]+) win ([\d.]+) ([\d.]+)ms/t "
    rb"(\d+)t/s ~([\d.]+)TF (\d+)MB")


def load_config() -> dict:
    local = ROOT / "dashboard.local.json"
    path = local if local.exists() else ROOT / "dashboard.json"
    if path != local:
        print(f"dashboard: no {local.name}, using the mockup {path.name}", flush=True)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {"host": os.environ.get("ANNI_DASH_HOST", cfg["host"]),
            "port": int(os.environ.get("ANNI_DASH_PORT", cfg["port"]))}


class Log:
    """Follow the trainer log; progress lines end in \\r, the rest in \\n."""

    def __init__(self, path: Path, window: int):
        self.path = path
        self.window = window
        self.pos = 0
        self.partial = b""
        self.points: list[list[float]] = []
        self.every = 1
        self.seen = 0
        self.last: dict = {}
        self.speed: list[float] = []
        self.lines: list[str] = []
        self.current = ""
        self.lock = threading.Lock()

    def line(self, raw: bytes) -> None:
        m = PROGRESS.search(raw)
        if not m:
            text = raw.decode("utf-8", "replace").strip()
            if text:
                self.lines = (self.lines + [text])[-40:]
            return
        self.current = raw.decode("utf-8", "replace").strip()
        done, total = int(m[1]), int(m[2])
        tok_s = int(m[6])
        self.speed = (self.speed + [tok_s])[-200:]
        self.last = {"windows": done, "total": total, "nll": float(m[3]),
                     "tok_s": tok_s, "tflops": float(m[7]), "mb": int(m[8]),
                     "tokens": done * self.window,
                     "total_tokens": total * self.window}
        self.seen += 1
        if self.seen % self.every == 0:
            self.points.append([done * self.window / 1e9, float(m[3])])
            if len(self.points) > MAX_POINTS:
                self.points = self.points[::2]
                self.every *= 2

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

    def state(self) -> dict:
        with self.lock:
            last = dict(self.last)
            if self.speed:
                avg = sum(self.speed) / len(self.speed)
                last["tok_s_avg"] = avg
                left = last["total_tokens"] - last["tokens"]
                last["eta_s"] = left / avg if avg > 0 else None
            return {"progress": last, "points": list(self.points),
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


def snapshot(log: Log, session: str) -> dict:
    log.update()
    state = log.state()
    state["gpu"] = gpu()
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
  --series:#2a78d6; --term:#1a1a19; --termink:#f0efec; }
@media (prefers-color-scheme: dark) { :root { --bg:#111110; --surface:#1a1a19; --ink:#f0efec;
  --ink2:#c3c2b7; --line:#33332f; --series:#3987e5; --term:#0d0d0c; } }
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
#chart { position:relative; } svg { display:block; width:100%; height:280px; }
#tip { position:absolute; display:none; pointer-events:none; background:var(--term); color:var(--termink);
  padding:4px 8px; border-radius:4px; font-size:13px; white-space:nowrap; }
pre { margin:0; padding:12px; background:var(--term); color:var(--termink); border-radius:6px;
  font:12.5px/1.4 ui-monospace,monospace; height:260px; overflow:auto; white-space:pre-wrap; word-break:break-all; }
.two { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:14px; }
</style></head><body><main>
<header><h1>ANNI 555M training</h1><div id="status">connecting</div></header>
<section class="card"><h2 id="pct">Progress</h2><div class="bar"><i id="fill"></i></div></section>
<section class="tiles">
<div class="card tile"><b id="nll">-</b><span>training loss (NLL)</span></div>
<div class="card tile"><b id="speed">-</b><span>tokens per second</span></div>
<div class="card tile"><b id="eta">-</b><span>time left</span></div>
<div class="card tile"><b id="vram">-</b><span>VRAM used</span></div>
<div class="card tile"><b id="gpu">-</b><span>GPU load, temperature</span></div>
</section>
<section class="card"><h2>Training loss by billions of tokens</h2>
<div id="chart"><svg id="svg" role="img" aria-label="Training loss over tokens"></svg><div id="tip"></div></div></section>
<section class="two">
<div class="card"><h2 id="panetitle">tmux</h2><pre id="pane"></pre></div>
<div class="card"><h2>Trainer log</h2><pre id="log"></pre></div>
</section></main>
<script>
const BASE = "", SNAPSHOT = null;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
let pts = [], geo = null;
function dur(s) { if (s == null) return "-"; const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600),
  m = Math.floor(s % 3600 / 60); return d ? d + " d " + h + " h" : h + " h " + m + " min"; }
function el(tag, at, text) { const e = document.createElementNS(NS, tag);
  for (const k in at) e.setAttribute(k, at[k]); if (text != null) e.textContent = text; return e; }
function ticks(lo, hi, n) { const raw = (hi - lo) / n || 1, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(k => k * p).find(s => s >= raw); const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out; }
function draw() {
  const svg = $("svg"), W = svg.clientWidth, H = svg.clientHeight, L = 44, R = 12, T = 10, B = 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.replaceChildren(); geo = null;
  const css = getComputedStyle(document.documentElement), ink = css.getPropertyValue("--ink2"),
    line = css.getPropertyValue("--line"), ser = css.getPropertyValue("--series");
  if (pts.length < 2) { svg.append(el("text", {x: W / 2, y: H / 2, "text-anchor": "middle", fill: ink, "font-size": 13},
    "Waiting for training steps")); return; }
  const x0 = pts[0][0], x1 = pts[pts.length - 1][0] || 1; let y0 = Infinity, y1 = -Infinity;
  for (const p of pts) { y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  y0 = Math.floor(y0); y1 = Math.ceil(y1);
  const X = v => L + (v - x0) / (x1 - x0 || 1) * (W - L - R), Y = v => T + (y1 - v) / (y1 - y0 || 1) * (H - T - B);
  for (const v of ticks(y0, y1, 5)) { svg.append(el("line", {x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: line}));
    svg.append(el("text", {x: L - 6, y: Y(v) + 4, "text-anchor": "end", fill: ink, "font-size": 12}, v)); }
  for (const v of ticks(x0, x1, 6)) svg.append(el("text", {x: X(v), y: H - 8, "text-anchor": "middle", fill: ink, "font-size": 12}, v + " G"));
  svg.append(el("path", {d: pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(""),
    fill: "none", stroke: ser, "stroke-width": 2, "stroke-linejoin": "round"}));
  const cross = el("line", {y1: T, y2: H - B, stroke: ink, visibility: "hidden"}),
    dot = el("circle", {r: 5, fill: ser, stroke: css.getPropertyValue("--surface"), "stroke-width": 2, visibility: "hidden"});
  svg.append(cross, dot); geo = {X, Y, cross, dot, W};
}
$("chart").addEventListener("mousemove", e => {
  if (!geo) return; const r = $("svg").getBoundingClientRect(), mx = (e.clientX - r.left) * geo.W / r.width;
  let best = pts[0], bd = Infinity; for (const p of pts) { const d = Math.abs(geo.X(p[0]) - mx); if (d < bd) { bd = d; best = p; } }
  const x = geo.X(best[0]), y = geo.Y(best[1]);
  geo.cross.setAttribute("x1", x); geo.cross.setAttribute("x2", x); geo.cross.setAttribute("visibility", "visible");
  geo.dot.setAttribute("cx", x); geo.dot.setAttribute("cy", y); geo.dot.setAttribute("visibility", "visible");
  const t = $("tip"); t.textContent = best[0].toFixed(3) + " G tokens, loss " + best[1].toFixed(3); t.style.display = "block";
  t.style.left = Math.min(x + 10, r.width - t.offsetWidth - 4) + "px"; t.style.top = Math.max(y - 34, 0) + "px";
});
$("chart").addEventListener("mouseleave", () => { $("tip").style.display = "none";
  if (geo) { geo.cross.setAttribute("visibility", "hidden"); geo.dot.setAttribute("visibility", "hidden"); } });
function follow(pre, text) { const end = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
  pre.textContent = text; if (end) pre.scrollTop = pre.scrollHeight; }
function show(s, label) {
    const p = s.progress, g = s.gpu;
    $("status").textContent = label + " " + new Date(s.now * 1000).toLocaleString();
    $("panetitle").textContent = "tmux " + s.session;
    if (p.total) { const f = p.windows / p.total * 100;
      $("pct").textContent = `Progress: ${f.toFixed(2)}%, ${(p.tokens / 1e9).toFixed(3)} of ${(p.total_tokens / 1e9).toFixed(2)} G tokens`;
      $("fill").style.width = f + "%"; $("nll").textContent = p.nll.toFixed(3);
      $("speed").textContent = Math.round(p.tok_s_avg).toLocaleString("en-US"); $("eta").textContent = dur(p.eta_s); }
    if (g.total) { $("vram").textContent = (g.used / 1024).toFixed(1) + " / " + (g.total / 1024).toFixed(1) + " GiB";
      $("gpu").textContent = Math.round(g.util) + "%, " + Math.round(g.temp) + " °C"; }
    follow($("pane"), s.pane || "(no tmux session)"); follow($("log"), s.log.join("\n"));
    if (s.points.length !== pts.length || (s.points.length && s.points.at(-1)[0] !== pts.at(-1)[0])) { pts = s.points; draw(); }
}
async function tick() {
  try { show(await (await fetch(BASE + "/state")).json(), "live, updated"); }
  catch (e) { if (SNAPSHOT) show(SNAPSHOT, "live server unreachable, snapshot of");
    else $("status").textContent = "disconnected"; }
  setTimeout(tick, 2000);
}
addEventListener("resize", draw); draw(); tick();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="run555m")
    ap.add_argument("--log", default=str(ROOT / "logs/run555m/pretrain.log"))
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--export", help="write a self-contained page and exit")
    args = ap.parse_args()
    cfg = load_config()
    log = Log(Path(args.log), args.window)
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
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
