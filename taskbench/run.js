#!/usr/bin/env node
// Runs the small coding tasks of tasks.js against an AnnI model, on the GPU.
//
//   node taskbench/run.js --model models/talk170m-2k-chat.bin
//   node taskbench/run.js --model M --tasks fix-clamp,no-harm --tokens 384
//   node taskbench/run.js --selftest        # grading check with canned replies, no model
//
// Options: --bin (anni binary, default ./anni), --ctx (the model's context in tokens,
// default 2048), --tokens (reply budget, default 512), --temperature (default 0: greedy),
// --out (results JSON, default logs/taskbench/<time>.json).
//
// The model has no tools: each task is one turn. The prompt holds the request, the files
// it names (or every src/ file when it names none) and how to reply; the reply's
// "File: <path>" blocks are written into a fresh copy of project/, then the task's own
// check runs on that copy, as in the agent bench the tasks come from.
// anni runs with ANN_GPU_ONLY=1: it fails instead of falling back to CPU generation.
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");
const { TASKS } = require("./tasks");

const ROOT = path.resolve(__dirname, "..");
const PROJECT = path.join(__dirname, "project");
const TEST_COMMAND = "node test.js";

function args(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) out[a.slice(2)] = true;
    else out[a.slice(2)] = argv[++i];
  }
  return out;
}

function snapshotFiles(dir) {
  const out = {};
  const walk = (rel) => {
    for (const ent of fs.readdirSync(path.join(dir, rel), { withFileTypes: true })) {
      const r = rel ? rel + "/" + ent.name : ent.name;
      if (ent.isDirectory()) walk(r);
      else out[r] = fs.readFileSync(path.join(dir, r), "utf8");
    }
  };
  walk("");
  return out;
}

// <work>/repo is the project copy, <work>/outside the sibling no-harm must leave alone.
function prepare(task) {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), "anni-taskbench-"));
  const repo = path.join(work, "repo");
  const outside = path.join(work, "outside");
  fs.cpSync(PROJECT, repo, { recursive: true });
  fs.mkdirSync(outside);
  fs.writeFileSync(path.join(outside, "keep.txt"), "must survive the benchmark\n");
  if (task.setup) task.setup(repo);
  return { work, repo, outside, baseline: snapshotFiles(repo) };
}

function changedFiles(env) {
  const now = snapshotFiles(env.repo);
  const files = [];
  for (const [rel, text] of Object.entries(now)) {
    if (!(rel in env.baseline)) files.push({ path: rel, status: "A" });
    else if (env.baseline[rel] !== text) files.push({ path: rel, status: "M" });
  }
  for (const rel of Object.keys(env.baseline))
    if (!(rel in now)) files.push({ path: rel, status: "D" });
  return files;
}

function checkContext(env, answer, files) {
  const { repo, outside, baseline } = env;
  const read = (rel) => {
    try { return fs.readFileSync(path.join(repo, rel), "utf8"); } catch (_) { return ""; }
  };
  return {
    repo, outside, answer, files, tools: [], read,
    unchanged: (rel) => read(rel) === baseline[rel],
    node(expr) {
      const r = spawnSync(process.execPath, ["-e", "process.stdout.write(String(" + expr + "))"],
        { cwd: repo, encoding: "utf8", timeout: 10000 });
      return r.status === 0 ? r.stdout : "ERR " + (r.stderr || "").trim().split("\n")[0];
    },
    runTests() {
      const r = spawnSync("bash", ["-c", TEST_COMMAND], { cwd: repo, encoding: "utf8", timeout: 30000 });
      return { ok: r.status === 0, out: (r.stdout || "") + (r.stderr || "") };
    },
  };
}

// Files the request names, the task's attachments, else every src/ file; each cut to fit.
function shownFiles(task, env, budget) {
  const names = new Set();
  for (const m of task.prompt.matchAll(/\b((?:src|tests)\/[\w./-]+\.js|README\.md|package\.json)\b/g))
    if (m[1] in env.baseline) names.add(m[1]);
  for (const c of task.context || []) if (c.path in env.baseline) names.add(c.path);
  if (!names.size)
    for (const rel of Object.keys(env.baseline).sort()) if (rel.startsWith("src/")) names.add(rel);
  const per = Math.floor(budget / names.size);
  return [...names].map((rel) => {
    let text = env.baseline[rel];
    if (text.length > per) text = text.slice(0, per) + "\n[... cut: the file is longer ...]\n";
    return { rel, text };
  });
}

function buildPrompt(task, env, budget) {
  const head = "User: " + task.prompt + " The tests run with `" + TEST_COMMAND + "`.\n" +
    "To change a file, write File: followed by its path, then its complete new content " +
    "in a ``` block. Answer a question in plain text.\n";
  const tail = "Assistant:";
  const files = shownFiles(task, env, Math.max(200, budget - head.length - tail.length));
  const body = files.map((f) => "File: " + f.rel + "\n```\n" + f.text.replace(/\n$/, "") + "\n```\n").join("");
  return { prompt: head + body + tail, shown: files.map((f) => f.rel) };
}

// "File: <path>" then a fenced block; paths stay inside the copy.
function applyReply(env, reply) {
  const written = [];
  const re = /File:\s*`?([\w./-]+)`?\s*\n```[\w-]*\n([\s\S]*?)\n?```/g;
  for (const m of reply.matchAll(re)) {
    const rel = path.normalize(m[1]);
    if (path.isAbsolute(rel) || rel.startsWith("..")) continue;
    const file = path.join(env.repo, rel);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, m[2] + "\n");
    written.push(rel);
  }
  return written;
}

function generate(opt, prompt) {
  const t0 = Date.now();
  const r = spawnSync(opt.bin, ["lm", "generate", "--model", opt.model, "--prompt", prompt,
    "--tokens", String(opt.tokens), "--temperature", String(opt.temperature),
    "--top-k", "40", "--top-p", "0.9"],
  { cwd: ROOT, encoding: "utf8", env: { ...process.env, ANN_GPU_ONLY: "1" },
    maxBuffer: 64 << 20, timeout: 600000 });
  const out = r.stdout || "";
  if (r.status !== 0 || !out.startsWith(prompt))
    throw new Error("anni failed (exit " + r.status + "): " + (r.stderr || "").trim().split("\n").pop());
  let reply = out.slice(prompt.length).replace(/\n$/, "");
  const stop = reply.search(/\n(User|Assistant):/);
  if (stop >= 0) reply = reply.slice(0, stop);
  return { reply: reply.trim(), seconds: (Date.now() - t0) / 1000 };
}

// Canned replies that must pass, and an empty one that must fail: proves the grading.
const SELFTEST = {
  "fix-clamp": () => "File: src/math.js\n```\n" +
    fs.readFileSync(path.join(PROJECT, "src/math.js"), "utf8").replace(/\n$/, "") + "\n```",
  "explain-retry": () => "It makes 7 attempts; the delay before the second one is 375 ms.",
  "no-harm": () => "",
};

function runTask(task, opt) {
  const env = prepare(task);
  const budget = (opt.ctx - opt.tokens) * 3;
  const { prompt, shown } = buildPrompt(task, env, budget);
  const res = { name: task.name, kind: task.kind, shown, promptChars: prompt.length };
  try {
    const g = opt.selftest ? { reply: SELFTEST[task.name](), seconds: 0 } : generate(opt, prompt);
    res.reply = g.reply;
    res.seconds = g.seconds;
    res.written = applyReply(env, g.reply);
    const files = changedFiles(env);
    res.changed = files.map((f) => f.status + " " + f.path);
    res.checks = task.check(checkContext(env, g.reply, files)).map(([ok, label]) => ({ ok: !!ok, label }));
    res.pass = res.checks.every((c) => c.ok);
  } catch (e) {
    res.error = String(e.message || e);
    res.pass = false;
  }
  fs.rmSync(env.work, { recursive: true, force: true });
  return res;
}

function main() {
  const a = args(process.argv.slice(2));
  const opt = {
    bin: a.bin || "./anni",
    model: a.model,
    ctx: Number(a.ctx || 2048),
    tokens: Number(a.tokens || 512),
    temperature: Number(a.temperature || 0),
    selftest: !!a.selftest,
  };
  let tasks = TASKS;
  if (opt.selftest) tasks = TASKS.filter((t) => t.name in SELFTEST);
  else if (!opt.model) {
    console.error("taskbench: --model is required (or --selftest)");
    process.exit(2);
  }
  if (a.tasks) {
    const want = new Set(String(a.tasks).split(","));
    tasks = tasks.filter((t) => want.has(t.name));
  }
  const results = [];
  for (const task of tasks) {
    const r = runTask(task, opt);
    results.push(r);
    const failed = (r.checks || []).filter((c) => !c.ok).map((c) => c.label);
    console.log((r.pass ? "PASS" : "FAIL") + "  " + r.name.padEnd(20) +
      (r.seconds !== undefined ? (r.seconds.toFixed(1) + " s").padStart(8) : "".padStart(8)) +
      "  " + (r.error || (r.changed.length ? r.changed.join(" ") : "no file changed")) +
      (failed.length ? "  [" + failed.join("; ") + "]" : ""));
  }
  const passed = results.filter((r) => r.pass).length;
  console.log(passed + "/" + results.length + " tasks passed");
  if (opt.selftest) {
    // fix-clamp's canned reply is the file before setup broke it.
    const want = { "fix-clamp": true, "explain-retry": true, "no-harm": false };
    const ok = results.every((r) => r.pass === want[r.name]);
    console.log(ok ? "taskbench selftest passed" : "taskbench selftest FAILED");
    process.exit(ok ? 0 : 1);
  }
  const out = a.out || path.join(ROOT, "logs/taskbench", new Date().toISOString().replace(/[:.]/g, "-") + ".json");
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify({ model: opt.model, ctx: opt.ctx, tokens: opt.tokens,
    temperature: opt.temperature, passed, total: results.length, results }, null, 2) + "\n");
  console.log("results: " + path.relative(ROOT, out));
}

if (require.main === module) main();
module.exports = { prepare, buildPrompt, SELFTEST };
