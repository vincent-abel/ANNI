#!/usr/bin/env node
// Asks an AnnI model the 20 general and 20 code questions of questions.js, one turn each,
// and grades the answers; run it on each new model to see what improved.
//
//   node qbench/run.js --model models/talk170m-2k-chat.bin
//   node qbench/run.js --model M --only code --tokens 256
//   node qbench/run.js --compare logs/qbench/A.json logs/qbench/B.json
//   node qbench/run.js --selftest        # grading check with reference answers, no model
//
// Options: --bin (anni binary, default ./anni), --tokens (reply budget, default 64 for
// general, 256 for code), --temperature (default 0: greedy), --only (general | code),
// --gpu-only (fail instead of generating on the CPU), --out (results JSON, default
// logs/qbench/<model>-<time>.json).
//
// The code of an answer runs on this machine (python3 -I or node, in a temporary directory,
// stdin closed, 10 s limit): read the answers of an unknown model before trusting it more.
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");
const { GENERAL, CODE } = require("./questions");

const ROOT = path.resolve(__dirname, "..");

function args(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) { out._.push(a); continue; }
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) out[a.slice(2)] = true;
    else out[a.slice(2)] = argv[++i];
  }
  return out;
}

function generate(opt, prompt, tokens) {
  const env = { ...process.env };
  if (opt.gpuOnly) env.ANN_GPU_ONLY = "1";
  const t0 = Date.now();
  const r = spawnSync(opt.bin, ["lm", "generate", "--model", opt.model, "--prompt", prompt,
    "--tokens", String(tokens), "--temperature", String(opt.temperature),
    "--top-k", "40", "--top-p", "0.9"],
  { cwd: ROOT, encoding: "utf8", env, maxBuffer: 64 << 20, timeout: 600000 });
  const out = r.stdout || "";
  if (r.status !== 0 || !out.startsWith(prompt))
    throw new Error("anni failed (exit " + r.status + "): " + (r.stderr || "").trim().split("\n").pop());
  let reply = out.slice(prompt.length).replace(/\n$/, "");
  const stop = reply.search(/\n(User|Assistant):/);
  if (stop >= 0) reply = reply.slice(0, stop);
  return { reply: reply.trim(), seconds: (Date.now() - t0) / 1000 };
}

// First fenced block, any language tag; a block cut by the token budget counts too.
function codeBlock(reply) {
  const m = reply.match(/```[\w+-]*\n([\s\S]*?)(?:```|$)/);
  return m ? m[1] : null;
}

function runCode(lang, code, tests) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anni-qbench-"));
  let file;
  let cmd;
  if (lang === "python") {
    file = path.join(dir, "answer.py");
    fs.writeFileSync(file, code + "\n\n" + tests + "\n");
    cmd = ["python3", ["-I", file]];
  } else {
    file = path.join(dir, "answer.js");
    // An answer written as a module still defines its functions.
    const body = code.replace(/^export\s+(default\s+)?/gm, "");
    fs.writeFileSync(file, 'const assert = require("assert");\n' + body + "\n\n" + tests + "\n");
    cmd = [process.execPath, [file]];
  }
  const r = spawnSync(cmd[0], cmd[1], { cwd: dir, encoding: "utf8", timeout: 10000,
    stdio: ["ignore", "pipe", "pipe"] });
  fs.rmSync(dir, { recursive: true, force: true });
  // The error line itself, not node's trailing version banner.
  const lines = (r.stderr || "").trim().split("\n").filter((l) => l);
  const err = lines.filter((l) => /Error/.test(l)).pop() || lines.pop() || "";
  // A syntax error means the code never ran; an assertion or other error means it ran.
  const ran = r.status === 0 || !/SyntaxError|IndentationError/.test(r.stderr || "");
  return { pass: r.status === 0, ran, error: r.status === 0 ? "" : (r.error ? "timeout" : err) };
}

function prompt(q) {
  return "User: " + q + "\nAssistant:";
}

function askGeneral(item, opt) {
  const res = { name: item.name, q: item.q };
  try {
    const g = opt.selftest ? { reply: item.selftest, seconds: 0 }
      : generate(opt, prompt(item.q), opt.tokens || 64);
    res.reply = g.reply;
    res.seconds = g.seconds;
    res.pass = new RegExp(item.want.source, "i").test(g.reply);
  } catch (e) {
    res.error = String(e.message || e);
    res.pass = false;
  }
  return res;
}

function askCode(item, opt) {
  const res = { name: item.name, lang: item.lang, q: item.q };
  try {
    const g = opt.selftest ? { reply: item.selftest, seconds: 0 }
      : generate(opt, prompt(item.q), opt.tokens || 256);
    res.reply = g.reply;
    res.seconds = g.seconds;
    const code = codeBlock(g.reply);
    res.block = code !== null;
    if (res.block) Object.assign(res, runCode(item.lang, code, item.tests));
    else { res.pass = false; res.ran = false; }
  } catch (e) {
    res.error = String(e.message || e);
    res.pass = false;
  }
  return res;
}

function line(r) {
  const extra = r.error ? "  [" + r.error.slice(0, 80) + "]" : "";
  const reply = (r.reply || "").replace(/\s+/g, " ").slice(0, 70);
  return (r.pass ? "PASS" : "FAIL") + "  " + r.name.padEnd(22) +
    (r.seconds !== undefined ? (r.seconds.toFixed(1) + " s").padStart(7) : "".padStart(7)) +
    "  " + reply + extra;
}

function summary(res) {
  const g = res.general || [];
  const c = res.code || [];
  const n = (xs, f) => xs.filter(f).length;
  return {
    general: n(g, (r) => r.pass) + "/" + g.length,
    code: n(c, (r) => r.pass) + "/" + c.length,
    codeBlock: n(c, (r) => r.block) + "/" + c.length,
    codeRan: n(c, (r) => r.ran) + "/" + c.length,
  };
}

function compare(files) {
  const runs = files.map((f) => JSON.parse(fs.readFileSync(f, "utf8")));
  const names = runs.map((r) => path.basename(r.model || "?"));
  const w = Math.max(...names.map((n) => n.length), 8);
  console.log("".padEnd(24) + names.map((n) => n.padStart(w + 2)).join(""));
  for (const part of ["general", "code"]) {
    for (const item of runs[0][part] || []) {
      const cells = runs.map((r) => {
        const x = (r[part] || []).find((y) => y.name === item.name);
        return (x ? (x.pass ? "PASS" : "-") : "?").padStart(w + 2);
      });
      console.log(item.name.padEnd(24) + cells.join(""));
    }
  }
  for (const key of ["general", "code", "codeBlock", "codeRan"])
    console.log(("total " + key).padEnd(24) + runs.map((r) => summary(r)[key].padStart(w + 2)).join(""));
}

function main() {
  const a = args(process.argv.slice(2));
  if (a.compare) {
    compare([a.compare, ...a._]);
    return;
  }
  const opt = {
    bin: a.bin || "./anni",
    model: a.model,
    tokens: a.tokens ? Number(a.tokens) : 0,
    temperature: Number(a.temperature || 0),
    gpuOnly: !!a["gpu-only"],
    selftest: !!a.selftest,
  };
  if (!opt.selftest && !opt.model) {
    console.error("qbench: --model is required (or --selftest, --compare)");
    process.exit(2);
  }
  let general = GENERAL;
  let code = CODE;
  if (opt.selftest) {
    // Reference answers must pass; a talkative answer and a wrong program must fail.
    general = GENERAL.map((x) => ({ ...x, selftest: "The answer: " + x.want.source.replace(/\\b|[()]/g, "").split("|")[0] + "." }));
    code = CODE.map((x) => ({ ...x, selftest: "Here it is:\n```" + x.lang + "\n" + x.ref + "\n```" }));
    general.push({ name: "selftest-wrong", q: "", want: /\bparis\b/, selftest: "Lyon.", expectFail: true });
    code.push({ name: "selftest-wrong", lang: "python", q: "", tests: "assert f() == 1",
      selftest: "```python\ndef f():\n    return 2\n```", expectFail: true });
    code.push({ name: "selftest-noblock", lang: "python", q: "", tests: "",
      selftest: "You can use a loop.", expectFail: true });
  }
  if (a.only === "general") code = [];
  if (a.only === "code") general = [];
  const res = { model: opt.model, temperature: opt.temperature, tokens: opt.tokens || "64/256",
    general: [], code: [] };
  for (const item of general) {
    const r = askGeneral(item, opt);
    r.expectFail = item.expectFail;
    res.general.push(r);
    console.log(line(r));
  }
  for (const item of code) {
    const r = askCode(item, opt);
    r.expectFail = item.expectFail;
    res.code.push(r);
    console.log(line(r));
  }
  if (opt.selftest) {
    const all = [...res.general, ...res.code];
    const ok = all.every((r) => r.pass === !r.expectFail);
    console.log(ok ? "qbench selftest passed" : "qbench selftest FAILED");
    process.exit(ok ? 0 : 1);
  }
  const s = summary(res);
  res.summary = s;
  console.log("general " + s.general + "  code " + s.code + "  (code block " + s.codeBlock +
    ", ran " + s.codeRan + ")");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const out = a.out || path.join(ROOT, "logs/qbench",
    path.basename(opt.model).replace(/\.bin$/, "") + "-" + stamp + ".json");
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(res, null, 2) + "\n");
  console.log("results: " + path.relative(ROOT, out));
}

main();
