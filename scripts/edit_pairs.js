#!/usr/bin/env node
// Contrastive replies for scripts/lm_edit (LM_EDIT_PAIRS): for each taskbench task, the
// bench prompt, an "edit" reply that starts in the format the bench applies, and a "talk"
// reply, the model's own greedy reply from a taskbench results file.
//
//   node scripts/edit_pairs.js logs/taskbench/talk170m-2k-chat.json > logs/edit/pairs.txt
//
// The edit reply of a task that changes files is "File: <first shown file>" and the start
// of that file in a ``` block; of the read-only question, the answer of the selftest.
// Records end with \x1e; name, prompt, edit and talk are split by \x1f.
"use strict";

const fs = require("fs");
const path = require("path");
const { TASKS } = require("../taskbench/tasks");
const { prepare, buildPrompt, SELFTEST } = require("../taskbench/run");

const LINES = 6;	// lines of the shown file in the edit reply

const results = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const ctx = results.ctx || 2048;
const tokens = results.tokens || 512;
const byName = Object.fromEntries(results.results.map((r) => [r.name, r]));
const out = [];
for (const task of TASKS) {
  const r = byName[task.name];
  if (!r || !r.reply) continue;
  const env = prepare(task);
  const { prompt, shown } = buildPrompt(task, env, (ctx - tokens) * 3);
  let edit;
  if (task.kind === "read-only question") edit = SELFTEST[task.name]();
  else {
    const head = env.baseline[shown[0]].split("\n").slice(0, LINES).join("\n");
    edit = "File: " + shown[0] + "\n```\n" + head + "\n";
  }
  fs.rmSync(env.work, { recursive: true, force: true });
  out.push([task.name, prompt, " " + edit, " " + r.reply].join("\x1f") + "\x1e");
}
process.stdout.write(out.join(""));
console.error("edit_pairs: " + out.length + " tasks");
