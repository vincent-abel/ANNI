"use strict";

// Runs every tests/*.test.js. Each file exports [name, fn] pairs; fn throws on failure.

const fs = require("fs");
const path = require("path");

let failed = 0;
let passed = 0;
const dir = path.join(__dirname, "tests");
for (const file of fs.readdirSync(dir).filter((f) => f.endsWith(".test.js")).sort()) {
  for (const [name, fn] of require(path.join(dir, file))) {
    try {
      fn();
      passed++;
    } catch (e) {
      failed++;
      console.log("FAIL " + file + ": " + name + "\n  " + String(e.message).split("\n").join("\n  "));
    }
  }
}
console.log(passed + " passed, " + failed + " failed");
process.exit(failed ? 1 : 0);
