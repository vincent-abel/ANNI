"use strict";

const assert = require("assert");
const { sum, mean, clamp } = require("../src/math");

module.exports = [
  ["sum", () => assert.strictEqual(sum([1, 2, 3]), 6)],
  ["mean", () => assert.strictEqual(mean([2, 4, 6]), 4)],
  ["mean of empty is NaN", () => assert.ok(Number.isNaN(mean([])))],
  ["clamp inside", () => assert.strictEqual(clamp(5, 0, 10), 5)],
  ["clamp low", () => assert.strictEqual(clamp(-1, 0, 10), 0)],
];
