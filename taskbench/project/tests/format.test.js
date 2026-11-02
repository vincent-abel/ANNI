"use strict";

const assert = require("assert");
const { formatDuration, formatTitle } = require("../src/format");
const { formatClock } = require("../src/time");

module.exports = [
  ["duration minutes", () => assert.strictEqual(formatDuration(65), "1:05")],
  ["duration hours", () => assert.strictEqual(formatDuration(3725), "1:02:05")],
  ["title", () => assert.deepStrictEqual(formatTitle(" intro to x "), { label: "Intro to x", anchor: "#intro-to-x" })],
  ["clock", () => assert.strictEqual(formatClock(9 * 60 + 5), "09:05")],
];
