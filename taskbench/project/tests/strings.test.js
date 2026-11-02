"use strict";

const assert = require("assert");
const { capitalize, slugify, truncate } = require("../src/strings");

module.exports = [
  ["capitalize", () => assert.strictEqual(capitalize("hello"), "Hello")],
  ["slugify", () => assert.strictEqual(slugify("  Hello, World! "), "hello-world")],
  ["truncate short", () => assert.strictEqual(truncate("abc", 5), "abc")],
];
