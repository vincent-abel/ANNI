"use strict";

const assert = require("assert");
const { withRetry } = require("../src/net/retry");

module.exports = [
  ["retry succeeds on third attempt", () => {
    const r = withRetry((n) => { if (n < 3) throw new Error("no"); return n; }, { baseDelayMs: 10 });
    assert.deepStrictEqual(r, { value: 3, delays: [10, 20] });
  }],
  ["retry gives up", () => {
    assert.throws(() => withRetry(() => { throw new Error("always"); }, { maxAttempts: 2 }), /always/);
  }],
];
