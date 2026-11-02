"use strict";

const DEFAULTS = {
  maxAttempts: 7,
  baseDelayMs: 375,
  factor: 2,
};

function delayFor(attempt, opts) {
  return opts.baseDelayMs * Math.pow(opts.factor, attempt - 1);
}

// Calls fn until it returns without throwing or the attempts run out. Synchronous: the
// delays are returned for the caller to honour, not slept.
function withRetry(fn, options) {
  const opts = { ...DEFAULTS, ...(options || {}) };
  const delays = [];
  for (let attempt = 1; ; attempt++) {
    try {
      return { value: fn(attempt), delays };
    } catch (e) {
      if (attempt >= opts.maxAttempts) throw e;
      delays.push(delayFor(attempt, opts));
    }
  }
}

module.exports = { withRetry, DEFAULTS };
