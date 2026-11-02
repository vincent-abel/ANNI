"use strict";

function sum(values) {
  let total = 0;
  for (const v of values) total += v;
  return total;
}

function mean(values) {
  if (!values.length) return NaN;
  return sum(values) / values.length;
}

function clamp(value, min, max) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

module.exports = { sum, mean, clamp };
