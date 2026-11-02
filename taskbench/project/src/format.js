"use strict";

const { slugify, capitalize } = require("./strings");

function pad2(n) {
  return n < 10 ? "0" + n : String(n);
}

// "M:SS" for a number of seconds, "H:MM:SS" from one hour.
function formatDuration(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return h ? h + ":" + pad2(m) + ":" + pad2(s) : m + ":" + pad2(s);
}

function formatTitle(title) {
  return { label: capitalize(title.trim()), anchor: "#" + slugify(title) };
}

module.exports = { formatDuration, formatTitle };
