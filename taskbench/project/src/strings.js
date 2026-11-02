"use strict";

function capitalize(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function slugify(s) {
  return String(s).toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) : s;
}

module.exports = { capitalize, slugify, truncate };
