"use strict";

function pad2(n) {
  return n < 10 ? "0" + n : String(n);
}

// "HH:MM" for a number of minutes since midnight.
function formatClock(minutes) {
  const h = Math.floor(minutes / 60) % 24;
  return pad2(h) + ":" + pad2(minutes % 60);
}

module.exports = { formatClock };
