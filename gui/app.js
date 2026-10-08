(function () {
  "use strict";

  var state = null;
  var viewStyle = "net";
  var pinned = null;
  var hover = null;
  var hit = {nodes: [], edges: []};
  var playing = false;
  var playLoopOn = false;
  var wantSample = null;
  var tail = Promise.resolve();
  var inflight = 0;
  var logLines = [];
  var formSynced = false;

  var form = document.getElementById("form");
  var datasetEl = document.getElementById("dataset");
  var inputsEl = document.getElementById("inputs");
  var sizesEl = document.getElementById("sizes");
  var lrEl = document.getElementById("lr");
  var seedEl = document.getElementById("seed");
  var stepsEl = document.getElementById("steps");
  var sampleInEl = document.getElementById("sample-in");
  var sampleTgEl = document.getElementById("sample-tg");
  var customIn = document.getElementById("custom-in-label");
  var customTg = document.getElementById("custom-tg-label");
  var errEl = document.getElementById("err");
  var hintEl = document.getElementById("hint");
  var launchBtn = document.getElementById("btn-launch");
  var stepBtn = document.getElementById("btn-step");
  var runBtn = document.getElementById("btn-run");
  var playBtn = document.getElementById("btn-play");
  var sampleBtn = document.getElementById("btn-sample");
  var accEl = document.getElementById("acc");
  var samplesEl = document.getElementById("samples");
  var fineEl = document.getElementById("fine");
  var inspectEl = document.getElementById("inspect");
  var logEl = document.getElementById("log");
  var mseEl = document.getElementById("mse");
  var epochEl = document.getElementById("epoch");
  var netCanvas = document.getElementById("net");
  var lossCanvas = document.getElementById("loss");
  var heatEl = document.getElementById("heat");
  var tipEl = document.getElementById("tip");
  var styleNet = document.getElementById("style-net");
  var styleFlow = document.getElementById("style-flow");

  function enqueue(task) {
    inflight++;
    var p = tail.then(task, task).then(function (v) {
      inflight--;
      return v;
    }, function (e) {
      inflight--;
      throw e;
    });
    tail = p.then(function () {}, function () {});
    return p;
  }

  function api(path, payload) {
    return fetch(path, {
      method: path === "/api/state" && !payload ? "GET" : "POST",
      headers: {"Content-Type": "application/json"},
      body: path === "/api/state" && !payload ? undefined : JSON.stringify(payload || {})
    }).then(function (res) {
      return res.json().catch(function () {
        throw new Error("The trainer returned something that is not JSON.");
      }).then(function (data) {
        if (!res.ok || data.ok === false)
          throw new Error(data.error || ("HTTP " + res.status));
        return data;
      });
    });
  }

  function onError(e) {
    errEl.hidden = false;
    if (e instanceof TypeError)
      errEl.textContent = "The trainer server is not running. Start it with python3 gui/server.py";
    else
      errEl.textContent = e.message || String(e);
  }

  function clearError() {
    errEl.hidden = true;
    errEl.textContent = "";
  }

  function logLine(text) {
    logLines.push(text);
    while (logLines.length > 14)
      logLines.shift();
    logEl.textContent = logLines.join("\n");
    logEl.scrollTop = logEl.scrollHeight;
  }

  function fmtNum(v) {
    if (!isFinite(v))
      return "0";
    if (Math.abs(v - Math.round(v)) < 1e-6)
      return String(Math.round(v));
    return v.toFixed(3);
  }

  function parseSizes(text) {
    var parts = text.trim().split(/[\s,]+/).filter(Boolean).map(Number);
    if (!parts.length || parts.some(function (n) { return !Number.isInteger(n) || n < 1; }))
      throw new Error("Layer sizes must be positive integers, for example 4,1.");
    if (parts.length > 8)
      throw new Error("At most 8 layers.");
    if (parts.some(function (n) { return n > 64; }))
      throw new Error("Each layer can have at most 64 neurons.");
    return parts;
  }

  function readVector(text, name) {
    var parts = text.trim().split(/[\s,]+/).filter(Boolean).map(Number);
    if (!parts.length || parts.some(function (n) { return !isFinite(n); }))
      throw new Error(name + " must be numbers separated by commas.");
    return parts;
  }

  function readLr() {
    var n = Number(lrEl.value);
    if (!(n > 0) || !isFinite(n))
      throw new Error("Learning rate must be a positive number.");
    return n;
  }

  function readSteps() {
    var n = Number(stepsEl.value);
    if (!Number.isInteger(n) || n < 1)
      throw new Error("Steps must be a positive integer.");
    if (n > 200000)
      throw new Error("Steps are capped at 200000 per run.");
    return n;
  }

  function readForm() {
    var dataset = datasetEl.value;
    var inputs = dataset === "xor" ? 2 : Number(inputsEl.value);
    var sizes = parseSizes(sizesEl.value);
    var seed = Number(seedEl.value);
    var sampleIn = null;
    var sampleTg = null;
    if (!Number.isInteger(inputs) || inputs < 1 || inputs > 784)
      throw new Error("Inputs must be an integer from 1 to 784.");
    if (!Number.isInteger(seed) || seed < 0)
      throw new Error("Seed must be a non-negative integer.");
    if (dataset === "xor" && sizes[sizes.length - 1] !== 1)
      throw new Error("XOR needs 2 inputs and a last layer of size 1.");
    if (dataset === "custom") {
      sampleIn = readVector(sampleInEl.value, "Sample inputs");
      sampleTg = readVector(sampleTgEl.value, "Targets");
      if (sampleIn.length !== inputs)
        throw new Error("Sample inputs must have " + inputs + " numbers.");
      if (sampleTg.length !== sizes[sizes.length - 1])
        throw new Error("Targets must have " + sizes[sizes.length - 1] + " numbers.");
    }
    return {
      dataset: dataset,
      inputs: inputs,
      sizes: sizes,
      seed: seed,
      lr: readLr(),
      sampleIn: sampleIn,
      sampleTg: sampleTg
    };
  }

  function updateHint(extra) {
    if (extra) {
      hintEl.textContent = extra;
      return;
    }
    var sizes = sizesEl.value.replace(/\s+/g, "");
    var lr = Number(lrEl.value);
    var seed = Number(seedEl.value);
    if (datasetEl.value === "xor" && sizes === "4,1" && seed === 1 && lr === 0.5)
      hintEl.textContent = "Seed 1, layers 4,1, learning rate 0.5 — same start as ./ann xor. A step is one epoch. Space runs the burst, N steps once.";
    else
      hintEl.textContent = "A step is one epoch over the samples. Space runs the burst, N steps once.";
  }

  function syncDataset() {
    var xor = datasetEl.value === "xor";
    inputsEl.disabled = xor;
    if (xor)
      inputsEl.value = "2";
    customIn.hidden = xor;
    customTg.hidden = xor;
    sampleBtn.hidden = xor;
  }

  function syncRunLabel() {
    var n = Number(stepsEl.value);
    runBtn.textContent = "Run " + (Number.isInteger(n) && n > 0 ? n : "…");
  }

  function syncPlayBtn() {
    var live = state && state.ready;
    if (playing) {
      playBtn.textContent = "Stop";
      playBtn.classList.add("active");
    } else if (playLoopOn) {
      playBtn.textContent = "Stopping";
      playBtn.classList.remove("active");
    } else {
      playBtn.textContent = "Play";
      playBtn.classList.remove("active");
    }
    stepBtn.disabled = !live || playLoopOn;
    runBtn.disabled = !live || playLoopOn;
    playBtn.disabled = !live;
    sampleBtn.hidden = datasetEl.value !== "custom";
    sampleBtn.disabled = !live;
    launchBtn.disabled = false;
  }

  function correct(row) {
    return row.tg.every(function (t, i) {
      return (row.out[i] >= 0.5) === (t >= 0.5);
    });
  }

  function accuracy(s) {
    var ok = 0;
    s.rows.forEach(function (r) {
      if (correct(r))
        ok++;
    });
    return {ok: ok, n: s.rows.length};
  }

  function sampleName(s, row) {
    if (s.dataset === "xor" && row.in.length === 2)
      return fmtNum(row.in[0]) + "  ⊕  " + fmtNum(row.in[1]) + "  →  " + fmtNum(row.tg[0]);
    return row.in.map(fmtNum).join("  ") + "  →  " + row.tg.map(fmtNum).join(" ");
  }

  function renderSamples() {
    samplesEl.replaceChildren();
    if (!state || !state.ready) {
      accEl.textContent = "";
      fineEl.hidden = true;
      return;
    }
    var acc = accuracy(state);
    accEl.textContent = acc.ok + " / " + acc.n + " correct";
    fineEl.hidden = state.rows[0].out.length !== 1;
    state.rows.forEach(function (row, i) {
      var btn = document.createElement("button");
      var top = document.createElement("div");
      var name = document.createElement("span");
      var meta = document.createElement("span");
      btn.type = "button";
      btn.className = "sample" + (i === state.sample ? " on" : "");
      top.className = "sample-top";
      name.textContent = sampleName(state, row);
      meta.textContent = row.out.map(function (v) { return v.toFixed(3); }).join("  ");
      meta.className = correct(row) ? "ok" : "bad";
      top.appendChild(name);
      top.appendChild(meta);
      btn.appendChild(top);
      if (row.out.length === 1) {
        var meter = document.createElement("div");
        var bar = document.createElement("span");
        var tick = document.createElement("i");
        var pred = Math.max(0, Math.min(1, row.out[0]));
        var tg = Math.max(0, Math.min(1, row.tg[0]));
        meter.className = "meter";
        bar.style.width = (pred * 100) + "%";
        tick.style.left = (tg * 100) + "%";
        meter.appendChild(bar);
        meter.appendChild(tick);
        btn.appendChild(meter);
      }
      btn.addEventListener("click", function () { onFocusSample(i); });
      samplesEl.appendChild(btn);
    });
  }

  function columnTitle(i, nCols) {
    if (i === 0)
      return "input";
    if (i === nCols - 1)
      return "output";
    if (nCols <= 3)
      return "hidden";
    return "hidden " + i;
  }

  function layerName(i, n) {
    if (n === 1 || i === n - 1)
      return "output";
    if (n === 2)
      return "hidden";
    return "hidden " + (i + 1);
  }

  function renderInspect() {
    var col, neuron, layer, title, weights, list;
    inspectEl.replaceChildren();
    if (!pinned || !state || !state.ready) {
      inspectEl.textContent = "Click a neuron or a wire.";
      return;
    }
    col = pinned.col;
    neuron = pinned.neuron;
    if (col === 0) {
      addKV("input " + neuron, state.rows[state.sample].in[neuron], false);
      return;
    }
    layer = state.layers[col - 1];
    if (!layer || neuron < 0 || neuron >= layer.a.length) {
      inspectEl.textContent = "Click a neuron or a wire.";
      pinned = null;
      return;
    }
    title = document.createElement("div");
    title.className = "inspect-title";
    title.textContent = columnTitle(col, state.layers.length + 1) + "  " + neuron;
    inspectEl.appendChild(title);
    addKV("activation", layer.a[neuron], false);
    addKV("bias", layer.b[neuron], true);
    weights = layer.w[neuron].map(function (w, i) { return {w: w, i: i}; });
    weights.sort(function (a, b) { return Math.abs(b.w) - Math.abs(a.w); });
    list = document.createElement("div");
    list.className = "wlist";
    weights.forEach(function (item) {
      var row = document.createElement("div");
      var src = col === 1 ? "input " + item.i : "neuron " + item.i;
      var sign = item.w >= 0 ? "+" : "";
      row.textContent = src + "    " + sign + item.w.toFixed(4);
      row.className = item.w >= 0 ? "pos" : "neg";
      list.appendChild(row);
    });
    inspectEl.appendChild(list);
  }

  function addKV(k, v, signed) {
    var d = document.createElement("div");
    var a = document.createElement("span");
    var b = document.createElement("span");
    d.className = "kv";
    a.textContent = k;
    b.textContent = (signed && v >= 0 ? "+" : "") + Number(v).toFixed(4);
    if (signed)
      b.className = v >= 0 ? "pos" : "neg";
    d.appendChild(a);
    d.appendChild(b);
    inspectEl.appendChild(d);
  }

  function mix(a, b, t) {
    return [
      Math.round(a[0] + (b[0] - a[0]) * t),
      Math.round(a[1] + (b[1] - a[1]) * t),
      Math.round(a[2] + (b[2] - a[2]) * t)
    ];
  }

  function renderHeat() {
    heatEl.replaceChildren();
    if (!state || !state.ready)
      return;
    state.layers.forEach(function (layer, i) {
      var fig = document.createElement("figure");
      var cap = document.createElement("figcaption");
      var canvas = document.createElement("canvas");
      var rows = layer.w.length;
      var cols = rows && layer.w[0] ? layer.w[0].length : 0;
      var max = 0.5;
      var cell, ctx, img, r, c, t, col, y, x, p;
      fig.appendChild(canvas);
      fig.appendChild(cap);
      heatEl.appendChild(fig);
      if (!rows || !cols) {
        cap.textContent = layerName(i, state.layers.length);
        return;
      }
      for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++)
          max = Math.max(max, Math.abs(layer.w[r][c]));
      }
      cell = Math.max(1, Math.min(14, Math.floor(260 / cols), Math.floor(110 / rows)));
      canvas.width = cols * cell;
      canvas.height = rows * cell;
      ctx = canvas.getContext("2d");
      img = ctx.createImageData(canvas.width, canvas.height);
      for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++) {
          t = Math.max(-1, Math.min(1, layer.w[r][c] / max));
          col = t >= 0 ? mix([22, 22, 24], [90, 190, 255], t) : mix([22, 22, 24], [255, 92, 70], -t);
          for (y = 0; y < cell; y++) {
            for (x = 0; x < cell; x++) {
              p = ((r * cell + y) * canvas.width + (c * cell + x)) * 4;
              img.data[p] = col[0];
              img.data[p + 1] = col[1];
              img.data[p + 2] = col[2];
              img.data[p + 3] = 255;
            }
          }
        }
      }
      ctx.putImageData(img, 0, 0);
      cap.textContent = layerName(i, state.layers.length) + "  " + rows + "×" + cols + "  max " + max.toFixed(2);
    });
  }

  function fit(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width * dpr));
    var h = Math.max(1, Math.round(rect.height * dpr));
    var ctx;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {ctx: ctx, w: rect.width, h: rect.height};
  }

  function drawLoss() {
    var box = fit(lossCanvas);
    var ctx = box.ctx;
    var w = box.w;
    var h = box.h;
    var xs, ys, padL, padR, padT, padB, ymax, xmin, xmax, span, i, x, y;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#070707";
    ctx.fillRect(0, 0, w, h);
    if (w < 2 || h < 2)
      return;
    xs = state && state.loss_epoch ? state.loss_epoch : [];
    ys = state && state.loss ? state.loss : [];
    padL = 52;
    padR = 14;
    padT = 14;
    padB = 22;
    ctx.font = "11px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    if (!ys.length) {
      ctx.fillStyle = "#8d887f";
      ctx.textAlign = "center";
      ctx.fillText("Loss shows up after launch.", w / 2, h / 2);
      return;
    }
    ymax = 0;
    for (i = 0; i < ys.length; i++)
      ymax = Math.max(ymax, ys[i]);
    if (ymax <= 0)
      ymax = 1;
    ymax *= 1.08;
    xmin = xs[0];
    xmax = xs[xs.length - 1];
    span = Math.max(1, xmax - xmin);
    function X(ep) { return padL + (ep - xmin) / span * (w - padL - padR); }
    function Y(v) { return padT + (1 - v / ymax) * (h - padT - padB); }
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#8d887f";
    ctx.textAlign = "right";
    for (i = 0; i <= 3; i++) {
      y = Y(ymax * i / 3);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillText((ymax * i / 3).toFixed(2), padL - 6, y);
    }
    ctx.beginPath();
    for (i = 0; i < ys.length; i++) {
      x = X(xs[i]);
      y = Y(ys[i]);
      if (i === 0)
        ctx.moveTo(x, y);
      else
        ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#e4c27a";
    ctx.lineWidth = 1.6;
    ctx.stroke();
    if (ys.length === 1) {
      ctx.beginPath();
      ctx.arc(X(xs[0]), Y(ys[0]), 3, 0, Math.PI * 2);
      ctx.fillStyle = "#e4c27a";
      ctx.fill();
    } else {
      ctx.lineTo(X(xs[xs.length - 1]), h - padB);
      ctx.lineTo(X(xs[0]), h - padB);
      ctx.closePath();
      ctx.fillStyle = "rgba(228, 194, 122, 0.13)";
      ctx.fill();
    }
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#8d887f";
    ctx.textAlign = "left";
    ctx.fillText(String(xmin), padL, h - 6);
    ctx.textAlign = "right";
    ctx.fillText("epoch " + xmax, w - padR, h - 6);
  }

  function gridShape(n) {
    var c, a;
    a = 1;
    for (c = 1; c * c <= n; c++) {
      if (n % c === 0)
        a = c;
    }
    return [a, n / a];
  }

  function visibleIndices(count) {
    var idx, i;
    if (count <= 12) {
      idx = [];
      for (i = 0; i < count; i++)
        idx.push(i);
      return idx;
    }
    idx = [];
    for (i = 0; i < 5; i++)
      idx.push(i);
    idx.push(-1);
    for (i = count - 5; i < count; i++)
      idx.push(i);
    return idx;
  }

  function spread(n, top, bottom) {
    var ys = [];
    var gap, y0, i, avail;
    if (n <= 1) {
      ys.push((top + bottom) / 2);
      return ys;
    }
    avail = bottom - top;
    gap = Math.min(76, avail / (n - 1));
    y0 = top + (avail - gap * (n - 1)) / 2;
    for (i = 0; i < n; i++)
      ys.push(y0 + gap * i);
    return ys;
  }

  function roundRect(ctx, x, y, w, h, r) {
    var rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function drawBrace(ctx, x, y0, y1) {
    var mid = (y0 + y1) / 2;
    ctx.beginPath();
    ctx.moveTo(x + 14, y0);
    ctx.bezierCurveTo(x + 2, y0, x + 2, y0 + 16, x + 2, mid - 18);
    ctx.bezierCurveTo(x + 2, mid - 8, x - 8, mid - 8, x - 8, mid);
    ctx.bezierCurveTo(x - 8, mid + 8, x + 2, mid + 8, x + 2, mid + 18);
    ctx.bezierCurveTo(x + 2, y1 - 16, x + 2, y1, x + 14, y1);
    ctx.stroke();
  }

  function nodeRadius(n, style) {
    if (style === "flow")
      return Math.max(11, Math.min(18, 210 / Math.max(3, n)));
    return Math.max(7, Math.min(16, 190 / Math.max(3, n)));
  }

  function drawNeuron(ctx, x, y, r, a) {
    var act = Math.max(0, Math.min(1, a));
    var g, grad;
    if (viewStyle === "flow") {
      grad = ctx.createLinearGradient(x - r, y - r, x + r, y + r);
      grad.addColorStop(0, "#ffffff");
      grad.addColorStop(0.42, act > 0.45 ? "#d7e8ff" : "#ffe6c2");
      grad.addColorStop(1, act > 0.45 ? "#3c7dff" : "#f0a24b");
      roundRect(ctx, x - r, y - r, r * 2, r * 2, Math.max(4, r * 0.28));
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#161616";
      ctx.stroke();
      return;
    }
    g = Math.round(42 + act * 213);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = "rgb(" + g + "," + g + "," + g + ")";
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = act > 0.55 ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.28)";
    ctx.stroke();
  }

  function drawEmpty() {
    var box = fit(netCanvas);
    var ctx = box.ctx;
    ctx.clearRect(0, 0, box.w, box.h);
    ctx.fillStyle = viewStyle === "flow" ? "#f3f1ec" : "#000";
    ctx.fillRect(0, 0, box.w, box.h);
    if (box.w < 2)
      return;
    ctx.fillStyle = viewStyle === "flow" ? "#5c6570" : "#8d887f";
    ctx.font = "15px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Launch the network to see weights and activations.", box.w / 2, box.h / 2);
  }

  function drawNet() {
    var box = fit(netCanvas);
    var ctx = box.ctx;
    var w = box.w;
    var h = box.h;
    var row, inputs, targets, layers, shape, colsN, rowsN, cell, gapPx, frameW, frameH;
    var fx, fy, maxFrame, c, r, t, gv, frameRight, frameTop, frameBot;
    var top, bottom, counts, visibles, nCols, netLeft, netRight, xs, colNodes, i, k;
    var ys, radius, ink, mute, pairs, maxAbs, mag, style, e, dst, src, W;
    var collapsed, colTop, colBot, rr, act, title, gap, nOut, node, ring, label;

    hit = {nodes: [], edges: []};
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = viewStyle === "flow" ? "#f3f1ec" : "#000";
    ctx.fillRect(0, 0, w, h);
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;
    if (!state || !state.ready || w < 2 || h < 2) {
      drawEmpty();
      return;
    }

    row = state.rows[state.sample];
    inputs = row.in;
    targets = row.tg;
    layers = state.layers;
    ink = viewStyle === "flow" ? "#243044" : "#f3f0e8";
    mute = viewStyle === "flow" ? "#5c6570" : "#9a958c";
    top = 64;
    bottom = h - 46;

    shape = gridShape(inputs.length);
    rowsN = shape[0];
    colsN = shape[1];
    maxFrame = Math.min(inputs.length <= 4 ? 120 : 150, Math.max(48, w * 0.2));
    cell = inputs.length <= 4 ? 42 : (inputs.length <= 16 ? 22 : 14);
    cell = Math.max(3, Math.min(cell, Math.floor(maxFrame / colsN)));
    gapPx = inputs.length <= 4 ? 8 : 0;
    frameW = colsN * cell + (colsN - 1) * gapPx;
    frameH = rowsN * cell + (rowsN - 1) * gapPx;
    if (frameH > bottom - top) {
      cell = Math.max(3, Math.floor((bottom - top) / rowsN) - gapPx);
      frameW = colsN * cell + (colsN - 1) * gapPx;
      frameH = rowsN * cell + (rowsN - 1) * gapPx;
    }
    fx = 26;
    fy = top + Math.max(0, (bottom - top - frameH) / 2);
    ctx.fillStyle = viewStyle === "flow" ? "#fbfbf8" : "#050505";
    ctx.fillRect(fx - 8, fy - 8, frameW + 16, frameH + 16);
    var inLo = inputs[0];
    var inHi = inputs[0];
    for (i = 1; i < inputs.length; i++) {
      if (inputs[i] < inLo) inLo = inputs[i];
      if (inputs[i] > inHi) inHi = inputs[i];
    }
    for (i = 0; i < inputs.length; i++) {
      c = i % colsN;
      r = Math.floor(i / colsN);
      t = inputs[i];
      if (inLo < -0.001 || inHi > 1.001)
        t = inHi === inLo ? 0.5 : (inputs[i] - inLo) / (inHi - inLo);
      t = Math.max(0, Math.min(1, t));
      gv = Math.round(32 + t * 223);
      ctx.fillStyle = "rgb(" + gv + "," + gv + "," + gv + ")";
      ctx.fillRect(fx + c * (cell + gapPx), fy + r * (cell + gapPx), cell, cell);
      ctx.strokeStyle = viewStyle === "flow" ? "rgba(0,0,0,0.18)" : "rgba(255,255,255,0.22)";
      ctx.lineWidth = 1;
      ctx.strokeRect(fx + c * (cell + gapPx) + 0.5, fy + r * (cell + gapPx) + 0.5, cell - 1, cell - 1);
    }
    if (inputs.length <= 4) {
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillStyle = mute;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(inputs.map(fmtNum).join("   "), fx + frameW / 2, fy + frameH + 14);
    }
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = viewStyle === "flow" ? "#1c1c1c" : "#3ec8ff";
    ctx.strokeRect(fx - 8, fy - 8, frameW + 16, frameH + 16);
    frameRight = fx + frameW + 8;
    frameTop = fy - 8;
    frameBot = fy + frameH + 8;

    counts = [inputs.length].concat(layers.map(function (L) { return L.a.length; }));
    visibles = counts.map(visibleIndices);
    nCols = counts.length;
    collapsed = visibles[0].some(function (idx) { return idx < 0; });
    netLeft = frameRight + (collapsed ? 86 : 48);
    netRight = w - 86;
    if (netRight < netLeft + 40)
      netRight = netLeft + 40;
    xs = [];
    for (i = 0; i < nCols; i++)
      xs.push(nCols === 1 ? (netLeft + netRight) / 2 : netLeft + (netRight - netLeft) * i / (nCols - 1));

    colNodes = visibles.map(function (idx, col) {
      ys = spread(idx.length, top, bottom);
      return idx.map(function (neuron, k) {
        return {neuron: neuron, x: xs[col], y: ys[k], col: col};
      });
    });

    if (collapsed && colNodes[0].length) {
      colTop = colNodes[0][0].y;
      colBot = colNodes[0][colNodes[0].length - 1].y;
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1.25;
      drawBrace(ctx, frameRight + 36, colTop, colBot);
      ctx.fillStyle = ink;
      ctx.font = "italic 28px Palatino, 'Times New Roman', serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(String(inputs.length), frameRight + 28, (colTop + colBot) / 2);
    }

    for (c = 0; c < layers.length; c++) {
      W = layers[c].w;
      maxAbs = 0.5;
      for (i = 0; i < W.length; i++) {
        for (k = 0; k < W[i].length; k++)
          maxAbs = Math.max(maxAbs, Math.abs(W[i][k]));
      }
      pairs = [];
      for (i = 0; i < colNodes[c + 1].length; i++) {
        dst = colNodes[c + 1][i];
        if (dst.neuron < 0 || !W[dst.neuron])
          continue;
        for (k = 0; k < colNodes[c].length; k++) {
          src = colNodes[c][k];
          if (src.neuron < 0 || src.neuron >= W[dst.neuron].length)
            continue;
          pairs.push({src: src, dst: dst, w: W[dst.neuron][src.neuron]});
        }
      }
      pairs.sort(function (a, b) { return Math.abs(a.w) - Math.abs(b.w); });
      if (pairs.length > 3500)
        pairs = pairs.slice(pairs.length - 3500);
      for (i = 0; i < pairs.length; i++) {
        e = pairs[i];
        mag = Math.min(1, Math.abs(e.w) / maxAbs);
        ctx.beginPath();
        ctx.moveTo(e.src.x, e.src.y);
        var mx = (e.src.x + e.dst.x) / 2;
        ctx.bezierCurveTo(mx, e.src.y, mx, e.dst.y, e.dst.x, e.dst.y);
        if (viewStyle === "flow") {
          ctx.strokeStyle = e.w >= 0
            ? "rgba(47,111,228," + (0.2 + 0.62 * mag) + ")"
            : "rgba(238,141,42," + (0.2 + 0.62 * mag) + ")";
          ctx.lineWidth = 1.3 + 4.2 * mag;
          ctx.setLineDash(e.w < 0 ? [7, 6] : []);
        } else {
          ctx.strokeStyle = e.w >= 0
            ? "rgba(121,200,255," + (0.14 + 0.76 * mag) + ")"
            : "rgba(255,93,74," + (0.14 + 0.76 * mag) + ")";
          ctx.lineWidth = 0.7 + 2.15 * mag;
          ctx.setLineDash([]);
        }
        ctx.stroke();
        hit.edges.push({
          kind: "edge",
          x1: e.src.x,
          y1: e.src.y,
          x2: e.dst.x,
          y2: e.dst.y,
          w: e.w,
          from: columnTitle(e.src.col, nCols) + " " + e.src.neuron,
          to: columnTitle(e.dst.col, nCols) + " " + e.dst.neuron,
          dstCol: e.dst.col,
          dst: e.dst.neuron
        });
      }
      ctx.setLineDash([]);
    }

    if (hover && hover.kind === "edge") {
      for (i = 0; i < hit.edges.length; i++) {
        e = hit.edges[i];
        if (e.from === hover.from && e.to === hover.to) {
          ctx.beginPath();
          ctx.moveTo(e.x1, e.y1);
          var hx = (e.x1 + e.x2) / 2;
          ctx.bezierCurveTo(hx, e.y1, hx, e.y2, e.x2, e.y2);
          ctx.strokeStyle = viewStyle === "flow" ? "#1b1b1b" : "#ffffff";
          ctx.lineWidth = 2.4;
          ctx.setLineDash([]);
          ctx.stroke();
          break;
        }
      }
    }

    for (c = 0; c < nCols; c++) {
      radius = nodeRadius(colNodes[c].length, viewStyle);
      for (i = 0; i < colNodes[c].length; i++) {
        node = colNodes[c][i];
        if (node.neuron < 0) {
          ctx.fillStyle = mute;
          ctx.font = "italic 18px Palatino, 'Times New Roman', serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("· · ·", node.x, node.y);
          ctx.font = "italic 16px Palatino, 'Times New Roman', serif";
          ctx.textAlign = "left";
          ctx.fillText(String(counts[c]), node.x + 22, node.y);
          continue;
        }
        act = c === 0 ? inputs[node.neuron] : layers[c - 1].a[node.neuron];
        if (c === 0)
          act = Math.max(0, Math.min(1, act));
        drawNeuron(ctx, node.x, node.y, radius, act);
        rr = viewStyle === "flow" ? radius * 1.15 : radius;
        if (c === nCols - 1 && targets[node.neuron] != null) {
          ring = (act >= 0.5) === (targets[node.neuron] >= 0.5);
          ctx.beginPath();
          if (viewStyle === "flow")
            roundRect(ctx, node.x - radius - 4, node.y - radius - 4, (radius + 4) * 2, (radius + 4) * 2, 8);
          else
            ctx.arc(node.x, node.y, radius + 5, 0, Math.PI * 2);
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = ring ? "#e4c27a" : "#ff5c45";
          ctx.stroke();
        }
        if (pinned && pinned.col === c && pinned.neuron === node.neuron) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 9, 0, Math.PI * 2);
          ctx.lineWidth = 1;
          ctx.strokeStyle = ink;
          ctx.stroke();
        }
        hit.nodes.push({
          kind: "node",
          x: node.x,
          y: node.y,
          r: rr,
          col: c,
          neuron: node.neuron,
          name: columnTitle(c, nCols) + " " + node.neuron,
          a: act
        });
      }
      ctx.fillStyle = mute;
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      title = columnTitle(c, nCols);
      ctx.fillText(title.toUpperCase(), xs[c], h - 16);
    }

    gap = 40;
    if (colNodes[nCols - 1].length > 1)
      gap = Math.abs(colNodes[nCols - 1][1].y - colNodes[nCols - 1][0].y);
    nOut = layers[layers.length - 1].a.length;
    ctx.textBaseline = "middle";
    for (i = 0; i < colNodes[nCols - 1].length; i++) {
      node = colNodes[nCols - 1][i];
      if (node.neuron < 0)
        continue;
      act = layers[layers.length - 1].a[node.neuron];
      radius = nodeRadius(colNodes[nCols - 1].length, viewStyle);
      ctx.textAlign = "left";
      if (nOut === 1) {
        ctx.font = "13px ui-monospace, monospace";
        ctx.fillStyle = ink;
        ctx.fillText(act.toFixed(2), node.x + radius + 14, node.y - 7);
        ctx.font = "11px ui-monospace, monospace";
        ctx.fillStyle = "#e4c27a";
        ctx.fillText("target " + fmtNum(targets[0]), node.x + radius + 14, node.y + 9);
      } else if (gap >= 18) {
        gv = Math.round(90 + Math.max(0, Math.min(1, act)) * 165);
        ctx.font = "14px Palatino, 'Times New Roman', serif";
        ctx.fillStyle = "rgb(" + gv + "," + gv + "," + gv + ")";
        ctx.fillText(String(node.neuron), node.x + radius + 12, node.y);
      }
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
  }

  function paint() {
    if (!state || !state.ready)
      drawEmpty();
    else
      drawNet();
    if (state && state.ready)
      drawLoss();
    else {
      var box = fit(lossCanvas);
      box.ctx.fillStyle = "#070707";
      box.ctx.fillRect(0, 0, box.w, box.h);
    }
  }

  function render(next) {
    if (!next || next.ok === false) {
      onError(new Error((next && next.error) || "request failed"));
      return;
    }
    clearError();
    state = next;
    if (!formSynced && next.ready) {
      formSynced = true;
      datasetEl.value = next.dataset;
      inputsEl.value = String(next.n_inputs);
      sizesEl.value = next.sizes.join(",");
      lrEl.value = String(next.lr);
      syncDataset();
      updateHint();
    }
    if (pinned && next.ready) {
      var nCols = next.layers.length + 1;
      var count = pinned.col === 0 ? next.n_inputs : (next.layers[pinned.col - 1] ? next.layers[pinned.col - 1].a.length : 0);
      if (pinned.col < 0 || pinned.col >= nCols || pinned.neuron < 0 || pinned.neuron >= count)
        pinned = null;
    }
    document.body.classList.toggle("flow", viewStyle === "flow");
    syncPlayBtn();
    if (!next.ready) {
      mseEl.textContent = "—";
      epochEl.textContent = "0";
      paint();
      renderHeat();
      renderSamples();
      renderInspect();
      return;
    }
    mseEl.textContent = next.mse.toFixed(6);
    epochEl.textContent = String(next.epoch);
    paint();
    renderHeat();
    renderSamples();
    renderInspect();
    requestAnimationFrame(paint);
  }

  function curvePoint(e, t) {
    var mx = (e.x1 + e.x2) / 2;
    var u = 1 - t;
    return [
      u * u * u * e.x1 + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * e.x2,
      u * u * u * e.y1 + 3 * u * u * t * e.y1 + 3 * u * t * t * e.y2 + t * t * t * e.y2
    ];
  }

  function distToCurve(e, x, y) {
    var best = 1e9;
    var i, p, dx, dy, d;
    for (i = 0; i <= 16; i++) {
      p = curvePoint(e, i / 16);
      dx = p[0] - x;
      dy = p[1] - y;
      d = dx * dx + dy * dy;
      if (d < best)
        best = d;
    }
    return Math.sqrt(best);
  }

  function pick(ev) {
    var rect = netCanvas.getBoundingClientRect();
    var x = ev.clientX - rect.left;
    var y = ev.clientY - rect.top;
    var i, n, dx, dy, best, bestD, d;
    for (i = hit.nodes.length - 1; i >= 0; i--) {
      n = hit.nodes[i];
      dx = x - n.x;
      dy = y - n.y;
      if (dx * dx + dy * dy <= (n.r + 4) * (n.r + 4))
        return n;
    }
    if (hit.edges.length > 2500)
      return null;
    best = null;
    bestD = 8;
    for (i = 0; i < hit.edges.length; i++) {
      d = distToCurve(hit.edges[i], x, y);
      if (d < bestD) {
        bestD = d;
        best = hit.edges[i];
      }
    }
    return best;
  }

  function sameHover(a, b) {
    if (!a && !b)
      return true;
    if (!a || !b || a.kind !== b.kind)
      return false;
    if (a.kind === "node")
      return a.col === b.col && a.neuron === b.neuron;
    return a.from === b.from && a.to === b.to;
  }

  function placeTip(ev, p) {
    var stage, x, y, sign;
    if (!p) {
      tipEl.hidden = true;
      tipEl.classList.remove("pos", "neg");
      return;
    }
    if (p.kind === "edge") {
      sign = p.w >= 0 ? "+" : "";
      tipEl.textContent = p.from + "  →  " + p.to + "    " + sign + p.w.toFixed(4);
      tipEl.classList.toggle("pos", p.w >= 0);
      tipEl.classList.toggle("neg", p.w < 0);
    } else {
      tipEl.textContent = p.name + "    " + Number(p.a).toFixed(4);
      tipEl.classList.remove("pos", "neg");
    }
    tipEl.hidden = false;
    stage = netCanvas.getBoundingClientRect();
    x = ev.clientX - stage.left + 14;
    y = ev.clientY - stage.top + 16;
    if (x > stage.width - 220)
      x = ev.clientX - stage.left - 180;
    if (y > stage.height - 36)
      y = ev.clientY - stage.top - 28;
    tipEl.style.left = x + "px";
    tipEl.style.top = y + "px";
  }

  function onFocusSample(i) {
    if (!state || !state.ready)
      return;
    if (playLoopOn) {
      wantSample = i;
      return;
    }
    clearError();
    enqueue(function () {
      return api("/api/sample", {index: i}).then(render);
    }).catch(onError);
  }

  function doTrain(steps) {
    updateHint("Training " + steps + " epoch" + (steps === 1 ? "" : "s") + "…");
    return api("/api/train", {steps: steps, lr: readLr()}).then(function (next) {
      if (wantSample != null && wantSample !== next.sample) {
        var i = wantSample;
        wantSample = null;
        return api("/api/sample", {index: i});
      }
      wantSample = null;
      return next;
    }).then(function (next) {
      var acc = accuracy(next);
      render(next);
      logLine("+" + steps + "  epoch " + next.epoch + "  mse " + next.mse.toFixed(6) + "  " + acc.ok + "/" + acc.n);
      updateHint();
    }, function (e) {
      updateHint();
      throw e;
    });
  }

  function doLaunch() {
    var body;
    playing = false;
    wantSample = null;
    pinned = null;
    hover = null;
    body = readForm();
    return api("/api/create", {
      dataset: body.dataset,
      inputs: body.inputs,
      sizes: body.sizes,
      seed: body.seed,
      lr: body.lr
    }).then(function (next) {
      if (body.dataset !== "custom")
        return next;
      return api("/api/values", {inputs: body.sampleIn, targets: body.sampleTg});
    }).then(function (next) {
      render(next);
      logLine("launch  " + body.dataset + "  " + body.inputs + " → " + body.sizes.join(",") + "  lr " + body.lr + "  seed " + body.seed);
      logLine("mse " + next.mse.toFixed(6));
    });
  }

  function onStep() {
    if (!state || !state.ready || playLoopOn)
      return;
    clearError();
    enqueue(function () { return doTrain(1); }).catch(onError);
  }

  function onRun() {
    var steps;
    if (!state || !state.ready || playLoopOn)
      return;
    clearError();
    try {
      steps = readSteps();
    } catch (e) {
      onError(e);
      return;
    }
    enqueue(function () { return doTrain(steps); }).catch(onError);
  }

  function pause(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function playLoop() {
    function again() {
      if (!playing)
        return Promise.resolve();
      return enqueue(function () {
        if (!playing)
          return;
        return doTrain(readSteps());
      }).then(function () {
        if (!playing)
          return;
        return pause(450);
      }).then(again);
    }
    return again();
  }

  function onPlay() {
    if (!state || !state.ready)
      return;
    if (playLoopOn) {
      playing = false;
      syncPlayBtn();
      return;
    }
    clearError();
    playing = true;
    playLoopOn = true;
    syncPlayBtn();
    playLoop().catch(onError).then(function () {
      playLoopOn = false;
      playing = false;
      syncPlayBtn();
    });
  }

  function onUseSample() {
    if (!state || !state.ready || state.dataset !== "custom" || playLoopOn)
      return;
    clearError();
    enqueue(function () {
      var inputs = readVector(sampleInEl.value, "Sample inputs");
      var targets = readVector(sampleTgEl.value, "Targets");
      var nOut = state.sizes[state.sizes.length - 1];
      if (inputs.length !== state.n_inputs)
        throw new Error("Sample inputs must have " + state.n_inputs + " numbers.");
      if (targets.length !== nOut)
        throw new Error("Targets must have " + nOut + " numbers.");
      return api("/api/values", {inputs: inputs, targets: targets}).then(function (next) {
        render(next);
        logLine("sample updated  mse " + next.mse.toFixed(6));
      });
    }).catch(onError);
  }

  function setStyle(next) {
    viewStyle = next;
    styleNet.classList.toggle("on", next === "net");
    styleFlow.classList.toggle("on", next === "flow");
    document.body.classList.toggle("flow", next === "flow");
    paint();
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    clearError();
    playing = false;
    syncPlayBtn();
    enqueue(doLaunch).catch(onError);
  });
  stepBtn.addEventListener("click", onStep);
  runBtn.addEventListener("click", onRun);
  playBtn.addEventListener("click", onPlay);
  sampleBtn.addEventListener("click", onUseSample);
  styleNet.addEventListener("click", function () { setStyle("net"); });
  styleFlow.addEventListener("click", function () { setStyle("flow"); });
  datasetEl.addEventListener("change", function () {
    syncDataset();
    updateHint();
    syncPlayBtn();
  });
  [inputsEl, sizesEl, lrEl, seedEl, stepsEl].forEach(function (el) {
    el.addEventListener("input", function () {
      syncRunLabel();
      updateHint();
    });
  });

  netCanvas.addEventListener("pointermove", function (ev) {
    var next = pick(ev);
    placeTip(ev, next);
    netCanvas.style.cursor = next ? "pointer" : "default";
    if (!sameHover(next, hover)) {
      hover = next;
      if (state && state.ready)
        drawNet();
    }
  });
  netCanvas.addEventListener("pointerleave", function () {
    hover = null;
    tipEl.hidden = true;
    netCanvas.style.cursor = "default";
    if (state && state.ready)
      drawNet();
  });
  netCanvas.addEventListener("click", function (ev) {
    var p = pick(ev);
    if (!p) {
      pinned = null;
    } else if (p.kind === "node") {
      pinned = {col: p.col, neuron: p.neuron};
    } else {
      pinned = {col: p.dstCol, neuron: p.dst};
    }
    renderInspect();
    if (state && state.ready)
      drawNet();
  });

  window.addEventListener("keydown", function (e) {
    if (e.target && e.target.closest && e.target.closest("input, select, textarea"))
      return;
    if (e.code === "Space") {
      e.preventDefault();
      onPlay();
    } else if (e.key === "n" || e.key === "N") {
      onStep();
    }
  });

  if (window.ResizeObserver) {
    var ro = new ResizeObserver(function () { paint(); });
    ro.observe(document.querySelector(".stage"));
    ro.observe(document.querySelector(".dock"));
  } else {
    window.addEventListener("resize", paint);
  }

  syncDataset();
  syncRunLabel();
  updateHint();
  syncPlayBtn();
  api("/api/state").then(render).catch(onError);
})();
