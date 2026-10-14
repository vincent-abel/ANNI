#!/usr/bin/env python3
"""Parse an AnnI train log into a pass/fail TFLOP/s gate.

The printer's t/s and ~TF are cumulative tokens/elapsed from the start of
the clock (lm_train.c fill_throughput / dbg_progress). That series only
rises, so its max is the last line and still includes graph capture.
This gate uses the median of interval tok/s after that warmup instead.

Meter mass is frozen at the compile-time count
P = L*(12*D*D + 2*D*D_FF) = 251658240 (about 2.52e8).
Printed TF must match 6*P*tok/s/1e12. P is not retuned.

Pad: in-memory and non-varlen stream windows are full real ids, and the
meter adds bb*T. --varlen counts sum(lens) but the progress line only has
window counts, so windows*T would include pad and the run is refused.
Graph replay stores mean_ce = -1 and the EMA does not move; a flat NLL
(or a lone -1) is a failed step. lm generate is host KV decode, not the
packed train forward, so it is not a parrot pass.
"""
import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# lang/lm_config.h D=1024, D_FF=4096, L=12. Do not replace with 2.52e8.
P_METER = 12 * (12 * 1024 * 1024 + 2 * 1024 * 4096)
VOCAB_SPECIALS = 4
MIN_INTERVALS = 3
OVERFIT_WALL_S = 150.0
NLL_FLAT = 0.001
NLL_DROP = 0.05

_PROG = re.compile(
    r"^ep (?P<ep>\d+)/(?P<epochs>\d+) \[(?P<bar>[#-]+)\]\s+"
    r"(?P<pct>\d+|\?)%\s+(?P<done>\d+)/(?P<total>\d+|\?)"
    r"(?: NLL (?P<nll>[-+]?(?:\d+(?:\.\d*)?|nan|inf|infinity)))?"
    r"(?: win (?P<win>[-+]?(?:\d+(?:\.\d*)?|nan|inf|infinity)))?"
    r"(?: (?P<ms>\d+(?:\.\d+)?)ms/t)?"
    r"(?: (?P<tps>\d+(?:\.\d+)?)t/s)?"
    r"(?: ~(?P<tf>\d+(?:\.\d+)?)TF)?"
    r"(?: (?P<mb>\d+(?:\.\d+)?)MB)?"
)
_PROBE = re.compile(
    r"computing start(?: window)? NLL|mid-epoch|ctx NLL|val NLL|window NLL",
    re.IGNORECASE,
)
_EPOCH_EMA = re.compile(
    r"epoch (\d+) ctx NLL ([-+]?\d+(?:\.\d+)?) \(train EMA ([-+]?\d+(?:\.\d+)?)\)"
)
_BATCH = re.compile(
    r"batch (?:fixed|auto): B=(\d+) T=(\d+) tokens/step=(\d+)"
)
_READY = re.compile(
    r"dataset ready: (\d+) token ids, (\d+) windows \(len (\d+)\)"
)
_CRASH = (
    "Segmentation",
    "dumped core",
    "train step failed",
    "lm train: failed",
)


def tf_of(tok_s):
    return 6.0 * P_METER * float(tok_s) / 1e12


def _fnum(text):
    if text is None:
        return None
    low = text.lower()
    if low in ("nan", "inf", "infinity", "+inf", "-inf", "+infinity", "-infinity"):
        return float(low)
    return float(text)


def arg_value(cmd, flag):
    for i, a in enumerate(cmd):
        if a == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def has_flag(cmd, flag):
    return flag in cmd


def split_chunks(blob):
    """Progress frames are separated by \\r, stamps by \\n."""
    parts = re.split(r"[\r\n]+", blob)
    return [p for p in parts if p.strip()]


def parse_stamped(text):
    samples = []
    probes = []
    body = []
    for raw in text.splitlines():
        if "\t" not in raw:
            body.append(raw)
            continue
        ts_s, line = raw.split("\t", 1)
        try:
            ts = float(ts_s)
        except ValueError:
            body.append(raw)
            continue
        line = line.strip()
        if not line:
            continue
        body.append(line)
        m = _PROG.match(line)
        if m:
            total = m.group("total")
            samples.append({
                "ts": ts,
                "ep": int(m.group("ep")),
                "done": int(m.group("done")),
                "total": None if total == "?" else int(total),
                "nll": _fnum(m.group("nll")),
                "tps": _fnum(m.group("tps")),
                "tf": _fnum(m.group("tf")),
                "mb": _fnum(m.group("mb")),
            })
            continue
        if _PROBE.search(line):
            probes.append(ts)
    return samples, probes, "\n".join(body)


def window_of(cmd, body):
    w = arg_value(cmd, "--window")
    if w and w.isdigit():
        return int(w)
    m = _READY.search(body)
    if m:
        return int(m.group(3))
    m = _BATCH.search(body)
    if m:
        return int(m.group(2))
    return None


def stream_mode(cmd, body):
    if has_flag(cmd, "--stream") or "streaming:" in body:
        return True
    return False


def meter_tokens(sample, T, stream):
    if stream:
        return sample["done"] * T
    if sample["total"] is None:
        return None
    return ((sample["ep"] - 1) * sample["total"] + sample["done"]) * T


def intervals_of(samples, probes, T, stream):
    out = []
    for i in range(1, len(samples)):
        a = samples[i - 1]
        b = samples[i]
        if stream and (b["ep"] != a["ep"] or b["done"] < a["done"]):
            continue
        ta = meter_tokens(a, T, stream)
        tb = meter_tokens(b, T, stream)
        if ta is None or tb is None:
            continue
        dt = b["ts"] - a["ts"]
        if dt <= 0.0 or tb < ta:
            continue
        dtok = tb - ta
        if dtok <= 0:
            continue
        probe = any(a["ts"] < p <= b["ts"] for p in probes)
        out.append({
            "rate": dtok / dt,
            "mb": b["mb"],
            "warmup": i == 1,
            "probe": probe,
        })
    return out


def scored_intervals(intervals):
    return [iv for iv in intervals if not iv["warmup"] and not iv["probe"]]


def nll_series(samples, body):
    vals = []
    for s in samples:
        nll = s["nll"]
        if nll is None:
            continue
        vals.append(nll)
    emas = [float(m.group(3)) for m in _EPOCH_EMA.finditer(body)]
    return vals, emas


def nll_failures(vals, emas, overfit):
    fails = []
    finite = [v for v in vals if v > 0.0 and v == v and v != float("inf")]
    bad = [v for v in vals if not (v > 0.0 and v == v and v != float("inf"))]
    if bad:
        fails.append("non-finite or sentinel NLL on a progress line")
    if not finite and not any(v > 0.0 for v in emas):
        fails.append(
            "no finite train NLL (stuck at -1 or hidden); "
            "a drop from the -1 sentinel does not count"
        )
        return fails
    if len(finite) >= 2 and max(finite) - min(finite) < NLL_FLAT:
        fails.append(
            "NLL is flat across the timed window "
            "(graph replay leaves mean_ce at -1 and the EMA does not move)"
        )
    if overfit:
        series = [v for v in emas if v > 0.0] or finite
        if len(series) < 2:
            fails.append("overfit NLL never moved; need two finite train losses")
        elif series[-1] > series[0] - NLL_DROP or series[-1] >= series[0]:
            fails.append(
                "train NLL did not fall "
                "(%.4f -> %.4f); a 0.5x drop from -1 does not count"
                % (series[0], series[-1])
            )
        if any(v <= 0.0 or v != v for v in emas):
            fails.append("epoch train EMA is missing or -1")
    return fails


def tf_consistency_failures(samples):
    fails = []
    saw = False
    for s in samples:
        if s["tps"] is None or s["tf"] is None or s["tps"] <= 0.0:
            continue
        saw = True
        exp = tf_of(s["tps"])
        err = abs(s["tf"] - exp)
        # %.2f rounding is about 0.005; a wrong P is much larger.
        if err > max(0.05 * exp, 0.015):
            fails.append(
                "printed ~%.2fTF disagrees with 6*P*%.0f/1e12 = %.2f "
                "(P=%d)" % (s["tf"], s["tps"], exp, P_METER)
            )
            break
    if not saw:
        fails.append("peak t/s and ~TF did not parse")
    return fails


def looks_tiled(blob):
    n = len(blob)
    if n < 20000:
        return False
    for span in (200, 400, 800, 1600):
        block = blob[:span]
        reps = n // span
        if reps >= 8 and blob[:span * reps] == block * reps:
            return True
    return False


def corpus_failures(path):
    fails = []
    p = Path(path)
    if not p.is_file():
        return ["overfit corpus missing: %s" % path]
    blob = p.read_bytes()
    if len(blob) < 20000:
        fails.append(
            "overfit corpus is %d bytes; need tens of thousands of tokens"
            % len(blob)
        )
    if looks_tiled(blob):
        fails.append("overfit corpus repeats one paragraph")
    simple = Path("data/wiki_simple.txt")
    if simple.is_file():
        src = simple.read_bytes()
        if src[:len(blob)] != blob:
            fails.append(
                "wiki_overfit.txt is not a contiguous head of wiki_simple.txt"
            )
    return fails


def unique_prompt(corpus, window, n_cont=8, prompt_len=40):
    """A byte prompt that occurs once and still leaves n_cont inside the window."""
    if prompt_len + n_cont > window:
        prompt_len = window - n_cont
    if prompt_len < 8 or len(corpus) < prompt_len + n_cont:
        return None
    step = 17
    for start in range(0, len(corpus) - prompt_len - n_cont, step):
        frag = corpus[start:start + prompt_len]
        if corpus.count(frag) != 1:
            continue
        return start, frag
    return None


def continuation_ids(text):
    return [VOCAB_SPECIALS + b for b in text]


def score_continuation(corpus, start, prompt, generated):
    """Score ids after the prompt. Echoing the prompt is not a match."""
    if generated.startswith(prompt):
        new = generated[len(prompt):]
    else:
        return False, "generate output does not start with the prompt"
    if not new:
        return False, "no tokens past the prompt"
    expect = corpus[start + len(prompt):start + len(prompt) + len(new)]
    got = continuation_ids(new)
    exp = continuation_ids(expect)
    if got != exp:
        return False, "token id mismatch past prompt"
    return True, "ok"


def parrot_failures(cmd, window):
    """The CLI cannot score greedy ids on the packed train forward."""
    fails = []
    tok = arg_value(cmd, "--tokenizer")
    if tok and tok != "byte":
        fails.append(
            "tokenizer %s: decoded text is not token-id equality; "
            "refusing a character match" % tok
        )
    found = None
    data = arg_value(cmd, "--data")
    if data and Path(data).is_file() and (tok is None or tok == "byte"):
        blob = Path(data).read_bytes()
        found = unique_prompt(blob, window)
        if found is None:
            fails.append(
                "no unique prompt that occurs once and fits in the window"
            )
    fails.append(
        "parrot refused: lm generate runs lm_generate_kv on host weights, "
        "not the packed training forward. Greedy text from that path is "
        "not a training-precision token-id check"
    )
    return fails


def device_failures(info):
    fails = []
    cap = (info.get("compute_cap") or "").strip()
    if cap != "12.0":
        fails.append(
            "device compute capability is %s; need sm_120 (12.0)"
            % (cap or "unknown")
        )
    archs = info.get("gpu_archs") or ""
    if "compute_120" not in archs and "sm_120" not in archs:
        fails.append(
            "CUDA toolkit cannot emit sm_120 (nvcc arch list has no compute_120)"
        )
    cubin = info.get("cubin_archs")
    if cubin is not None and "sm_120" not in cubin:
        shown = ", ".join(cubin) if cubin else "none"
        fails.append("binary cubin archs (%s) do not include sm_120" % shown)
    if info.get("cuda_linked") is False:
        fails.append("anni is not linked to libcudart")
    return fails


def crash_failures(text, exit_code):
    fails = []
    if exit_code not in (0, None):
        fails.append("process exit %s" % exit_code)
    for pat in _CRASH:
        if pat in text:
            fails.append("log contains %r" % pat)
    return fails


def pad_failures(cmd, body):
    if has_flag(cmd, "--varlen"):
        return [
            "varlen tok/s is sum(lens), but the progress line only has "
            "window counts; windows*T would count pad, so the run is refused"
        ]
    m = _BATCH.search(body)
    if m:
        b, t, step = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if step != b * t and not has_flag(cmd, "--varlen"):
            return [
                "tokens/step %d is not B*T (%d); refusing to guess which "
                "slots are pad" % (step, b * t)
            ]
    return []


def vram_failures(scored, memory_total_mib):
    if not scored or not memory_total_mib or memory_total_mib <= 0:
        return []
    rates = [iv["rate"] for iv in scored]
    med = statistics.median(rates)
    if med <= 0.0:
        return []
    full = 0.98 * memory_total_mib
    for iv in scored:
        mb = iv["mb"]
        if mb is None:
            continue
        if mb >= full and iv["rate"] < med / 3.0:
            return [
                "VRAM glued near %.0f MiB while interval tok/s fell to %.0f "
                "(median %.0f)" % (memory_total_mib, iv["rate"], med)
            ]
    return []


def evaluate(text, cmd, mode, min_tf, device, exit_code=0, wall_s=None):
    """Return (fails, warns, summary dict). device is a probe dict."""
    fails = []
    warns = []
    samples, probes, body = parse_stamped(text)
    whole = body + "\n" + text
    fails.extend(crash_failures(whole, exit_code))
    fails.extend(device_failures(device))
    data = arg_value(cmd, "--data") or ""
    if Path(data).name == "toy_micro.txt":
        fails.append("toy_micro is smoke only; this gate does not score it")
    if "-d" not in cmd and "--debug" not in cmd:
        fails.append("train command has no -d, so the meter line is not printed")
    fails.extend(pad_failures(cmd, body))
    T = window_of(cmd, body)
    if not T:
        fails.append("window T is unknown; pass --window or keep the batch stamp")
        T = 0
    stream = stream_mode(cmd, body)
    ivs = intervals_of(samples, probes, T, stream) if T else []
    scored = scored_intervals(ivs)
    median = statistics.median([iv["rate"] for iv in scored]) if scored else None
    printed = [s["tps"] for s in samples if s["tps"]]
    peak_printed = max(printed) if printed else None
    vals, emas = nll_series(samples, body)
    overfit = mode == "overfit"
    fails.extend(nll_failures(vals, emas, overfit))
    fails.extend(tf_consistency_failures(samples))
    if mode == "speed":
        if len(scored) < MIN_INTERVALS:
            fails.append(
                "need %d interval tok/s samples after warmup; got %d "
                "(cumulative t/s is not the gate)"
                % (MIN_INTERVALS, len(scored))
            )
        elif median is not None and tf_of(median) < min_tf:
            fails.append(
                "median interval %.0f tok/s is %.2f TF, under --min-tf %.2f"
                % (median, tf_of(median), min_tf)
            )
    mem = device.get("memory_total_mib")
    fails.extend(vram_failures(scored, mem))
    if overfit:
        if data:
            fails.extend(corpus_failures(data))
        else:
            fails.append("overfit mode needs --data")
        if wall_s is not None and wall_s > OVERFIT_WALL_S:
            warns.append(
                "overfit wall %.0fs exceeds %.0fs (soft; NLL still has to fall)"
                % (wall_s, OVERFIT_WALL_S)
            )
        if T:
            fails.extend(parrot_failures(cmd, T))
    # Sentinel -1 on the epoch stamp, even when progress hid it.
    if re.search(r"NLL -1\.0+\b", body) and not any(v > 0.0 for v in vals):
        if not any("sentinel" in f for f in fails):
            fails.append("epoch stamp NLL is -1")
    summary = {
        "median_tok_s": median,
        "median_tf": tf_of(median) if median is not None else None,
        "peak_printed_tok_s": peak_printed,
        "intervals": len(scored),
        "wall_s": wall_s,
    }
    return fails, warns, summary


def format_report(fails, warns, summary, device):
    lines = []
    cap = device.get("compute_cap") or "?"
    lines.append(
        "device %s compute_cap=%s driver=%s toolkit=%s clocks_sm=%s clocks_mem=%s"
        % (
            device.get("name") or "?",
            cap,
            device.get("driver") or "?",
            device.get("nvcc") or "?",
            device.get("clocks_sm") or "?",
            device.get("clocks_mem") or "?",
        )
    )
    med = summary.get("median_tok_s")
    if med is not None:
        lines.append(
            "median interval %.0f tok/s -> %.2f TF (P=%d, 6*P*tok/s/1e12); "
            "peak printed cumulative %.0f tok/s; intervals %d"
            % (
                med,
                summary["median_tf"],
                P_METER,
                summary.get("peak_printed_tok_s") or -1,
                summary.get("intervals") or 0,
            )
        )
    for w in warns:
        lines.append("WARN " + w)
    if fails:
        lines.append("FAIL")
        for f in fails:
            lines.append("- " + f)
    else:
        lines.append("PASS")
    return "\n".join(lines) + "\n"


def _query(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.strip()


def probe_device(binary):
    info = {
        "name": "",
        "compute_cap": "",
        "driver": "",
        "clocks_sm": "",
        "clocks_mem": "",
        "nvcc": "",
        "gpu_archs": "",
        "cubin_archs": None,
        "cuda_linked": None,
        "memory_total_mib": None,
    }
    csv = _query([
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,clocks.sm,clocks.mem,memory.total",
        "--format=csv,noheader",
    ])
    if csv:
        parts = [p.strip() for p in csv.split(",")]
        if len(parts) >= 6:
            info["name"] = parts[0]
            info["compute_cap"] = parts[1]
            info["driver"] = parts[2]
            info["clocks_sm"] = parts[3]
            info["clocks_mem"] = parts[4]
            mem = parts[5].split()
            if mem and mem[0].isdigit():
                info["memory_total_mib"] = float(mem[0])
    ver = _query(["nvcc", "--version"])
    rel = ""
    for line in ver.splitlines():
        if "release" in line:
            rel = " ".join(line.split())
            break
    info["nvcc"] = rel or " ".join(ver.split())
    info["gpu_archs"] = _query(["nvcc", "--list-gpu-arch"])
    if binary and Path(binary).is_file():
        ldd = _query(["ldd", binary])
        if ldd:
            info["cuda_linked"] = "libcudart" in ldd
        dump = _query(["cuobjdump", "-lelf", binary])
        if dump:
            info["cubin_archs"] = sorted(set(re.findall(r"sm_\d+", dump)))
    return info


def read_process(cmd, log_path):
    """Run cmd, stamp each \\r/\\n frame, return (exit, wall, stamped text)."""
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
    )
    frames = []
    buf = b""
    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                n = buf.find(b"\n")
                r = buf.find(b"\r")
                cuts = [i for i in (n, r) if i >= 0]
                if not cuts:
                    break
                i = min(cuts)
                piece = buf[:i].decode("utf-8", "replace").strip()
                buf = buf[i + 1:]
                if piece:
                    frames.append((time.time(), piece))
        if buf.strip():
            frames.append((time.time(), buf.decode("utf-8", "replace").strip()))
        rc = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    wall = time.time() - t0
    stamped = "".join("%0.6f\t%s\n" % (ts, line) for ts, line in frames)
    Path(log_path).write_text(stamped, encoding="utf-8")
    return rc, wall, stamped


def run_gate(cmd, mode, min_tf, log_path):
    binary = cmd[0] if cmd else "./anni"
    device = probe_device(binary)
    early = device_failures(device)
    header = format_report(early, [], {
        "median_tok_s": None,
        "median_tf": None,
        "peak_printed_tok_s": None,
        "intervals": 0,
        "wall_s": None,
    }, device)
    if early:
        sys.stderr.write(header)
        Path(log_path).write_text(header, encoding="utf-8")
        return 1
    rc, wall, stamped = read_process(cmd, log_path)
    fails, warns, summary = evaluate(
        stamped, cmd, mode, min_tf, device, exit_code=rc, wall_s=wall
    )
    report = format_report(fails, warns, summary, device)
    sys.stderr.write(report)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(report)
    return 0 if not fails else 1


def _line(ep, done, total, nll, tps, tf, mb=1000.0):
    pct = 100.0 * done / total
    return (
        "ep %d/1 [####--------] %4.0f%% %d/%d NLL %.3f %.2fms/t %.0ft/s ~%.2fTF %.0fMB"
        % (ep, pct, done, total, nll, 1000.0 / tps if tps else 0.0, tps, tf, mb)
    )


def _stamp(rows):
    return "".join("%0.6f\t%s\n" % (ts, line) for ts, line in rows)


def _device_ok():
    return {
        "name": "NVIDIA GeForce RTX 5070 Ti",
        "compute_cap": "12.0",
        "driver": "616.64",
        "clocks_sm": "1800 MHz",
        "clocks_mem": "14001 MHz",
        "nvcc": "release 13.4",
        "gpu_archs": "compute_120\nsm_120",
        "cubin_archs": ["sm_120"],
        "cuda_linked": True,
        "memory_total_mib": 16384.0,
    }


def _rising_cumulative_case():
    """Steady interval tok/s while the printed cumulative t/s is still climbing."""
    # 100 windows * 64 tokens each second. First gap is warmup and is dropped.
    T = 64
    rows = []
    nlls = (8.500, 8.200, 7.900, 7.500, 7.100, 6.700)
    for i, nll in enumerate(nlls):
        done = 100 * (i + 1)
        ts = 10.0 + i
        # Cumulative from t=0, so printed t/s rises toward the interval rate.
        tokens = done * T
        tps = tokens / ts
        rows.append((ts, _line(1, done, 10000, nll, tps, tf_of(round(tps)))))
    text = _stamp(rows)
    text += "0.000000\tdataset ready: 640000 token ids, 10000 windows (len 64)\n"
    text += (
        "0.000000\tbatch fixed: B=128 T=64 tokens/step=%d\n" % (128 * 64)
    )
    cmd = [
        "./anni", "lm", "train", "--data", "data/wiki_ids.bin",
        "--window", "64", "--batch", "128", "-d",
    ]
    return text, cmd


def self_test():
    assert P_METER == 251658240
    # 60 printed TF on this frozen P is about 39.7k tok/s.
    assert 39000 < (60e12 / (6.0 * P_METER)) < 41000

    text, cmd = _rising_cumulative_case()
    fails, warns, summary = evaluate(
        text, cmd, "speed", 8.0, _device_ok(), exit_code=0, wall_s=10.0
    )
    if fails:
        sys.stderr.write("expected pass, got:\n" + "\n".join(fails) + "\n")
        return 1
    med = summary["median_tok_s"]
    peak = summary["peak_printed_tok_s"]
    # Interval is 6400 tok/s. Cumulative printed t/s is lower and still rising.
    if abs(med - 6400.0) > 1.0:
        sys.stderr.write("median interval %s, want 6400\n" % med)
        return 1
    if peak is None or peak >= med:
        sys.stderr.write("peak printed %s should stay under the interval\n" % peak)
        return 1
    # Gating on the last cumulative line would miss min-tf 8; the interval hits it.
    if tf_of(peak) >= 8.0 or tf_of(med) < 8.0:
        sys.stderr.write("threshold check used the wrong series\n")
        return 1

    flat = text.replace("NLL 8.500", "NLL 8.000").replace("NLL 8.200", "NLL 8.000")
    flat = flat.replace("NLL 7.900", "NLL 8.000").replace("NLL 7.500", "NLL 8.000")
    flat = flat.replace("NLL 7.100", "NLL 8.000").replace("NLL 6.700", "NLL 8.000")
    fails, _, _ = evaluate(flat, cmd, "speed", 1.0, _device_ok())
    if not any("flat" in f for f in fails):
        sys.stderr.write("flat NLL was not rejected: %s\n" % fails)
        return 1

    blown = []
    for line in text.splitlines():
        blown.append(re.sub(r"~(\d+(?:\.\d+)?)TF", lambda m: "~%.2fTF" % (float(m.group(1)) * 3), line))
    fails, _, _ = evaluate("\n".join(blown) + "\n", cmd, "speed", 1.0, _device_ok())
    if not any("disagrees" in f for f in fails):
        sys.stderr.write("TF mismatch was not rejected: %s\n" % fails)
        return 1

    sentinel = "1.0\tepoch 1/1 done: 10 windows  NLL -1.0000\n"
    fails, _, _ = evaluate(sentinel, cmd, "speed", 1.0, _device_ok(), exit_code=1)
    if not any("sentinel" in f or "exit" in f or "-1" in f for f in fails):
        sys.stderr.write("sentinel NLL was not rejected: %s\n" % fails)
        return 1

    var_cmd = cmd + ["--varlen"]
    fails, _, _ = evaluate(text, var_cmd, "speed", 1.0, _device_ok())
    if not any("pad" in f for f in fails):
        sys.stderr.write("varlen pad count was not rejected: %s\n" % fails)
        return 1

    toy = ["./anni", "lm", "train", "--data", "data/toy_micro.txt", "-d", "--window", "64"]
    fails, _, _ = evaluate(text, toy, "speed", 1.0, _device_ok())
    if not any("toy_micro" in f for f in fails):
        sys.stderr.write("toy_micro was scored: %s\n" % fails)
        return 1

    bad_dev = _device_ok()
    bad_dev["compute_cap"] = "8.9"
    bad_dev["gpu_archs"] = "compute_89"
    bad_dev["cubin_archs"] = ["sm_90"]
    fails, _, _ = evaluate(text, cmd, "speed", 1.0, bad_dev)
    if not any("sm_120" in f for f in fails):
        sys.stderr.write("wrong device was accepted: %s\n" % fails)
        return 1

    crash = text + "2.0\tSegmentation fault (core dumped)\n"
    fails, _, _ = evaluate(crash, cmd, "speed", 1.0, _device_ok(), exit_code=139)
    if not any("Segmentation" in f for f in fails):
        sys.stderr.write("crash text was ignored: %s\n" % fails)
        return 1

    corpus = b"A" * 100 + b"unique-prompt-slice-0123456789abcdef" + b"CONT" + b"Z" * 30000
    # Make it long enough and not tiled.
    corpus = b"April is a month.\n" + corpus + b"\nThe end of the article.\n"
    corpus = corpus + b"more distinct text " * 2000
    hit = unique_prompt(corpus, 64, n_cont=4, prompt_len=32)
    if hit is None:
        sys.stderr.write("unique prompt was not found\n")
        return 1
    start, prompt = hit
    if corpus.count(prompt) != 1:
        sys.stderr.write("prompt is not unique\n")
        return 1
    if len(prompt) + 4 > 64:
        sys.stderr.write("prompt does not fit in the window\n")
        return 1
    ok, _ = score_continuation(corpus, start, prompt, prompt)
    if ok:
        sys.stderr.write("echoing the prompt counted as a parrot hit\n")
        return 1
    expect = corpus[start + len(prompt):start + len(prompt) + 4]
    ok, why = score_continuation(corpus, start, prompt, prompt + expect)
    if not ok:
        sys.stderr.write("real continuation ids failed: %s\n" % why)
        return 1
    ok, _ = score_continuation(corpus, start, prompt, prompt + b"XXXX")
    if ok:
        sys.stderr.write("wrong continuation ids passed\n")
        return 1

    if not looks_tiled(b"para\n" * 5000):
        sys.stderr.write("tiled paragraph was accepted\n")
        return 1
    wiki = Path("data/wiki_overfit.txt")
    if wiki.is_file():
        blob = wiki.read_bytes()
        if len(blob) < 20000 or looks_tiled(blob):
            sys.stderr.write("committed overfit head is too small or tiled\n")
            return 1

    parts = split_chunks("ep 1/1 [----]  0% 1/2\r\rep 1/1 [####] 50% 2/2\n")
    if len(parts) != 2:
        sys.stderr.write("CR frames did not split: %s\n" % parts)
        return 1

    # Probe stamps between samples are dropped, so they cannot prop up the median.
    rows = []
    for i in range(6):
        done = 100 * (i + 1)
        ts = 10.0 + i
        tps = (done * 64) / ts
        rows.append((ts, _line(1, done, 10000, 8.0 - i * 0.4, tps, tf_of(round(tps)))))
    rows.insert(3, (12.4, "epoch 1 ctx NLL 7.0000 (train EMA 6.5000)"))
    fails, _, summary = evaluate(
        _stamp(rows) + "0\tdataset ready: 1 token ids, 10000 windows (len 64)\n",
        cmd, "speed", 1.0, _device_ok(),
    )
    if fails:
        sys.stderr.write("probe case failed closed for the wrong reason: %s\n" % fails)
        return 1
    # 5 gaps, minus warmup, minus the ctx-NLL probe sitting in one gap.
    if summary["intervals"] != 3:
        sys.stderr.write("probe interval was kept: %s\n" % summary)
        return 1

    sys.stdout.write("self-test ok\n")
    return 0


def main(argv):
    p = argparse.ArgumentParser(description="AnnI training TFLOP/s gate")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--mode", choices=("speed", "overfit"), default="speed")
    p.add_argument("--min-tf", type=float, default=60.0)
    p.add_argument("--log", default="")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        p.error("missing train command")
    log_path = args.log or ("/tmp/bench_tf_gate-%s.log" % args.mode)
    return run_gate(cmd, args.mode, args.min_tf, log_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
