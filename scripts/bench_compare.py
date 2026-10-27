#!/usr/bin/env python3
"""Compare two `anni lm bench` JSONL result files (see docs/bench-harness.md).

Usage: scripts/bench_compare.py BASE.jsonl NEW.jsonl [--threshold 0.03]

Rows are matched by mode + GPU + model dims + config (warmup/steps/runs and
the requested B are ignored). The latest row per key wins in each file.
Exit 1 when a matched row regresses by more than the threshold, fails the
loss check, or (decode) changes its greedy-output hash.
"""
from __future__ import annotations

import argparse
import json
import sys

IGNORED_CFG = {"warmup", "steps", "runs", "B_req", "lr"}
# mode -> (metric path, higher_is_better)
METRIC = {
    "step": (("ms", "p50"), False),
    "gemm": (("tf_best",), True),
    "decode": (("steady_tok_s",), True),
}


def key_of(row: dict) -> tuple:
    cfg = {k: v for k, v in row.get("cfg", {}).items() if k not in IGNORED_CFG}
    return (
        row.get("mode"),
        row.get("gpu"),
        json.dumps(row.get("model", {}), sort_keys=True),
        json.dumps(cfg, sort_keys=True),
    )


def load(path: str) -> dict:
    rows = {}
    with open(path, encoding="utf-8") as fp:
        for n, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"{path}:{n}: skipping bad JSON", file=sys.stderr)
                continue
            if row.get("mode") in METRIC:
                rows[key_of(row)] = row
    return rows


def metric(row: dict) -> float | None:
    path, _ = METRIC[row["mode"]]
    v = row
    for p in path:
        v = v.get(p) if isinstance(v, dict) else None
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def label(key: tuple) -> str:
    cfg = json.loads(key[3])
    parts = [key[0]] + [f"{k}={cfg[k]}" for k in sorted(cfg)]
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--threshold", type=float, default=0.03)
    args = ap.parse_args()

    base = load(args.base)
    new = load(args.new)
    common = [k for k in new if k in base]
    if not common:
        print("bench_compare: no matching rows (same GPU, model and config)")
        return 1
    bad = 0
    for k in sorted(common, key=label):
        b, n = base[k], new[k]
        mb, mn = metric(b), metric(n)
        _, higher = METRIC[k[0]]
        notes = []
        if n.get("busy"):
            notes.append("busy GPU")
        if k[0] == "step" and not n.get("loss_ok", False):
            notes.append("LOSS CHECK FAILED")
            bad += 1
        if k[0] == "decode" and b.get("hash") != n.get("hash"):
            notes.append(f"greedy hash {b.get('hash')} -> {n.get('hash')}")
            bad += 1
        if mb is None or mn is None:
            print(f"?    {label(k)}  (missing metric) {' '.join(notes)}")
            continue
        gain = (mn / mb - 1.0) if higher else (mb / mn - 1.0)
        status = "ok  "
        if gain < -args.threshold:
            status = "SLOW"
            bad += 1
        elif gain > args.threshold:
            status = "FAST"
        print(f"{status} {gain * 100:+6.1f}%  {mb:10.3f} -> {mn:10.3f}  "
              f"{label(k)}  {' '.join(notes)}".rstrip())
    only_new = len(new) - len(common)
    if only_new:
        print(f"({only_new} new row(s) without a baseline)")
    print("FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
