#!/usr/bin/env python3
"""Parse anni train logs for TF / tok/s / NLL gates (see scripts/bench_tf_gate.sh)."""
from __future__ import annotations

import argparse
import math
import re
import sys

# Frozen meter: P ≈ LM_N_BLOCKS * (12*D*D + 2*D*D_FF) with D=1024, D_FF=4096, blocks=12
# 12 * (12*1024^2 + 2*1024*4096) = 12 * (12_582_912 + 8_388_608) = 251_658_240
P_PARAMS = 251_658_240.0

RE_PROG = re.compile(
    r"NLL\s+([-\d.]+|"
    r"nan|inf).*?"
    r"([\d.]+)t/s\s+~([\d.]+)TF",
    re.I,
)
RE_NLL = re.compile(r"NLL\s+([-\d.]+|nan|inf)", re.I)
RE_TS = re.compile(r"([\d.]+)t/s", re.I)
RE_TF = re.compile(r"~([\d.]+)TF", re.I)
RE_BAD = re.compile(
    r"Segmentation|dumped core|train step failed|CUDA error|out of memory",
    re.I,
)


def parse_float(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--min-tf", type=float, default=60.0)
    ap.add_argument("--mode", choices=("speed", "overfit"), default="speed")
    ap.add_argument("--exit-code", type=int, default=0)
    ap.add_argument("--warmup-lines", type=int, default=8)
    args = ap.parse_args()

    text = open(args.log, encoding="utf-8", errors="replace").read()
    if RE_BAD.search(text):
        print("FAIL: crash / CUDA / train step failed marker in log", file=sys.stderr)
        return 2
    if args.exit_code != 0:
        print(f"FAIL: process exit {args.exit_code}", file=sys.stderr)
        return 2

    rows: list[tuple[float, float, float]] = []
    nlls: list[float] = []
    for line in text.splitlines():
        m = RE_PROG.search(line)
        if m:
            nll = parse_float(m.group(1))
            ts = parse_float(m.group(2))
            tf = parse_float(m.group(3))
            if nll is not None:
                nlls.append(nll)
            if ts is not None and tf is not None and ts > 0 and tf > 0:
                rows.append((nll if nll is not None else -1.0, ts, tf))
            continue
        # Partial lines during early progress
        nll_m = RE_NLL.search(line)
        if nll_m:
            nll = parse_float(nll_m.group(1))
            if nll is not None:
                nlls.append(nll)

    if not rows:
        print("FAIL: no progress lines with t/s and TF", file=sys.stderr)
        return 2

    # Drop warmup; median of remaining interval tok/s (ignore first-frame spikes).
    timed = rows[args.warmup_lines :] if len(rows) > args.warmup_lines else rows[-3:]
    # Drop NLL-probe / absurd spikes (>3x median of timed window).
    raw_ts = [r[1] for r in timed]
    raw_ts_sorted = sorted(raw_ts)
    rough = raw_ts_sorted[len(raw_ts_sorted) // 2]
    timed = [r for r in timed if r[1] <= rough * 3.0 or rough <= 0]
    if not timed:
        timed = rows[-3:]
    tss = sorted(r[1] for r in timed)
    tfs = sorted(r[2] for r in timed)
    med_ts = tss[len(tss) // 2]
    med_tf = tfs[len(tfs) // 2]
    peak_ts = max(r[1] for r in timed)
    peak_tf = max(r[2] for r in timed)

    expect_tf = 6.0 * P_PARAMS * med_ts / 1e12
    if expect_tf > 0 and abs(med_tf - expect_tf) / expect_tf > 0.05:
        print(
            f"FAIL: TF meter mismatch med_tf={med_tf:.3f} vs 6P*tok/1e12={expect_tf:.3f}",
            file=sys.stderr,
        )
        return 2

    finite_nll = [n for n in nlls if n >= 0]
    if not finite_nll:
        print("FAIL: no finite NLL (stuck at -1 / missing)", file=sys.stderr)
        return 2

    if args.mode == "overfit":
        if finite_nll[-1] >= finite_nll[0] * 0.95 and finite_nll[-1] >= finite_nll[0] - 0.05:
            # Require clear drop from first finite to last
            if finite_nll[-1] > min(finite_nll) + 1e-6 and finite_nll[-1] >= finite_nll[0]:
                print(
                    f"FAIL: overfit NLL did not fall ({finite_nll[0]:.4f} -> {finite_nll[-1]:.4f})",
                    file=sys.stderr,
                )
                return 2
            if finite_nll[-1] >= finite_nll[0]:
                print(
                    f"FAIL: overfit NLL did not fall ({finite_nll[0]:.4f} -> {finite_nll[-1]:.4f})",
                    file=sys.stderr,
                )
                return 2

    if med_tf < args.min_tf:
        print(
            f"FAIL: median TF {med_tf:.3f} < --min-tf {args.min_tf} "
            f"(peak {peak_tf:.3f} TF, med {med_ts:.0f} t/s, peak {peak_ts:.0f} t/s)",
            file=sys.stderr,
        )
        return 1

    print(
        f"PASS: med_tf={med_tf:.3f} peak_tf={peak_tf:.3f} "
        f"med_t/s={med_ts:.0f} peak_t/s={peak_ts:.0f} "
        f"nll={finite_nll[0]:.4f}->{finite_nll[-1]:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
