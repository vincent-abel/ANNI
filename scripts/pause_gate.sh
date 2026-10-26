#!/usr/bin/env bash
# Pause-window deploy: gate a candidate binary and resume the talk170m run.
#
#   scripts/pause_gate.sh CANDIDATE FALLBACK [flags...]
#   NO_RESTART=1 scripts/pause_gate.sh ...   # gate only, leave the run stopped
#
# Run right after the training process was interrupted (Ctrl-C in tmux
# talk170m following a checkpoint). Installs CANDIDATE as ./anni, runs the
# deploy gate (scripts/prof_step.sh gate: validation probes, checkpoint save,
# peak VRAM against the 85% cap) and a plain 12-step bench, compares the
# bench loss trajectory with logs/prof/bench_reference.log when present,
# keeps CANDIDATE on PASS or restores FALLBACK, then restarts the driver.
# FALLBACK is benched first (same 12 steps): a candidate slower than 98% of
# its tok/s is rejected even when the gate passes. Prints a one-paragraph
# report. Total under 6 minutes.
set -uo pipefail
cd "$(dirname "$0")/.."
CAND=${1:?candidate binary}
FALL=${2:?fallback binary}
shift 2
FLAGS=${*:---bf16 --flash-attn}
T0=$(date +%H:%M:%S)
if pgrep -f '^\./anni lm train' >/dev/null; then
	echo "training still running; pause it first"
	exit 1
fi
TAG=$(echo "$FLAGS" | tr -c 'a-z0-9' '_' | sed 's/_*$//')
cp "$FALL" anni
FBENCH=$(scripts/prof_step.sh bench $FLAGS 2>&1 | tr '\r' '\n' | grep -a '^ep 1/1' | sed -n '9p' | grep -o '[0-9]*t/s ~[0-9.]*TF')
FTPS=$(echo "$FBENCH" | grep -o '^[0-9]*')
cp "$CAND" anni
GATE=$(scripts/prof_step.sh gate $FLAGS 2>&1 | tail -3)
PEAK=$(echo "$GATE" | grep -o 'peak VRAM [0-9]*' | grep -o '[0-9]*$')
VERDICT=$(echo "$GATE" | grep -o 'GATE [A-Z]*')
BENCH=$(scripts/prof_step.sh bench $FLAGS 2>&1 | tr '\r' '\n' | grep -a '^ep 1/1' | sed -n '9p' | grep -o '[0-9]*t/s ~[0-9.]*TF')
CTPS=$(echo "$BENCH" | grep -o '^[0-9]*')
SPEED=ok
if [ -n "$FTPS" ] && [ -n "$CTPS" ] && [ "$((CTPS * 100))" -lt "$((FTPS * 98))" ]; then
	SPEED="SLOWER than fallback ($FBENCH)"
fi
LOSS=same
if [ -f logs/prof/bench_reference.log ]; then
	A=$(tr '\r' '\n' < logs/prof/bench_reference.log | grep -a '^ep 1/1' | grep -a -o 'NLL [0-9.]*' | head -12 | tr '\n' ' ')
	B=$(tr '\r' '\n' < "logs/prof/bench_$TAG.log" | grep -a '^ep 1/1' | grep -a -o 'NLL [0-9.]*' | head -12 | tr '\n' ' ')
	[ "$A" = "$B" ] || LOSS="DIFFERENT (ref: $A | new: $B)"
fi
if [ "$VERDICT" = "GATE PASS" ] && [ "$LOSS" = same ] && [ "$SPEED" = ok ]; then
	DEPLOYED=candidate
else
	cp "$FALL" anni
	DEPLOYED=fallback
fi
RESTARTED="driver restarted"
if [ "${NO_RESTART:-0}" = 1 ]; then
	RESTARTED="driver NOT restarted (NO_RESTART=1: run scripts/talk170m_run.sh)"
else
	scripts/talk170m_run.sh >/dev/null
fi
echo "pause $T0 -> $(date +%H:%M:%S): gate $VERDICT peak ${PEAK:-?} MiB, bench $BENCH (fallback $FBENCH), loss $LOSS, speed $SPEED, deployed $DEPLOYED, $RESTARTED"
