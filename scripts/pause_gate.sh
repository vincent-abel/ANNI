#!/usr/bin/env bash
# Pause-window deploy: gate a candidate binary and resume the talk170m run.
#
#   scripts/pause_gate.sh CANDIDATE FALLBACK [flags...]
#
# Run right after the training process was interrupted (Ctrl-C in tmux
# talk170m following a checkpoint). Installs CANDIDATE as ./anni, runs the
# deploy gate (scripts/prof_step.sh gate: validation probes, checkpoint save,
# peak VRAM against the 85% cap) and a plain 12-step bench, compares the
# bench loss trajectory with logs/prof/bench_reference.log when present,
# keeps CANDIDATE on PASS or restores FALLBACK, then restarts the driver.
# Prints a one-paragraph report. Total under 5 minutes.
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
cp "$CAND" anni
GATE=$(scripts/prof_step.sh gate $FLAGS 2>&1 | tail -3)
PEAK=$(echo "$GATE" | grep -o 'peak VRAM [0-9]*' | grep -o '[0-9]*$')
VERDICT=$(echo "$GATE" | grep -o 'GATE [A-Z]*')
BENCH=$(scripts/prof_step.sh bench $FLAGS 2>&1 | tr '\r' '\n' | grep -a '^ep 1/1' | sed -n '9p' | grep -o '[0-9]*t/s ~[0-9.]*TF')
TAG=$(echo "$FLAGS" | tr -c 'a-z0-9' '_' | sed 's/_*$//')
LOSS=same
if [ -f logs/prof/bench_reference.log ]; then
	A=$(tr '\r' '\n' < logs/prof/bench_reference.log | grep -a '^ep 1/1' | grep -a -o 'NLL [0-9.]*' | head -12 | tr '\n' ' ')
	B=$(tr '\r' '\n' < "logs/prof/bench_$TAG.log" | grep -a '^ep 1/1' | grep -a -o 'NLL [0-9.]*' | head -12 | tr '\n' ' ')
	[ "$A" = "$B" ] || LOSS="DIFFERENT (ref: $A | new: $B)"
fi
if [ "$VERDICT" = "GATE PASS" ] && [ "$LOSS" = same ]; then
	DEPLOYED=candidate
else
	cp "$FALL" anni
	DEPLOYED=fallback
fi
scripts/talk170m_run.sh >/dev/null
echo "pause $T0 -> $(date +%H:%M:%S): gate $VERDICT peak ${PEAK:-?} MiB, bench $BENCH, loss $LOSS, deployed $DEPLOYED, driver restarted"
