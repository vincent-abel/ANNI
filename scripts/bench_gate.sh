#!/usr/bin/env bash
# Run the standard `anni lm bench sweep` matrix and compare it with the stored
# baseline for this GPU (docs/bench-harness.md). Never kills other processes;
# `lm bench` refuses to run on a busy GPU unless --allow-busy is passed.
# Usage:
#   scripts/bench_gate.sh --record     run the matrix, store it as the baseline
#   scripts/bench_gate.sh              run the matrix, fail on a >3% regression
#   extra args (e.g. --allow-busy, --steps 50) are passed to every sweep
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

ANN=${ANN:-./anni}
THRESHOLD=${BENCH_THRESHOLD:-0.03}
MODE=compare
EXTRA=()
for a in "$@"; do
	case "$a" in
	--record) MODE=record ;;
	*) EXTRA+=("$a") ;;
	esac
done

mkdir -p bench
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' /' '__')
gpu=${gpu:-cpu}
sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
RUN="bench/run-$sha-$(date +%Y%m%d-%H%M%S).jsonl"
BASE="bench/baseline-$gpu.jsonl"

sweep() { "$ANN" lm bench sweep --json "$RUN" "$@" "${EXTRA[@]}"; }

# Short windows want large B, long windows smaller B; plus one act-checkpoint
# point at a B that only fits with it.
rc=0
sweep --windows 64 --batches 128,256 --precs fp32,bf16,fp8 --graphs 0,1 || rc=1
sweep --windows 256 --batches 32,64 --precs fp32,bf16,fp8 --graphs 0,1 || rc=1
sweep --windows 256 --batches 128 --precs bf16 --graphs 1 --act-checkpoint || rc=1
echo "results: $RUN"

if [[ $MODE == record ]]; then
	cp "$RUN" "$BASE"
	echo "baseline stored: $BASE"
	exit $rc
fi
if [[ ! -f $BASE ]]; then
	echo "no baseline $BASE; run scripts/bench_gate.sh --record first" >&2
	exit 2
fi
python3 scripts/bench_compare.py "$BASE" "$RUN" --threshold "$THRESHOLD" || rc=1
exit $rc
