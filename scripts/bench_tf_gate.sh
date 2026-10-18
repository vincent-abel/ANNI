#!/usr/bin/env bash
# Gate AnnI train runs on printed TF / tok/s / NLL sanity.
# Usage: scripts/bench_tf_gate.sh [--min-tf N] [--overfit] [--] anni-args...
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MIN_TF=60
MODE=speed
EXTRA=()
while [[ $# -gt 0 ]]; do
	case "$1" in
	--min-tf) MIN_TF=$2; shift 2 ;;
	--overfit) MODE=overfit; shift ;;
	--) shift; EXTRA+=("$@"); break ;;
	*) EXTRA+=("$1"); shift ;;
	esac
done

ANN=${ANN:-./anni}
LOG=$(mktemp /tmp/anni_tf_XXXXXX.log)
trap 'rm -f "$LOG"' EXIT
# OpenMP host attn + shared cuBLAS scratch races without serialization; pin to 1.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

pkill -x anni 2>/dev/null || true
sleep 0.2

if [[ ${#EXTRA[@]} -eq 0 ]]; then
	if [[ $MODE == overfit ]]; then
		EXTRA=(lm train --data data/wiki_overfit.txt --epochs 20 --batch 120
			--window 64 --flash-attn --bf16 --force --out /tmp/overfit.bin
			--eval-every 0 -d --lr 3e-4)
	else
		# Prefer pretokenized ids if present; else toy_micro smoke.
		if [[ -f data/wiki_ids.bin ]]; then
			EXTRA=(lm train --data data/wiki_ids.bin --epochs 1 --batch 120
				--window 64 --tokenizer bpe --vocab model-wiki-16k.bin.vocab
				--flash-attn --bf16 --force --out /tmp/tf.bin
				--eval-every 0 -d --lr 3e-4)
		else
			EXTRA=(lm train --data data/toy_micro.txt --epochs 40 --batch 120
				--window 64 --flash-attn --bf16 --force --out /tmp/tf.bin
				--eval-every 0 -d)
		fi
	fi
fi

echo "bench_tf_gate: min-tf=$MIN_TF mode=$MODE"
echo "+ $ANN ${EXTRA[*]}"
set +e
# Normalize CR progress lines; keep going until process exits.
stdbuf -oL -eL "$ANN" "${EXTRA[@]}" 2>&1 | tr '\r' '\n' | tee "$LOG"
# PIPESTATUS[0] is anni; under timeout/SIGPIPE it may be empty with set -u.
RC=${PIPESTATUS[0]:-1}
set -e

python3 "$ROOT/scripts/parse_tf_gate.py" --log "$LOG" --min-tf "$MIN_TF" \
	--mode "$MODE" --exit-code "$RC"
