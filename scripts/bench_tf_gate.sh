#!/usr/bin/env bash
# Training TFLOP/s gate.
#
# Scores the median interval tok/s after warmup, not the cumulative
# tokens/elapsed series the progress line prints (that series only rises).
# Frozen meter: 6*P*tok/s/1e12 with P=251658240 (~2.52e8). Pad slots are not
# tokens. A flat NLL under CUDA graphs is a fail. toy_micro is smoke only.
#
#   scripts/bench_tf_gate.sh [--min-tf N] [--log PATH]
#   scripts/bench_tf_gate.sh --overfit [--log PATH]
#   scripts/bench_tf_gate.sh --self-test
#   scripts/bench_tf_gate.sh --min-tf 20 -- ./anni lm train ...
set -euo pipefail
cd "$(dirname "$0")/.."

MIN_TF=60
MODE=speed
LOG=""
SELF=0
CMD=()

usage() {
	sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
	case "$1" in
	--min-tf)
		MIN_TF="${2:?--min-tf needs a number}"
		shift 2
		;;
	--overfit)
		MODE=overfit
		shift
		;;
	--log)
		LOG="${2:?--log needs a path}"
		shift 2
		;;
	--self-test)
		SELF=1
		shift
		;;
	--help|-h)
		usage
		exit 0
		;;
	--)
		shift
		CMD=("$@")
		break
		;;
	*)
		echo "bench_tf_gate: unknown argument: $1" >&2
		usage >&2
		exit 2
		;;
	esac
done

if [ "$SELF" -eq 1 ]; then
	exec python3 scripts/bench_tf_parse.py --self-test
fi

if [ "${#CMD[@]}" -eq 0 ]; then
	if [ "$MODE" = overfit ]; then
		# Byte vocab (no --tokenizer bpe). Full next-token loss. Not toy_micro.
		CMD=(
			./anni lm train
			--data data/wiki_overfit.txt
			--epochs 20
			--batch 128
			--window 64
			--flash-attn
			--bf16
			--force
			--out /tmp/overfit.bin
			--eval-every 0
			-d
			--lr 3e-4
		)
	else
		# Pretokenized BPE-16k head. toy_micro cannot stand in for this.
		CMD=(
			./anni lm train
			--data data/wiki_ids.bin
			--epochs 1
			--batch 128
			--window 64
			--tokenizer bpe
			--flash-attn
			--bf16
			--force
			--out /tmp/tf.bin
			--eval-every 0
			-d
		)
	fi
fi

# A leftover train would share the GPU and the meter.
pkill -x anni 2>/dev/null || true

LOG_ARGS=()
if [ -n "$LOG" ]; then
	LOG_ARGS=(--log "$LOG")
fi

exec python3 scripts/bench_tf_parse.py \
	--mode "$MODE" \
	--min-tf "$MIN_TF" \
	"${LOG_ARGS[@]}" \
	-- "${CMD[@]}"
