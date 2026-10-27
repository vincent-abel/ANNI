#!/usr/bin/env bash
# Short FP32 / BF16 / FP8 A/B on the fixed training path (docs/gpu-training-audit.md).
#
#   scripts/ab_precision.sh [STEPS] [fp32|bf16|fp8 ...]   # default 150 steps, all three
#
# Same seed, same data slice (the first STEPS*16*256*32 tokens of the mix),
# same flags as scripts/talk170m_run.sh pretrain(). Needs the GPU to itself
# (stop janni-llama-server and the training run first). Writes
# logs/ab_<prec>.log and prints the final train NLL, tok/s and TFLOP/s of each.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-150}
shift || true
PRECS=${*:-fp32 bf16 fp8}
TOK_PER_STEP=$((16 * 256 * 32))
BYTES=$((STEPS * TOK_PER_STEP * 4 + 4096 * 4))
SLICE=data/ab_train_ids.bin
mkdir -p logs models/ab
head -c "$BYTES" data/mix_train_ids.bin > "$SLICE"
export ANN_GNORM_EVERY=10
run() {	# run NAME [extra flags...]
	local name=$1
	shift
	rm -f "models/ab/$name.bin"
	echo "== $name"
	./anni lm train --force --out "models/ab/$name.bin" \
	    --data "$SLICE" --val-data data/mix_val_ids.bin \
	    --tokenizer bpe --vocab data/mix.vocab \
	    --window 256 --stride 256 --batch 16 --grad-accum 32 --epochs 1 \
	    --lr 3e-4 --lr-warmup 1000 --lr-decay cosine --lr-decay-steps 28100 \
	    --adam-beta2 0.95 --grad-clip 1.0 --weight-decay 0.1 \
	    --flash-attn --eval-every 1 --eval-windows 256 --seed 1 -d "$@" \
	    2>&1 | tee "logs/ab_$name.log" | tr '\r' '\n' | grep -a --line-buffered '\[adam\]\|NLL' | tail -3
}
for p in $PRECS; do
	case $p in
	fp32) run fp32 ;;
	bf16) run bf16 --bf16 ;;
	fp8) run fp8 --bf16 --fp8 ;;
	esac
done
echo "== summary (last progress line of each)"
for p in $PRECS; do
	printf '%-5s ' "$p"
	tr '\r' '\n' < "logs/ab_$p.log" | grep -a 'NLL' | tail -1 | sed 's/^ *//'
done
