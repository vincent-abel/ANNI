#!/usr/bin/env bash
# From-scratch 555M run: tied 100k table, then the chat fine-tune.
#
#   scripts/run555m.sh --check         # CHECK_SECONDS of training, prints peak VRAM
#   BATCH=2 ACCUM=256 EXTRA=--act-checkpoint scripts/run555m.sh --check
#   scripts/run555m.sh                 # start (or reattach to) tmux session run555m
#   tmux attach -t run555m             # watch; Ctrl-b d to detach
#   tail -f logs/run555m/pretrain.log
#
# After scripts/build_code.sh. Builds a copy of HEAD in build/555m with the
# 555M sizes, so ./anni stays as it is. Each step is skipped when its marker
# exists, and pretraining resumes from models/run555m.bin. The script never
# starts or stops llama-server: free the GPU before running it.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SESSION=run555m
LOGS=logs/run555m
BUILD=build/555m
OUT=models/run555m.bin
CHAT=models/run555m-chat.bin
VOCAB=data/mix100k.vocab
TRAIN=data/mix100k_train_ids.bin
WINDOW=${WINDOW:-256}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-128}
EXTRA=${EXTRA:-}
CHECK_SECONDS=${CHECK_SECONDS:-600}
CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-13* | tail -1)}
mkdir -p "$LOGS" models

log() { echo "[$(date '+%F %T')] run555m: $*"; }

build() {
	rm -rf "$BUILD"
	mkdir -p "$BUILD"
	git archive HEAD | tar -x -C "$BUILD"
	sed -i -e 's/\(define LM_D_MODEL\t\).*/\11536/' \
	    -e 's/\(define LM_N_HEADS\t\).*/\124/' \
	    -e 's/\(define LM_D_FF\t\).*/\16272/' \
	    -e 's/\(define LM_N_BLOCKS\t\).*/\114/' "$BUILD/lang/lm_config.h"
	make -C "$BUILD" -j6 cuda CUDA_HOME="$CUDA_HOME"
}

# Stop unless the GPU is free: nothing here may push another job out.
need_free_gpu() {
	local vram
	vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
	if [ "$vram" -ge 2048 ]; then
		log "GPU holds ${vram} MiB; free it first"
		exit 1
	fi
}

# train START OUT LOG: the pretraining command.
train() {
	local steps
	steps=$(( $(stat -c %s "$TRAIN") / 4 / (WINDOW * BATCH * ACCUM) ))
	ANN_GNORM_EVERY=50 ANN_TWIN_EMIT=1 "$BUILD/anni" lm train $1 --out "$2" \
	    --data "$TRAIN" --val-data data/mix100k_val_ids.bin \
	    --tokenizer bpe --vocab "$VOCAB" --tied \
	    --window "$WINDOW" --stride "$WINDOW" --batch "$BATCH" \
	    --grad-accum "$ACCUM" --epochs 1 \
	    --lr 3e-4 --lr-warmup 1000 --lr-decay cosine --lr-decay-steps "$steps" \
	    --adam-beta2 0.95 --grad-clip 1.0 --weight-decay 0.1 \
	    --flash-attn --bf16 --eval-every 1 --eval-windows 256 \
	    --ckpt-minutes 30 --seed 1 $EXTRA -d
}

pretrain() {
	local start=--force
	[ -s "$OUT" ] && start="--resume $OUT"
	train "$start" "$OUT"
}

# Constant LR: the schedule follows Adam's step count, already past the decay.
finetune() {
	"$BUILD/anni" lm train --resume "$OUT" --out "$CHAT" --force \
	    --data data/chat100k_train_mids.bin --val-data data/chat100k_val_mids.bin \
	    --tokenizer bpe --vocab "$VOCAB" \
	    --window "$WINDOW" --stride "$WINDOW" --batch "$BATCH" --grad-accum 32 \
	    --epochs 2 --lr 5e-5 --lr-decay none --grad-clip 1.0 --weight-decay 0.1 \
	    --flash-attn --bf16 --eval-every 1 --eval-windows 256 --seed 2 -d
}

step() {	# step NAME MARKER CMD...
	local name=$1 marker=$2
	shift 2
	if [ -e "$marker" ]; then
		log "skip $name ($marker exists)"
		return 0
	fi
	log "run $name -> $LOGS/$name.log"
	"$@" >> "$LOGS/$name.log" 2>&1
	touch "$marker"
	log "done $name"
}

if [ "${1:-}" = "--check" ]; then
	need_free_gpu
	[ -x "$BUILD/anni" ] || build > "$LOGS/build.log" 2>&1
	nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 \
	    > "$LOGS/check_vram.log" &
	smi=$!
	timeout "$CHECK_SECONDS" bash -c "$(declare -f train); $(declare -p TRAIN WINDOW BATCH ACCUM BUILD VOCAB EXTRA); \
	    train --force models/run555m-check.bin" > "$LOGS/check.log" 2>&1 || true
	kill "$smi"
	rm -f models/run555m-check.bin
	log "peak VRAM $(sort -n "$LOGS/check_vram.log" | tail -1) MiB"
	tail -5 "$LOGS/check.log"
	exit 0
fi
if [ "${1:-}" != "--inside" ]; then
	if tmux has-session -t "=$SESSION" 2>/dev/null; then
		log "already running in tmux session $SESSION"
		exit 0
	fi
	tmux new-session -d -s "$SESSION" "$ROOT/scripts/run555m.sh --inside 2>&1 | tee -a $ROOT/$LOGS/driver.log"
	log "started tmux session $SESSION (tmux attach -t $SESSION)"
	exit 0
fi

need_free_gpu
step build "$LOGS/.built" build
step pretrain "$LOGS/.pretrained" pretrain
step finetune "$LOGS/.chat" finetune
log "all steps done"
