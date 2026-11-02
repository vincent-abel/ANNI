#!/usr/bin/env bash
# Extend talk-170m from a 256-token to a 2048-token context (issue #3):
# continued pretraining on data/ctx_train_ids.bin at window 2048, then the
# chat fine-tune again at 2048.
#
#   scripts/ctx2048_run.sh             # start (or reattach to) tmux session ctx2048
#   tmux attach -t ctx2048             # watch; Ctrl-b d to detach
#   tail -f logs/ctx2048/extend.log
#
# data/ctx_train_ids.bin is a 0.4B-token cut of data/mix_train_ids.bin
# (340M web ids from offset 1.5B, then the last 60M ids, which are code).
# Each step is skipped when its marker exists, and an interrupted step
# resumes from its own output checkpoint. The script never starts or stops
# llama-server: free the GPU before running it.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SESSION=ctx2048
LOGS=logs/ctx2048
ANNI=${ANNI:-./anni}
mkdir -p "$LOGS" models

if [ "${1:-}" != "--inside" ]; then
	if tmux has-session -t "=$SESSION" 2>/dev/null; then
		echo "ctx2048: already running in tmux session $SESSION"
		exit 0
	fi
	tmux new-session -d -s "$SESSION" "$ROOT/scripts/ctx2048_run.sh --inside 2>&1 | tee -a $ROOT/$LOGS/driver.log"
	echo "ctx2048: started tmux session $SESSION (tmux attach -t $SESSION)"
	exit 0
fi

BASE=models/talk170m.bin
EXT=models/talk170m-2k.bin
CHAT=models/talk170m-2k-chat.bin
COMMON="--tokenizer bpe --vocab data/mix.vocab --window 2048 --stride 2048 \
    --batch 2 --grad-clip 1.0 --weight-decay 0.1 --flash-attn --bf16 \
    --eval-every 1 --ckpt-minutes ${CKPT_MIN:-27} -d"

log() { echo "[$(date '+%F %T')] ctx2048: $*"; }

# start IN OUT: a fresh run from IN, or the resume of an interrupted OUT.
start() {
	if [ -s "$2" ]; then
		echo "--resume $2"
	else
		echo "--resume $1 --force"
	fi
}

if [ ! -e models/.ctx2048_extended ]; then
	log "extend: $(start $BASE $EXT) -> $LOGS/extend.log"
	# 400M ids / 2048 = 195312 windows, 64 per Adam step: 3052 steps,
	# warmup and cosine counted from talk170m.bin's last Adam step (29084).
	env ANN_TWIN_EMIT=1 $ANNI lm train $(start $BASE $EXT) --out "$EXT" \
	    --data data/ctx_train_ids.bin --val-data data/mix_val_ids.bin \
	    $COMMON --grad-accum 32 --epochs 1 \
	    --lr 1e-4 --lr-warmup 100 --lr-decay cosine --lr-decay-steps 3052 \
	    --lr-sched-from 29084 \
	    --adam-beta2 0.95 --eval-windows 64 --seed 3 \
	    >> "$LOGS/extend.log" 2>&1
	touch models/.ctx2048_extended
	log "done extend"
fi

if [ ! -e models/.ctx2048_chat ]; then
	log "chat: $(start $EXT $CHAT) -> $LOGS/chat.log"
	env ANN_TWIN_EMIT=1 $ANNI lm train $(start $EXT $CHAT) --out "$CHAT" \
	    --data data/chat_train_mids.bin --val-data data/chat_val_mids.bin \
	    $COMMON --grad-accum 8 --epochs 2 \
	    --lr 5e-5 --lr-decay none --eval-windows 64 --seed 4 \
	    >> "$LOGS/chat.log" 2>&1
	touch models/.ctx2048_chat
	log "done chat"
fi
log "all steps done"
