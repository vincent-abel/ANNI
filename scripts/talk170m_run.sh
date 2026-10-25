#!/usr/bin/env bash
# Resumable driver for the talk-170m pipeline (docs/talk-170m-plan.md).
#
#   scripts/talk170m_run.sh            # start (or reattach to) tmux session talk170m
#   scripts/talk170m_run.sh --boot     # same, after a 5 min wait (for @reboot)
#   tmux attach -t talk170m            # watch; Ctrl-b d to detach
#   tail -f logs/talk170m_pretrain.log
#
# Crash recovery (installed by hand): crontab -e and add
#   @reboot /home/kolymine/claude-projects/ann/scripts/talk170m_run.sh --boot
# After WSL comes back it waits 5 min, stops llama-server and resumes.
#
# Every step is idempotent: it is skipped when its marker exists, and
# pretraining resumes from models/talk170m.bin (a mid-epoch checkpoint keeps
# the window it reached; --ckpt-minutes sets how often it is written).
# Before any GPU or heavy step, janni-llama-server is stopped with systemctl.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SESSION=talk170m
LOGS=logs
mkdir -p "$LOGS" models

if [ "${1:-}" = "--boot" ]; then
	sleep 300
	shift
fi
if [ "${1:-}" != "--inside" ]; then
	if tmux has-session -t "=$SESSION" 2>/dev/null; then
		echo "talk170m: already running in tmux session $SESSION"
		exit 0
	fi
	tmux new-session -d -s "$SESSION" "$ROOT/scripts/talk170m_run.sh --inside 2>&1 | tee -a $ROOT/$LOGS/talk170m_driver.log"
	echo "talk170m: started tmux session $SESSION (tmux attach -t $SESSION)"
	exit 0
fi

PY="$HOME/.venvs/anni-data/bin/python"
SMI=/usr/lib/wsl/lib/nvidia-smi
command -v nvidia-smi >/dev/null && SMI=nvidia-smi
NVCC="$HOME/cuda13/usr/local/cuda-13.0/bin/nvcc"
OUT=models/talk170m.bin
CHAT=models/talk170m-chat.bin

log() { echo "[$(date '+%F %T')] talk170m: $*"; }

stop_llama() {
	export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
	if systemctl --user is-active --quiet janni-llama-server 2>/dev/null; then
		log "stopping janni-llama-server (restart: systemctl --user start janni-llama-server)"
		systemctl --user stop janni-llama-server
	fi
}

# Wait until the GPU holds < 2 GB and RAM has > 30 GB available.
wait_free() {
	local vram avail
	for _ in $(seq 1 60); do
		vram=$($SMI --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
		avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
		if [ "$vram" -lt 2048 ] && [ "$avail" -gt 30 ]; then
			log "free: VRAM used ${vram} MiB, RAM available ${avail} GiB"
			return 0
		fi
		log "waiting: VRAM used ${vram} MiB, RAM available ${avail} GiB"
		sleep 10
	done
	log "GPU or RAM still busy after 10 min; stopping"
	exit 1
}

step() {	# step NAME MARKER CMD...
	local name=$1 marker=$2
	shift 2
	if [ -e "$marker" ]; then
		log "skip $name ($marker exists)"
		return 0
	fi
	log "run $name -> $LOGS/talk170m_$name.log"
	"$@" >> "$LOGS/talk170m_$name.log" 2>&1
	touch "$marker"
	log "done $name"
}

build_cuda() {
	# No header dependencies in the Makefile: always rebuild from clean.
	make clean
	make -j6 cuda NVCC="$NVCC -Xcompiler -U_GNU_SOURCE"
}

chat_ids() {
	gcc -O2 -std=c99 -I. -Ilang -Inum -Icompat scripts/pretok_chat.c \
	    lang/bpe.c lang/vocab.c lang/tokenize.c -o scripts/pretok_chat -lm
	scripts/pretok_chat data/mix.vocab data/chat_train.txt data/chat_train_mids.bin
	scripts/pretok_chat data/mix.vocab data/chat_val.txt data/chat_val_mids.bin
}

# Chat fine-tune from the pretrained weights and Adam state. Short, so a
# crash restarts it from the pretrained file. Constant LR: the schedule
# follows Adam's step count, which is already past the pretraining decay.
finetune() {
	./anni lm train --resume "$OUT" --out "$CHAT" --force \
	    --data data/chat_train_mids.bin --val-data data/chat_val_mids.bin \
	    --tokenizer bpe --vocab data/mix.vocab \
	    --window 256 --stride 256 --batch 16 --grad-accum 8 --epochs 2 \
	    --lr 5e-5 --lr-decay none --grad-clip 1.0 --weight-decay 0.1 \
	    --flash-attn --eval-every 1 --eval-windows 256 --seed 2 -d
}

pretrain() {
	export ANN_GNORM_EVERY=50
	local start=--force
	[ -s "$OUT" ] && start="--resume $OUT"
	log "pretrain start flag: $start"
	./anni lm train $start --out "$OUT" \
	    --data data/mix_train_ids.bin --val-data data/mix_val_ids.bin \
	    --tokenizer bpe --vocab data/mix.vocab \
	    --window 256 --stride 256 --batch 16 --grad-accum 32 --epochs 1 \
	    --lr 3e-4 --lr-warmup 1000 --lr-decay cosine --lr-decay-steps 28100 --adam-beta2 0.95 \
	    --grad-clip 1.0 --weight-decay 0.1 \
	    --flash-attn \
	    --eval-every 1 --eval-windows 256 --ckpt-minutes 30 --seed 1 -d
}

# Finished: leave llama-server alone.
if [ -e models/.talk170m_chat ]; then
	log "pipeline finished; nothing to do"
	exit 0
fi
stop_llama
step fetch_mix data/mix.SOURCE "$PY" -I scripts/fetch_mix.py
step build_mix data/.mix_built scripts/build_mix.sh
step fetch_chat data/chat.SOURCE "$PY" scripts/fetch_chat.py
step build_cuda models/.cuda_built build_cuda
stop_llama
wait_free
step pretrain models/.talk170m_pretrained pretrain
step build_cuda_mask models/.cuda_built_mask build_cuda
step chat_ids data/.chat_ids chat_ids
stop_llama
wait_free
step finetune models/.talk170m_chat finetune
log "all steps done; restarting janni-llama-server"
systemctl --user start janni-llama-server || log "llama-server start failed"
