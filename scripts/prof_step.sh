#!/usr/bin/env bash
# Profile / bench one short pretraining configuration (GPU must be free).
#
#   scripts/prof_step.sh bench [flags...]   # ~12 steps, prints tok/s, TF, fwd/bwd ms
#   scripts/prof_step.sh nsys  [flags...]   # ~4 steps under nsys, kernel time table
#   scripts/prof_step.sh gate  [flags...]   # deploy gate: 12 steps with validation
#                                           # probes and a checkpoint save while
#                                           # nvidia-smi samples VRAM; prints the
#                                           # peak and PASS/FAIL against 85% (13.9 GB)
#
# flags default to the pretrain() precision/attention flags of talk170m_run.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-bench}
shift || true
FLAGS=${*:---bf16 --flash-attn}
STEPS=$([ "$MODE" = nsys ] && echo 4 || echo 12)
SMI=/usr/lib/wsl/lib/nvidia-smi
command -v nvidia-smi >/dev/null && SMI=nvidia-smi
VRAM_CAP_MIB=13900
SLICE=data/prof_ids.bin
head -c $((STEPS * 16*256*32 * 4 + 16384)) data/mix_train_ids.bin > "$SLICE"
mkdir -p logs/prof
TAG=$(echo "$FLAGS" | tr -c 'a-z0-9' '_' | sed 's/_*$//')
CMD=(./anni lm train --force --out logs/prof/model.bin --data "$SLICE" \
    --tokenizer bpe --vocab data/mix.vocab --window 256 --stride 256 \
    --batch 16 --grad-accum 32 --epochs 1 --lr 3e-4 --lr-warmup 1000 \
    --adam-beta2 0.95 --grad-clip 1.0 --weight-decay 0.1 \
    --eval-every 0 --eval-windows 8 --seed 1 -d $FLAGS)
if [ "$MODE" = gate ]; then
	# Same flags as the run, plus a 32-window validation probe at the start
	# and every 5%, and a checkpoint save after one minute.
	CMD=(./anni lm train --force --out logs/prof/model.bin --data "$SLICE" \
	    --val-data data/mix_val_ids.bin \
	    --tokenizer bpe --vocab data/mix.vocab --window 256 --stride 256 \
	    --batch 16 --grad-accum 32 --epochs 1 --lr 3e-4 --lr-warmup 1000 \
	    --adam-beta2 0.95 --grad-clip 1.0 --weight-decay 0.1 \
	    --eval-every 1 --eval-windows 32 --ckpt-minutes 1 --seed 1 -d $FLAGS)
	( while :; do $SMI --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 0.5; done ) > "logs/prof/gate_vram_$TAG.txt" &
	SMI_PID=$!
	"${CMD[@]}" > "logs/prof/gate_$TAG.log" 2>&1 || true
	kill $SMI_PID 2>/dev/null
	PEAK=$(sort -n "logs/prof/gate_vram_$TAG.txt" | tail -1)
	tr '\r' '\n' < "logs/prof/gate_$TAG.log" | grep -a 'val NLL\|saved checkpoint\|failed\|^ep 1/1' | sed -n '1,3p;$p' | cut -c1-120
	tr '\r' '\n' < "logs/prof/gate_$TAG.log" | grep -a '^ep 1/1' | sed -n '9p' | cut -c40-110
	echo "peak VRAM ${PEAK:-?} MiB (cap $VRAM_CAP_MIB)"
	if [ -n "$PEAK" ] && [ "$PEAK" -le "$VRAM_CAP_MIB" ] && ! grep -aq 'failed' "logs/prof/gate_$TAG.log"; then
		echo "GATE PASS"
	else
		echo "GATE FAIL"
	fi
elif [ "$MODE" = nsys ]; then
	"${NSYS:-$HOME/cuda13/usr/local/cuda-13.0/bin/nsys}" profile -t cuda,nvtx --stats=false -f true -o "logs/prof/nsys_$TAG" \
	    "${CMD[@]}" > "logs/prof/nsys_$TAG.log" 2>&1 || true
	"${NSYS:-$HOME/cuda13/usr/local/cuda-13.0/bin/nsys}" stats --report cuda_gpu_kern_sum --format csv "logs/prof/nsys_$TAG.nsys-rep" 2>/dev/null \
	    | python3 -c '
import sys,csv
rows=[r for r in csv.reader(sys.stdin) if r and r[0].replace(".","").isdigit()]
tot=sum(float(r[1]) for r in rows)
print("total kernel ns %.3e" % tot)
for r in sorted(rows,key=lambda r:-float(r[1]))[:25]:
    print("%6.2f%% %8.1fms %6s  %s" % (100*float(r[1])/tot, float(r[1])/1e6, r[2], r[-1][:90]))'
else
	ANN_STEP_PROF=1 "${CMD[@]}" 2>&1 | tee "logs/prof/bench_$TAG.log" | tr '\r' '\n' \
	    | grep -a --line-buffered 'stepprof\|\[loop\]\|\[adam\] [0-9.]*ms\|^ep 1/1' | tail -14 | cut -c1-150
fi
rm -f logs/prof/model.bin logs/prof/model.bin.best
