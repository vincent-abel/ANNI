#!/usr/bin/env bash
# Train the 100k vocab on the whole 555M mix and encode everything with it.
#
#   scripts/build_code.sh                # after scripts/fetch_code.py
#   JOBS=4 SAMPLE_MB=1500 CODE_PCT=35 VOCAB=100000 scripts/build_code.sh
#
# Output (gitignored):
#   data/mix100k.vocab               BPE trained on a web/code sample
#   data/mix100k_train_ids.bin       all train text, int32 LE; the trainer
#                                    shuffles windows, so the order is harmless
#   data/mix100k_val_ids.bin         held-out web + code
#   data/mix100k_val_web_ids.bin     held-out web only (per-source loss)
#   data/mix100k_val_code_ids.bin    held-out code only
#   data/chat100k_{train,val}_mids.bin   chat fine-tune set
#
# pretok_ids reads a whole file and counts ids in an int, so the text is
# encoded in pieces split at line ends, JOBS at a time.
set -euo pipefail

cd "$(dirname "$0")/.."
JOBS="${JOBS:-4}"
PIECE_MB="${PIECE_MB:-512}"
SAMPLE_MB="${SAMPLE_MB:-1500}"
CODE_PCT="${CODE_PCT:-35}"
VOCAB="${VOCAB:-100000}"
WORK="${WORK:-data/code2_work}"
OUT=data/mix100k
CFLAGS="-O2 -std=c99 -I. -Ilang -Inum -Icompat"
BPE_SRC="lang/bpe.c lang/vocab.c lang/tokenize.c"

for f in data/code2.txt data/web2.txt data/mix_web.txt data/mix_code.txt \
    data/mix_val_web.txt data/mix_val_code.txt data/chat_train.txt data/chat_val.txt; do
	[ -s "$f" ] || { echo "build_code: missing $f" >&2; exit 1; }
done
for tool in bpe_train pretok_ids pretok_chat; do
	gcc $CFLAGS "scripts/$tool.c" $BPE_SRC -o "scripts/$tool" -lm
done

# sample_slices SRC MB: ten evenly spaced slices.
sample_slices() {
	local size step i
	size=$(stat -c %s "$1")
	step=$((size / 10))
	for i in $(seq 0 9); do
		dd if="$1" iflag=skip_bytes,count_bytes skip=$((i * step)) \
		    count=$(($2 * 1024 * 1024 / 10)) status=none
	done
}
if [ ! -s "$OUT.vocab" ]; then
	rm -rf "$WORK"
	mkdir -p "$WORK"
	code_mb=$((SAMPLE_MB * CODE_PCT / 100))
	{ sample_slices data/web2.txt $((SAMPLE_MB - code_mb))
	  sample_slices data/code2.txt "$code_mb"; } > "$WORK/sample.txt"
	echo "build_code: training BPE vocab $VOCAB on $(du -h "$WORK/sample.txt" | cut -f1) sample"
	scripts/bpe_train "$WORK/sample.txt" "$VOCAB" "$OUT.vocab.part"
	mv "$OUT.vocab.part" "$OUT.vocab"
	rm -rf "$WORK"
fi

# encode SRC OUT
encode() {
	[ -s "$2" ] && return
	rm -rf "$WORK"
	mkdir -p "$WORK"
	split -C "${PIECE_MB}M" -d -a 3 "$1" "$WORK/piece_"
	ls "$WORK"/piece_* | xargs -P "$JOBS" -I{} sh -c \
	    'scripts/pretok_ids "$2" "$1" "$1.ids" >/dev/null && rm -f "$1"' _ {} "$OUT.vocab"
	cat "$WORK"/piece_*.ids > "$2.part"
	mv "$2.part" "$2"
	rm -rf "$WORK"
	echo "build_code: $2 $(( $(stat -c %s "$2") / 4 )) ids"
}
encode data/mix_val_web.txt "${OUT}_val_web_ids.bin"
encode data/mix_val_code.txt "${OUT}_val_code_ids.bin"
cat "${OUT}_val_web_ids.bin" "${OUT}_val_code_ids.bin" > "${OUT}_val_ids.bin"
encode data/code2.txt data/code2_ids.bin
encode data/mix_code.txt data/code1_ids.bin
encode data/web2.txt data/web2_ids.bin
encode data/mix_web.txt data/web1_ids.bin
cat data/web1_ids.bin data/web2_ids.bin data/code1_ids.bin data/code2_ids.bin \
    > "${OUT}_train_ids.bin.part"
mv "${OUT}_train_ids.bin.part" "${OUT}_train_ids.bin"
rm -f data/web1_ids.bin data/web2_ids.bin data/code1_ids.bin data/code2_ids.bin
echo "build_code: ${OUT}_train_ids.bin $(( $(stat -c %s "${OUT}_train_ids.bin") / 4 )) ids"
scripts/pretok_chat "$OUT.vocab" data/chat_train.txt data/chat100k_train_mids.bin
scripts/pretok_chat "$OUT.vocab" data/chat_val.txt data/chat100k_val_mids.bin
