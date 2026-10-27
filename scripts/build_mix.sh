#!/usr/bin/env bash
# Turn the fetched talk-170m mix into a BPE vocab and packed int32 id files.
#
#   scripts/build_mix.sh                 # after scripts/fetch_mix.py
#   SAMPLE_MB=300 VOCAB=16384 scripts/build_mix.sh
#
# Output (gitignored):
#   data/mix.vocab              16k BPE trained on a web/code sample
#   data/mix_train_ids.bin      all train text, int32 LE
#   data/mix_val_ids.bin        held-out web + code
#   data/mix_val_web_ids.bin    held-out web only (per-source loss)
#   data/mix_val_code_ids.bin   held-out code only
#
# pretok_ids reads a whole file and counts ids in an int, so text is encoded
# in pieces of PIECE_MB (split at line ends) and the id files concatenated.
set -euo pipefail

cd "$(dirname "$0")/.."
SAMPLE_MB="${SAMPLE_MB:-300}"
CODE_PCT="${CODE_PCT:-15}"
VOCAB="${VOCAB:-16384}"
PIECE_MB="${PIECE_MB:-1024}"
WORK="${WORK:-data/mix_work}"
CFLAGS="-O2 -std=c99 -I. -Ilang -Inum -Icompat"
BPE_SRC="lang/bpe.c lang/vocab.c lang/tokenize.c"

for f in data/mix_web.txt data/mix_code.txt data/mix_val_web.txt data/mix_val_code.txt; do
	[ -s "$f" ] || { echo "build_mix: missing $f (run scripts/fetch_mix.py)" >&2; exit 1; }
done
mkdir -p "$WORK"
gcc $CFLAGS scripts/bpe_train.c $BPE_SRC -o scripts/bpe_train -lm
gcc $CFLAGS scripts/pretok_ids.c $BPE_SRC -o scripts/pretok_ids -lm

# Sample: ten evenly spaced slices per source, CODE_PCT percent code.
sample_slices() {
	local src=$1 total_mb=$2 size step i
	size=$(stat -c %s "$src")
	step=$((size / 10))
	for i in $(seq 0 9); do
		dd if="$src" iflag=skip_bytes,count_bytes skip=$((i * step)) \
		    count=$((total_mb * 1024 * 1024 / 10)) status=none
	done
}
if [ ! -s data/mix.vocab ]; then
	code_mb=$((SAMPLE_MB * CODE_PCT / 100))
	{ sample_slices data/mix_web.txt $((SAMPLE_MB - code_mb))
	  sample_slices data/mix_code.txt "$code_mb"; } > "$WORK/sample.txt"
	echo "build_mix: training BPE vocab $VOCAB on $(du -h "$WORK/sample.txt" | cut -f1) sample"
	scripts/bpe_train "$WORK/sample.txt" "$VOCAB" data/mix.vocab
fi

# encode SRC OUT: split at line ends, encode each piece, concatenate ids.
encode() {
	local src=$1 out=$2 piece
	rm -f "$WORK"/piece_* "$out.part"
	split -C "${PIECE_MB}M" -d -a 3 "$src" "$WORK/piece_"
	for piece in "$WORK"/piece_*; do
		scripts/pretok_ids data/mix.vocab "$piece" "$piece.ids" >/dev/null
		cat "$piece.ids" >> "$out.part"
		rm -f "$piece" "$piece.ids"
	done
	mv "$out.part" "$out"
	echo "build_mix: $out $(( $(stat -c %s "$out") / 4 )) ids from $src"
}
encode data/mix_val_web.txt data/mix_val_web_ids.bin
encode data/mix_val_code.txt data/mix_val_code_ids.bin
cat data/mix_val_web_ids.bin data/mix_val_code_ids.bin > data/mix_val_ids.bin
encode data/mix_code.txt "$WORK/code_ids.bin"
encode data/mix_web.txt "$WORK/web_ids.bin"
cat "$WORK/web_ids.bin" "$WORK/code_ids.bin" > data/mix_train_ids.bin
rm -f "$WORK/web_ids.bin" "$WORK/code_ids.bin"
echo "build_mix: data/mix_train_ids.bin $(( $(stat -c %s data/mix_train_ids.bin) / 4 )) ids"
