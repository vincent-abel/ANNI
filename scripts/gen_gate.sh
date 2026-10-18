#!/usr/bin/env bash
# Gate `anni lm generate` changes on greedy text and steady t/s.
# Usage:
#   scripts/gen_gate.sh --record     write references + baseline median
#   scripts/gen_gate.sh              text must match, median must beat last by >5%
#   scripts/gen_gate.sh --no-drop    text must match, median must stay >= 95% of last
#   add --keep to store this median as the new "last kept" on PASS
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

ANN=${ANN:-./anni}
MODEL=${GEN_GATE_MODEL:-/tmp/overfit_fp8.bin}
DIR=${GEN_GATE_DIR:-/tmp/gen_gate}
TOKENS=${GEN_GATE_TOKENS:-200}
RUNS=${GEN_GATE_RUNS:-3}
PROMPTS=("April (Apr.) is the fourth" "April always begins on the same day")
MODE=gain
KEEP=0
for a in "$@"; do
	case "$a" in
	--record) MODE=record ;;
	--no-drop) MODE=nodrop ;;
	--keep) KEEP=1 ;;
	*) echo "gen_gate: unknown arg $a" >&2; exit 2 ;;
	esac
done
mkdir -p "$DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# run_one PROMPT OUT ERR
run_one() {
	"$ANN" lm generate --model "$MODEL" --prompt "$1" --tokens "$TOKENS" \
	    --temperature 0 >"$2" 2>"$3" </dev/null
}

steady_of() {
	sed -n 's/.*steady \([0-9.]*\) t\/s.*/\1/p' "$1" | tail -1
}

fail=0
for i in "${!PROMPTS[@]}"; do
	out="$DIR/out$i.txt"
	err="$DIR/err$i.txt"
	if ! run_one "${PROMPTS[$i]}" "$out" "$err"; then
		echo "FAIL text: prompt $i exited non-zero"; tail -3 "$err"; fail=1
		continue
	fi
	if [[ $MODE == record ]]; then
		cp "$out" "$DIR/ref$i.txt"
		echo "recorded ref$i: $(head -c 80 "$out" | tr '\n' ' ')"
	elif ! cmp -s "$out" "$DIR/ref$i.txt"; then
		pos=$(cmp "$out" "$DIR/ref$i.txt" 2>/dev/null | sed -n 's/.*byte \([0-9]*\).*/\1/p')
		echo "FAIL text: prompt $i differs at byte ${pos:-EOF}"
		echo "  got: $(tail -c +"$(( ${pos:-1} > 20 ? ${pos:-1} - 20 : 1 ))" "$out" | head -c 60 | tr '\n' ' ')"
		echo "  ref: $(tail -c +"$(( ${pos:-1} > 20 ? ${pos:-1} - 20 : 1 ))" "$DIR/ref$i.txt" | head -c 60 | tr '\n' ' ')"
		fail=1
	else
		echo "ok   text: prompt $i matches ($(wc -c <"$out") bytes)"
	fi
done

rates=()
for ((r = 0; r < RUNS; r++)); do
	run_one "${PROMPTS[0]}" "$DIR/speed_out.txt" "$DIR/speed_err.txt" || true
	rates+=("$(steady_of "$DIR/speed_err.txt")")
done
median=$(printf '%s\n' "${rates[@]}" | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
echo "runs: ${rates[*]}  median steady: ${median} t/s"

if [[ $MODE == record ]]; then
	echo "$median" >"$DIR/last_median"
	[[ $fail -eq 0 ]] && echo "PASS (recorded)" || echo "FAIL (record)"
	exit $fail
fi

last=$(cat "$DIR/last_median" 2>/dev/null || echo 0)
if [[ $MODE == gain ]]; then
	ok=$(awk -v m="$median" -v l="$last" 'BEGIN {print (m > l * 1.05) ? 1 : 0}')
	need=">${last}*1.05"
else
	ok=$(awk -v m="$median" -v l="$last" 'BEGIN {print (m >= l * 0.95) ? 1 : 0}')
	need=">=${last}*0.95"
fi
if [[ $ok -ne 1 ]]; then
	echo "FAIL speed: ${median} t/s, need ${need}"
	fail=1
else
	echo "ok   speed: ${median} t/s vs last ${last}"
fi

if [[ $fail -eq 0 ]]; then
	[[ $KEEP -eq 1 ]] && echo "$median" >"$DIR/last_median"
	echo "PASS"
else
	echo "FAIL"
fi
exit $fail
