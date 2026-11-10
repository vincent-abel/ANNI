#!/usr/bin/env bash
# Keep screenshots of the dashboard while run555m lives, so its last state
# survives the end of the run. Runs in its own tmux session, which leaves
# scripts/run555m.sh alone.
#
#   scripts/dashboard_shots.sh             # start tmux session shots555m
#   scripts/dashboard_shots.sh --now NAME  # one screenshot, logs/run555m/shots/NAME.png
#
# latest.png is replaced every hour, YYYY-MM-DD.png is kept once a day,
# final-pretrain.png and final-chat.png are taken when those steps end, and
# final-stopped.png when the tmux session goes away early. Headless Chrome on
# the CPU only.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
LOGS=logs/run555m
SHOTS=$LOGS/shots
SESSION=shots555m
CHROME=${CHROME:-google-chrome}
mkdir -p "$SHOTS"

shot() {	# shot NAME
	local profile
	profile=$(mktemp -d)
	timeout 60 "$CHROME" --headless=new --no-sandbox --disable-gpu \
	    --user-data-dir="$profile" --window-size=1200,2600 \
	    --virtual-time-budget=5000 --screenshot="$SHOTS/$1.png.part.png" \
	    "$(python3 scripts/dashboard.py --print-url)" >/dev/null 2>&1 || true
	rm -rf "$profile"
	[ -s "$SHOTS/$1.png.part.png" ] && mv "$SHOTS/$1.png.part.png" "$SHOTS/$1.png"
	echo "[$(date '+%F %T')] shot $1"
}

if [ "${1:-}" = "--now" ]; then
	shot "${2:?name}"
	exit 0
fi
if [ "${1:-}" != "--inside" ]; then
	if tmux has-session -t "=$SESSION" 2>/dev/null; then
		echo "shots: already running in tmux session $SESSION"
		exit 0
	fi
	tmux new-session -d -s "$SESSION" "$ROOT/scripts/dashboard_shots.sh --inside 2>&1 | tee -a $ROOT/$SHOTS/shots.log"
	echo "shots: started tmux session $SESSION"
	exit 0
fi

last=0
while :; do
	if [ -e "$LOGS/.pretrained" ] && [ ! -e "$SHOTS/final-pretrain.png" ]; then
		shot final-pretrain
	fi
	if [ -e "$LOGS/.chat" ]; then
		shot final-chat
		exit 0
	fi
	if ! tmux has-session -t "=run555m" 2>/dev/null; then
		shot final-stopped
		exit 0
	fi
	now=$(date +%s)
	if [ $((now - last)) -ge 3600 ]; then
		shot latest
		[ -e "$SHOTS/$(date +%F).png" ] || cp "$SHOTS/latest.png" "$SHOTS/$(date +%F).png"
		last=$now
	fi
	sleep 20
done
