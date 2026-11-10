#!/usr/bin/env bash
# Bring back the live page, the screenshot taker and the step checks of the
# 555M run, after a reboot or a crash. None touches the training, and the curve
# goes on after its last row.
#
#   scripts/dashboard_up.sh            # start whichever is missing
#   scripts/dashboard_up.sh --restart  # stop them first, to load new code
#
# Screenshots wait for the run: without tmux session run555m the taker would
# write final-stopped.png and exit, so start scripts/run555m.sh first.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
LOGS=logs/run555m
DASH=dash555m
SHOTS=shots555m
CHECKS=checks555m
mkdir -p "$LOGS"

alive() { tmux has-session -t "=$1" 2>/dev/null; }

if [ "${1:-}" = "--restart" ]; then
	for session in "$DASH" "$SHOTS" "$CHECKS"; do
		if alive "$session"; then
			tmux kill-session -t "=$session"
		fi
	done
elif [ $# -gt 0 ]; then
	echo "usage: scripts/dashboard_up.sh [--restart]" >&2
	exit 2
fi

url=$(python3 scripts/dashboard.py --print-url)
if alive "$DASH"; then
	echo "dashboard: already running in tmux session $DASH"
else
	tmux new-session -d -s "$DASH" -c "$ROOT" \
	    "python3 scripts/dashboard.py 2>&1 | tee -a $LOGS/dashboard.log"
	echo "dashboard: started tmux session $DASH"
fi

# The first screenshot needs the page, and a taken port only shows up here.
for _ in $(seq 20); do
	curl -fsS -o /dev/null "$url" 2>/dev/null && break
	sleep 0.5
done
if ! curl -fsS -o /dev/null "$url" 2>/dev/null; then
	echo "dashboard: no answer on $url, see $LOGS/dashboard.log" >&2
	exit 1
fi
echo "dashboard: $url"

if alive "$CHECKS"; then
	echo "checks: already running in tmux session $CHECKS"
else
	tmux new-session -d -s "$CHECKS" -c "$ROOT" \
	    "python3 scripts/run555m_checks.py 2>&1 | tee -a $LOGS/checks.log"
	echo "checks: started tmux session $CHECKS"
fi

if alive run555m; then
	scripts/dashboard_shots.sh
else
	echo "shots: not started, tmux session run555m is gone; run scripts/run555m.sh, then this again"
fi
