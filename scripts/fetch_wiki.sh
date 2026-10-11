#!/usr/bin/env bash
# Fetch a license-clean Wikipedia slice into data/ (never vendored in git).
#
#   scripts/fetch_wiki.sh                       # Simple English Wikipedia, up to 100 MB of text
#   WIKI_LANG=en MAX_MB=500 scripts/fetch_wiki.sh
#   WIKI_DUMP=20260101 scripts/fetch_wiki.sh    # pin a dated dump (reproducible)
#
# Output: data/wiki_<lang>.txt        one document per article, blank-line separated
#         data/wiki_<lang>.SOURCE     dump URL, retrieval time, license, attribution
#
# License: Wikipedia text is CC BY-SA 4.0 (+ GFDL). Keep the .txt out of git
# (data/wiki_* is in .gitignore) and keep the .SOURCE file next to any model you
# share. The dump is streamed and cut at MAX_MB, so a huge dump is never stored.
set -euo pipefail

cd "$(dirname "$0")/.."
LANG_CODE="${WIKI_LANG:-simple}"
DUMP="${WIKI_DUMP:-latest}"
MAX_MB="${MAX_MB:-100}"
OUT="${OUT:-data/wiki_${LANG_CODE}.txt}"
BASE="${WIKI_MIRROR:-https://dumps.wikimedia.org}"
FILE="${LANG_CODE}wiki-${DUMP}-pages-articles.xml.bz2"
URL="${BASE}/${LANG_CODE}wiki/${DUMP}/${FILE}"

command -v curl >/dev/null || { echo "fetch_wiki: curl is required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "fetch_wiki: python3 is required" >&2; exit 1; }
mkdir -p data

echo "fetch_wiki: streaming $URL (cap ${MAX_MB} MB of text) -> $OUT"
# Extractor exits early at MAX_MB; curl then gets SIGPIPE (exit 23/141), which is fine.
set +e
curl -fsSL "$URL" | python3 scripts/wiki_extract.py - "$OUT" --max-mb "$MAX_MB"
status=("${PIPESTATUS[@]}")
set -e
if [ "${status[1]}" -ne 0 ]; then
	echo "fetch_wiki: extraction failed (curl=${status[0]} python=${status[1]})" >&2
	exit 1
fi

cat > "${OUT%.txt}.SOURCE" <<EOT
source:    $URL
retrieved: $(date -u +%Y-%m-%dT%H:%M:%SZ)
text cap:  ${MAX_MB} MB
license:   CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/) and GFDL
credit:    Wikipedia contributors; per-article history on the matching wiki page
EOT
wc -c "$OUT"
echo "fetch_wiki: done. Train with e.g.:"
echo "  ./ann lm train --data $OUT --tokenizer bpe --vocab-size 16384 --val-frac 0.01 --stream --out wiki.bin"
