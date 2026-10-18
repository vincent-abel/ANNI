#!/usr/bin/env python3
"""Pretokenize wiki text with AnnI BPE vocab → packed int32 LE ids."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# Minimal BPE load matching lang/bpe.c vocab file format is heavy; shell out
# to a tiny C helper if present, else use anni lm path via subprocess encode.
# Prefer: write ids with the same merges as model-wiki-16k.bin.vocab by calling
# a one-shot encode binary built into the tree.

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=str(ROOT / "data/wiki_simple.txt"))
    ap.add_argument("--vocab", default=str(ROOT / "model-wiki-16k.bin.vocab"))
    ap.add_argument("--out", default=str(ROOT / "data/wiki_ids.bin"))
    ap.add_argument("--max-bytes", type=int, default=0, help="0 = full file")
    args = ap.parse_args()

    helper = ROOT / "scripts" / "pretok_ids"
    anni = ROOT / "anni"
    if not helper.exists():
        # Build helper from inline C if missing
        csrc = ROOT / "scripts" / "pretok_ids.c"
        if not csrc.exists():
            print("missing pretok_ids.c", file=sys.stderr)
            return 2
        import subprocess

        subprocess.check_call(
            [
                "gcc",
                "-O2",
                "-std=c99",
                f"-I{ROOT}",
                f"-I{ROOT}/lang",
                f"-I{ROOT}/num",
                f"-I{ROOT}/compat",
                str(csrc),
                str(ROOT / "lang/bpe.c"),
                str(ROOT / "lang/vocab.c"),
                str(ROOT / "lang/tokenize.c"),
                "-o",
                str(helper),
                "-lm",
            ]
        )
    import subprocess

    cmd = [str(helper), args.vocab, args.text, args.out]
    if args.max_bytes > 0:
        cmd.append(str(args.max_bytes))
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
