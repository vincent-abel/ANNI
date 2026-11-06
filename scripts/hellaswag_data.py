#!/usr/bin/env python3
"""Fetch the HellaSwag validation set for `anni lm hellaswag`.

    scripts/hellaswag_data.py [--out data/hellaswag_val.tsv]

Writes one question per line: answer, context, then the four endings, separated
by tabs. The text is cleaned as in the usual log-likelihood evaluation of the
benchmark, so scores can be compared with published ones. Needs pyarrow.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = ("https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/"
       "validation-00000-of-00001.parquet")
QUESTIONS = 10042


def clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return " ".join(text.replace("\t", " ").split("\n")).replace("  ", " ")


def line(row: dict) -> str:
    context = f"{row['ctx_a']} {row['ctx_b'].capitalize()}"
    fields = [str(row["label"]), clean(f"{row['activity_label']}: {context}")]
    fields += [clean(ending) for ending in row["endings"]]
    if len(fields) != 6 or not all(fields):
        raise ValueError(f"bad row: {row}")
    return "\t".join(fields)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/hellaswag_val.tsv")
    args = ap.parse_args()
    import pyarrow.parquet as parquet	# only this script needs it

    with urllib.request.urlopen(URL, timeout=120) as reply:
        rows = parquet.read_table(io.BytesIO(reply.read())).to_pylist()
    if len(rows) != QUESTIONS:
        sys.exit(f"got {len(rows)} questions, expected {QUESTIONS}")
    lines = [line(r) for r in rows]
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(f"{args.out}: {len(lines)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
