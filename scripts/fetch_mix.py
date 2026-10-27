#!/usr/bin/env python3
"""Fetch the talk-170m pretraining mix (web text + code) into data/.

    ~/.venvs/anni-data/bin/python scripts/fetch_mix.py            # full mix
    ~/.venvs/anni-data/bin/python scripts/fetch_mix.py --web-files 1 --code-files 1

Needs pyarrow (python3 -m venv --without-pip ~/.venvs/anni-data;
python3 -m pip --python ~/.venvs/anni-data/bin/python install pyarrow).

Sources (open, no login):
  web   HuggingFaceFW/fineweb-edu, sample/10BT parquet (ODC-By 1.0)
  code  codeparrot/github-code-clean parquet export, Python/C/C++/JavaScript,
        rows kept only under permissive licenses (MIT, Apache-2.0, BSD, ISC)

Output (never vendored, see .gitignore):
  data/raw/...               downloaded parquet, reused on re-runs
  data/mix_web.txt           train documents, one blank line between them
                             (documents keep their own blank lines)
  data/mix_code.txt
  data/mix_val_web.txt       every --val-every-th document, held out
  data/mix_val_code.txt
  data/mix.SOURCE            URLs, licenses, counts, retrieval time
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets"
WEB_URL = HF + "/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{:03d}_00000.parquet"
CODE_URL = (HF + "/codeparrot/github-code-clean/resolve/refs%2Fconvert%2Fparquet/"
            "{lang}-all/partial-train/{i:04d}.parquet")
CODE_LANGS = ("Python", "C", "C++", "JavaScript")
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"}
MAX_CODE_BYTES = 100_000


def download(url: str, dest: Path) -> Path:
    """Fetch url to dest once; a .part file makes an interrupted run restart cleanly."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"fetch_mix: {url}", flush=True)
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(part, dest)
    return dest


def clean(text: str) -> str:
    """Keep inner blank lines (paragraphs, code layout); trim the ends."""
    return text.replace("\r\n", "\n").strip("\n")


class Sink:
    """Split documents into train / val files by a fixed stride."""

    def __init__(self, train: Path, val: Path, val_every: int):
        self.train = open(train, "w", encoding="utf-8")
        self.val = open(val, "w", encoding="utf-8")
        self.val_every = val_every
        self.docs = 0
        self.bytes = 0

    def add(self, text: str) -> None:
        text = clean(text)
        if not text:
            return
        self.docs += 1
        out = self.val if self.docs % self.val_every == 0 else self.train
        out.write(text)
        out.write("\n\n")
        self.bytes += len(text.encode("utf-8")) + 2

    def close(self) -> None:
        self.train.close()
        self.val.close()


def rows(path: Path, columns: list[str]):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=columns, batch_size=2048):
        yield from batch.to_pylist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-files", type=int, default=4,
                    help="fineweb-edu sample/10BT files (~2.15 GB, ~0.7B tokens each)")
    ap.add_argument("--code-files", type=int, default=2,
                    help="parquet files per code language (~175 MB each)")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    data = Path(args.data)
    raw = data / "raw"
    web = Sink(data / "mix_web.txt", data / "mix_val_web.txt", args.val_every)
    for i in range(args.web_files):
        p = download(WEB_URL.format(i), raw / "fineweb-edu" / f"{i:03d}_00000.parquet")
        for r in rows(p, ["text"]):
            web.add(r["text"])
        print(f"fetch_mix: web {web.docs} docs {web.bytes / 1e9:.2f} GB", flush=True)
    web.close()

    code = Sink(data / "mix_code.txt", data / "mix_val_code.txt", args.val_every)
    kept = {}
    for lang in CODE_LANGS:
        kept[lang] = 0
        for i in range(args.code_files):
            p = download(CODE_URL.format(lang=lang, i=i),
                         raw / "github-code-clean" / f"{lang}-{i:04d}.parquet")
            for r in rows(p, ["code", "license", "size"]):
                if r["license"] not in PERMISSIVE or r["size"] > MAX_CODE_BYTES:
                    continue
                code.add(r["code"])
                kept[lang] += 1
        print(f"fetch_mix: code {lang} {kept[lang]} files, total "
              f"{code.bytes / 1e9:.2f} GB", flush=True)
    code.close()

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(data / "mix.SOURCE", "w", encoding="utf-8") as f:
        f.write(f"retrieved: {now}\n")
        f.write(f"web: {WEB_URL.format(0)} .. {args.web_files} files, "
                f"{web.docs} docs, {web.bytes} bytes. License: ODC-By 1.0.\n")
        f.write(f"code: {CODE_URL} langs {','.join(CODE_LANGS)}, {args.code_files} "
                f"files each, licenses {','.join(sorted(PERMISSIVE))}, "
                f"files kept {kept}, {code.bytes} bytes.\n")
        f.write(f"val: every {args.val_every}th document of each source.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
