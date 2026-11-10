#!/usr/bin/env python3
"""Fetch more code and prose for the 555M run into data/.

    ~/.venvs/anni-data/bin/python scripts/fetch_code.py --target-gb 17.3 --web-files 10

Prose: the next fineweb-edu sample/10BT files after the 4 fetch_mix.py took.

French: the start of fineweb-2 fra_Latn, every 1000th document held out.

Code: the full codeparrot/github-code-clean shards, which mix all languages,
read from the last shard backwards because fetch_mix.py took the start of the
per-language exports. Same filter as fetch_mix.py: Python, C, C++, JavaScript
under MIT, Apache-2.0, BSD or ISC, files up to 100 KB.

A file whose text is already in data/raw/github-code-clean, train or val, is
dropped, so the held-out code stays held out, and so is an exact duplicate
among the new files.

Output (gitignored):
  data/raw/github-code-full/  downloaded shards, reused on re-runs
  data/code2.txt              one blank line between documents
  data/web2.txt               same, prose
  data/fr2.txt, data/fr2_val.txt   same, French
  data/code2.SOURCE, data/web2.SOURCE, data/fr2.SOURCE
                              URLs, licenses, counts, retrieval time
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import sys
from pathlib import Path

import pyarrow.parquet as pq

from fetch_mix import (CODE_LANGS, HF, MAX_CODE_BYTES, PERMISSIVE, WEB_URL, Sink, clean,
                       download, rows)

ROOT = Path(__file__).resolve().parents[1]
SHARDS = 880
SHARD_URL = (HF + "/codeparrot/github-code-clean/resolve/main/data/"
             "train-{:05d}-of-00880.parquet")


FR_URL = HF + "/HuggingFaceFW/fineweb-2/resolve/main/data/fra_Latn/train/{:03d}_00000.parquet"


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=12).digest()


def seen_before(raw: Path) -> set[bytes]:
    seen = set()
    for path in sorted((raw / "github-code-clean").glob("*.parquet")):
        for r in rows(path, ["code"]):
            seen.add(digest(clean(r["code"])))
    print(f"fetch_code: {len(seen)} files already in the mix", flush=True)
    return seen


def fetch_web(raw: Path, out_path: Path, first: int, count: int) -> tuple[int, int]:
    docs = size = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i in range(first, first + count):
            path = download(WEB_URL.format(i), raw / "fineweb-edu" / f"{i:03d}_00000.parquet")
            for r in rows(path, ["text"]):
                text = clean(r["text"])
                if not text:
                    continue
                out.write(text)
                out.write("\n\n")
                docs += 1
                size += len(text.encode("utf-8", "surrogatepass")) + 2
            print(f"fetch_code: web {docs} docs, {size / 1e9:.2f} GB", flush=True)
    return docs, size


def fetch_fr(data: Path, target_gb: float) -> None:
    sink = Sink(data / "fr2.txt", data / "fr2_val.txt", 1000)
    i = 0
    while sink.bytes < target_gb * 1e9:
        path = download(FR_URL.format(i), data / "raw" / "fineweb-2-fr" / f"{i:03d}_00000.parquet")
        for r in rows(path, ["text"]):
            sink.add(r["text"])
            if sink.bytes >= target_gb * 1e9:
                break
        print(f"fetch_code: fr {sink.docs} docs, {sink.bytes / 1e9:.2f} GB", flush=True)
        i += 1
    sink.close()
    with open(data / "fr2.SOURCE", "w", encoding="utf-8") as f:
        f.write(f"retrieved: {utc_now()}\n")
        f.write(f"fr: {FR_URL.format(0)} .. {i} files, {sink.docs} docs, "
                f"{sink.bytes} bytes. License: ODC-By 1.0.\n")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def fetch_code(data: Path, target_gb: float, prefetch: int) -> None:
    raw = data / "raw"
    seen = seen_before(raw)
    target = int(target_gb * 1e9)
    langs = set(CODE_LANGS)
    kept, dropped, size, used = {}, 0, 0, 0

    def fetch(i: int) -> Path:
        return download(SHARD_URL.format(i), raw / "github-code-full" / f"{i:05d}.parquet")

    order = iter(range(SHARDS - 1, -1, -1))
    with concurrent.futures.ThreadPoolExecutor(prefetch) as pool, \
            open(data / "code2.txt", "w", encoding="utf-8") as out:
        queue = [pool.submit(fetch, next(order)) for _ in range(prefetch)]
        while size < target and queue:
            path = queue.pop(0).result()
            nxt = next(order, None)
            if nxt is not None:
                queue.append(pool.submit(fetch, nxt))
            for r in rows(path, ["code", "language", "license", "size"]):
                if (r["language"] not in langs or r["license"] not in PERMISSIVE
                        or r["size"] > MAX_CODE_BYTES):
                    continue
                text = clean(r["code"])
                d = digest(text)
                if not text or d in seen:
                    dropped += 1
                    continue
                seen.add(d)
                out.write(text)
                out.write("\n\n")
                size += len(text.encode("utf-8", "surrogatepass")) + 2
                kept[r["language"]] = kept.get(r["language"], 0) + 1
            used += 1
            print(f"fetch_code: {used} shards, {size / 1e9:.2f} GB, "
                  f"{sum(kept.values())} files, {dropped} duplicates", flush=True)
        for f in queue:
            f.cancel()

    with open(data / "code2.SOURCE", "w", encoding="utf-8") as f:
        f.write(f"retrieved: {utc_now()}\n")
        f.write(f"code: {SHARD_URL} last {used} shards, langs {','.join(CODE_LANGS)}, "
                f"licenses {','.join(sorted(PERMISSIVE))}, files kept {kept}, "
                f"{dropped} duplicates dropped, {size} bytes.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gb", type=float, default=17.3,
                    help="code to keep, in GB; 0 skips the code")
    ap.add_argument("--web-files", type=int, default=10,
                    help="fineweb-edu files to add; 0 skips the prose")
    ap.add_argument("--web-first", type=int, default=4, help="first fineweb-edu file")
    ap.add_argument("--fr-gb", type=float, default=6.0,
                    help="French to keep, in GB; 0 skips it")
    ap.add_argument("--prefetch", type=int, default=3, help="shards downloaded ahead")
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    data = Path(args.data)
    if args.target_gb > 0:
        fetch_code(data, args.target_gb, args.prefetch)
    if args.web_files > 0:
        docs, size = fetch_web(data / "raw", data / "web2.txt", args.web_first, args.web_files)
        with open(data / "web2.SOURCE", "w", encoding="utf-8") as f:
            f.write(f"retrieved: {utc_now()}\n")
            f.write(f"web: {WEB_URL.format(args.web_first)} .. {args.web_files} files, "
                    f"{docs} docs, {size} bytes. License: ODC-By 1.0.\n")
    if args.fr_gb > 0:
        fetch_fr(data, args.fr_gb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
