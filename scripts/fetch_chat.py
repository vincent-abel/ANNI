#!/usr/bin/env python3
"""Fetch chat data for the talk-170m fine-tune, rendered as User:/Assistant: text.

    ~/.venvs/anni-data/bin/python scripts/fetch_chat.py

Sources (open, no login):
  HuggingFaceTB/smoltalk subsets (Apache-2.0)
  OpenAssistant/oasst2, English threads, best-ranked reply per turn (Apache-2.0)

Each conversation becomes one document in the format cli/chat_session.c uses:

  User: <text>
  Assistant: <text>
  User: <text>
  Assistant: <text>

System messages are dropped. Conversations longer than --max-bytes (the
model's 256-token context is about 1 KB of English) are cut after the last
whole exchange that fits, and dropped if not even one fits.

Output (gitignored): data/chat_train.txt, data/chat_val.txt (every
--val-every-th conversation), data/chat.SOURCE.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_mix import ROOT, Sink, download  # noqa: E402

HF = "https://huggingface.co/datasets"
SMOLTALK = (HF + "/HuggingFaceTB/smoltalk/resolve/refs%2Fconvert%2Fparquet/"
            "{cfg}/train/{i:04d}.parquet")
SMOLTALK_CFGS = {
    "everyday-conversations": 1,
    "self-oss-instruct": 1,
    "openhermes-100k": 1,
    "explore-instruct-rewriting": 1,
    "smol-constraints": 1,
    "smol-magpie-ultra": 6,
}
OASST2 = (HF + "/OpenAssistant/oasst2/resolve/refs%2Fconvert%2Fparquet/"
          "default/train/0000.parquet")
ROLE = {"user": "User", "prompter": "User", "assistant": "Assistant"}


def render(turns: list[tuple[str, str]], max_bytes: int) -> str:
    """User/Assistant pairs, cut after the last whole exchange under max_bytes."""
    out: list[str] = []
    size = 0
    pending = None
    for role, text in turns:
        name = ROLE.get(role)
        if not name or not text.strip():
            continue
        line = f"{name}: {text.strip()}"
        if name == "User":
            pending = line
            continue
        if pending is None:
            continue
        pair = pending + "\n" + line
        if size + len(pair.encode("utf-8")) + 1 > max_bytes:
            break
        out.append(pair)
        size += len(pair.encode("utf-8")) + 1
        pending = None
    return "\n".join(out)


def smoltalk(raw: Path):
    for cfg, n in SMOLTALK_CFGS.items():
        for i in range(n):
            p = download(SMOLTALK.format(cfg=cfg, i=i),
                         raw / "smoltalk" / f"{cfg}-{i:04d}.parquet")
            for batch in pq.ParquetFile(p).iter_batches(columns=["messages"]):
                for row in batch.to_pylist():
                    yield cfg, [(m["role"], m["content"]) for m in row["messages"]]


def oasst2(raw: Path):
    """Walk each English tree along the best-ranked reply at every turn."""
    p = download(OASST2, raw / "oasst2" / "train-0000.parquet")
    cols = ["message_id", "parent_id", "role", "text", "lang", "rank"]
    msgs = pq.read_table(p, columns=cols).to_pylist()
    kids = defaultdict(list)
    for m in msgs:
        kids[m["parent_id"]].append(m)
    for root in kids[None]:
        if root["lang"] != "en":
            continue
        turns = []
        node = root
        while node is not None:
            turns.append((node["role"], node["text"]))
            nxt = [k for k in kids[node["message_id"]] if k["lang"] == "en"]
            nxt.sort(key=lambda k: k["rank"] if k["rank"] is not None else 99)
            node = nxt[0] if nxt else None
        yield "oasst2", turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bytes", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    data = Path(args.data)
    sink = Sink(data / "chat_train.txt", data / "chat_val.txt", args.val_every)
    kept: dict[str, int] = defaultdict(int)
    seen: dict[str, int] = defaultdict(int)
    for gen in (smoltalk(data / "raw"), oasst2(data / "raw")):
        for name, turns in gen:
            seen[name] += 1
            text = render(turns, args.max_bytes)
            if text:
                sink.add(text)
                kept[name] += 1
    sink.close()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(data / "chat.SOURCE", "w", encoding="utf-8") as f:
        f.write(f"retrieved: {now}\n")
        f.write(f"smoltalk: {SMOLTALK} subsets {SMOLTALK_CFGS}. License: Apache-2.0.\n")
        f.write(f"oasst2: {OASST2}, English, best-ranked path. License: Apache-2.0.\n")
        for name in seen:
            f.write(f"{name}: kept {kept[name]} of {seen[name]} conversations\n")
        f.write(f"max bytes per conversation {args.max_bytes}, "
                f"{sink.docs} kept, {sink.bytes} bytes, val every {args.val_every}\n")
    for name in seen:
        print(f"fetch_chat: {name} kept {kept[name]} of {seen[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
