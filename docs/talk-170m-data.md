# Talking model data: mix, tokenizer, ids, chat (talk-170m)

## Abstract

The pretraining mix is 3.81B tokens: 3.21B of FineWeb-Edu prose and 0.60B of permissively licensed code, encoded with a 16k BPE retrained on an 85/15 sample of that mix and packed as int32 ids. A held-out set of every 1000th document (3.8M tokens) is kept apart for validation. The chat fine-tune set is 150k conversations (93 MB) rendered as `User:` / `Assistant:` turns that fit the 256-token context.

## What problem it solves

[The plan](talk-170m-plan.md) needs about 20 tokens per parameter, and the model must see code and conversations, not only Wikipedia. The repo could already encode text and train on an id file, but three limits stood in the way at this size: id counts in an `int` (2.1B max), a loader that copied the whole file into the heap, and a pretokenizer that holds a whole text file plus its ids in RAM.

## Numbers

| Set | Text | Tokens | Bytes / token |
|---|---|---|---|
| Web train (FineWeb-Edu, 4 files) | 13.9 GB | 3,208,727,249 | 4.33 |
| Code train (Python, C, C++, JS) | 1.43 GB | 603,362,012 | 2.37 |
| **Train total** `data/mix_train_ids.bin` | 15.3 GB | **3,812,089,261** | |
| Val web `data/mix_val_web_ids.bin` | 14.2 MB | 3,268,677 | 4.34 |
| Val code `data/mix_val_code_ids.bin` | 1.2 MB | 504,224 | 2.43 |
| Chat train `data/chat_train.txt` | 93 MB | about 22M | |

Code is 16% of the tokens but 9% of the bytes: it costs nearly twice as many tokens per byte, mostly because the pre-tokenizer makes every indentation space its own word (see [the plan](talk-170m-plan.md#tokenizer)).

Code rows were kept only under MIT, Apache-2.0, BSD-2/3 or ISC, and only files under 100 KB: 234k of the files in 8 parquet shards.

Chat conversations kept, from `data/chat.SOURCE`:

| Source | Kept | Of |
|---|---|---|
| everyday-conversations | 2,260 | 2,260 |
| self-oss-instruct (code) | 16,183 | 48,127 |
| openhermes-100k | 38,471 | 95,000 |
| explore-instruct-rewriting | 30,286 | 30,400 |
| smol-constraints | 22,344 | 34,424 |
| smol-magpie-ultra | 38,843 | 409,537 |
| oasst2 (English, best-ranked path) | 2,140 | 5,126 |

A conversation longer than 1000 bytes is cut after its last whole exchange; one whose first exchange does not fit is dropped. That is why long-answer sets such as smol-magpie-ultra keep under 10%.

## Method

1. `scripts/fetch_mix.py` streams each parquet file once into `data/raw/` (a `.part` file is renamed when complete) and writes documents separated by a blank line. Every 1000th document goes to the val file instead.
2. `scripts/build_mix.sh` samples 300 MB (ten evenly spaced slices per source, 255 MB web and 45 MB code) and trains the vocab with `scripts/bpe_train` (`bpe_train_from_file`, 16384 tokens).
3. It then splits each text file at line ends into 1 GB pieces, encodes each with `scripts/pretok_ids`, and concatenates the ids. A piece costs about 1 GB of text plus up to 4 GB of ids in RAM, far below the old whole-file path.
4. `scripts/fetch_chat.py` renders SmolTalk subsets and oasst2 trees in the format `cli/chat_session.c` builds at inference.

## Code changes

- [`lang/dataset_text.c`](../lang/dataset_text.c): `n_ids` is a `long`, window starts are computed in `long`, and the window count clamps at `INT_MAX` (use `--stride` equal to `--window`: 14.9M windows here). `*ids.bin` files are mapped read-only with `mmap` instead of read into the heap, so the 15 GB file is held once, in the page cache.
- [`main.c`](../main.c): `--val-data PATH` scores a separate held-out id file.
- [`cli/chat_session.c`](../cli/chat_session.c): `chat_reply_cut` ends a reply at the next `User:` or `Assistant:` turn, so the model cannot write the user's next line.
- Unit tests: 3B-id window count, id file load and slicing, reply cut.

## Pitfalls

- The machine's WSL VM went down three times during this step, at 23:04, 00:20 and 00:44. Inside WSL there was no OOM-killer trace and no kernel panic. The only faults were the WSLg compositor `weston` segfaulting in `rdp-backend.so` every one to two minutes. Windows ended the VM from outside (WSL may use 48 GB of the 64 GB host). One crash cut the last web piece mid-copy; it was re-encoded on its own and the total checked against the per-piece counts in `logs/build_mix.log`.
- A file written just before a crash can exist with 0 bytes even after its rename. `fetch_chat.py` hit eight such parquet files; deleting 0-byte files in `data/raw/` and rerunning fixed it.
- The Makefile has no header dependencies: after changing a struct in a header, `make clean` before building, or objects disagree on its layout.

## How it shows up in this repo

- [`scripts/fetch_mix.py`](../scripts/fetch_mix.py), [`scripts/build_mix.sh`](../scripts/build_mix.sh), [`scripts/bpe_train.c`](../scripts/bpe_train.c), [`scripts/fetch_chat.py`](../scripts/fetch_chat.py)
- Data under `data/` (gitignored): `mix.vocab`, `mix_train_ids.bin`, `mix_val*_ids.bin`, `chat_train.txt`, `chat_val.txt`, and `mix.SOURCE` / `chat.SOURCE` with URLs and licenses
