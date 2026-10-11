# Real corpora at scale

## Abstract

Fixtures like `data/toy.txt` prove correctness; they cannot teach language. Track D documents how to fetch a real corpus (Wikipedia is the reference example), prepare it, and size the run.

## What problem it solves

A 16k-token vocabulary on a few kilobytes of text is mostly unseen tokens. The model needs on the order of tens to hundreds of megabytes before BPE merges and attention patterns mean anything. Users also need a reproducible, legal way to obtain text.

## How it works for a human

Download a dump, extract plain text, strip markup, and write one UTF-8 file. Then train BPE on it, encode it once to ids, and train on the id stream. The docs give the commands; the repo does not vendor the data.

## Formal view

Chinchilla-style rule of thumb: about 20 tokens per parameter for compute-optimal training. A 10M-parameter model wants roughly \(2\times10^8\) tokens; at ~4 bytes per BPE token that is ~800 MB of text. Throughput bounds the epoch count: \(\text{time}\approx \text{tokens}/\text{tokens-per-second}\).

## Complexity / pitfalls / design choices

- **Licensing:** Wikipedia text is CC BY-SA; keep it out of git and record the dump date.
- **Cleaning:** drop templates, tables, and non-prose; normalize whitespace and UTF-8.
- **Storage:** loading 1 GB into RAM is fine for text, but not as `double` ids; see [streaming](streaming-dataloader.md).
- **Smaller first:** start with a 50–100 MB slice to validate the pipeline.

## How it shows up in this repo

- Fetch: [`scripts/fetch_wiki.sh`](../scripts/fetch_wiki.sh) (or `make data-wiki`) streams a Wikimedia dump through [`scripts/wiki_extract.py`](../scripts/wiki_extract.py) into `data/wiki_<lang>.txt` plus a `.SOURCE` file recording URL, retrieval time and license. Defaults to Simple English Wikipedia capped at `MAX_MB=100`; `WIKI_LANG=en`, `MAX_MB=500`, and `WIKI_DUMP=YYYYMMDD` (pin the dump) are environment overrides. `data/wiki_*` is git-ignored; nothing large is vendored.
- Format: one article per document, documents separated by a blank line, paragraphs inside an article by single newlines, so a blank line is always a document boundary for the [held-out split](held-out-document-split.md).
- Train on it: `./ann lm train --data data/wiki_simple.txt --tokenizer bpe --vocab-size 16384 --val-frac 0.01 --stream --out wiki.bin`
- [Text dataset loading](text-dataset-loading.md), [`lang/dataset_text.c`](../lang/dataset_text.c), [`lang/dataset_stream.c`](../lang/dataset_stream.c)
- [Scale toward target size](scale-toward-target-size.md)
- `--data PATH` in [`main.c`](../main.c)

## Tiny worked example

A 100 MB English slice ≈ 25M BPE tokens. At 5k tokens/s on CPU, one epoch takes about 5,000 s (≈1.4 h); a GPU path is needed for more than a few epochs.

## What comes next

[Held-out document split](held-out-document-split.md) decides what part of this text the model never trains on.
