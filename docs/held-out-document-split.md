# Held-out document split

## Abstract

Validation text must be disjoint from training text. With real corpora the split is done **by document** (article), not by sliding window or mid-document byte offset (the `offset` mode only assigns whole documents).

## What problem it solves

Neighboring windows from the same article share names, phrasing, and facts. If some land in train and others in validation, held-out NLL looks better than true generalization. A pure tail split is also biased by corpus ordering.

## How it works for a human

Treat each article as a card. Shuffle the cards with a fixed seed, deal a small fraction (for example 1%) to validation, and train on the rest. No article ever appears on both piles. The tokenizer is trained from the training pile only.

## Formal view

Documents \(D=\{d_1..d_N\}\). Assign \(d_i\) to val iff \(h(\text{id}_i, s)\bmod 10000 < \rho\cdot 10000\), where \(h\) is a stable hash and \(\rho\) the fraction. Expected val size is \(\rho N\) documents with variance \(\rho(1-\rho)N\); no state needs to be stored, and the split is reproducible and stable if documents are appended.

## Complexity / pitfalls / design choices

- **Hash on document id**, not position, so reruns and appends keep the split.
- **Duplicates:** near-duplicate articles can still leak; dedupe first if possible.
- **Windows** are formed *after* splitting, never across document boundaries.
- **Size:** val should be big enough for a stable estimate; combine with [cheaper eval](cheaper-window-eval.md).

## How it shows up in this repo

- [`lang/dataset_stream.c`](../lang/dataset_stream.c): `text_docs_*` reads blank-line separated documents; `text_split_init` / `text_split_is_val` decide the side of each document.
  - `--split doc` (default): val iff a hash of the document bytes and `--split-seed` lands in `--val-frac`. Hashing content (not position) keeps the split stable under reordering and appends, and identical duplicate articles stay on one side.
  - `--split offset`: val = documents that start in the last `--val-frac` of the file's bytes (documents are never cut, so there is no mid-document boundary).
- `text_dataset_load_split` tokenizes each document separately and records valid window starts (`text_dataset.starts`), so **no window crosses a document and none can span train and val**. Documents shorter than `--window` contribute no windows.
- `text_split_train_bpe` learns BPE merges from the train side only.
- Metric: `lm_nll_windows` / `lm_nll_stream` in [`lang/lm_train.h`](../lang/lm_train.h); `lm_train_corpus_val` reports the per-epoch window NLL on the held-out set.
- CLI: `./ann lm train --data FILE --val-frac 0.01 [--split doc|offset] [--split-seed N] [--eval-windows M]` prints `LM val window NLL`.
- [Held-out loss / perplexity](held-out-loss-perplexity.md) (a separate fixture such as `data/toy_val.txt` is still fine for the byte demo)
- Test: `test_split_and_stream` in [`tests/test_ann.c`](../tests/test_ann.c) checks disjointness and that no window crosses documents in both modes.

## Tiny worked example

10,000 articles, \(\rho=0.01\) → about 100 validation articles. If articles average 500 tokens at \(W=64\), that is roughly 45k windows, subsampled by `--eval-windows`.

## What comes next

[Streaming dataloader](streaming-dataloader.md) feeds the train side without loading it all.
