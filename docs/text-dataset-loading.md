# Text dataset loading

## Abstract

Phrase-only training (`lm_train_steps(model, "hello", ...)`) cannot expose **long-range statistics** or **held-out evaluation**. A **text dataset loader** reads a UTF-8/byte file such as **`data/toy.txt`**, tokenizes into byte ids, and exposes **sliding windows** of length up to `LM_MAX_CTX` as training examples for next-token prediction. Planned API: `text_dataset_open`, `text_dataset_next_window`, `text_dataset_reset`.

## What problem it solves (plain language)

Models learn from **many contexts**, not one repeated string. A file-backed dataset turns arbitrary prose (or synthetic toy corpora) into an endless stream of `(prefix → next byte)` tasks, which is how `lm_train_corpus` scales beyond demo phrases.

## How it works for a human (intuition, analogy)

Treat the corpus as a **long ribbon of bytes**. A window is a scissors cut of length \(T\) (≤ 32). Each cut gives you several supervision points: at every position inside the cut except the last, the model predicts the following byte. Slide the scissors one byte (or stride \(s\)) forward and cut again.

## Formal view (definitions, formulas, shapes, algorithms)

**Corpus.** Byte sequence \(c[0..N-1]\) from file bytes (no UTF-8 decoding in v1—raw bytes match [`tokenizer.md`](tokenizer.md)).

**Window at start index \(s\).** Length \(L = \min(T_{\max}, N-s)\), ids \(x_i = c[s+i]\) for \(i=0..L-1\), \(T_{\max} =\) `LM_MAX_CTX`.

**Valid loss positions.** \(t = 0..L-2\); target at \(t\) is \(x_{t+1}\).

**Sliding.** stride `stride` (default 1): next window starts at \(s + \text{stride}\). Stop when \(s + 2 > N\) (need at least 2 bytes for one CE term).

**Planned struct (`lang/text_dataset.h`).**

```c
typedef struct text_dataset {
	char  *path;
	int   *ids;       /* full corpus ids, length n_ids */
	int    n_ids;
	int    cursor;    /* next start index */
	int    max_len;   /* LM_MAX_CTX */
	int    stride;
} text_dataset;

int  text_dataset_load(text_dataset *ds, const char *path);
void text_dataset_free(text_dataset *ds);
void text_dataset_reset(text_dataset *ds);

/* Copy up to max_len ids starting at cursor; advance cursor by stride.
   Returns L (window length), or 0 if exhausted. */
int  text_dataset_next_window(text_dataset *ds, int *ids_out, int max_len);
```

**Algorithm `text_dataset_load`.**

```text
read entire file into buffer
allocate ids[n_bytes]
for i in 0..n_bytes-1: ids[i] = (unsigned char)buffer[i]
cursor = 0
```

**Optional shuffle (epoch boundary).** `text_dataset_shuffle_starts` permutes start indices `{0, stride, 2*stride, ...}` for SGD noise—not required for toy.txt.

## Complexity / numerical pitfalls / design choices (CS depth)

**Memory.** Load **whole file** into RAM—fine for toy.txt (kilobytes); streaming for GB corpora is future work.

**Vocab.** Byte ids 0–255 always in vocab; no UNK for raw bytes.

**Windows crossing file end.** Last window may be short; loss normalizes by `L-1` per window inside training loop.

**Empty / tiny files.** `n_ids < 2` → load error; trainer should skip or fail loudly.

**CRLF.** Windows line endings add `\r` bytes—model may learn `\r\n`; document or normalize when building toy data.

**Path.** Default corpus path `data/toy.txt` relative to repo root ([lm-cli.md](lm-cli.md)).

## How it shows up in this repo

- Today: [`lang/tokenize.c`](../lang/tokenize.c) `tokenize_encode` on C strings in memory only.
- Planned: `lang/text_dataset.c`, sample file `data/toy.txt` (user-provided or generated).
- Consumer: `lm_train_corpus` ([mini-batch-corpus-training.md](mini-batch-corpus-training.md)).

## Worked example

**File contents** `"abc"` → ids `[97, 98, 99]`, `max_len=32`, `stride=1`.

| Window start | ids | CE positions |
|--------------|-----|--------------|
| 0 | `[97,98,99]` | t=0→98, t=1→99 |
| 1 | `[98,99]` | t=0→99 |
| 2 | exhausted (need ≥2 ids) | — |

Two windows, three scalar CE terms total if both used in one epoch pass.

## What next

- Add `data/toy.txt` and loader implementation.
- Hook `text_dataset_reset` at each epoch in `lm_train_corpus`.
- Report corpus **byte count** and **windows per epoch** in CLI logs.
