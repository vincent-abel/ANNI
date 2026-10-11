# Vocab in checkpoints and CLI

## Abstract

A model is meaningless without the tokenizer that produced its ids. Checkpoints will embed the vocabulary (specials, merge table, mode) and the CLI will read it back so `generate` needs no extra files.

## What problem it solves

Weights trained with one merge table and decoded with another produce nonsense that looks like a model bug. Shipping a separate `vocab.bin` invites drift between the two files.

## How it works for a human

When you save, the file holds weights plus a small vocabulary section. When you load for `generate` or `--resume`, the program rebuilds the tokenizer from that section. Training flags choose a tokenizer once (`byte` or `bpe`, vocab size); after that the checkpoint decides.

## Formal view

Checkpoint \(=\) header (magic, version, dims, \(|V|\), tokenizer mode) \(\Vert\) vocab payload (\(M\) merges as pairs \((a_i,b_i)\), specials) \(\Vert\) tensors \(\Vert\) optimizer state. The vocab payload size is \(O(M)\): about \(8M\) bytes for 32-bit pairs, roughly 128 KB at 16k.

## Complexity / pitfalls / design choices

- **Version bump** in the magic/header so old byte-only files still load as byte mode.
- **Consistency check:** header \(|V|\) must equal the embedding row count.
- **Resume** requires the same vocab; mismatch with `--data`-derived vocab is an error.
- **CLI:** `--tokenizer byte|bpe`, `--vocab PATH`, `--vocab-size N`; all are checked against (never override) the checkpoint on `--resume`.

## How it shows up in this repo

- [`lang/lm_serialize.c`](../lang/lm_serialize.c) / [`lm_serialize.h`](../lang/lm_serialize.h): BPE models are written with magic `LM_MAGIC_BPE` (`LMCV`): same header as `LMCF`, then the merge table (`i32 n_merges` + `n_merges` pairs of `i32`), then tensors and the optional Adam section. Byte models still write `LMCF`, so old files load as byte mode. Header \(|V|\) must equal the merge table size and the embedding rows.
- `lm_load`, `lm_load_checkpoint`, `lm_merge_average` carry the tokenizer (`lm->bpe`); merging models with different tokenizers is rejected.
- CLI in [`main.c`](../main.c):
  - `./ann lm train --tokenizer bpe [--vocab-size N]` trains merges on `--data` (train split only if `--val-frac` is set), saves them to `OUT.vocab`, and embeds them in `OUT`.
  - `--vocab PATH` loads that vocab file if it exists, otherwise trains and writes it; `--vocab-size` defaults to 16384 and `--bpe-max-bytes N` caps the bytes used for merge training.
  - `--resume`: the checkpoint's embedded vocab wins; a `--vocab` file must be identical or training aborts.
  - `lm generate` / `lm load` need no flag for a BPE checkpoint. `--vocab PATH` is verified against the embedded table (mismatch is an error) or attached to a vocab-less model of equal size.
  - No vocab flag and a byte checkpoint: unchanged byte behavior.
- [LM weight serialization](lm-weight-serialization.md), [Checkpoint and resume](checkpoint-and-resume.md)
- Test: `test_bpe_lm_checkpoint` saves/loads a BPE model and re-encodes with the loaded tokenizer.

## Tiny worked example

Train with `--tokenizer bpe --vocab-size 16000`, save `model.bin`. Later `./ann lm generate --load model.bin "Hello"` encodes the prompt with the embedded merges and decodes the output with the same table.

## What comes next

[Real corpora at scale](real-corpora-scale.md) supplies text big enough to justify 16k tokens.
