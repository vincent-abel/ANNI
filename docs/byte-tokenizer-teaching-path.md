# Byte tokenizer as teaching path

## Abstract

Subword BPE becomes the default for real training, but the **byte tokenizer stays** as a first-class, selectable path. It is the smallest tokenizer that works and the best one for learning and debugging.

## What problem it solves

A 16k BPE vocabulary hides what happens between text and ids. A newcomer cannot hand-trace it, and tests that depend on a trained merge table are heavy. The byte path keeps a fully transparent, dependency-free pipeline.

## How it works for a human

One byte, one id. No training step, no merge file, no vocab to ship. Choosing the mode is a switch: `byte` for lessons, tests, and tiny corpora; `bpe` for real runs. Checkpoints record which was used so you cannot mix them up.

## Formal view

Byte mode has \(V=256\) (plus specials), \(\operatorname{enc}\) is the identity on bytes, and sequence length equals byte length \(n\). BPE with \(V=16\text{k}\) shortens sequences by a compression ratio \(r>1\) (typically 3–4× on English), so attention cost drops by roughly \(r^2\), while the embedding and head tables grow by \(V/256\).

## Complexity / pitfalls / design choices

- **Same interface** (`encode` / `decode`) so the trainer does not care which is active.
- **Tests stay on bytes** for speed and determinism.
- **Mismatch guard:** loading a BPE checkpoint with byte mode is an error, not silent garbage.
- **Byte mode on large text** is slow: very long sequences for the same content.

## How it shows up in this repo

- [Tokenizer](tokenizer.md) and [`lang/tokenize.c`](../lang/tokenize.c)
- [Vocabulary](vocabulary.md)
- Sizes in [`lang/lm_config.h`](../lang/lm_config.h) (`LM_BYTE_VOCAB`)

## Tiny worked example

`"hello"` in byte mode is 5 ids. A BPE vocab that merged `he`, `ll`, `o` might give 3. Both decode back to `"hello"`.

## What comes next

[Subword BPE tokenizer](subword-bpe-tokenizer.md) is the default for real corpora.
