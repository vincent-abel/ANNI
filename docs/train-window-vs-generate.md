# Train window vs generate

## Abstract

Training reads text in fixed-length windows; generation grows a sequence one token at a time. The default `--window` for training becomes **64**, while the model's hard limit stays `LM_MAX_CTX` (128). This note explains how the two lengths relate.

## What problem it solves

The old default window was 8, which teaches the model almost no long-range structure. But a window that is too large costs \(O(T^2)\) attention per sample. Without a clear rule, users also confuse "how much context the model trained on" with "how much it can accept."

## How it works for a human

Training chops the corpus into snippets of `window` tokens and asks "predict each next token." The model learns patterns up to that length. At generate time you can feed a prompt up to the model's maximum, but beyond the trained window the quality degrades because positions it never saw are out of distribution.

## Formal view

Let window \(W\), max context \(C=\) `LM_MAX_CTX`, with \(W \le C\). Each training sample gives \(W-1\) next-token targets. Attention cost per sample is \(O(W^2 d)\); cost per target token grows linearly in \(W\). Generation at position \(t\) needs \(\min(t, C)\) tokens of history.

## Complexity / pitfalls / design choices

- **Default 64:** half of the max context, big enough for phrases and sentences, cheap enough for CPU runs.
- **Validation:** `--window` must be \(\le\) `LM_MAX_CTX`; reject larger values.
- **Prompt truncation:** keep the *last* \(C\) tokens when a prompt is too long.
- **Eval windows** should use the same \(W\) as training.

## How it shows up in this repo

- `--window` in [`main.c`](../main.c) (currently default 8)
- Context cap in [`lang/lm_config.h`](../lang/lm_config.h)
- [Text dataset loading](text-dataset-loading.md) for sliding windows

## Tiny worked example

A 1000-token file with \(W=64\) yields 937 sliding windows (stride 1). Asking generate for 200 tokens from a 10-token prompt crosses \(C=128\) around token 118, so history is cropped from the left.

## What comes next

[Generate sampling](generate-sampling.md) controls how each next token is chosen.
