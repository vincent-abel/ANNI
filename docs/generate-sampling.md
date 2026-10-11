# Generate sampling

## Abstract

`lm generate` gains three knobs, `--temperature`, `--top-k`, and `--top-p`, that reshape the next-token distribution before one id is drawn.

## What problem it solves

Greedy decoding always picks the argmax and tends to loop. Pure sampling at temperature 1 sometimes picks very unlikely tokens and derails. Users need a small, standard set of controls between those extremes.

## How it works for a human

Think of the model's scores as a ranked menu. **Temperature** flattens or sharpens the odds. **Top-k** crosses off everything except the \(k\) best items. **Top-p** keeps the shortest list of best items whose combined probability reaches \(p\). Then one item is drawn at random from what remains.

## Formal view

Logits \(z\). Temperature \(T>0\): \(p_i=\operatorname{softmax}(z/T)_i\). Top-k keeps the \(k\) largest \(p_i\). Top-p sorts descending and keeps the smallest prefix with \(\sum p_i\ge p\). Renormalize survivors and sample. \(T\to0\) approaches greedy; \(T=0\) is treated as argmax.

## Complexity / pitfalls / design choices

- **Order:** temperature, then top-k, then top-p, then renormalize.
- **Cost:** sorting \(V\) logits per step is \(O(V\log V)\); fine for \(V\le16\text{k}\), partial selection could replace it later.
- **Edge cases:** \(k\ge V\) or \(p=1\) is a no-op; always keep at least one token.
- **Reproducibility:** seed the RNG explicitly.

## How it shows up in this repo

- Sampling in [`lang/generate.c`](../lang/generate.c), where temperature sampling already exists
- Flags are parsed in [`main.c`](../main.c) for `lm generate`
- Background: [Text generation](text-generation.md), [Generation from checkpoints](generation-from-checkpoints.md)

## Tiny worked example

Probabilities `[0.5, 0.3, 0.15, 0.05]`. Top-k=2 keeps `[0.5, 0.3]` → renormalized `[0.625, 0.375]`. Top-p=0.9 keeps the first three (cumulative 0.95 ≥ 0.9).

## What comes next

[`--out` resume confirm](out-resume-confirm.md) covers safer CLI behavior when writing checkpoints.
