# `lm merge`: averaging the weights of several models

## Abstract

`anni lm merge --out OUT.bin a.bin b.bin [...]` writes a model whose every weight is the mean of the same weight in the inputs. It takes 2 to 32 models of the same shape and tokenizer.

## What problem it solves (plain language)

The last checkpoint of a run is one point on a noisy path. The mean of a few nearby checkpoints is often a bit better than any of them, and it costs one pass over the files, no training and no GPU.

## How it works for a human (intuition, analogy)

Several photos of the same scene, each with its own grain: stack them and the grain fades while the scene stays. It only works if the photos show the same scene from the same spot.

## Formal view (definitions, formulas, shapes, algorithms)

For `n` models with weights `theta_1 .. theta_n`:

```text
theta = (theta_1 + ... + theta_n) / n
```

applied to the embedding table, every block (`Wq`, `Wk`, `Wv`, `Wo`, `W1`, `b1`, `W2`, `b2`, both LayerNorm `gamma` and `beta`), the head and its bias.

## Complexity / numerical pitfalls / design choices

- **Same basin only.** Average checkpoints of one run, or fine-tunes that started from the same weights. Two models trained from different seeds store the same functions in different neurons, and their mean is noise.
- **Checks before merging:** same number of blocks, same vocabulary size, same BPE merges (`bpe_equal`), and all tied or all untied ([Weight tying](weight-tying.md)). Anything else is refused.
- **Tied models:** the shared table is averaged once, through the head.
- **Weights only.** The output has no Adam state; training from it starts a fresh Adam.
- Memory: all inputs are loaded at once, so `n + 1` models in RAM.

## How it shows up in this repo

- [`lang/lm_serialize.c`](../lang/lm_serialize.c): `lm_merge_average(models, n, &out)` copies the first model, adds the others, scales by `1/n`.
- [`main.c`](../main.c): `run_lm_merge` loads weight files or training checkpoints alike.
- The merged model keeps a clone of the tokenizer ([Vocab checkpoint CLI](vocab-checkpoint-cli.md)).

## Use

```text
./anni lm merge --out models/avg.bin models/a.bin models/b.bin models/c.bin
```

## What next

- Weighted means, and a running average kept during training (EMA), are not there.
