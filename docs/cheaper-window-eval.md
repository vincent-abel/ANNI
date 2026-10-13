# Cheaper window eval

## Abstract

`lm_nll_windows` scores held-out text by running a forward pass on every sliding window. Two CLI flags, `--eval-every` and `--eval-windows`, make that cheaper during training.

## What problem it solves

Held-out NLL is useful, but a full pass over a large validation set after every epoch can cost more than the training step it is measuring. On a big corpus the eval, not the gradient, becomes the bottleneck.

## How it works for a human

Two dials. `--eval-every N` says "only evaluate every N epochs (or steps)." `--eval-windows K` says "when you do evaluate, look at at most K windows." Fewer, spread-out windows give a noisier but still honest estimate of generalization.

## Formal view

With windows \(w_1,\ldots,w_M\) and per-window mean NLL \(\ell_i\), the full metric is \(\bar\ell=\frac1M\sum_i \ell_i\). Sampling \(K\) windows gives \(\hat\ell=\frac1K\sum_{j}\ell_{i_j}\) with standard error about \(\sigma/\sqrt K\). Evaluation cost drops from \(O(M)\) forwards to \(O(K)\) per eval, and total eval cost by a further factor of \(N\).

## Complexity / pitfalls / design choices

- **Fixed subset:** use the same evenly strided windows each time so curves stay comparable between evals.
- **Final eval:** always run one full (or large) eval at the end of training.
- **Perplexity:** \(\exp(\hat\ell)\) inherits the noise; do not compare two runs that used different \(K\).
- **Defaults:** `--eval-every 1` and `--eval-windows 32`. Pass `--eval-windows 0` for a full pass (can dominate tiny smokes).

## How it shows up in this repo

- Metric: `lm_nll_windows` in [`lang/lm_train.h`](../lang/lm_train.h)
- CLI parsing in [`main.c`](../main.c) next to `--epochs` and `--window`
- Concept: [Held-out loss / perplexity](held-out-loss-perplexity.md)

## Tiny worked example

1000 validation windows, 20 epochs. Full eval each epoch is 20,000 forwards. With `--eval-every 5 --eval-windows 100` it is 4 × 100 = 400 forwards, 50× less.

## What comes next

[Train window vs generate](train-window-vs-generate.md) clarifies which context length these windows use.
