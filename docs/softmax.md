# Softmax

## What problem it solves

Classifiers need a list of raw scores (“logits”) turned into **probabilities that add to 1**. Softmax does that: bigger scores become larger shares of the pie, and nothing goes negative.

## How it works (layman)

Exponentiate each score so everything is positive, then divide by the total so the pieces sum to one. Large logits can overflow `exp`, so we first **subtract the maximum** logit. That shifts all scores down without changing the final ratios, and keeps `exp` in a safer range.

## The math

For logits \(\mathbf{z}\):

\[
\operatorname{softmax}(z)_i = \frac{e^{z_i - \max_j z_j}}{\sum_k e^{z_k - \max_j z_j}}
\]

The \(\max\) shift cancels in the ratio but prevents overflow.

## How it shows up in this repo

- [`ops.h`](../num/ops.h) / [`ops.c`](../num/ops.c): `vec_softmax(out, in)` writes probabilities into `out` (same length as `in`).
- Used as a building block for cross-entropy and later language-model heads.

## Tiny example

Logits `[1, 2]` → after stable softmax ≈ `[0.269, 0.731]`, summing to 1.
