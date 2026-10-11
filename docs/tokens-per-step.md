# Tokens per step (E9)

## Abstract

`--tokens-per-step N` derives `grad_accum` from packed `B*T` so each Adam update sees a target token budget.

## Formal view

\(\mathrm{accum}=\lceil N/(B\cdot T)\rceil\).

## How it shows up in this repo

- [`lang/lm_train.c`](../lang/lm_train.c): `apply_tokens_per_step`
- [`main.c`](../main.c): `--tokens-per-step`
