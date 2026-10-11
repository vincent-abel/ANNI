# Packed activation arena (E4)

## Abstract

One `pack_ctx` holds steady `(Bmax,T)` caches and a tail slot so packed train never reallocates activations inside the step loop.

## What problem it solves

Per-step `mat_create` / cache rebuilds add latency and fragment VRAM. A persistent arena keeps the working set warm.

## How it works for a human

At train start, allocate the big pack once. The last shorter microbatch of an epoch uses the second slot. Shapes stay fixed until B or T changes.

## Formal view

Amortized alloc cost \(O(1)\) per run vs \(O(\#\mathrm{steps})\) if rebuilt each microbatch.

## How it shows up in this repo

- [`lang/lm_train.c`](../lang/lm_train.c): `pack_ctx`, `pack_slot[2]`

## What comes next

[Pinned host staging](pinned-host-staging.md) (E5) for the ids buffer.
