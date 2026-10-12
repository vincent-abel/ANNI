# Huge batch / VRAM fill (R3–R4)

## Abstract

`--batch auto` was capped at 64, so a 16 GB card often sat near 2 GB used. R3 raises the auto cap to 1024 and relaxes the packed-bytes estimate; R4 keeps OOM backoff for large fixed `--batch`.

## What problem it solves

Small B leaves GEMMs thin and util low even after residency. Larger B raises tokens per step and amortizes launch overhead.

## How it works for a human

Ask for `--batch auto` (or a big `--batch 512`). The trainer estimates activation bytes, picks the largest B under `vram_frac × free`, then halves B if allocation fails.

## Formal view

Budget \(B_{\max}=\max\{B: \widehat{M}(B,T,V)\le f\cdot M_{\mathrm{free}}\}\) with cap `LM_BATCH_AUTO_MAX`. On OOM, \(B\leftarrow\lfloor B/2\rfloor\) until alloc succeeds.

## Complexity / pitfalls / design choices

- Estimate is conservative; real peak may differ with graphs / checkpointing later.
- Tiny corpora still clamp B by window count.
- CPU builds keep a small fallback B.

## How it shows up in this repo

- [`lang/lm_train.h`](../lang/lm_train.h): `LM_BATCH_AUTO_MAX` 1024
- [`lang/lm.c`](../lang/lm.c): `lm_packed_bytes` — **S14** safety ×1.35; flash counts one layer of scores; act-checkpoint uses ~2 live blocks
- [`lang/lm_train.c`](../lang/lm_train.c): `resolve_batch`, `pack_ctx_init` OOM halve

## Tiny worked example

Free 14 GB, frac 0.85 → budget ≈ 12 GB. At T=64, D=256 the estimate may allow B in the hundreds where the old cap stopped at 64. With `--flash-attn --act-checkpoint` the estimate frees more budget for auto-B.

## What comes next

[activation checkpointing](activation-checkpointing.md) / [adam offload](adam-cpu-offload.md) when still VRAM-bound; SOTA memory notes in [sota-memory.md](sota-memory.md).
