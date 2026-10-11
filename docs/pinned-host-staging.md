# Pinned host staging (E5)

## Abstract

Ids (and any remaining H↔D staging) use `cudaHostAlloc` when CUDA is available so uploads can be faster and overlap-ready for R8 streams.

## What problem it solves

Pageable `malloc` ids force the driver to bounce through a staging buffer on every H2D of token ids.

## How it works for a human

Pack context prefers pinned ids; falls back to `malloc` on CPU builds.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h): `gpu_host_alloc` / `gpu_host_free`
- [`lang/lm_train.c`](../lang/lm_train.c): `pack_ctx.ids_pinned`

## What comes next

[CUDA streams overlap](cuda-streams-overlap.md) (R8/E6).
