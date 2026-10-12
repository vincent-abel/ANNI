# Activation checkpointing (R9 / S11)

## Abstract

`--act-checkpoint` drops device twins of per-block activations after each block forward (keeping `x_in`), then recomputes the block in backward before `tblock_backward`. Trades compute for VRAM so larger B can fit.

## How it shows up in this repo

- CLI: `--act-checkpoint` → `lm_train_set_act_checkpoint` → `block_set_act_checkpoint`
- [`lang/block.c`](../lang/block.c): `block_cache_drop_device` after each forward block; recompute from `x_in` in `transformer_backward`
- Host `x_in` mats remain allocated; only GPU twins are freed between blocks

## Pitfalls

- Roughly +1 forward per block on the backward pass
- Combine with `--flash-attn` for the largest VRAM cut
- CUDA Graphs may need re-capture if launch sets change
