# Flash-style attention (R11)

## Abstract

Fused causal attn kernels now accept T≤256 (shared-memory block). Full Flash tiling that avoids materializing `B·H·T·T` weights remains future work; raising the fuse cap unblocks R6 ctx=256.

## How it shows up in this repo

- `ATT_BLOCK 256` in [`num/gpu_cuda.cu`](../num/gpu_cuda.cu)
- [`lang/attn.c`](../lang/attn.c) pcache fuse gate
