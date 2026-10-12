# Flash-style attention (R11 / S7)

## Abstract

`--flash-attn` skips persistent `B·H·T·T` attention weight storage across the forward. The fused causal kernel still computes softmax in shared memory per row; backward recomputes weights into a temporary buffer before the existing fused bwd.

## How it shows up in this repo

- CLI: `--flash-attn` → `lm_train_set_flash_attn` → `attn_set_flash`
- [`lang/attn.c`](../lang/attn.c): no `w_all` / host `seq[]` when flash; `fused == 2` path
- `gpu_attn_flash_fwd` / `gpu_attn_recompute_w` in [`num/gpu_cuda.cu`](../num/gpu_cuda.cu)
- Cap remains **T ≤ 256** (`ATT_BLOCK`)

## Status vs full FlashAttention

This is Flash-*style* residency (no persistent scores), not IO-aware tiled FlashAttention-2/3. Backward still materializes one full `B·H·T·T` temp. Materialized fused attn (`w_all`) remains the default and fallback.
