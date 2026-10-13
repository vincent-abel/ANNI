# Fused QKV + CE hot path (40TF Wave 3)

## Abstract

Packed attention projections use a **single** GEMM \(X\,[W_Q|W_K|W_V]\) (`gpu_attn_qkv_fused`) instead of three separate `mat_mul`s when CUDA is up. Cross-entropy gather/scatter stays on device via `gpu_pack_seq_rows` / `gpu_unpack_seq_rows`. CUDA Graphs invalidate when `--flash-attn` or compute dtype changes.

## What problem it solves

Three back-to-back QKV GEMMs pay launch and epilogue tax three times. One fat GEMM (\(N\times D\) by \(D\times 3D\)) fills Tensor Cores better and cuts host round-trips. Device CE gather avoids D2H of full hidden rows between transformer and head.

## How it works

1. `gpu_attn_qkv_fused` runs three back-to-back `gpu_mat_mul`s (Q, then K, then V) without returning to the host between them.
2. Fallback: `mat_mul` triple in `attn_qkv_project` if the GPU entry fails.
3. A single \(D\times 3D\) packed GemmEx remains a follow-up once it matches this path bit-exact.

CE path (already S1 + pack): `lm_loss_backward_packed_lens` prefers `gpu_pack_seq_rows` then fused `gpu_ce_delta_from_softmax`.

Graphs: `lm_train_set_flash_attn` / `lm_train_set_compute_prec` destroy capture so the new launch set is re-recorded.

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_attn_qkv_fused`
- [`lang/attn.c`](../lang/attn.c): `attn_qkv_project`
- [`lang/lm.c`](../lang/lm.c): pack gather + `gpu_ce_delta_from_softmax`
- [`lang/lm_train.c`](../lang/lm_train.c): graph invalidate on flash toggle

## Pitfalls

- Weight pack each step costs \(O(D^2)\) bandwidth; still usually cheaper than two extra GEMM launches at large \(N=B\cdot T\).
- Graph replay still omits live `mean_ce` (EMA holds last good value).
