# Fused epilogues (S8)

## Abstract

Combine common post-GEMM elementwise steps into one kernel launch to cut launch tax after Wave 1 residency.

## How it shows up in this repo

- `gpu_bias_relu_forward` in [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): bias add + ReLU in one launch
- [`lang/ffn.c`](../lang/ffn.c) `ffn_forward_cache` prefers the fused path; keeps a separate pre-ReLU `pre_act` for the backward mask
- Residual add stays on `gpu_mat_add` (already device-resident); LN stays on the fused LN path from E7

## Pitfalls

- Backward still needs pre-activation values, so the fused forward is not a pure single-buffer win
- Host fallback remains if the GPU epilogue fails
