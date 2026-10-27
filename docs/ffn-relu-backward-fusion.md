# FFN ReLU backward in one pass (80 TF, candidate 5)

## Abstract

In BF16 mode the FFN backward ran three memory-bound passes over the \(T \times D_{FF}\) hidden gradient `dh` after the W2 dgrad GEMM: the ReLU mask (`relu_bwd_kernel`), the FP32 to BF16 conversion of `dh` for the W1 wgrad and again for the W1 dgrad (the one-shot gradient twin refused buffers over 16 M elements, and `dh` has 16.8 M at \(T = 4096\)), and the column sums for \(\partial b_1\). One kernel now does all of it: it reads `dh` and `hidden` once (float4), writes the masked BF16 twin that both W1 GEMMs then consume, and accumulates the column sums with one atomic per column and 64-row slab, exactly as `col_sum_tiled_kernel` did. The FP32 `dh` is left unmasked; nothing reads it in this mode.

## Why not a cuBLASLt epilogue

The first attempt used cuBLASLt's `RELU_AUX_BIAS` epilogue on the W1 forward GEMM (bit mask out) and `DRELU_BGRAD` on the W2 dgrad GEMM (mask in, bias gradient out). It passed the correctness gate but was 11% slower: on the RTX 5070 Ti with CUDA 13.3 the aux-mask epilogue makes the W1 GEMM 7.5 ms slower per microbatch (22.8 to 30.4 ms forward) and the `DRELU` dgrad is slower than the plain GEMM plus the separate mask kernel. Measured with the bench of `scripts/prof_step.sh`, 12 steps each, native build 51.1k tok/s, epilogue build 45.8k, epilogue build with the mask disabled 51.0k. The epilogue route was dropped.

## How it works

`gpu_ffn_relu_bwd_fused(dh, hidden, db1)` in [`num/gpu_cuda.cu`](../num/gpu_cuda.cu):

1. reserves a gradient-twin slot for `dh`'s device buffer (`grad_twin_reserve`, no size cap);
2. launches `relu_bwd_pack_colsum_kernel`: one thread per 4 columns and 64 rows, mask `hidden > 0`, `__floats2bfloat162_rn` stores, FP32 partial sums, `atomicAdd` into the `db1` device accumulator of `gpu_mat_col_sum_add`;
3. in eager mode flushes the accumulator to the host like `gpu_mat_col_sum_add`.

`cublas_gemm_rm_at` (the wgrad) now checks `grad_twin_peek` before converting its gradient operand, and the dgrad takes the twin as before. [`lang/ffn.c`](../lang/ffn.c) `ffn_backward` calls the fused entry first and keeps the three-step path as the fallback (FP32, FP8, CPU).

## Results (pause-window gate, 2026-10-26)

| Build | Bench tok/s | Meter TF | fwd / bwd ms per microbatch | Peak VRAM |
|---|---|---|---|---|
| native (2775c4f) | 50.0k | 75.4 | 23.0 / 48.8 | 13408 MiB |
| fused ReLU backward | 52.8k | 79.8 | 22.9 / 46.3 | 13263 MiB |

Loss trajectory over the 12 gate steps identical to the reference; `make test-gpu` BF16 losses identical to the native build's at every step.

## Candidate 6: accumulate instead of add

The same profile showed 48 `axpy_kernel` launches per microbatch (1.5 ms): two per layer in the attention backward (`dx = dq Wq^T + dk Wk^T + dv Wv^T` built with a scratch product and an add) and two per layer for the block residual gradients (LayerNorm backward into a scratch, then added). Both now accumulate in place: `gpu_mat_mul_bt_acc` runs the dgrad GEMM with \(\beta = 1\) (`mat_mul_wt_add` in [`lang/gradutil.c`](../lang/gradutil.c)), and `gpu_layernorm_backward_acc` makes `ln_bwd_kernel` add its result (`layernorm_backward_add`). The values are the same sums in the same precision; only the scratch write and read go away. Two conversions went with it: the attention output `concat` is published as a BF16 twin by the fused attention forward (the Wo GEMM and its wgrad both read it), and the head's delta gets a dedicated gradient-twin slot (134 MB) so its dgrad reuses the wgrad's conversion.

| Build | Bench tok/s | Meter TF | Peak VRAM |
|---|---|---|---|
| fused ReLU backward (candidate 5) | 52.7k | 79.6 | 13263 MiB |
| + accumulate, concat twin, head twin (candidate 6) | 54.7k | 82.5 | 13566 MiB |

## Pitfalls

- The FP32 `dh` is unmasked after the fused call: a later CPU or FP32 consumer of it would be wrong. Only the BF16 GEMM paths run after it; the fallback branch masks in place as before.
- `hidden` and `dh` must be device-resident in place (`MAT_DEV_DEVICE`): the twin is keyed on `dh`'s device pointer, and the mask reads `hidden` on the device.
- `scripts/pause_gate.sh` now benches the fallback first and rejects a candidate slower than 98% of it, after the epilogue build slipped through on correctness alone.
