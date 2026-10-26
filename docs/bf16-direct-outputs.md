# BF16 outputs straight from the GEMMs (candidates 8 and 9)

## Abstract

In BF16 mode every activation was produced in FP32 by a GEMM, then converted (and for Q, K, V also re-laid out head-major) before the next tensor-core consumer read it. Two of those round trips are gone: the Q/K/V projection is now one cuBLASLt GEMM \(X\,[W_Q|W_K|W_V]\) whose output is BF16 rows `[B·T][3D]` that the FA2 attention kernels read strided (candidate 8), and the FFN's \(\mathrm{ReLU}(X W_1 + b_1)\) is written once as the BF16 twin that the W2 GEMM, its wgrad and the fused ReLU backward consume (candidate 9). The FP32 `q`, `k`, `v` and `hidden` mats of the layer caches are no longer written: their device buffers only serve as the keys of the BF16 slots.

## Candidate 8: fused QKV projection

- `cublas_lt_gemm_half` takes `out_bf16`: the C/D layout is BF16 and the epilogue bias, when there is one, must be BF16 as well (cuBLASLt refuses an FP32 bias with a BF16 output; `g_lt_bias16` is that copy).
- `[W_Q|W_K|W_V]` is one BF16 `[D][3D]` matrix packed from the three Adam-flat twins by `qkv_wt_pack_kernel`, rebuilt once per optimizer step (keyed on the weight epoch).
- The attention layer tells the projection that the fused attention will follow (`gpu_attn_qkv_bf16_hint(T, dh)`, one shot); `gpu_attn_qkv_fused` then writes the layer's `qkv_bf16` slot (24 MB at 4096 x 3072) with one Lt GEMM instead of three GemmEx calls on three streams, each followed by a host `cudaStreamSynchronize`.
- `attn_fa2.cuh` addresses its operands through `fa2_lay` strides, so the same kernels read head-major packs (the harness, the fallback) or row-major slices (`ld = 3D`, head offset `h·dh`).

Removed per layer and microbatch: three FP32 `[4096][1024]` writes, three head-packing passes in the forward, and the three host syncs.

## Candidate 9: BF16-only ReLU output

- `act_bf16_reserve` hands the W1 GEMM the layer's activation-twin buffer; `gpu_mat_mul_epi` runs the `RELU_BIAS` epilogue with `out_bf16` into it (batch mode only; `ANN_HID_BF16=0` restores the FP32 output plus conversion). The slot is marked `bf16_only`, which `ANN_TWIN_CHECK` skips and `gpu_act_bf16_only` reports.
- `relu_bwd_pack_colsum_kernel` takes the mask from that twin (8 bytes per 4 elements instead of 16): `bf16(h) > 0 ⟺ h > 0` except for FP32 subnormals below the BF16 range, which are irrelevant to a ReLU gradient.
- `ffn_backward` refuses to fall back to the FP32 mask when the hidden is BF16-only, so a failure is an error, never a silent wrong gradient.

Removed per layer and microbatch: one 64 MB FP32 write, one 16.8 M-element conversion, half of the mask read.

## Results

Pause-window gate 2026-10-26 20:52 (both candidates in one binary, fallback candidate 7): PASS, peak VRAM 13485 MiB, 12-step loss trajectory identical to the reference.

| Build | microbatch fwd / bwd | Bench tok/s | Meter TF |
|---|---|---|---|
| FA2 attention (candidate 7) | 19.7 / 38.3 ms | 61.4k | 92.7 |
| + fused BF16 QKV, BF16-only ReLU output (8 + 9) | 17.3 / 38.4 ms | 64.3k | 97.2 |
| + lazy pre-activation buffer, BF16-only LayerNorm output (10 + 12) | | 64.7k | 97.7 |
| + producer-written gradient twins, per-layer QKV weight pack (13) | 16.7 / 38.9 ms | 64.8k | 97.8 |
| + dO as a BF16 twin, warp-per-row LayerNorm (14 + 15) | 16.4 / 39.9 ms | 65.5k (idle-GPU bench; 13 = 65.2k) | 99.0 |
| + fused head softmax / CE / BF16 delta (16) | 16.4 / 38.4 ms, loss 7.2 -> 5.7 ms | 67.3k | 101.7 |

Candidate 12 adds the LayerNorm output to the same scheme (`ln_fwd_kernel` stores the BF16 twin instead of FP32 `y`; `ANN_LN_BF16=0` restores it) and stops allocating the FFN pre-activation buffer that only the fallback paths use (64 MB per layer): the gate's peak VRAM went from 13485 to 12604 MiB, the live run from 86.9% to under 82% of the card. The gain in speed is within the bench noise (+1%); the memory was the point.

## Candidate 13: gradient twins written by their producers

The gradients that the next GEMMs need in BF16 were still converted by the wgrad (`grad_twin_put`): dK and dV from the attention backward, and the block's residual-path gradients from the accumulating LayerNorm backward. `attn_fa2_bwd_kernel` now also stores BF16 dK / dV into two reserved twin slots, and `ln_bwd_kernel` (accumulate mode) stores the BF16 twin of its result; `grad_twin_peek` in the wgrad and `grad_twin_take` in the dgrad pick them up. The packed `[W_Q|W_K|W_V]` matrix is cached per layer instead of being repacked for every layer (its single cache thrashed between the twelve layers, 12 extra kernels per microbatch).

Two rules came out of the test suite while doing this. A twin keyed on a device buffer must never outlive the microbatch (`gpu_grad_twins_clear` at the end of `lm_backward_packed`): the first block's LayerNorm backward has no GEMM consumer, and the next microbatch reuses that buffer for the last block's incoming gradient. And a dgrad that does not consume the twin (the FP8 path) must forget it, or with a larger slot pool a later wgrad peeks a stale one; `make test-gpu` caught both as wrong `dw1` / `dw2` gradients.

## Candidate 14: the attention output's gradient as a BF16 twin

`gpu_mat_mul_bt_twin` runs the Wo dgrad (\(dO = dY\,W_O^T\)) as a cuBLASLt GEMM with a BF16 output into a reserved gradient-twin slot of `dconcat`; the FP32 `dconcat` is never written. `gpu_attn_fa_bwd` takes that twin, reads it strided (`fa2_lay_rows(T, D, dh)`) instead of packing heads, and computes \(D_i = \langle dO_i, O_i \rangle\) from it (`attn_fa_dvec16_kernel`). The FP8 dgrad path is excluded (it writes FP32 and would leave the twin unwritten; `make test-gpu`'s FP8 section caught that). Removed per layer: one 16 MB FP32 write, one head-packing pass, one FP32 read in dvec.

## Candidates 15 and 16: LayerNorm kernels and the fused head loss

Candidate 15 (worker branch `worker/ln-colsum`) runs the LayerNorm forward and backward as one warp per row with float4 traffic and folds the dgamma / dbeta column sums into the backward (two launches fewer per LayerNorm); `ANN_LN_FAST=0` restores the old kernels. On the full step it is a wash (the forward is at DRAM bandwidth when cold, the backward gains ~0.3 ms), kept for the launch count.

Candidate 16 (worker branch `worker/ce-bf16`): `ce_softmax_delta_bf16_kernel` reads each FP32 logits row once (held in registers), computes the softmax, the CE term and the db_head column sums, and writes the delta \(p - \mathrm{onehot}\) only as BF16 in place over the first half of the row's own FP32 storage (row stride \(2V\)); the gradient-twin slots carry a row stride so the head wgrad and dgrad consume it without any conversion. `lang/lm.c` fails the step if either GEMM did not use the twin. It replaced softmax + CE + column sum + two 67 M-element conversions (about 1.2 GB of traffic) by one 268 MB read and one 134 MB write: the loss phase went from 7.2 to 5.7 ms per microbatch. `ANN_CE_BF16=0` restores the old path; eager mode, FP32, FP8 and the quantized head keep it.

## Pitfalls

- The FP32 `q`, `k`, `v`, `hidden` buffers hold nothing in BF16 training: any new consumer must read the twins (`act_bf16_get`, the `qkv_bf16` slot). Eager evaluation (`lm info`, generation) keeps the FP32 outputs.
- cuBLASLt returns `CUBLAS_STATUS_NOT_SUPPORTED` for a cached algorithm when a later call's pointers are less aligned than the heuristic assumed (seen in `make test-gpu` on per-head slices); the wrappers fall back to the FP32 path or GemmEx, so this costs speed, not correctness.
