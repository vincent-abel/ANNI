# Fused attention on mma.sync (candidate 7, 93 TF)

## Abstract

The WMMA attention kernels (forward 0.32 ms, backward 0.90 ms per layer at B=16, T=256, 16 heads of 64) ran at about 8 TFLOP/s: every product went through shared memory, dQ was built by two of four warps, and 42 KB of shared memory per block left two blocks per SM. [`num/attn_fa2.cuh`](../num/attn_fa2.cuh) rewrites both in the FlashAttention-2 structure on raw `mma.sync.m16n8k16` (BF16 in, FP32 accumulate) with `ldmatrix` loads: the forward keeps the 64 query rows of a block in A fragments, streams 64-key tiles of K and V through shared memory, keeps the running max and sum in registers (quad shuffles), repacks P from the accumulator fragments straight into the A fragments of \(O \mathrel{+}= P V\) and rescales O in registers; the backward keeps K and V rows in fragments, streams Q and dO tiles from the diagonal on, recomputes \(P^T\) from the stored log-sum-exp, accumulates dV and dK in fragments, and gets dQ by transposing the BF16 \(dS^T\) tile through shared memory (`ldmatrix.trans`) and FP32 atomics. Templated on the head size (16, 32, 48, 64) so the 64-wide test model runs the same code.

## Checks

`tests/bench_attn.cu` includes `gpu_cuda.cu` and runs both kernel pairs on random data at the training shape: the backward is bit-identical to the WMMA kernels on dq, dk and dv (same products in the same order), the forward differs by at most 8.5e-4 on O (values up to 1; the BF16 rounding of P happens at a different point in the online softmax) and 5e-7 on the log-sum-exp. `make test-gpu` passes with the kernels active; `ANN_FA2=0` restores the WMMA kernels.

## Results

| | per layer, forward | per layer, backward | microbatch fwd / bwd | Bench tok/s | Meter TF |
|---|---|---|---|---|---|
| WMMA (candidate 6) | 0.32 ms | 0.90 ms | 23.2 / 45.0 ms | 54.6k | 82.4 |
| FA2 mma.sync (candidate 7) | ~0.07 ms | ~0.25 ms | 19.7 / 38.3 ms | 61.8k | 93.3 |

Pause-window gate 2026-10-26 20:16: PASS, peak VRAM 13426 MiB, 12-step loss trajectory identical to the reference. The per-layer times were measured while sharing the GPU with the run and are approximate.

## Pitfalls

- Fragment layouts are hard-wired for `m16n8k16` (sm_80+): A/B/C element positions and the `ldmatrix` row addressing (`lane & 7`, `(lane >> 3) & 1`, `lane >> 4`) encode them; changing a tile shape means re-deriving them. The harness is the safety net.
- dQ uses FP32 atomics across the key tiles of a block row; results can differ in the last bit between runs, as they did with the WMMA kernel.
- `T % 64 == 0` and no variable-length sequences (`gpu_attn_set_seqlens` turns the fused path off, as before).
