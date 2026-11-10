# cuBLASLt: choosing the GEMM algorithm, and remembering it

## Abstract

cuBLASLt offers several algorithms for one matrix product and its own ranking of them is often wrong on this card. The wrapper times the candidates once per kind of call, keeps the winner in a small cache, and reuses it for the rest of the run. The cache key had to grow twice: by the alignment of the pointers, then by the leading dimensions. Getting the key right was worth about 9% of training speed.

## What problem it solves (plain language)

Almost all of a training step is matrix products. A slow algorithm on one shape is a tax paid thousands of times per minute. And a cached choice that the library refuses at run time is worse: the call silently falls back to a slower path.

## How it works for a human (intuition, analogy)

Several roads lead to work and the map's advice is unreliable. Drive each one once with a stopwatch, then stop thinking about it. But the best road depends on the vehicle: a route timed with a car is useless the day you tow a trailer that does not fit under the bridge.

## Formal view (definitions, formulas, shapes, algorithms)

A cache entry is keyed on everything that changes which algorithms are legal or fast:

```text
M, N, K            shape
opa, opb           transposes
epilogue           none, BIAS, RELU_BIAS
out_bf16           BF16 or FP32 output
lda, ldb, ldc      leading dimensions (strided views differ from dense ones)
align              alignment of the four pointers A, B, C, D (log2, up to 256 bytes)
```

On a miss:

1. Ask the heuristic for up to 16 candidates, telling it the real alignment of each pointer.
2. Run each candidate that fits the 128 MB workspace once, timed with CUDA events.
3. For each candidate that splits `K` in several parts, also try its split-K 1 version (same tile, no reduction), if `cublasLtMatmulAlgoCheck` accepts it.
4. Keep the fastest, with its descriptor and layouts, so later calls create nothing.

There are 128 slots; a training step uses far fewer.

## Complexity / numerical pitfalls / design choices

- **Alignment.** The heuristic assumes 256-byte aligned operands. A BF16 weight copy that lives inside the optimizer's flat buffer starts at its parameter's offset, which can be 2-byte aligned. The algorithm chosen on the first call was then refused (`CUBLAS_STATUS_NOT_SUPPORTED`) from the second optimizer step on, and every such GEMM fell back. Keying by alignment fixed it: most of the +9.3% of candidate 23 in [BF16 outputs](bf16-direct-outputs.md).
- **Timing runs are real products.** With `beta = 0` repeating one is harmless. With `beta = 1` (`dW += X^T dY`, or a [fused residual](residual-fused-gemm.md)) each timed candidate would add the product again, so those are timed into a scratch output.
- **Never choose inside a CUDA graph capture.** Nothing can be timed there, and the choice would differ from the one made outside. The capture is refused instead ([CUDA graphs](cuda-graphs-train.md)).
- **Different algorithms round differently.** `ANN_LT_HEURISTIC=1` skips the timing and takes the first fitting candidate, so two runs under different load make the same choice. Tests that compare runs use it.
- `ANN_LT_SPLIT1=0` turns step 3 off. `ANN_LT_LOG=1` prints when a cached algorithm is refused or a split-K 1 variant wins.

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `cublas_lt_gemm_half` (BF16) and its FP8 sibling, the `lt_algo_entry` cache, `lt_ptr_align` and `lt_align_key`.
- History: defect list in the [GPU training audit](gpu-training-audit.md), item **B6** in the [code audit](code-audit-dead-paths-and-compute.md), candidates 22 to 24 in [BF16 outputs](bf16-direct-outputs.md).

## What next

- A miss costs a few timed products. A buffer whose alignment changes at every step would thrash the cache, so new device buffers should come from the aligned pools.
