# Residual add inside the projection GEMM

## Abstract

Each block adds its input back twice: after attention and after the FFN. Both adds now happen inside the matrix product that comes just before them (`Wo` and `W2`), as one cuBLASLt call. Two kernel launches and two full passes over the activations are gone per block.

## What problem it solves (plain language)

`x + f(x)` was done as: compute `f(x)` into a buffer, then run a second small kernel that reads two buffers and writes a third. That second kernel does no real math, it only moves memory, and at these sizes moving memory is what the step is short of.

## How it works for a human (intuition, analogy)

A cashier who adds the tip while totalling the bill, instead of printing the bill and then a second receipt with the tip on it.

## Formal view (definitions, formulas, shapes, algorithms)

A GEMM is defined as `D = alpha * A B + beta * C`. Most calls use `beta = 0`. Here:

```text
x_mid = concat * Wo           + x_in      alpha = 1, beta = 1, C = x_in,  D = x_mid
x_out = hidden * W2 + b2      + x_mid     alpha = 1, beta = 1, C = x_mid, D = x_out
```

`C` and `D` are different buffers (cuBLASLt allows that), so the residual input is read, never overwritten. The FFN call keeps its bias epilogue: bias and residual are both folded in.

## Complexity / numerical pitfalls / design choices

- **Saved per block and microbatch:** two launches, and for each one a read of two `[B*T][D]` buffers and a write of a third.
- **The input must stay intact.** The backward needs `x_in` and `x_mid` as they were, hence the out-of-place `D`. The wrapper refuses `D == C`.
- **BF16 mode only.** In FP32, FP8, or when the call fails, the code falls back to the product followed by `resid_add`. Same result, one more launch.
- **Timing the candidates** of a `beta = 1` call must not add the product several times: see [cuBLASLt algorithm choice](cublaslt-algo-choice.md).

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_mat_mul_res(d, a, b, bias, res)`; `cublas_lt_gemm_half` takes an optional out-of-place `D`.
- [`lang/attn.c`](../lang/attn.c): `attn_mha_forward_packed_res` and `attn_project_out`.
- [`lang/ffn.c`](../lang/ffn.c): `ffn_forward_cache_res`.
- [`lang/block.c`](../lang/block.c): the packed forward writes `x_mid` and the block output straight from the two fused projections.

Older notes ([Fused epilogues](fused-epilogues.md), [Residual connections](residual-connections.md)) describe the separate add, which is now the fallback.

## What next

- The backward's residual gradients are already added in place (**B4** in the [code audit](code-audit-dead-paths-and-compute.md)).
