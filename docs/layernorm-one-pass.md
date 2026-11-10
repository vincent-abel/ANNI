# LayerNorm in one pass per row

## Abstract

LayerNorm runs twice per block, forward and backward. The GPU kernels now give each row to one warp (32 threads), keep the row in registers, and do everything on it in that one visit: the forward also writes the BF16 copy the next matrix product reads, and the backward also computes the gamma and beta gradients and the BF16 copy of its output. The normalized values (`x_hat`) are no longer stored at all.

## What problem it solves (plain language)

The math of LayerNorm is cheap. What cost time was the passes around it: one kernel to normalize, another pass to convert the result to BF16, two more launches to sum the gamma and beta gradients over the rows, and a stored `x_hat` per LayerNorm that the backward read back. At these sizes each pass over a `[B*T][D]` buffer is 16 MB of memory traffic.

## How it works for a human (intuition, analogy)

A teacher marking a pile of papers picks one up once: reads it, writes the mark, adds it to the class total, puts it down. The old way went through the pile once to read, once more to mark, and twice more for the totals.

## Formal view (definitions, formulas, shapes, algorithms)

The formulas are those of [Layer normalization](layer-normalization.md). Per row, with `d` columns split as 4 values (`float4`) per thread and load:

**Forward.** Load the row. Sum with warp shuffles for the mean, again for the variance. Write `y = gamma * x_hat + beta` in FP32 and in BF16, and keep `mean` and `1/std` (two numbers per row).

**Backward.** Load `dy` and the row's input `x`. Recompute `x_hat = (x - mean) * inv_std`. Then in the same visit:

- `dx` from the usual three sums, written in FP32 and as a BF16 [gradient twin](bf16-twins.md);
- `dgamma += dy * x_hat` and `dbeta += dy`, kept per column in registers while the warp walks its rows, and written out once at the end.

## Complexity / numerical pitfalls / design choices

- **Recomputing must match the forward bit for bit.** A compiler may merge a multiply and an add into one fused instruction, which rounds differently. The kernel uses `__fmul_rn(__fsub_rn(x, mean), inv_std)` so the backward's `x_hat` is exactly the forward's.
- **What dropping `x_hat` saves:** 384 MB of VRAM at the 185M training shape (24 LayerNorms x 16 MB) and as many stores per microbatch ([VRAM diet](vram-diet.md)).
- **Order of the column sums.** Each warp's partial goes to its own slot and a small kernel adds the slots in a fixed order ([Deterministic gradients](deterministic-gradients.md)). The backward handles 8 rows per warp by default (`ANN_LN_RPW`) to keep the number of partials low.
- **Limits.** `D` must be a multiple of 4, at most 2048, and the buffers 16-byte aligned; otherwise the older kernels run.
- **Honest result.** On the full step the warp-per-row kernels alone were a wash in time. They are kept for the two launches fewer per LayerNorm backward. The BF16 output and the dropped `x_hat` are where the gain is.
- Switches: `ANN_LN_FAST=0` (older kernels), `ANN_LN_BF16=0` (no BF16 output), `ANN_LN_RECOMPUTE=0` (store `x_hat` again).

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `ln_fwd_warp_kernel`, `ln_bwd_warp_kernel`, behind `gpu_layernorm_forward_ex`, `gpu_layernorm_backward_x` and `gpu_layernorm_xhat` (rebuilds `x_hat` for a caller that still wants it).
- [`lang/layernorm.c`](../lang/layernorm.c): the cache records the input (`x_src`) instead of `x_hat`.
- [`tests/bench_ln.cu`](../tests/bench_ln.cu): times old against new at the training shape and checks the recomputing backward bit for bit.
- Candidates 12, 15 and 20 in [BF16 outputs](bf16-direct-outputs.md).
