# Adam on the device, with the BF16 weights written in the same pass

## Abstract

The optimizer step runs as one GPU kernel over one flat buffer that holds every parameter. That kernel updates the FP32 weight and the two moments, zeroes the gradient, and writes the BF16 copy of the new weight that the next forward will read. Nothing crosses to the host, and no later pass converts weights.

## What problem it solves (plain language)

Adam touches every parameter and three more numbers beside it, at every step. For 555M parameters that is about 19 GB of memory traffic per step, so the step is bound by how fast memory moves, not by the arithmetic. Any extra pass over the weights (copying them into place, converting them to BF16, zeroing gradients) is another slice of the same cost.

## How it works for a human (intuition, analogy)

One trip down the warehouse aisle with the full checklist: count the shelf, restock it, update the ledger, print the new label. Not four trips with one job each.

## Formal view (definitions, formulas, shapes, algorithms)

All parameters sit end to end in one device buffer `param`, with `grad`, `m`, `v` of the same length and `param_bf16` at half the size. Each weight matrix's device pointer is an offset into `param`; its gradient an offset into `grad`. For each element:

```text
g   = grad * gscale                      1 / (batch x accumulation), and the clip factor
m   = b1 * m + (1 - b1) * g
v   = b2 * v + (1 - b2) * g * g
p   = param - lr * (m / bc1) / (sqrt(v / bc2) + eps)
p   = p - lr * wd * p                    decoupled weight decay
param = p ;  param_bf16 = bf16(p) ;  grad = 0
```

`bc1 = 1 - b1^t` and `bc2 = 1 - b2^t` are the usual bias corrections ([Adam on the LM weights](adam-lm-weights.md)).

Before the update, the clip norm `sqrt(sum g^2)` is summed over the same flat gradient by a second kernel.

## Complexity / numerical pitfalls / design choices

- **Traffic:** 34 bytes per parameter and step (four FP32 reads, four FP32 writes, one BF16 write). The kernel handles four parameters per thread with `float4` loads and stores so the memory transactions are wide.
- **Bit-identical to the scalar kernel.** The compiler fuses some multiply-adds; the vector version spells out the same fusions so no moment drifts by a last bit. `ANN_ADAM_V4=0` runs the scalar kernel.
- **The norm.** The first version did one atomic add per 256 values onto a single address: about 720k atomics and 5.3 ms per step at 185M parameters. The current one has each block sum a share in double precision and add once, in block order ([Deterministic gradients](deterministic-gradients.md)).
- **A non-finite norm skips the step** and zeroes the gradient, so one bad batch cannot poison `m` and `v`.
- **Weight twins are never stale.** The BF16 buffer is written by the only code that changes weights. Loading a checkpoint reseeds it (`gpu_adam_seed_bf16_flat`). Derived copies (packed `[Wq|Wk|Wv]`, device biases) are rebuilt when the weight epoch counter moves.
- **Alignment.** A matrix inside the flat starts wherever the previous one ended, which is why the GEMM algorithm cache is keyed on pointer alignment ([cuBLASLt algorithm choice](cublaslt-algo-choice.md)).
- `ANN_GNORM_EVERY=N` prints the gradient norm every `N` steps.

## How it shows up in this repo

- [`lang/lm_adam.c`](../lang/lm_adam.c): `adam_step_gpu`, the flat buffers and the aliasing of each weight and gradient into them.
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_adam_step_dev_bf16` (`adam_kernel_bf16_v4`), `gpu_grad_sumsq` (`sumsq_gs_kernel`), `gpu_adam_set_bf16_flat`, `half_wt_get`.
- Earlier steps: [Adam step on GPU](gpu-adam.md), [GPU-resident Adam](gpu-adam-residency.md). Items **A1** and **A3** of the [code audit](code-audit-dead-paths-and-compute.md).
