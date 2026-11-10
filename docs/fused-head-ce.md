# Head and loss in one kernel

## Abstract

The last step of the forward turns `rows x vocab` logits into a loss and a gradient. One kernel now reads each row of logits once and does all of it: adds the head bias, takes the softmax, picks the loss of the target, sums the bias gradient, and writes the gradient `p - onehot` as BF16 on top of the logits it just read. No second buffer of that size exists.

## What problem it solves (plain language)

With a 16k vocabulary and 4096 rows, the logits are 268 MB; with 100k tokens they are far more. The old path walked that block again and again: bias add, softmax, subtract the target, column sums, then two conversions to BF16 for the two matrix products of the head's backward. About 1.2 GB of memory traffic per microbatch, for math that fits in one visit.

## How it works for a human (intuition, analogy)

Grading a multiple-choice sheet and writing the corrections in the margin of the same sheet, instead of copying every answer onto a new page first.

## Formal view (definitions, formulas, shapes, algorithms)

For each row `z` (FP32, length `V`) with target `y`:

```text
z      = z + b                      head bias, added at the load
p      = softmax(z)
loss  += -log p[y]
delta  = p - onehot(y)              the gradient of the loss w.r.t. z
db    += delta                      column sums over the rows
store delta as BF16 over the first half of this row's own storage
```

A BF16 value is half an FP32, so row `r` of the gradient starts where row `r` of the logits started and the rows are `2 V` BF16 elements apart. The two head GEMMs (`dW += h^T delta` and `dh = delta W^T`) read it with that row stride.

## Complexity / numerical pitfalls / design choices

- **The logits are destroyed.** After the kernel only the BF16 gradient is there, so both head GEMMs must read it. The caller checks afterwards that they did (`gpu_ce_bf16_delta_done`).
- **In place is safe** because a thread block holds its row in registers between the single load and the store.
- **Same numbers.** Probabilities, loss and gradient are bitwise those of the separate kernels it replaces.
- **Measured** at the 185M shape: the loss phase went from 7.2 to 5.7 ms per microbatch (candidate 16 in [BF16 outputs](bf16-direct-outputs.md)).
- **A read-only sibling for probes.** `gpu_ce_nll_rows` computes `logsumexp(z) - z[y]` per row and writes nothing else, so validation can score many windows from one head GEMM ([Batched validation probe](batched-validation-probe.md)).
- Switches: `ANN_CE_BF16=0` (separate FP32 steps), `ANN_CE_BIAS=0` (bias added in its own pass).

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `ce_softmax_delta_bf16_kernel` behind `gpu_ce_softmax_delta_bf16`; `gpu_ce_bias_fused`; `gpu_ce_nll_rows`.
- [`lang/lm.c`](../lang/lm.c): `ce_rows_backward` tries the fused kernel first and falls back to the FP32 steps.
- Earlier steps of the same road: [LM head + cross-entropy backward](gpu-head-cross-entropy.md), [Device-fused CE](device-fused-ce.md).

## What next

- With the 100k vocabulary the head is the largest single product of the step; a sampled or factored softmax would be the next lever, at a cost in exactness.
