# Device-fused CE (S1)

## Abstract

After the LM-head softmax, subtract the one-hot target, accumulate `db_head`, and compute mean CE **on the GPU** so training does not D2H the full `rows × V` logits matrix every microbatch.

## How it shows up in this repo

- `gpu_ce_delta_from_softmax` in [`num/gpu.h`](../num/gpu.h) / [`num/gpu_cuda.cu`](../num/gpu_cuda.cu)
- Wired from `ce_rows_backward` in [`lang/lm.c`](../lang/lm.c); host loop remains the fallback
