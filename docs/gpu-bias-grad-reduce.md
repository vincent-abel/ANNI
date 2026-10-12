# GPU bias grad column-sum (S3)

## Abstract

FFN `db1` / `db2` accumulate via a GPU column-sum over activation grads so backward does not `mat_ensure_host` those tensors solely for the bias reduce.

## How it shows up in this repo

- `gpu_mat_col_sum_add` in [`num/gpu.h`](../num/gpu.h) / [`num/gpu_cuda.cu`](../num/gpu_cuda.cu)
- [`lang/ffn.c`](../lang/ffn.c) `ffn_backward`; host loops remain the fallback
