# Mixed precision train (R10)

## Abstract

Enable TF32 for cuBLAS FP32 GEMMs when the toolkit supports it. BF16 GEMMs with FP32 master weights remain the next increment on this path.

## How it shows up in this repo

- `cublasSetMathMode(...TF32...)` in `cublas_ensure`
