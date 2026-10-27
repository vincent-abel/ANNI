# Mixed precision train (R10 / S5a / C13)

## Abstract

TF32 is enabled for FP32 cuBLAS GEMMs when the toolkit supports it. **`--bf16`** runs cuBLAS `GemmEx` with BF16 inputs and FP32 accumulate (CUDA ≥ 11); BF16 needs CUDA >= 11 (older toolkits are no longer supported). Weights and Adam moments stay **FP32 masters**.

## How it shows up in this repo

- `cublasSetMathMode(...TF32...)` in `cublas_ensure`
- `gpu_set_compute_prec` / `GPU_PREC_BF16` in [`num/gpu.h`](../num/gpu.h)
- CLI: `./anni lm train --bf16 …` (`--fp16` was removed)
