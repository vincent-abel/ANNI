# Mixed precision train (R10 / S5a / C13)

## Abstract

TF32 is enabled for FP32 cuBLAS GEMMs when the toolkit supports it. **`--bf16`** runs cuBLAS `GemmEx` with BF16 inputs and FP32 accumulate (CUDA ≥ 11); on older toolkits BF16 falls back to the FP16 GemmEx path. Weights and Adam moments stay **FP32 masters**.

## How it shows up in this repo

- `cublasSetMathMode(...TF32...)` in `cublas_ensure`
- `gpu_set_compute_prec` / `GPU_PREC_BF16` in [`num/gpu.h`](../num/gpu.h)
- CLI: `./anni lm train --bf16 …` (mutually exclusive with `--fp16`)
- See also [fp16-loss-scaling.md](fp16-loss-scaling.md) for `--fp16`
