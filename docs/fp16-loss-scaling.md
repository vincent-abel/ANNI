# FP16 + dynamic loss scaling (S5b)

## Abstract

`--fp16` casts GEMM operands to FP16 for `cublasGemmEx` (FP32 accumulate). A **dynamic loss scale** starts at 1024: Adam uses `grad_scale / loss_scale`; on non-finite train NLL the step is skipped and the scale halves; after 2000 good steps the scale doubles (cap 65536).

## How it shows up in this repo

- `GPU_PREC_FP16` + `lm_train_set_compute_prec` / `lm_train_loss_scale`
- CLI: `./anni lm train --fp16 …` (not with `--bf16`)
