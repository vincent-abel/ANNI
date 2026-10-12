# GPU positional encoding (S2)

## Abstract

Packed forward adds sinusoidal positional encodings on device after embed gather, without pulling the full `(B·T) × D` hidden matrix to the host.

## How it shows up in this repo

- `gpu_posenc_add_packed` in [`num/gpu.h`](../num/gpu.h) / [`num/gpu_cuda.cu`](../num/gpu_cuda.cu)
- [`lang/lm.c`](../lang/lm.c) `lm_forward_cache_packed`; host `posenc_add_row` remains the fallback
