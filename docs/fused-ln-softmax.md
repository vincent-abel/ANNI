# Fused LN / softmax (E7)

## Abstract

Hand CUDA LN and row-softmax kernels in `gpu_cuda.cu` already fuse the hot elementwise paths after R1 residency; no cuDNN dependency.

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `ln_*_kernel`, `softmax_rows_kernel`
