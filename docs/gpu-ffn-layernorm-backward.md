# FFN + LayerNorm backward kernels

## Abstract

FFN backward uses **GEMM** (`dout W_2^\top`, `dh W_1^\top`) plus a host ReLU mask so CUDA mat_mul applies. LayerNorm backward tries [`gpu_layernorm_backward`](../num/gpu.h) (CUDA row kernel with atomics on \(d\gamma,d\beta\)) and falls back to the CPU loop.

## What problem it solves (plain language)

FFN and LN sit on every block’s backward path. Expressing FFN as matrix multiplies and offering a LN CUDA kernel keeps more of that path on the GPU.

## Formal view

\[
dh = \mathrm{ReLUMask}(dO W_2^\top),\quad
dW_2 \mathrel{+}= H^\top dO,\quad
dx = dh W_1^\top
\]

## How it shows up in this repo

- [`lang/ffn.c`](../lang/ffn.c), [`lang/layernorm.c`](../lang/layernorm.c)
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `ln_bwd_kernel`

## What next

[LM head + CE](gpu-head-cross-entropy.md).
