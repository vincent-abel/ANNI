# KV-cache generate on GPU

## Abstract

[`attn_decode_row`](../lang/kv.c) tries [`gpu_kv_attn_decode`](../num/gpu.h) before the host path. The CUDA path projects Q/K/V, appends into the K/V cache, scores each head, softaxes on the host for the short score row, gathers \(AV\), and applies \(W_o\). Token sampling stays on the host.

## What problem it solves (plain language)

Incremental generate spends most of its time in decode-step attention. Running that step on the GPU keeps K/V append and the dense maps off the CPU.

## How it works for a human (intuition, analogy)

The cache is a growing notebook of past keys and values. Each new token adds one line on the GPU and attends against all prior lines there; only the chosen next-id comes back to the host.

## Formal view (definitions, formulas, shapes, algorithms)

\[
q=xW_q,\; k=xW_k,\; v=xW_v,\quad
K\leftarrow\mathrm{append}(K,k),\; V\leftarrow\mathrm{append}(V,v)
\]

\[
\mathrm{scores}_h=\mathrm{softmax}\!\left(\frac{q_h K_h^\top}{\sqrt{d}}\right),\quad
o_h=\mathrm{scores}_h V_h,\quad
\mathrm{out}=o W_o
\]

## Complexity / numerical pitfalls / design choices

**Upload cost.** Each decode uploads weights and the full cache (toy \(T\le 128\)). Later batch-boundary work can pin them.

**Softmax.** Score length equals current cache length; host softmax keeps numerics identical to CPU for the tiny row.

**Fallback.** Stub returns `-1`; CPU `attn_decode_row` unchanged.

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gemv_kernel`, `kv_append_kernel`, `kv_scores_hd`, `kv_av_hd`, `gpu_kv_attn_decode`
- [`lang/kv.c`](../lang/kv.c): try GPU in `attn_decode_row`; `mat_try_pin` on K/V mats

## Worked example

`make test` (CPU stub) still runs `test_lm_generate_kv`. With `make cuda`, the same generate path hits device kernels when a GPU is present.

## What next

[Optional cuBLAS / cuDNN](cublas-cudnn.md).
