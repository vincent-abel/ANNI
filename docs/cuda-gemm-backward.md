# CUDA GEMM backward

## Abstract

For \(C = AB\), gradients are \(dA = dC\,B^\top\) and \(dB = A^\top\,dC\). [`gpu_mat_mul_bt`](../num/gpu.h) and [`gpu_mat_mul_at`](../num/gpu.h) compute them on device when CUDA is linked (the combined `gpu_mat_mul_backward` was removed). [`gpu_mat_mul_bt`](../num/gpu.h) and [`gpu_mat_outer_add`](../num/gpu.h) cover \(AB^\top\) and weight outer products used by LM backward. CPU stubs return `-1` so [`gradutil`](../lang/gradutil.c) / [`mat_mul_bt`](../num/mat.c) fall back to host loops.

## What problem it solves (plain language)

Training needs the same matrix multiplies run “backward.” Doing those on the GPU matches where forward GEMM already runs and cuts host time on the big weight updates.

## How it works for a human (intuition, analogy)

If forward is “mix ingredients with a recipe,” backward asks “how should each ingredient change?” using the same mixing bowls on the GPU stove.

## Formal view (definitions, formulas, shapes, algorithms)

\[
C=AB \implies dA=dC B^\top,\quad dB=A^\top dC
\]

\[
C=AB^\top \quad\text{(scores)},\quad
dW \mathrel{+}= A^\top B \quad\text{(outer_add)}
\]

## Complexity / numerical pitfalls / design choices

**Teaching kernels** (not cuBLAS yet). Transposes for `dA`/`dB` may stage on host then call `gpu_mat_mul`.

**Fallback.** Any CUDA failure leaves CPU path unchanged.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h), [`gpu_stub.c`](../num/gpu_stub.c), [`gpu_cuda.cu`](../num/gpu_cuda.cu)
- [`lang/gradutil.c`](../lang/gradutil.c): `mat_outer_add` tries GPU first
- [`num/mat.c`](../num/mat.c): `mat_mul_bt` tries GPU first

## Worked example

On CPU builds, `gpu_mat_mul_bt` returns `-1` and outer_add uses the host loop. With `make cuda`, the same call runs device kernels and matches CPU within float noise on tiny mats.

## What next

[Attention backward on GPU](gpu-attention-backward.md).
