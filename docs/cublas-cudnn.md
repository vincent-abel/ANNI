# Optional cuBLAS / cuDNN

## Abstract

With `make cuda` (default `ANN_CUBLAS=1`), [`gpu_mat_mul`](../num/gpu_cuda.cu) tries **`cublasDgemm`** on row-major doubles, then falls back to the teaching `gemm_kernel`. Backward GEMMs reuse the same forward path. cuDNN is not linked yet; LayerNorm/softmax stay on teaching kernels or the host.

## What problem it solves (plain language)

Naive CUDA GEMM is correct and readable but slow. cuBLAS is the production double-precision multiply vendors tune; swapping it in speeds forward and backward without changing the LM math.

## How it works for a human (intuition, analogy)

Keep the student notebook (teaching kernel) on the shelf; when the library is linked, call the expert chef (cuBLAS) for the big stir-fries and only cook yourself if the chef is out.

## Formal view (definitions, formulas, shapes, algorithms)

Row-major \(C=AB\) via column-major cuBLAS:

\[
C_{\mathrm{rm}} = AB \;\equiv\;
\texttt{cublasDgemm}(N,M,K;\; B,N;\; A,K;\; C,N)
\]

(with \(\alpha=1,\beta=0\)).

## Complexity / numerical pitfalls / design choices

**Build.** `ANN_CUBLAS=1` adds `-DANN_CUBLAS` and `-lcublas`. `ANN_CUBLAS=0` keeps teaching kernels only.

**cuDNN.** Softmax / LN epilogues could use cuDNN later; this checklist item focuses on the GEMM hot path that dominates toy LM FLOPs.

**Backend name.** `gpu_backend_name()` reports `cuda+cublas` when the flag is on.

## How it shows up in this repo

- [`Makefile`](../Makefile): `cuda` target, `ANN_CUBLAS`, `-lcublas`
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `cublas_gemm_rm`, preferred path inside `gpu_mat_mul`

## Worked example

```bash
make cuda              # cuBLAS GEMM
make cuda ANN_CUBLAS=0 # teaching gemm_kernel only
make test              # still CPU stub, no nvcc
```

## What next

[Host↔device sync at batch boundaries](gpu-batch-sync.md).
