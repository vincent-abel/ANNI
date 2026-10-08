# GPU support

## Abstract

Optional **CUDA** `mat_mul` lives behind [`num/gpu.h`](../num/gpu.h). The default build links [`num/gpu_stub.c`](../num/gpu_stub.c) (`gpu_available() == 0`). `make cuda` compiles [`num/gpu_cuda.cu`](../num/gpu_cuda.cu) with `nvcc` when present. CPU `mat_mul` remains the path for `make` / `make test` so CI never needs a GPU.

## What problem it solves (plain language)

Big matrix multiplies dominate Transformer time. A GPU can run thousands of multiplies in parallel when you opt into a CUDA build — without forcing every developer to install CUDA.

## How it works for a human (intuition, analogy)

The kitchen (CPU) always works. If you have an industrial oven (GPU) and the right plug (`nvcc`), `make cuda` uses it for the heavy baking (`mat_mul`). Otherwise dinner still cooks on the stove.

## Formal view (definitions, formulas, shapes, algorithms)

```text
mat_mul(C, A, B):
  if gpu_available() and gpu_mat_mul(C,A,B) == 0:
    return
  else:
    blocked CPU GEMM (OpenMP optional)
```

**CUDA kernel (naive teaching GEMM).** One thread per \(C_{ij}\):

\[
C_{ij} = \sum_k A_{ik} B_{kj}
\]

**Build.**

```bash
make              # CPU stub
make test         # never requires nvcc
make cuda         # fails clearly if nvcc missing
```

## Complexity / numerical pitfalls / design choices

**Dependency-light default.** No CUDA headers in the normal compile.

**Naive kernel.** Not cuBLAS — educational and sufficient for toy shapes; production would call cuBLAS.

**Fallback.** Any CUDA error returns `-1` so CPU GEMM runs.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h)
- [`num/gpu_stub.c`](../num/gpu_stub.c) / [`num/gpu_cuda.cu`](../num/gpu_cuda.cu)
- [`num/mat.c`](../num/mat.c): tries GPU before CPU
- [`Makefile`](../Makefile): `cuda` target

## Worked example

`gpu_backend_name()` is `"cpu"` after `make`. After a successful `make cuda` on a machine with a device, it reports `"cuda"` and `mat_mul` may run on device.

## What next

Performance checklist complete. Further work is scale (vocab, data, params) rather than new roadmap boxes.
