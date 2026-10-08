# SIMD

## Abstract

**SIMD** (Single Instruction Multiple Data) applies one arithmetic op to several lanes of a vector register. This repo prepares for that with **contiguous `restrict` kernels** (`ann_dot`, `ann_axpy`, `ann_scale`) and default **`-O2`** so GCC can auto-vectorize before explicit AVX intrinsics land.

## What problem it solves (plain language)

Scalar loops touch one double at a time. Vector units can multiply several doubles per cycle when data is contiguous and the compiler can prove there is no aliasing.

## How it works for a human (intuition, analogy)

Instead of filling cups one by one, you use a **ladle that pours four cups at once**. `restrict` tells the compiler the ladle will not spill into another bowl mid-pour.

## Formal view (definitions, formulas, shapes, algorithms)

**Dot product.**

\[
\mathrm{ann\_dot}(a,b,n)=\sum_{i=0}^{n-1} a_i b_i
\]

**AXPY.**

\[
y_i \leftarrow y_i + \alpha x_i
\]

**Build.**

```make
OPT ?= -O2
CFLAGS=... $(OPT) ...
# OPT= make   # debug without optimization if needed
```

**Alias contract.** `ANN_RESTRICT` maps to `__restrict__` on GCC so `a` and `b` are assumed not to overlap.

## Complexity / numerical pitfalls / design choices

**No intrinsics yet.** This commit stays portable C; [AVX2 / AVX-512](avx2-avx-512.md) adds explicit kernels.

**`-ansi` + `-std=c99`.** Language is C99; `restrict` via the GNU attribute macro keeps pedantic builds happy.

**Correctness.** Results match scalar loops bit-for-bit at `-O0`; at `-O2` floating reductions may reorder slightly within ulp — tests use loose tolerances where needed.

## How it shows up in this repo

- [`num/simd.h`](../num/simd.h) / [`num/simd.c`](../num/simd.c)
- [`num/vec.c`](../num/vec.c): `vec_dot` / `vec_axpy` / `vec_scale` call `ann_*`
- [`Makefile`](../Makefile): `OPT ?= -O2`

## Worked example

`ann_dot` on `[1,2,3]` and `[4,5,6]` yields `32`. Building with `OPT=-O2` keeps `make test` green.

## What next

[Multithreading](multithreading.md) parallelizes independent GEMM tiles and attention heads with OpenMP.
