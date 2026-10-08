# Multithreading

## Abstract

**OpenMP** parallelizes blocked `mat_mul` row-tiles and multi-head attention heads. Each head owns private Q/K/O scratch and a softmax temp slice so threads do not race. Training still runs one logical batch at a time (parallel GEMMs inside the forward/backward); generation remains correct with `OMP_NUM_THREADS=1` or many.

## What problem it solves (plain language)

Attention heads and matrix tiles are independent work. Spreading them across CPU cores finishes a forward pass sooner without changing the math.

## How it works for a human (intuition, analogy)

Eight cashiers (heads) ring up separate carts at once, then one clerk merges the receipts (`W_O`). Shared shopping bags would clash — so each cashier gets their own.

## Formal view (definitions, formulas, shapes, algorithms)

**GEMM.**

```c
#pragma omp parallel for private(...)
for (i0 = 0; i0 < M; i0 += BLOCK)
  /* blocked C[i0:i0+B,:] += A * B */
```

**MHA.**

```text
for each head h in parallel:
  Qh,Kh,Vh <- split(Q,K,V)
  Oh <- softmax(scale Qh Kh^T) Vh   /* private softmax_tmp[h] */
  write Oh into concat
concat * Wo
```

**Build.**

```make
OPENMP ?= -fopenmp
# OPENMP= make   # force single-threaded build
```

**Env.** `OMP_NUM_THREADS` caps worker count.

## Complexity / numerical pitfalls / design choices

**No parallel batches yet.** Sharing `dw_*` across batch items would race; keep batch serial and parallelize math.

**Reduction.** Head writes land in disjoint concat column ranges — safe without atomics.

**Generation.** Safe under OpenMP; optional single-thread via `OMP_NUM_THREADS=1`.

## How it shows up in this repo

- [`num/mat.c`](../num/mat.c): OpenMP over `i0` tiles in `mat_mul`
- [`lang/attn.c`](../lang/attn.c): per-head scratch arrays; `#pragma omp parallel for` over heads
- [`Makefile`](../Makefile): `-fopenmp` in `CFLAGS` / `LDFLAGS`

## Worked example

`OMP_NUM_THREADS=4 ./ann lm bench --tokens 32 --iters 10` should be no slower than one thread on this toy size and often faster as `T` and `D` grow.

## What next

[AVX2 / AVX-512](avx2-avx-512.md) replaces the scalar inner products with explicit vector intrinsics.
