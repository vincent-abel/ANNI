# Optimized matrix multiplication

## What problem it solves

Naive triple loops for \(C = AB\) are correct but slow on large matrices: they jump around memory and thrash the CPU cache. **Blocking** (tiling) multiplies small tiles that fit in cache so each loaded number gets reused many times before being thrown away.

## How it works (layman)

Imagine assembling a big mosaic. Instead of walking the whole wall for every tiny tile color, you finish one square patch at a time with the paints already on your palette. Blocking does the same for matrix multiply: it cuts \(A\), \(B\), and \(C\) into `MAT_BLOCK × MAT_BLOCK` patches (here 32) and accumulates products patch-by-patch. The math is identical; only the visit order changes.

## The math

Still \(C_{ij} = \sum_k A_{ik} B_{kj}\). Blocking partitions indices into ranges \([i_0,i_0+B)\), \([k_0,k_0+B)\), \([j_0,j_0+B)\) and accumulates:

\[
C_{ij} \mathrel{+}= A_{ik} B_{kj}
\quad\text{for } (i,k,j) \text{ inside those blocks.}
\]

`mat_mul_naive` keeps the classic \(i,j,k\) order as a reference; `mat_mul` uses the blocked order.

## How it shows up in this repo

- [`mat.c`](../mat.c): `mat_mul` zeros \(C\) then runs three outer block loops (`MAT_BLOCK` 32) and inner micro-kernels; `mat_mul_naive` remains for checking.
- Tests compare blocked vs naive on random small matrices.

## Tiny example

For 64×64 matrices, blocking walks 2×2 tiles of size 32 along each axis — same final \(C\), better cache reuse on bigger sizes.
