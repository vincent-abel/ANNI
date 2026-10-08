# Tensors

## What problem it solves

Vectors are 1D and matrices are 2D, but models often need batches, sequences, or channels — **more axes**. A tensor is a single contiguous block of numbers plus a **shape** that tells you how to read those axes, so the same storage can mean “a vector,” “a matrix,” or “a stack of matrices.”

## How it works (layman)

Imagine a warehouse shelf of boxes (the `data` array). The **shape** is the labeling system: “3 shelves × 4 rows × 5 columns.” You still walk one long line of boxes in memory; the shape only decides how multi-index addresses map to that line. A **view** reuses the same boxes without copying: a 1D tensor can look like a `vec`, a 2D tensor like a `mat`.

## The math

Shape \((d_0, d_1, \ldots, d_{n-1})\) has size \(N = \prod_i d_i\).

With row-major (last index fastest) flat index:

\[
\text{flat} = i_{n-1} + d_{n-1}(i_{n-2} + d_{n-2}(\cdots))
\]

Rank-1 tensors behave as vectors in \(\mathbb{R}^{d_0}\); rank-2 as matrices in \(\mathbb{R}^{d_0 \times d_1}\).

## How it shows up in this repo

- [`tensor.h`](../tensor.h) / [`tensor.c`](../tensor.c) define `tensor` as `{ data, shape, ndim, size }`.
- `tensor_create` / `tensor_free` / `tensor_fill` manage storage; `tensor_at` returns a pointer to one element.
- `tensor_as_vec` / `tensor_as_mat` fill a view struct that points at the same `data` (no copy) when `ndim` is 1 or 2.

## Tiny example

Shape `(2, 3)` is six numbers. Index `(1, 2)` sits at flat offset `1*3+2 = 5`. Viewing it as a matrix lets `mat_get` read the same memory.
