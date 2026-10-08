# Matrices

## What problem it solves

A layer maps many inputs to many outputs. That map is naturally a **grid of weights** — a matrix — times a vector of activations. Matrices let you write “transform this whole list of numbers” as one operation instead of a nest of per-neuron loops.

## How it works (layman)

A matrix is a table of numbers with rows and columns. Multiplying matrix \(A\) by vector \(\mathbf{x}\) means: each output entry is a weighted mix of all inputs (one row of \(A\) dotted with \(\mathbf{x}\)). Multiplying two matrices chains those transforms: the columns of the first become inputs to the second.

This repo stores matrices **row-major**: the whole first row sits next to itself in memory, then the next row, and so on. That matches how we walk “for each row, mix all columns.”

## The math

For \(A \in \mathbb{R}^{m \times n}\), \(\mathbf{x} \in \mathbb{R}^n\), \(\mathbf{y} \in \mathbb{R}^m\):

\[
y_i = \sum_{j=1}^{n} A_{ij}\, x_j
\]

For \(A \in \mathbb{R}^{m \times k}\), \(B \in \mathbb{R}^{k \times n}\), \(C = AB\):

\[
C_{ij} = \sum_{p=1}^{k} A_{ip}\, B_{pj}
\]

Index in memory: `data[i * cols + j]` is \(A_{ij}\).

## How it shows up in this repo

- [`mat.h`](../num/mat.h) / [`mat.c`](../num/mat.c) define `mat` as `{ data, rows, cols }`.
- `mat_get` / `mat_set` / `mat_fill` access the grid; `mat_vec_mul` computes \(\mathbf{y} = A\mathbf{x}\).
- `mat_mul` / `mat_mul_naive` compute \(C = AB\) (same algorithm for now; a later commit blocks/tiles it).

## Tiny example

\(A = \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix}\), \(\mathbf{x}=\begin{bmatrix}1\\0\end{bmatrix}\) gives \(\mathbf{y}=\begin{bmatrix}1\\3\end{bmatrix}\).  
\(A\) times \(B=\begin{bmatrix}0 & 1\\1 & 0\end{bmatrix}\) swaps the columns of \(A\).
