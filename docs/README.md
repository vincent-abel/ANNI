# Numerical Engine docs

Short guides for each Numerical Engine piece, written for humans first and math second.

### [Vectors](vectors.md)
A vector is a contiguous list of numbers the engine can scale, add, and combine. The core score between two lists is the dot product. This repo stores them in `vec` with helpers like `vec_dot` and `vec_axpy`.

### [Matrices](matrices.md)
A matrix is a row-major grid of weights that transforms vectors in one shot. Matrix–vector multiply mixes every input into each output; matrix–matrix multiply chains those transforms. `mat_vec_mul` and `mat_mul` implement those products here.

### [Tensors](tensors.md)
A tensor is contiguous storage plus a multi-axis shape for batches and higher ranks. Flat indexing maps coordinates into one array; rank-1/2 views expose the same memory as `vec` or `mat` without copying.

### [Optimized matrix multiplication](optimized-matrix-multiplication.md)
Blocked multiply does the same \(C=AB\) math as the naive loops but in cache-friendly tiles. `mat_mul` uses 32×32 blocks while `mat_mul_naive` stays as a correctness reference. Only the visit order changes, not the result.
