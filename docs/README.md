# Numerical Engine docs

Short guides for each Numerical Engine piece, written for humans first and math second.

### [Vectors](vectors.md)
A vector is a contiguous list of numbers the engine can scale, add, and combine. The core score between two lists is the dot product. This repo stores them in `vec` with helpers like `vec_dot` and `vec_axpy`.

### [Matrices](matrices.md)
A matrix is a row-major grid of weights that transforms vectors in one shot. Matrix–vector multiply mixes every input into each output; matrix–matrix multiply chains those transforms. `mat_vec_mul` and `mat_mul` implement those products here.
