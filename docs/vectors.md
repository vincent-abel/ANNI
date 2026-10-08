# Vectors

## What problem it solves

A neural network spends most of its time combining lists of numbers: inputs, weights, activations, gradients. A **vector** is the simplest building block for that — an ordered list of numbers you can add, scale, and combine in bulk instead of writing the same loop by hand every time.

## How it works (layman)

Think of a vector as a row of dials. Each dial holds one `double`. Operations move all dials together: scale every dial by the same factor, add two rows dial-by-dial, or score how aligned two rows are with a single number (the dot product). Keeping the numbers in one contiguous memory block makes those loops fast and simple.

## The math

For vectors \(\mathbf{a}, \mathbf{b} \in \mathbb{R}^n\):

- **Dot product:** \(\mathbf{a}\cdot\mathbf{b} = \sum_{i=1}^{n} a_i b_i\)
- **Scale:** \((s\mathbf{a})_i = s\,a_i\)
- **Add:** \((\mathbf{a}+\mathbf{b})_i = a_i + b_i\)
- **AXPY** (“a times x plus y”): \(\mathbf{y} \leftarrow \mathbf{y} + a\mathbf{x}\)

These are the same primitives used inside layer math and later matrix products.

## How it shows up in this repo

- [`vec.h`](../num/vec.h) / [`vec.c`](../num/vec.c) define `vec` as `{ double *data; int n; }`.
- `vec_create` / `vec_free` own the contiguous block; `vec_fill` / `vec_copy` set values.
- `vec_dot`, `vec_axpy`, `vec_scale`, `vec_add` implement the operations above.

## Tiny example

`[1, 2, 3] · [4, 5, 6] = 1·4 + 2·5 + 3·6 = 32`.  
`axpy` with `a=2`, `x=[1,1]`, `y=[3,0]` yields `y=[5,2]`.
