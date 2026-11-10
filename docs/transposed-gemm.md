# Transposed GEMMs without transpose copies (H2)

## Abstract

The backward pass needs products with a transposed matrix: `dX = dY W^T` and `dW = X^T dY`. cuBLAS can read an operand transposed in place, so no transposed copy of a weight or an activation is ever built, on the host or on the device.

## What problem it solves (plain language)

The first GPU backward transposed `W` on the host, uploaded the copy, and multiplied. For a `1024 x 4096` weight that is a 16 MB shuffle and an upload for every layer of every microbatch, and it forced the weights back to the host.

## How it works for a human (intuition, analogy)

A spreadsheet read by rows or by columns is the same sheet. Nobody retypes it to read it the other way; you just turn your head.

## Formal view (definitions, formulas, shapes, algorithms)

The repo stores matrices by rows; cuBLAS reads memory by columns. A row-major `M x K` buffer, read by columns, is the `K x M` matrix `A^T`. Transposing a product swaps its operands, which gives three recipes:

| Wanted (row-major) | Identity | cuBLAS call (first, second operand) |
|---|---|---|
| `C = A B` | `C^T = B^T A^T` | buffer of `B`, plain; buffer of `A`, plain |
| `C = A B^T` (data gradient) | `C^T = B A^T` | buffer of `B` with `OP_T`; buffer of `A`, plain |
| `C = A^T B` (weight gradient) | `C^T = B^T A` | buffer of `B`, plain; buffer of `A` with `OP_T` |

So the forward needs no transpose flag at all, and each backward product needs exactly one.

## Complexity / numerical pitfalls / design choices

- **Accumulate in the same call.** `dW += X^T dY` is the third recipe with `beta = 1`: gradient accumulation costs nothing extra.
- **Leading dimensions.** The "rows" cuBLAS sees are the repo's columns, so the leading dimension passed is the row-major column count. A strided view (one head inside a `[B*T][3D]` row) only changes that number.
- **FP8 is the exception.** cuBLASLt's FP8 kernels accept one layout only, so that path keeps a transposed quantized copy of the weights ([FP8 training path](fp8-train-path.md)).

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `cublas_gemm_rm_bt` (`C = A B^T`) and `cublas_gemm_rm_at` (`C (+)= A^T B`), behind `gpu_mat_mul_bt`, `gpu_mat_mul_bt_acc`, `gpu_mat_mul_at` and `gpu_mat_outer_add`.
- [`num/mat.h`](../num/mat.h): `mat_mul_bt` is the host twin with the same meaning.
- Without cuBLAS, hand-written kernels (`gemm_at_kernel` and friends) do the same indexing.

## Worked example

`A` is `2 x 3` row-major: memory `a b c d e f`. Read by columns with 3 rows, the same six numbers are the `3 x 2` matrix with columns `(a b c)` and `(d e f)`: that is `A^T`. Nothing moved.
