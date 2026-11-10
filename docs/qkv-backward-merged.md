# One backward for the Q, K and V projections

## Abstract

Attention projects its input three times (queries, keys, values). The forward already does that as one matrix product with the three weights side by side. The backward now does the same: one product for the gradient going down to the input and one batched product for the three weight gradients, instead of six.

## What problem it solves (plain language)

Six medium products cost more than two large ones: each launch has a fixed cost, and the input `x` and the three gradients were each read from memory several times. Two of the three data-gradient products also had to add into a buffer the first one had written.

## How it works for a human (intuition, analogy)

Three forms that ask for the same name and address. Fill them side by side in one sitting instead of three trips to the same desk.

## Formal view (definitions, formulas, shapes, algorithms)

Forward, with `x` of shape `[N][D]` (`N = B*T`):

```text
[Q | K | V] = x [Wq | Wk | Wv]            one GEMM, output [N][3D]
```

Backward, with the three gradients stored side by side as BF16 rows `[N][3D]`:

```text
dx           = [dQ | dK | dV] [Wq | Wk | Wv]^T     one GEMM, K = 3D
dWq, dWk, dWv += x^T dQ,  x^T dK,  x^T dV          one strided-batched GEMM, batch of 3
```

The first line is exact algebra: `dQ Wq^T + dK Wk^T + dV Wv^T` is the product of the concatenated matrices. The second uses the fact that the three right-hand operands are column blocks of one buffer, `D` columns apart, which is what a strided batch describes.

The attention backward kernel writes `dK` and `dV` straight into their column blocks. `dQ` is built with atomic adds in FP32, so it is converted into its block afterwards.

## Complexity / numerical pitfalls / design choices

- **Launches:** 6 GEMMs become 2 per block and microbatch, and `x` is read once for the weight gradients instead of three times.
- **Rounding.** Same BF16 operands; only the order of the FP32 additions inside the products changes.
- **Needs the gradient twins.** The merged rows are a [BF16 twin](bf16-twins.md) with a one-microbatch life. With `ANN_TWIN_EMIT=0` the six separate products run.
- **On failure** the merged rows were the only copy of `dK` and `dV`, so the call returns an error and the step is abandoned, not silently continued.
- Switches: `ANN_QKV_BWD=0` (separate GEMMs), `ANN_QKV_WBATCH=0` (merged data gradient, three separate weight gradients).

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_attn_qkv_bwd`; the packed BF16 `[Wq|Wk|Wv]` it shares with `gpu_attn_qkv_fused`.
- [`num/attn_fa2.cuh`](../num/attn_fa2.cuh): the backward kernel takes the row stride of its `dK` / `dV` outputs.
- [`lang/attn.c`](../lang/attn.c): `attn_mha_backward_packed` calls it after the fused backward and falls back to the six GEMMs when nothing is pending.
- Item **B3** of the [code audit](code-audit-dead-paths-and-compute.md), candidate 22 in [BF16 outputs](bf16-direct-outputs.md); the forward side is in [Fused QKV + CE hot path](fused-qkv-ce-40tf.md).
