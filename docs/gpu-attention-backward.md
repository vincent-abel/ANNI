# Attention backward on GPU

## Abstract

[`attn_mha_backward`](../lang/attn.c) now expresses weight and activation grads as **GEMM / outer_add / mul_bt / mul_at**, so the CUDA path from [cuda-gemm-backward.md](cuda-gemm-backward.md) runs automatically when `make cuda` is linked. Softmax Jacobian rows stay on the host (small `T`).

## What problem it solves (plain language)

Attention backward was nested `mat_get` loops. Rewriting it as matrix multiplies lets the same GPU GEMMs that speed forward also speed training.

## How it works for a human (intuition, analogy)

Instead of hand-multiplying every score, you ship whole scoreboards to the GPU stack that already knows how to multiply boards.

## Formal view (definitions, formulas, shapes, algorithms)

\[
dW_O \mathrel{+}= C^\top dO,\quad
dC = dO\,W_O^\top
\]

Per head:

\[
dV = A^\top dO_h,\quad
dS_{\text{pre}}=dO_h V^\top,\quad
dS=\mathrm{softmax\_bwd}(A,dS_{\text{pre}})\cdot\mathrm{scale}
\]

\[
dQ=dS\,K,\quad dK=dS^\top Q
\]

## Complexity / numerical pitfalls / design choices

**Softmax bwd** remains CPU per row — cheap vs \(T\times T\times d\).

**Causal mask** zeros upper scores after softmax bwd.

## How it shows up in this repo

- [`lang/attn.c`](../lang/attn.c): GEMM-shaped MHA backward

## Worked example

`make test` still passes finite-difference and train corpus checks. With CUDA, `nvidia-smi` may show brief util during train steps that run attention backward.

## What next

[FFN + LayerNorm backward](gpu-ffn-layernorm-backward.md).
