# Transformer blocks

## Abstract

A **Transformer block** is one repeatable layer: **pre-norm multi-head self-attention** with a residual, then **pre-norm position-wise FFN (ReLU)** with a second residual. Stack `LM_N_BLOCKS` copies to build the toy decoder body between embeddings and the language-model head.

## What problem it solves (plain language)

Token embeddings alone know “which word” and “where,” but not how words **relate in context**. One attention+FFN pass is a single round of “listen to the sentence, then think locally.” Stacking blocks lets **higher layers** build on **lower layers’** mixed representations—syntax, then longer patterns, then logits-ready features—without one monolithic matrix doing everything.

## How it works for a human (intuition, analogy)

Each block is a **floor in a factory**. Floor input: a conveyor belt of token vectors. Station 1 (MHA): workers compare notes across positions and update each package. Station 2 (FFN): each package goes through the same two-room workshop ([feed-forward-layers.md](feed-forward-layers.md)) without talking to neighbors. After each station, the original package is **added back** ([residual-connections.md](residual-connections.md)) so nothing is lost by accident. The building has `LM_N_BLOCKS = 2` floors in the toy config.

## Formal view (definitions, formulas, shapes, algorithms)

**Input / output.** Block input \(X^{(\ell)} \in \mathbb{R}^{T \times d}\), output \(X^{(\ell+1)}\) same shape, \(d =\) `LM_D_MODEL = 32`, \(T \le\) `LM_MAX_CTX = 32`.

**Block \(\ell\) (pre-norm, causal MHA):**

\[
\begin{aligned}
U^{(\ell)} &= X^{(\ell)} + \mathrm{MHA}\bigl(\mathrm{LN}_1^{(\ell)}(X^{(\ell)});\ \text{causal}\bigr) \\
X^{(\ell+1)} &= U^{(\ell)} + \mathrm{FFN}\bigl(\mathrm{LN}_2^{(\ell)}(U^{(\ell)})\bigr)
\end{aligned}
\]

**MHA** ([multi-head-attention.md](multi-head-attention.md)): 4 heads, \(d_h = 8\), includes \(W_O\).

**FFN** ([feed-forward-layers.md](feed-forward-layers.md)): ReLU, `LM_D_FF = 64`:

\[
\mathrm{FFN}(x) = \mathrm{ReLU}(x W_1 + b_1) W_2 + b_2
\]

**Full stack:**

\[
X^{(0)} = \mathrm{Embed}(tokens) + \mathrm{PosEnc},\quad
X^{(L)} = \mathrm{Block}_{L-1}\circ\cdots\circ \mathrm{Block}_0(X^{(0)}),\quad L = \texttt{LM\_N\_BLOCKS}
\]

**Algorithm (`block_forward`).**

1. Copy `x` to residual stream (or use in place with temps).
2. `layernorm_forward(ln1, tmp_ln, x)`; `attn_mha(tmp_attn, attn, tmp_ln, causal=1)`; `resid_add(x, tmp_attn)`.
3. `layernorm_forward(ln2, tmp_ln, x)`; `ffn_forward(ffn, tmp_ln)` (FFN overwrites buffer); `resid_add(x, tmp_ffn)`.

**`lm_forward`** loops `block_forward` for each block in the array `blocks[LM_N_BLOCKS]`.

## Complexity / numerical pitfalls / design choices (CS depth)

**Per block (order of magnitude).** MHA: \(O(T d^2 + T^2 d)\); FFN: \(O(T d \cdot d_{\text{ff}})\). With \(L=2\), \(d=32\), \(d_{\text{ff}}=64\), \(T=32\), FLOPs are still small—clarity over fusion.

**Pre-norm vs post-norm.** This repo commits to **pre-norm** for training stability (see [residual-connections.md](residual-connections.md)); LayerNorm weights are **per block** (separate `ln1`, `ln2` per layer).

**Causal everywhere.** Decoder-only LM: every block’s MHA uses `causal = 1` ([causal-attention.md](causal-attention.md)).

**ReLU FFN.** Simpler than GELU/SwiGLU; dead ReLU units can produce zero-variance rows—LayerNorm’s \(\varepsilon\) helps downstream.

**Parameter count (rough).** Each block: 4× `32×32` attention matrices + `32×64` + `64×32` FFN + 2× LayerNorm `32` scales/shifts—dominated by attention+FFN weights.

**No cross-block weight sharing.** Block 0 and block 1 have independent parameters unless explicitly tied (not in toy plan).

## How it shows up in this repo

- [`lang/lm_config.h`](../lang/lm_config.h): `LM_N_BLOCKS`, `LM_D_MODEL`, `LM_N_HEADS`, `LM_D_HEAD`, `LM_D_FF`, `LM_MAX_CTX`.
- [`lang/block.h`](../lang/block.h) / [`lang/block.c`](../lang/block.c): struct `block` { `attn *`, `ffn *`, `layernorm *ln1`, `layernorm *ln2` }; `block_create`, `block_forward`, `block_free`.
- [`lang/lm.h`](../lang/lm.h): struct `lm` holds `block blocks[LM_N_BLOCKS]`, embedding, pos enc, `lm_head`; `lm_forward`, `lm_train`, `lm_generate`.
- Wiring: [`lang/attn.h`](../lang/attn.h), [`lang/ffn.h`](../lang/ffn.h), [`lang/layernorm.h`](../lang/layernorm.h), [`lang/resid.h`](../lang/resid.h).

## Tiny worked example

`T = 4` tokens, one block, shapes only:

| Step | Tensor shape | Notes |
|------|--------------|-------|
| Input `x` | `4 × 32` | after embed + pos |
| LN1 out | `4 × 32` | per-row normalize |
| MHA out | `4 × 32` | causal, 4 heads |
| After resid 1 | `4 × 32` | stream + MHA |
| FFN out | `4 × 32` | ReLU inner `4 × 64` |
| Block out | `4 × 32` | second resid |

Second block repeats with **new** weights, same shapes. Final `4 × 32` feeds [next-token prediction](next-token-prediction.md) at each row.

## What comes next

[Next-token prediction](next-token-prediction.md) maps the last (or each) hidden row to `LM_BYTE_VOCAB` logits; [text-generation.md](text-generation.md) runs the stack autoregressively via `./ann lm`.
