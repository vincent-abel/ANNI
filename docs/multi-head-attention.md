# Multi-head attention

## Abstract

Multi-head attention (MHA) runs **several self-attention operations in parallel** on different subspaces of the model dimension, then **concatenates** and **re-projects** the results. In this repo: project once to full \(Q,K,V\), split into `LM_N_HEADS` slices of `LM_D_HEAD`, apply `attn_self` per head, concat, multiply by `W_O`.

## What problem it solves (plain language)

One attention pattern might track syntax, another coreference, another local bigrams. A single \(d \times d\) mixer forces one compromise. Multiple heads let the model learn **different match rules at once**, each in a smaller \(d_{\text{head}}\)-dimensional space, then fuse back into the main `LM_D_MODEL` stream.

## How it works for a human (intuition, analogy)

Imagine four specialist committees (`LM_N_HEADS = 4`). Everyone still speaks in the full 32-dimensional “company language,” but each committee only listens along **8 of those coordinates** (`LM_D_HEAD = 8`). Each committee runs its own “who matches whom?” vote ([self-attention](self-attention.md)). Their reports are stapled side-by-side (concatenation) and sent through one final editor matrix \(W_O\) to produce a single updated sentence-wide representation per token.

## Formal view (definitions, formulas, shapes, algorithms)

**Hyperparameters (this repo).**

| Symbol | Value | Meaning |
|--------|-------|---------|
| \(d\) | 32 | `LM_D_MODEL` |
| \(h\) | 4 | `LM_N_HEADS` |
| \(d_h\) | 8 | `LM_D_HEAD` (= \(d/h\)) |

**Input.** \(X \in \mathbb{R}^{T \times d}\).

**Shared projections** (see [query-key-value.md](query-key-value.md)):

\[
Q = X W_Q,\quad K = X W_K,\quad V = X W_V,\quad W_* \in \mathbb{R}^{d \times d}
\]

**Head split.** For head index \(k \in \{0,\ldots,h-1\}\), columns \([k d_h,\,(k+1)d_h)\) of \(Q,K,V\) form \(Q^{(k)}, K^{(k)}, V^{(k)} \in \mathbb{R}^{T \times d_h}\).

**Per-head attention:**

\[
O^{(k)} = \mathrm{Attention}\bigl(Q^{(k)}, K^{(k)}, V^{(k)}\bigr) \in \mathbb{R}^{T \times d_h}
\]

**Concatenate along features:**

\[
\mathrm{Concat} = \big[\, O^{(0)} \,\|\, O^{(1)} \,\|\, \cdots \,\|\, O^{(h-1)} \,\big] \in \mathbb{R}^{T \times d}
\]

**Output projection:**

\[
Y = \mathrm{Concat}\, W_O,\quad W_O \in \mathbb{R}^{d \times d}
\]

**Algorithm (`attn_mha`).**

1. `attn_qkv_project` → \(Q,K,V\) each `T × 32`.
2. For `h = 0 .. LM_N_HEADS-1`: copy head slices into `T × 8` buffers; `attn_self(..., causal)`; write head output back into `concat`.
3. `mat_mul(proj, concat, wo)` → `T × 32` copied to `out`.

Causal decoding passes `causal = 1` so every head respects the same lower-triangular mask.

## Complexity / numerical pitfalls / design choices (CS depth)

**FLOPs.** One QKV triple: \(3 \cdot O(T d^2)\). Per head: \(O(T^2 d_h)\) attention; \(h\) heads → \(O(T^2 d)\) (same order as one big head if \(h d_h = d\)). Output proj: \(O(T d^2)\). **Total** dominated by QKV + \(W_O\) GEMMs at small \(T\); at long context, \(O(T^2 d)\) attention per layer matters.

**Head divisibility.** `LM_D_MODEL % LM_N_HEADS == 0` is enforced in `attn_create`; otherwise head split is impossible.

**Layout.** This code uses **fused** \(d \times d\) projections then manual column slicing (`copy_head` / `write_head`). Some libraries fuse QKV into one `3d × d` multiply for cache efficiency—equivalent math.

**\(W_O\).** Without it, concat heads would be a block-diagonal mix only; \(W_O\) allows **cross-head interaction** in one linear map.

**Dropout / bias.** Production Transformers often add biases and attention dropout; this toy stack uses no bias on projections and no dropout—simpler numerics for learning C.

## How it shows up in this repo

- [`lang/attn.h`](../lang/attn.h): struct `attn` holds `wq`, `wk`, `wv`, `wo`, `n_heads`, `d_head`.
- `attn_create()` allocates four `32×32` matrices, init \(\pm 0.02\).
- `attn_mha(out, a, x, causal)` — full MHA forward; `out` is `T × LM_D_MODEL`.
- Used inside each transformer block (see [transformer-blocks.md](transformer-blocks.md)) after pre-norm LayerNorm.

## Tiny worked example

Take `T = 2`, `d = 32`, `h = 4`, `d_h = 8`. After QKV project, row 0 of `q` is a 32-vector; head 0 uses elements `q[0..7]`, head 1 uses `q[8..15]`, etc. Each head produces a `2×8` `o_h`; concat row 0 is 32 numbers (four 8-blocks). Multiplying concat by `wo` yields the MHA output row—same width as `x`, ready for [residual-connections.md](residual-connections.md).

If head 0 alone matched strongly token 0→0 and token 1→0, its 8-d slice would resemble row 0 of `v`; other heads can disagree until \(W_O\) blends them.

## What comes next

Stack MHA with residuals and FFN in [transformer-blocks.md](transformer-blocks.md). For autoregressive LMs, always call with `causal = 1` ([causal-attention.md](causal-attention.md)).
