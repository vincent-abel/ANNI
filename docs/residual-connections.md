# Residual connections (pre-norm)

## Abstract

Residual connections add the **sub-layer input** back to the **sub-layer output** so gradients and information can flow across depth without being rewritten every time. This repo uses **pre-norm**: normalize first, apply attention or FFN, then add the skip: \(Y = X + \mathrm{SubLayer}(\mathrm{LN}(X))\).

## What problem it solves (plain language)

Deep stacks of matrix multiplies and nonlinearities can **attenuate signals** and make training fragile: early layers change slowly or vanishing gradients block learning. A skip path lets each block learn **incremental updates** (“adjust what you already have”) instead of re-encoding the entire representation from scratch.

## How it works for a human (intuition, analogy)

Picture editing a document. **Post-norm** (classic paper diagram): write a draft, edit heavily, then compare to the original and keep a blend. **Pre-norm** (this project): lightly normalize the draft for consistency, edit, then **add the edit on top of the original page** you started from. If the editor’s change is useless, the block can learn to output near zero and the page stays almost unchanged—an easy default.

## Formal view (definitions, formulas, shapes, algorithms)

**Pre-norm residual around a sub-layer** \(F\) (MHA or FFN):

\[
Y = X + F(\mathrm{LN}(X))
\]

Shapes: \(X, Y \in \mathbb{R}^{T \times d}\) with \(d =\) `LM_D_MODEL`; `LN` is [layer normalization](layer-normalization.md) applied **row-wise** (per token).

**Full block pattern (two residuals):**

\[
\begin{aligned}
H_1 &= X + \mathrm{MHA}(\mathrm{LN}(X)) \\
Y   &= H_1 + \mathrm{FFN}(\mathrm{LN}(H_1))
\end{aligned}
\]

**Element-wise add (`resid_add`).** For matrices `dst` and `src` with identical `rows × cols`:

\[
\texttt{dst}_{ij} \leftarrow \texttt{dst}_{ij} + \texttt{src}_{ij}
\]

Typically `dst` is the residual stream (`X` or `H_1`); `src` is the sub-layer output written into a temp buffer, then accumulated in place.

**Contrast with post-norm:**

\[
Y = \mathrm{LN}(X + F(X))
\]

Pre-norm moves normalization **inside** the residual branch; optimization often tolerates deeper stacks without special learning-rate schedules.

## Complexity / numerical pitfalls / design choices (CS depth)

**Complexity.** \(O(T d)\) per add—negligible vs attention \(O(T^2 d)\) and FFN \(O(T d \cdot d_{\text{ff}})\).

**In-place discipline.** `resid_add(dst, src)` must not alias incorrectly if `src` was derived from `dst` without a buffer; blocks usually compute sub-layer output into `tmp`, then add into the stream.

**Scale growth.** Skips preserve vector norm growth across layers; LayerNorm before each sub-layer keeps activations bounded entering \(F\).

**Initialization synergy.** Small random weights (\(\pm 0.02\)) on attention and FFN make early \(F(\mathrm{LN}(X)) \approx 0\), so the network starts **near identity** depth-wise—a practical trick for training stability.

**No gating.** Some architectures (Highway, GRU-style) use learned gates on skips; vanilla Transformer residuals are pure add.

## How it shows up in this repo

- [`lang/resid.h`](../lang/resid.h) / [`lang/resid.c`](../lang/resid.c): `resid_add(mat *dst, const mat *src)` — in-place accumulation.
- [`lang/block.h`](../lang/block.h) (planned): `block_forward` calls `layernorm_forward`, `attn_mha`, `resid_add`, then second LN + `ffn_forward` + `resid_add`.
- [`lang/layernorm.h`](../lang/layernorm.h): normalization before each sub-layer, not after the add.

Config: `LM_N_BLOCKS = 2` stacks two such residual pairs.

## Tiny worked example

Single token (\(T=1\)), \(d=3\) for readability. Let \(X = [1, 2, 3]\). Suppose \(\mathrm{LN}(X) = [0, 0, 0]\) (unlikely literally, but imagine a small normalized vector) and MHA returns \([0.1, -0.1, 0.05]\).

\[
H_1 = [1, 2, 3] + [0.1, -0.1, 0.05] = [1.1, 1.9, 3.05]
\]

Second sub-layer: FFN on \(\mathrm{LN}(H_1)\) might return \([0.02, 0.02, -0.01]\); final \(Y = H_1 + \text{that}\). With `LM_D_MODEL = 32`, the same add runs row-wise for all `T` positions in parallel.

## What comes next

[Causal attention](causal-attention.md) inside MHA for decoder LMs; full stack in [transformer-blocks.md](transformer-blocks.md).
