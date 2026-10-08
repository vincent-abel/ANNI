# Self-attention (scaled dot-product)

## Abstract

Self-attention lets each token in a sequence **re-weight every other token’s value vector** using content-based scores. This note defines the core operator: scaled dot-product attention maps \((Q, K, V)\) to an output matrix of the same shape as \(Q\), with optional causal masking handled elsewhere in depth.

## What problem it solves (plain language)

Fixed-weight layers mix features the same way at every position. Language is not like that: the word “bank” should listen to “river” or “money” depending on context. Self-attention gives each position a **dynamic mixer**: it decides how much to copy from every other position’s representation before moving on.

## How it works for a human (intuition, analogy)

Think of a meeting where everyone holds a **question** (query), a **name tag** others can match (key), and a **handout** to share (value). Person \(i\) compares their question to everyone’s name tags; high matches mean “I want more of that handout.” The answers are blended handouts, one updated bundle per person. Scaling by \(\sqrt{d_k}\) keeps the match scores from exploding when many dimensions contribute.

## Formal view (definitions, formulas, shapes, algorithms)

**Inputs.** For sequence length \(T\) and head dimension \(d_k\):

| Tensor | Shape | Role |
|--------|-------|------|
| \(Q\) | \(T \times d_k\) | queries, one row per token |
| \(K\) | \(T \times d_k\) | keys |
| \(V\) | \(T \times d_k\) | values |
| output | \(T \times d_k\) | mixed values |

**Scores and weights.** Row \(i\) of the score matrix:

\[
S_{ij} = \frac{1}{\sqrt{d_k}} \sum_{t=1}^{d_k} Q_{it}\, K_{jt}
\]

**Attention weights** (row-wise softmax):

\[
A_{ij} = \frac{e^{S_{ij} - \max_j S_{ij}}}{\sum_{j'} e^{S_{ij'} - \max_j S_{ij}}}
\]

**Output:**

\[
\mathrm{Out}_{i,:} = \sum_{j=1}^{T} A_{ij}\, V_{j,:}
\]

In matrix form: \(\mathrm{Out} = \mathrm{softmax}(Q K^\top / \sqrt{d_k})\, V\) (softmax applied **per row** of the score matrix).

**Algorithm (`attn_self`).**

1. Allocate scores \(T \times T\).
2. For each pair \((i, j)\), compute dot product of row \(i\) of \(Q\) with row \(j\) of \(K\), multiply by \(1/\sqrt{d_k}\).
3. If `causal` is set, positions with \(j > i\) are forced to \(-10^9\) before softmax (see [causal-attention.md](causal-attention.md)).
4. Softmax each row of scores → \(A\).
5. For each output row \(i\), \(\mathrm{Out}_{i,:} = \sum_j A_{ij} V_{j,:}\).

Toy config: full model width is `LM_D_MODEL = 32`; inside one head, \(d_k =\) `LM_D_HEAD = 8`.

## Complexity / numerical pitfalls / design choices (CS depth)

**Complexity.** Naive triple loop: \(O(T^2 d_k)\) for scores plus \(O(T^2 d_k)\) for the value aggregate → **\(O(T^2 d_k)\)** time and **\(O(T^2)\)** memory for the score matrix. For `LM_MAX_CTX = 32`, that is tiny; at 4k context, memory dominates.

**Scaling.** Without \(1/\sqrt{d_k}\), dot products grow in variance with \(d_k\) (random vectors in high dimension have larger typical inner products), pushing softmax into near one-hot extremes and starving gradients.

**Softmax stability.** Subtract row max before `exp` (see [softmax.md](softmax.md)); `attn_self` uses `vec_softmax` per row.

**Mask value.** Large negative logits (\(-10^9\)) approximate zero mass after softmax; use finite constants compatible with `double`.

**Alternatives.** Additive attention (Bahdanau) uses a feed-forward scorer instead of dot product; linear attention and flash-attention variants reduce \(O(T^2)\) memory—out of scope for this educational C loop.

## How it shows up in this repo

- [`lang/attn.h`](../lang/attn.h) / [`lang/attn.c`](../lang/attn.c): `attn_self(out, q, k, v, causal)` — one head, `q/k/v` shaped `T × LM_D_HEAD`.
- Q/K/V for the full model come from `attn_qkv_project` (see [query-key-value.md](query-key-value.md)).
- `causal` is an `int` flag; when non-zero, future keys are masked before softmax. Multi-head wiring calls `attn_self` per head inside `attn_mha`.

## Tiny worked example

Let \(T = 3\), \(d_k = 2\) (smaller than production `LM_D_HEAD = 8` for pencil-and-paper).

\[
Q = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
K = Q,\quad V = \begin{bmatrix}1&0\\0&1\\2&2\end{bmatrix}
\]

Raw dots \(Q_i \cdot K_j\) (before scale):

| \(i \setminus j\) | 0 | 1 | 2 |
|------------------|---|---|---|
| 0 | 1 | 0 | 1 |
| 1 | 0 | 1 | 1 |
| 2 | 1 | 1 | 2 |

With scale \(1/\sqrt{2}\), row 0 softmax weights are roughly \([0.45, 0.18, 0.37]\); output row 0 is that convex combination of the three value rows. With `causal = 1`, row 0 only sees column 0; row 1 sees columns 0–1; row 2 sees all three.

At full head size (`d_k = 8`, `T = 32`), the same formulas apply; only dimensions change.

## What comes next

[Multi-head attention](multi-head-attention.md) runs several independent self-attention passes in parallel and merges them with \(W_O\). [Causal attention](causal-attention.md) explains the decoder mask in detail.
