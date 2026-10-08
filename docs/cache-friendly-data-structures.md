# Cache-friendly data structures

## Abstract

Hot LM paths now walk **contiguous row pointers** and express attention as **GEMM**: \(QK^\top\) via `mat_mul_bt`, \(AV\) via `mat_mul`. Embeddings copy rows with `memcpy`. Logits use a 1×D view times `w_head` through blocked `mat_mul`. Accessor-heavy `mat_get` / `mat_set` loops leave the forward hot path.

## What problem it solves (plain language)

Calling `mat_get` for every multiply jumps around with bounds checks and hurts the CPU cache. Row-major layouts and matrix multiply reuse already-loaded tiles so the same math finishes faster without changing the math itself.

## How it works for a human (intuition, analogy)

Reading a book **left to right on one line** is easier than hopping randomly between pages for each letter. Contiguous rows and blocked GEMM keep the “eye” on neighboring numbers.

## Formal view (definitions, formulas, shapes, algorithms)

**Row helper.**

```text
mat_row(m, i) -> &m.data[i * cols]
```

**Attention scores.**

\[
S = \frac{Q K^\top}{\sqrt{d}},\quad
S_{ij} \leftarrow -\infty \text{ if causal and } j>i,\quad
A = \mathrm{softmax}_{\text{row}}(S),\quad
O = A V
\]

Implemented as `mat_mul_bt` → scale/mask → row softmax → `mat_mul`.

**Logits.**

```text
hrow = view of hidden[t,:] as 1 x D
lrow = view of logits as 1 x V
mat_mul(lrow, hrow, w_head)   /* blocked GEMM */
logits += b_head
```

**Embed.**

```text
memcpy(out[i], table[id], D * sizeof(double))
```

## Complexity / numerical pitfalls / design choices

**Layout.** Weights stay `(D × V)` for the head so LMCK files remain compatible; GEMM still streams `w_head` in blocked tiles.

**Numerical.** Causal mask still uses `-1e9` before softmax — same as before.

**gradutil.** `mat_outer_add` / `mat_mul_at` use direct `data[]` indexing; `mat_mul_bt` is the `num/mat.c` implementation.

## How it shows up in this repo

- [`num/mat.h`](../num/mat.h): `mat_row`, `mat_mul_bt`
- [`lang/attn.c`](../lang/attn.c): GEMM attention
- [`lang/embed.c`](../lang/embed.c): row `memcpy`
- [`lang/lm.c`](../lang/lm.c): `lm_logits_at` via `mat_mul`
- [`lang/gradutil.c`](../lang/gradutil.c): pointer-based outer/AT products

## Worked example

For `T=4`, `d_head=8`, `mat_mul_bt` fills a 4×4 score matrix matching nested-loop dots within `1e-12`. Softmax + `mat_mul` match the old accessor `attn_self` outputs in tests.

## What next

[SIMD](simd.md) tunes the same contiguous loops so the compiler (and later AVX) can vectorize them.
