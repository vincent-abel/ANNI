# Quantization

## Abstract

**Int8 quantization** packs a full-precision weight matrix into `int8` values plus one **scale** so inference uses less memory bandwidth. `qmat_from_mat` chooses `scale = max|w|/127`; `qmat_gemv_t` reconstructs approximately \(y \approx scale \cdot Q^\top x\). Training stays in `double`; call `lm_quantize_head` before generate to switch the LM head.

## What problem it solves (plain language)

Weights as 64-bit doubles waste cache when you only need rough amplitudes for decoding. Storing them as small integers (with one scale) is like writing prices in cents instead of high-precision floats.

## How it works for a human (intuition, analogy)

Photograph → **JPEG**: smaller file, tiny visual loss. Quantization is JPEG for matrices: good enough for looking (inference), not for painting over (training).

## Formal view (definitions, formulas, shapes, algorithms)

\[
s = \frac{\max |W_{ij}|}{127},\quad
Q_{ij} = \mathrm{round}\big(\mathrm{clip}(W_{ij}/s, -127, 127)\big)
\]

\[
y_j = s \sum_i x_i Q_{ij}
\]

**API.**

```text
q = qmat_from_mat(W)
qmat_gemv_t(logits, q, hidden_row)
lm_quantize_head(model)  /* sets model->w_head_q */
```

`lm_logits_at` uses `w_head_q` when present.

## Complexity / numerical pitfalls / design choices

**Per-tensor scale** is simple but less accurate than per-channel scales.

**Asymmetric zero-points** omitted — symmetric int8 around zero.

**Training.** Gradients still update `w_head` doubles; re-quantize after training if needed.

## How it shows up in this repo

- [`lang/quant.h`](../lang/quant.h) / [`lang/quant.c`](../lang/quant.c)
- [`lang/lm.c`](../lang/lm.c): `lm_quantize_head`, quantized branch in `lm_logits_at`

## Worked example

Quantize a random `4×4` mat, GEMV a unit vector, compare to full-precision GEMV — relative error typically well under a few percent for this toy scale.

## What next

[Optimized inference](optimized-inference.md) adds a KV cache so generate does not reforward the whole prefix each token.
