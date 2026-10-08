# Positional encoding (sinusoidal)

## Abstract

Transformers have no built-in notion of **order**: self-attention treats a bag of tokens unless you tell the model where each token sits in the sequence. **Positional encoding (PE)** adds a position-dependent vector to each token embedding so layer stacks can use “position 3” differently from “position 17.” This note explains the classic **sinusoidal PE** from *Attention Is All You Need* (Vaswani et al., 2017) and how this repo applies it in place on a token matrix.

## What problem it solves (layman)

Imagine reading a sentence with every word on a separate card, shuffled on a table. You still know *what* was said, but not *in what order*. Attention alone is like that: it mixes tokens but does not mark “first word vs second word.” PE writes a small, fixed **position fingerprint** onto each row of the sequence so the network can learn patterns like “the verb often follows the subject two slots ahead.”

Sinusoidal PE picks fingerprints made of sines and cosines at different frequencies so nearby positions look similar and far-apart positions look different, without learning a huge lookup table for every possible length.

## Formal view (CS / math)

Let the input sequence be token embeddings \(X \in \mathbb{R}^{T \times d_{\text{model}}}\) (rows = time steps \(0..T-1\), columns = model width). The encoder adds a matrix \(P\) of the same shape:

\[
X'_{t,j} = X_{t,j} + P_{t,j}.
\]

For dimension index \(j = 2i\) (even):

\[
P_{t,2i} = \sin\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right).
\]

For \(j = 2i+1\) (odd), when that column exists:

\[
P_{t,2i+1} = \cos\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right).
\]

Here \(t\) is the **absolute position** (0-based), \(i \in \{0,1,\ldots\}\) indexes frequency bands, and \(d_{\text{model}}\) is the embedding width. Low \(i\) → slow oscillation across positions; high \(i\) → fast oscillation. The base \(10000\) sets the wavelength ladder (same as in the original paper and most reference implementations).

**Odd \(d_{\text{model}}\):** the last column index is \(d_{\text{model}}-1\). Pairs \((2i, 2i+1)\) are filled while \(2i+1 < d_{\text{model}}\); the final even index (when width is odd) gets **sin only**, with no paired cos column—matching PyTorch’s `pe[:, 0::2]` / `pe[:, 1::2]` behavior.

**Shapes in this repo**

| Object | Shape | Meaning |
|--------|--------|---------|
| `seq` (`mat *`) | `T × d_model` | One row per token position after embedding |
| `seq->rows` | `T` | Sequence length (≤ `LM_MAX_CTX` in toy config) |
| `seq->cols` | `d_model` | Typically `LM_D_MODEL` (32 in [`lang/lm_config.h`](../lang/lm_config.h)) |

**Complexity:** \(O(T \cdot d_{\text{model}})\) time, \(O(1)\) extra memory (in-place add).

**Numerics:** angles use `exp(2i · (-log(10000)/d))` instead of `pow(10000, 2i/d)` for equivalent \(t / 10000^{2i/d}\) with stable exponents.

## API

```c
void posenc_sinusoidal_add(mat *seq);
```

- **Input/output:** `seq` must be a valid `mat` with `rows = T`, `cols = d_model`. Existing entries are **incremented** by PE (not replaced).
- **Null/empty:** no-op if `seq` is NULL, has no `data`, or non-positive dimensions.

## How it shows up in this repo

- Implementation: [`lang/posenc.h`](../lang/posenc.h), [`lang/posenc.c`](../lang/posenc.c)
- Matrix layout and accessors: [`num/mat.h`](../num/mat.h) (`mat_get` / `mat_set`, row-major storage)
- Toy LM dimensions: [`lang/lm_config.h`](../lang/lm_config.h) (`LM_D_MODEL`, `LM_MAX_CTX`)
- Related pipeline pieces: byte tokenizer [`lang/tokenize.c`](../lang/tokenize.c), vocabulary [`lang/vocab.c`](../lang/vocab.c)

To link, add `lang/posenc.c` to `LANG_SRC` in the Makefile (not done automatically in this change).

## Tiny worked example

Toy config: `d_model = 4`, sequence length `T = 2`. Before PE, suppose row 0 is all zeros and row 1 is all zeros (only PE visible):

| \(t\) | col 0 (\(\sin\)) | col 1 (\(\cos\)) | col 2 (\(\sin\)) | col 3 (\(\cos\)) |
|------|------------------|------------------|------------------|------------------|
| 0 | 0 | 1 | 0 | 1 |
| 1 | \(\sin(1)\) | \(\cos(1)\) | \(\sin(1/100)\) | \(\cos(1/100)\) |

After `posenc_sinusoidal_add`, each cell is **previous value + table entry**. For a non-zero embedding row, you add these values on top of the embedding.

**Odd width:** `d_model = 5`, position `t = 0`: columns 0–3 get the sin/cos pairs for \(i=0,1\); column 4 gets \(\sin(0/10000^{4/5}) = 0\) only.

## Design choices and pitfalls

- **Add vs concat:** Original Transformer **adds** PE to embeddings; some models concatenate or use rotary (RoPE) / learned PE instead. This module is fixed sinusoidal **add**.
- **Relative length:** Sinusoidal PE generalizes to lengths seen at training time in theory; very long extrapolation can still degrade—`LM_MAX_CTX` caps the toy model.
- **Batch dimension:** API is a single `T × d` matrix; batched training would apply the same function per sequence row block or wrap a batch loop outside.
- **Learned PE alternative:** a `T_max × d` parameter matrix; sinusoidal avoids extra weights and was the paper default for the encoder stack.

## What comes next

A decoder-only LM stack will typically: embed token ids → `posenc_sinusoidal_add` on the embedding matrix → blocks of attention + FFN. Learned or rotary encodings can be documented alongside when added.

## References

- Vaswani, Shazeer, Parmar, et al., “Attention Is All You Need,” NeurIPS 2017.
- Original PE discussion: Section 3.5 in that paper (sin/cos functions of position and dimension).
