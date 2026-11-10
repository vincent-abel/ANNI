# Weight tying: one table for the embedding and the head

## Abstract

With `--tied`, the input embedding and the output head share one `D x vocab` matrix. The embedding owns no memory: it reads columns of the head. At a 100k vocabulary and `D = 1536` that saves about 154M parameters, which is what lets the 555M run spend its size on blocks instead of two word tables.

## What problem it solves (plain language)

A model has two big tables about words: one turns a token into a vector on the way in, the other scores every token on the way out. With a large vocabulary each one is bigger than several Transformer blocks, and they learn nearly the same thing: which tokens are alike.

## How it works for a human (intuition, analogy)

One dictionary instead of two. You look a word up in it to read, and you look the meaning up in the same one to pick the next word. Every training step improves that single dictionary from both sides.

## Formal view (definitions, formulas, shapes, algorithms)

The head is `W` of shape `D x V` and the logits are `z = h W + b`, so column `id` of `W` is the output vector of token `id`.

- **Lookup:** `x_t = sqrt(D) * W[:, id_t]`, then the sinusoidal position is added.
- **Gradient:** `dW[:, id_t] += sqrt(D) * dx_t`, added to the same `dW` the head already filled. One gradient, one Adam update.

The `sqrt(D)` factor is there because the head's values are small (they are sized to make sane logits), and without it the positions, which have values up to 1, would drown the token.

## Complexity / numerical pitfalls / design choices

- **Columns, not rows.** The untied table is `V x D` and a token is a row. Tied, a token is a column of the head, so the lookup walks the matrix with a stride of `V`. The kernels take two strides (per id, per dimension) and work for both layouts.
- **Zero the gradient once.** The head clears `dW`; the embedding must not clear it again after the head has written to it.
- **Count it once.** The parameter count, the Adam state, the checkpoint and `lm merge` all skip the embedding table when tied.
- **Scatter order.** Several positions can hold the same token, so the scatter adds into one column from many threads; see [Deterministic gradients](deterministic-gradients.md).

## How it shows up in this repo

- [`lang/embed.c`](../lang/embed.c): `embed_create_tied(w_head, dw_head)` borrows the two matrices and sets `scale = sqrt(LM_D_MODEL)`; `embed_free` leaves them alone; `embed_zero_grad` does nothing when tied.
- [`lang/lm.c`](../lang/lm.c): `lm_create_bpe_tied(tok)`.
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_embed_lookup` and `gpu_embed_scatter` take the layout and the scale. [`lang/decode.c`](../lang/decode.c) passes the same to the GPU decoder.
- [`lang/lm_adam.c`](../lang/lm_adam.c): the shared table is registered once, through the head.
- [`lang/lm_serialize.c`](../lang/lm_serialize.c): tied checkpoints start with the magic `LMCT` and hold no embedding table. Loading one builds a tied model; merging tied with untied models is refused.
- `anni lm train --tied` (in [`main.c`](../main.c)); `anni lm info` prints `LMCT (float+bpe, tied head)`.
- Tests in [`tests/test_ann.c`](../tests/test_ann.c): tied gradient against finite differences, scatter, checkpoint round trip.

## Worked example

`run555m`: `D = 1536`, 14 blocks, 100k BPE tokens. The table is `1536 x 100,000 ≈ 154M` numbers. Untied, the model would carry it twice and weigh about 709M; tied it is about 555M. Adam keeps two moments per parameter, so the optimizer state shrinks by about 308M values too.

## What next

- A tied model cannot load an untied checkpoint (and the reverse); converting one would need a choice between the two tables.
