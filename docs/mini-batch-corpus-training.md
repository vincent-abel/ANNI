# Mini-batch corpus training

## Abstract

After loading windows from a text file ([text-dataset-loading.md](text-dataset-loading.md)), **`lm_train_corpus`** should not update weights on every window independently with unscaled noise. **Mini-batch training** runs forward+backward on **`batch_size` windows**, **accumulates gradients** on shared parameter buffers, **averages** by batch size, then applies one **Adam** step—mirroring [`network_train_mb`](../net/layer.c) for the XOR net ([mini-batches.md](mini-batches.md)).

## What problem it solves (plain language)

Single-window updates jitter when adjacent windows overlap heavily (stride 1). Averaging gradients over a small batch **smooths** the direction of weight changes and better matches the **mean loss** over the corpus chunk you saw before moving parameters.

## How it works for a human (intuition, analogy)

Instead of one student shouting homework corrections after each problem, the teacher collects **four papers**, marks them all, **averages the suggested fixes**, and applies **one** set of edits to the textbook (the weights).

## Formal view (definitions, formulas, shapes, algorithms)

**Batch** \(\mathcal{B} = \{ \text{window}_1, \ldots, \text{window}_B \}\).

**Per-window loss** \(\mathcal{L}_b\) = mean CE over valid positions in window \(b\) (length \(L_b-1\) terms).

**Batch objective (typical).**

\[
\mathcal{L}_{\mathcal{B}} = \frac{1}{B}\sum_{b=1}^{B} \mathcal{L}_b
\]

**Gradients.**

\[
\nabla_\theta \mathcal{L}_{\mathcal{B}} = \frac{1}{B}\sum_{b=1}^{B} \nabla_\theta \mathcal{L}_b
\]

**Algorithm (`lm_train_corpus`, planned).**

```text
opt = lm_adam_create(model)
for epoch in 1..epochs:
  text_dataset_reset(ds)
  lm_zero_grads(model)
  b = 0
  while (L = text_dataset_next_window(ds, ids, LM_MAX_CTX)) > 0:
    lm_forward_cache(H, cache, model, ids, L)
    lm_backward(model, cache, ids, L, 1.0 / (L - 1))   /* per-window mean */
    b += 1
    if b == batch_size:
      lm_scale_grads(model, 1.0 / batch_size)
      lm_adam_step(opt, lr)
      lm_zero_grads(model)
      b = 0
  if b > 0:   /* tail batch */
    lm_scale_grads(model, 1.0 / b)
    lm_adam_step(opt, lr)
return lm_perplexity(model, ds_eval) or train NLL
```

**`lm_scale_grads`.** Multiply every entry in every `d_*` tensor by scalar `scale`.

**Cache reuse.** Allocate one `lm_fwd_cache` sized for `LM_MAX_CTX`; reuse per window (overwrite activations each forward).

## Complexity / numerical pitfalls / design choices (CS depth)

**Variable window lengths.** Normalize each backward by `1/(L-1)` **before** batch sum so short and long windows contribute comparable per-token gradient mass—or use **token-weighted** average: scale by `(L-1)` and divide total grad by sum of `(L-1)` in batch (closer to “mean over all tokens”).

**Last partial batch.** Smaller batch ⇒ noisier step; acceptable or drop remainder.

**batch_size=1.** Recovers online SGD on windows.

**Memory.** `batch_size` affects **time**, not peak activation memory (sequential forwards)—unless batched matmul is added later.

**Epoch definition.** One epoch = one pass of `text_dataset_next_window` until exhaustion after `reset`.

## How it shows up in this repo

- Pattern: [`docs/mini-batches.md`](mini-batches.md) (`network_scale_grads`, accumulate backward).
- Planned: `lang/lm_train.c` — `lm_train_corpus`, `lm_scale_grads`.
- Entry: `./ann lm train --corpus data/toy.txt --batch 4` ([lm-cli.md](lm-cli.md)).

## Token-parallel packing (`--batch B`)

`--batch B` packs **B windows of T tokens into one forward/backward**: embed, LN, QKV/Wo, FFN and the LM head are single `(B·T)×D` GEMMs, while attention is causal **within each sequence only** (B score caches of `T×T`, never `(B·T)²`). The CE head gathers hidden rows `0..T-2` of every sequence into one `B·(T-1)×D` matrix (position `t` of sequence `b` predicts `ids[b*T+t+1]`).

- API: `lm_cache_create_packed(B, T)`, `lm_forward_cache_packed`, `lm_loss_backward_packed`, `lm_backward_packed` (`lang/lm.h`). `B == 1` is the classic single-window path.
- `--grad-accum K`: K packed microbatches are summed before one Adam step (scale `1/(B·K)`); activations do not grow with K.
- `--batch auto` (with `--vram-frac F`, default 0.85): largest B (cap 64, and windows per epoch) whose `lm_packed_bytes` estimate fits `F × cudaMemGetInfo free`; CPU builds fall back to 8. `-d` prints free/total VRAM, B, T and the estimate. A failed allocation halves B.
- GPU builds run attention for all `(b, h)` in one fused kernel launch (`gpu_attn_causal_fwd/bwd`); CPU builds loop per sequence/head under OpenMP.

## Worked example

**batch_size=2**, two windows from `"abcd"` with max_len=4, stride=1.

- Window1 `[97,98,99,100]`: 3 CE terms.
- Window2 `[98,99,100]`: 2 CE terms.

Backward each with `1/(L-1)` scale; add grads; multiply sum by `0.5`; Adam once. Two windows, **one** weight update.

## What next

- Implement `lm_scale_grads` alongside `lm_zero_grads`.
- Log **tokens/sec** and **updates/epoch** in CLI.
- Compare perplexity: batch 1 vs 8 on same toy corpus ([held-out-loss-perplexity.md](held-out-loss-perplexity.md)).
