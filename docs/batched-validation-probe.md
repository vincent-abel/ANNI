# Batched validation probe: held-out loss without stalling the run

## Abstract

Twenty times per epoch the trainer measures the loss on held-out windows. The probe now reuses the training's packed cache: `B` windows per forward, one head GEMM, and a small kernel that writes the loss of each row without touching the logits. A probe went from about 40 s to about 2 s (candidate 18 in [BF16 outputs](bf16-direct-outputs.md)).

## What problem it solves (plain language)

The old probe ran one window at a time, built its own cache, and did the softmax on the host in double precision, row by row. It was written for small models. On a big one it froze training for most of a minute at every 5% mark.

## How it works for a human (intuition, analogy)

Grading a quiz with the machine already on the desk instead of walking each sheet to another room. The training pipeline is right there and already handles 16 windows at once; the probe borrows it for a forward and only asks one extra question at the end: how wrong was each row?

## Formal view (definitions, formulas, shapes, algorithms)

For each chunk of `B` held-out windows of `T` tokens:

1. Packed forward, exactly as in training (see [Packed batches](packed-batch-attention.md)).
2. One head GEMM over the `B*(T-1)` rows that have a target.
3. Per row, on the device: `nll = logsumexp(z) - z[target]`, in FP32.
4. The host sums the rows in FP64 and divides by the row count.

No backward, no gradient buffers written, weights untouched.

## Complexity / numerical pitfalls / design choices

- **Read-only on the logits.** Training's CE kernel overwrites the logits with the gradient ([Fused head and CE](fused-head-ce.md)). The probe uses a separate kernel so nothing downstream can mistake a probe for a training step.
- **Not bit-equal to the old probe.** FP32 log-sum-exp and a head GEMM in the training precision: the value moves by about `1e-5`. Curves before and after the change line up.
- **Subsample by default.** 32 windows per mid-epoch probe unless `--eval-windows` says otherwise; the full set 20 times per epoch would cost more than it tells.
- **Tails and odd cases fall back.** A last chunk smaller than a window batch, a host head, or `B == 1` use the per-window `lm_nll`.
- **Resume probed twice.** A resumed run used to probe at start and again at the first 5% mark it had already passed; the mark is now advanced first.

## How it shows up in this repo

- [`lang/lm.c`](../lang/lm.c): `lm_nll_rows_packed` (one value per row, `-1` when the batched path does not apply) and `lm_nll_packed` (per-window means).
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_ce_nll_rows` and `ce_nll_rows_kernel`.
- [`lang/lm_train.c`](../lang/lm_train.c): `probe_windows` runs the chunks on the training cache; `train_mid_val_due` fires every 5% of the epoch; `TRAIN_MID_VAL_DEFAULT_WINS` is the 32.
- `ANN_PROBE_BATCH=0` restores the per-window path.
- CLI: `--val-data PATH` or `--val-frac F` pick the held-out set, `--eval-windows N` the sample size.

Related: [Cheaper window eval](cheaper-window-eval.md), [Held-out loss and perplexity](held-out-loss-perplexity.md).
