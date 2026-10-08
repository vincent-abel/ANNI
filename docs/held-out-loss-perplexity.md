# Held-out loss and perplexity

## Abstract

Training loss on the **same bytes you train on** can look excellent while the model **memorizes**. **Held-out evaluation** computes mean next-token **negative log-likelihood (NLL)** on text **not used in gradient updates** (e.g. last 20% of `data/toy.txt` or a separate `data/toy_holdout.txt`). **Perplexity** is \(\exp(\text{mean NLL})\)—“effective branching factor” of the model’s predictive distribution.

## What problem it solves (plain language)

NLL alone on `"hello"` tells you the demo works; it does not tell you whether the LM **generalizes** within a corpus. Held-out metrics catch **overfitting** and compare checkpoints during scaling experiments.

## How it works for a human (intuition, analogy)

If a student memorizes answers to **practice problems**, give them a **new sheet** with similar but unseen prompts. Perplexity measures how **surprised** they are on average: lower is better; random guessing over 256 bytes is near \(\exp(\ln 256) \approx 256\).

## Formal view (definitions, formulas, shapes, algorithms)

**Per-position NLL** (same as training CE):

\[
\ell_t = -\log p_\theta(x_{t+1} \mid x_{\le t})
\]

**Mean NLL** over held-out token positions \(\mathcal{T}\):

\[
\overline{\mathcal{L}} = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \ell_t
\]

**Perplexity:**

\[
\mathrm{PPL} = \exp(\overline{\mathcal{L}})
\]

Use **natural log** consistently (`vec_softmax_cross_entropy` / nats); perplexity in nats base is standard in ML tooling.

**Planned API.**

```c
double lm_nll(const lm *model, const int *ids, int n);  /* exists */
double lm_perplexity(const lm *model, const int *ids, int n);
double lm_perplexity_corpus(const lm *model, const text_dataset *ds);
```

**`lm_perplexity`.** `ppl = exp(lm_nll(...))` when NLL is mean over positions; return -1 on error.

**`lm_perplexity_corpus`.** Reset dataset, iterate all windows **without backward**, sum \((L-1)\cdot \text{nll_window}\), divide by total \((L-1)\) sum—token-weighted corpus perplexity.

**Train/holdout split (simple).**

```text
load full ids[0..N-1]
train_ids = ids[0 .. floor(0.8*N)-1]
hold_ids  = ids[floor(0.8*N) .. N-1]
```

Evaluate only on `hold_ids` windows.

**Eval mode.** Dropout not present; forward only (`lm_forward`, no grad).

## Complexity / numerical pitfalls / design choices (CS depth)

**Exp overflow.** If mean NLL > ~700, `exp` overflows—cap reported PPL or print NLL only for broken models.

**Short holdout.** Few bytes ⇒ high variance; toy metrics jump step-to-step.

**Train/eval leakage.** Overlapping windows with stride 1 leak context across split if split is within sliding region—split **by byte offset** before windowing, or use disjoint files.

**Teacher forcing in eval.** Same as training: inputs are ground-truth prefix; matches standard PPL definition for autoregressive LMs.

**Compare to [`lm_nll`](../lang/lm.c).** Already implements mean CE on one id sequence—perplexity is monotonic transform for logging humans prefer.

## How it shows up in this repo

- Today: `lm_nll` in [`lang/lm.h`](../lang/lm.h) / [`lang/lm.c`](../lang/lm.c).
- Planned: `lm_perplexity*`, logged each epoch in `lm_train_corpus`.
- CLI: `./ann lm eval --checkpoint m.bin --corpus data/toy_holdout.txt` ([lm-cli.md](lm-cli.md)).

## Worked example

Holdout ids length 1000 positions → 999 CE terms. Suppose mean NLL = 2.0 nats (model decent on bytes). PPL = \(e^2 \approx 7.39\): on average roughly **7–8 equally likely** next-byte choices—good for byte-level toy data, far from 256.

Random uniform next-byte: NLL = \(\ln 256 \approx 5.545\), PPL = 256.

## What next

- Implement `lm_perplexity` wrapper and corpus aggregator.
- Print train vs holdout NLL each epoch during `lm_train_corpus`.
- Add tiny holdout file in `data/` for tests.
