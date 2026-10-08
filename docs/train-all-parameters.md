# Train all parameters

## Abstract

The current toy trainer in [`lang/lm.c`](../lang/lm.c) updates only the **LM head** (`w_head`, `b_head`) with a correct softmax–cross-entropy gradient, plus a **heuristic embedding nudge** that assumes the Transformer is the identity map. Full language-model training **replaces that shortcut** with **`lm_forward_cache` + `lm_backward`**, accumulating gradients for **every learnable tensor**: token embeddings, all attention projections, FFN weights, LayerNorm γ/β per block, and the head—then applying **Adam** (or SGD) across the unified parameter vector.

## What problem it solves (plain language)

If only the head learns, the hidden states feeding it stay a **random function** of the input bytes. The model might memorize tiny phrases by shoving mass into embeddings and a linear classifier, but it cannot learn **patterns that require context** (e.g. “after `hello` expect space”) because attention and FFN weights never move. End-to-end training aligns the whole stack so representations at each layer become useful for next-token prediction.

## How it works for a human (intuition, analogy)

Think of the LM as a **team** where only the goalie (head) was allowed to practice. Full training lets **every player** adjust: embeddings choose better “word vectors,” attention learns who should listen to whom, FFN learns local fixes, LayerNorm keeps everyone on the same scale. One loss signal at the end tells each player how they contributed to the missed goal ([transformer-backpropagation.md](transformer-backpropagation.md)).

## Formal view (definitions, formulas, shapes, algorithms)

**Parameter set \(\theta\).** For default [`lm_config.h`](../lang/lm_config.h):

| Component | Tensors | Shape (each block unless noted) |
|-----------|---------|----------------------------------|
| Embed | `E` | \(V \times d\), \(V=256\), \(d=32\) |
| Block \(\ell\) | `Wq,Wk,Wv,Wo` | \(d \times d\) |
| Block \(\ell\) | `W1,b1,W2,b2` | \(d \times d_{\text{ff}}\), \(d_{\text{ff}}\), \(d_{\text{ff}} \times d\), \(d\) |
| Block \(\ell\) | `ln1.gamma,beta`, `ln2.gamma,beta` | \(d\) each |
| Head | `W_head`, `b_head` | \(d \times V\), \(V\) |

**Count (order of magnitude).** Two blocks ≈ **2×(4d² + 2d·d_ff + d_ff·d + 4d) + dV + dV** ≈ tens of thousands of doubles—tiny by modern LM standards, still **two orders of magnitude** more than head-only (~8k head weights).

**Objective (unchanged).** Mean next-token CE over sequence ([next-token-prediction.md](next-token-prediction.md)).

**Training step (target design).**

\[
\theta \leftarrow \theta - \eta \cdot \mathrm{Optimizer}\Big(\nabla_\theta \mathcal{L}\Big)
\]

**Algorithm (`lm_train_step_full`, planned).**

```text
lm_zero_grads(model)
lm_forward_cache(H, cache, model, ids, n)
lm_backward(model, cache, ids, n, 1.0 / (n - 1))   /* mean loss */
lm_adam_step(model, opt, lr)   /* or lm_sgd_step */
```

**Contrast: current `lm_train_steps`.**

```text
for each step:
  lm_forward (no cache)
  for each position t:
    delta = softmax(logits) - one_hot(target)
    update w_head, b_head with delta
    dh = W_head @ delta
    update embed row ids[t] with dh   /* WRONG: ignores transformer Jacobian */
```

**Gradient flow requirement.** `dh` at row \(t\) must be **`lm_backward`’s** \(\partial \mathcal{L}/\partial H_{t,:}\), which already includes head, blocks, and embed path through attention.

**Positional encoding.** Sinusoidal PE in [`lang/posenc.c`](../lang/posenc.c) is **fixed**—no parameters, no grad.

**Weight tying (optional future).** Tying \(W_{\text{head}}\) to \(E^\top\) merges embed and head grads on shared storage; not required for “train all” baseline.

## Complexity / numerical pitfalls / design choices (CS depth)

**Optimizer.** Plain SGD on full \(\theta\) with lr=0.05 (current head lr) often **destabilizes** attention; Adam with lr ≈ 1e-3–3e-4 is typical ([adam-lm-weights.md](adam-lm-weights.md)).

**Gradient clipping.** Optional global norm cap (e.g. 1.0) before Adam helps when \(T\) or lr is large.

**Overfitting on phrases.** With full params, memorizing `"hello"` in dozens of steps is easy—use **held-out NLL / perplexity** ([held-out-loss-perplexity.md](held-out-loss-perplexity.md)) to detect pure memorization.

**Dead ReLU + LayerNorm.** FFN units can die; LN gradients still flow; monitor fraction of zero FFN pre-activations in debug builds.

**Embed vs head learning rate.** Some frameworks use **embed_lr_mult**; single Adam with per-tensor slots is enough for toy scale.

**Zero grad hygiene.** `lm_zero_grads` must run **once per step** before backward; Adam `m/v` persist ([checkpoint-and-resume.md](checkpoint-and-resume.md)).

**Symmetry breaking.** Attention is permutation-equivariant without PE; PE breaks it—do not remove PE when training all weights.

## How it shows up in this repo

**Current.**

- [`lang/lm.h`](../lang/lm.h): `lm_train_steps(lm *, const char *text, int steps, double lr)` — head SGD + embed hack.
- [`lang/lm.c`](../lang/lm.c): lines ~188–214 implement head update; comment *“rough embed update (treat transformer as identity)”*.

**Planned replacements / additions.**

```c
void lm_zero_grads(lm *model);

int lm_train_step(lm *model, lm_fwd_cache *cache, lm_adam *opt,
    const int *ids, int n, double lr);

double lm_train_corpus(lm *model, lm_adam *opt,
    const text_dataset *ds, int epochs, int batch_size, double lr);
```

**Grad storage (planned).** Extend `lm` or parallel `lm_grads` with pointers matching:

- `embed->d_table`
- per `attn`: `d_wq`, `d_wk`, `d_wv`, `d_wo`
- per `ffn`: `d_w1`, `d_b1`, `d_w2`, `d_b2`
- per `layernorm`: `d_gamma`, `d_beta`
- `d_w_head`, `d_b_head`

Module `*_backward` functions write into these buffers ([transformer-backpropagation.md](transformer-backpropagation.md)).

**CLI / tests.**

- [`main.c`](../main.c) `run_lm`: will call full trainer when `./ann lm train` grows corpus support ([lm-cli.md](lm-cli.md)).
- [`tests/test_ann.c`](../tests/test_ann.c): extend LM tests to assert **NLL drops** when backward enabled and **attention weight norm changes** after one step.

**Related docs.** [mini-batch-corpus-training.md](mini-batch-corpus-training.md), [text-dataset-loading.md](text-dataset-loading.md).

## Worked example

**Phrase `"ab"`, ids `[97, 98]`, \(T=2\), one training step.**

1. Forward with cache: embed rows 97, 98 → posenc → 2 blocks → \(H \in \mathbb{R}^{2 \times 32}\).
2. Loss: only \(t=0\) (predict 98). CE gradient feeds `dH[0,:]`.
3. Backward: block1 → block0 → embed rows get `dE[97,:]` from stream grad at position 0; position 1’s embed row receives grad only if later positions attend to it (here row 0’s causal attn may still touch key from position 0 only at query 0).
4. After Adam: check **`mat_frob_diff(attn->wq, wq_snapshot) > 0`** — impossible with head-only training.

**Before/after metric.** `lm_nll` on `"ab"` should fall faster with full training than head-only at same lr (until both overfit).

## What next

- Land **`lm_backward`** and grad buffers; delete embed hack from `lm_train_steps` or deprecate in favor of `lm_train_step`.
- Hook **`lm_adam_create`** to walk all parameter pointers ([adam-lm-weights.md](adam-lm-weights.md)).
- Train on **`data/toy.txt`** windows ([text-dataset-loading.md](text-dataset-loading.md)) with mini-batches ([mini-batch-corpus-training.md](mini-batch-corpus-training.md)).
- Save checkpoints after epochs ([lm-weight-serialization.md](lm-weight-serialization.md), [checkpoint-and-resume.md](checkpoint-and-resume.md)).
