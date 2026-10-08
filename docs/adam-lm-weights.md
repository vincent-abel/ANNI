# Adam for LM weights

## Abstract

The feedforward XOR network already uses **Adam** via [`net/optim.c`](../net/optim.c) (`adam_create`, `adam_step`) over `network` neurons. Language-model training introduces **thousands of matrix elements** in `lang/` that are not `neuron` structs. This note specifies **`lm_adam`**: a flat array of **Adam slots**—each slot points at one scalar **parameter** and its matching **gradient buffer**—so the same \(\beta_1,\beta_2,\varepsilon\) update rule drives embeddings, attention, FFN, LayerNorm, and the LM head in one `lm_adam_step` call.

## What problem it solves (plain language)

Hand-written SGD loops (like today’s head update in `lm_train_steps`) need a learning rate that works for every tensor shape. Adam **adapts per-weight step sizes** using running averages of the gradient and its square, which makes full Transformer training much less twitchy than a single global lr on all matmul weights.

## How it works for a human (intuition, analogy)

Each weight keeps two **exponential memories**: “which direction have I been pushed lately?” (momentum-ish) and “how big are those pushes usually?” (volatility). Big, noisy weights get smaller effective steps; quiet, consistent gradients get larger ones—within bounds set by the base learning rate.

## Formal view (definitions, formulas, shapes, algorithms)

**Reuse slot shape** from [`net/optim.h`](../net/optim.h):

```c
typedef struct adam_slot {
	double *param;
	double *grad;
	double m;
	double v;
} adam_slot;
```

**LM optimizer wrapper (planned, `lang/lm_adam.h`).**

```c
typedef struct lm_adam {
	adam_slot *slots;
	int        n_slots;
	int        t;          /* timestep for bias correction */
	double     beta1;      /* e.g. 0.9 */
	double     beta2;      /* e.g. 0.999 */
	double     eps;        /* e.g. 1e-8 */
} lm_adam;

lm_adam *lm_adam_create(lm *model);
void     lm_adam_free(lm_adam **opt);
void     lm_adam_step(lm_adam *opt, double lr);
```

**Registration order (`lm_adam_create`).** Walk `model` deterministically so save/load and tests are stable:

1. `embed->table->data[0 .. V*d-1]`
2. For each block `ℓ = 0 .. L-1`: `wq, wk, wv, wo` row-major; `w1, b1, w2, b2`; `ln1 gamma, beta`; `ln2 gamma, beta`
3. `w_head`, `b_head`

Each scalar gets `slot.param = &tensor->data[i]`, `slot.grad = &matching_d_tensor->data[i]`, `m=v=0`.

**Update (same as `adam_step` on net).** For each slot at time `t`:

\[
m \leftarrow \beta_1 m + (1-\beta_1) g,\quad
v \leftarrow \beta_2 v + (1-\beta_2) g^2
\]

\[
\hat{m} = \frac{m}{1-\beta_1^t},\quad
\hat{v} = \frac{v}{1-\beta_2^t},\quad
\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
\]

Then **zero `grad`** (or expect `lm_zero_grads` before next backward—pick one convention and stick to it; net code clears in `adam_step`).

**Algorithm.**

```text
lm_adam_create(model):
  n = count_params(model)
  allocate slots[n]
  idx = 0
  for each param scalar p in registration order:
    slots[idx++] = { &p, &grad(p), 0, 0 }

lm_adam_step(opt, lr):
  opt->t += 1
  for each slot: adam_update_scalar(slot, lr, opt->t, beta1, beta2, eps)
  optionally zero all grads
```

## Complexity / numerical pitfalls / design choices (CS depth)

**Slot count.** ~40k scalars at default config—**O(n_slots)** per step, fine on CPU; no need for fused CUDA kernels here.

**Bias correction early steps.** Small `t` inflates steps; first 10–100 updates behave larger—warmup lr optional for production LMs; toy runs often ignore.

**Weight decay.** Not in baseline `adam_step`; add decoupled WD later if regularizing large `d_model`.

**Shared pointers.** Weight tying would register **one** slot per shared scalar; double-registration would double-update.

**Grad staleness.** If backward forgets to fill a grad, Adam still moves weight using old/zero grad—assert `lm_check_finite_grads` in debug.

**Separate from `adam_create(network *)`.** Do not mix LM mats into `network` walker; keep **`lm_adam_*`** in `lang/lm_adam.c` to avoid layer/neuron coupling.

## How it shows up in this repo

- Today: [`net/optim.h`](../net/optim.h), [`net/optim.c`](../net/optim.c) — reference implementation.
- LM training today: manual lr SGD in [`lang/lm.c`](../lang/lm.c) — **no Adam**.
- Planned: `lang/lm_adam.c`, included from `lm_train_corpus` ([train-all-parameters.md](train-all-parameters.md)).
- Checkpoints: persist `m`, `v`, `t` per slot ([checkpoint-and-resume.md](checkpoint-and-resume.md)).

## Worked example

**One scalar weight** \(w = 0.01\), grad \(g = 0.1\), first step, defaults \(\beta_1=0.9\), \(\beta_2=0.999\), \(\eta=0.001\), \(\epsilon=10^{-8}\).

- \(m = 0.01\), \(v = 0.001\)
- \(\hat{m} = 0.01/(1-0.9) = 0.1\), \(\hat{v} = 0.001/(1-0.999) = 1\)
- \(\Delta w = 0.001 \times 0.1 / (1 + \epsilon) \approx 0.0001\)

Apply across all slots in parallel after one full backward on a 32-token window.

## What next

- Implement `lm_adam_create` / `lm_adam_step` mirroring `optim.c` math in double.
- Unit test: optimize \(w\) on scalar \( (w-3)^2 \) to near zero in <100 steps.
- Wire into `lm_train_corpus` and serialize with LM checkpoints.
