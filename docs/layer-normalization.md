# Layer normalization

## Abstract

**LayerNorm** normalizes each token’s feature vector to approximately **zero mean and unit variance**, then applies learnable **scale γ** and **shift β** per dimension. In this LM, it runs **before** each sub-layer (pre-norm): MHA and FFN always receive stabilized rows of the `T × LM_D_MODEL` stream.

## What problem it solves (plain language)

Deep stacks multiply matrices and squash activations; different tokens and layers can end up with wildly different numeric ranges. The next sub-layer’s weights expect reasonable inputs. LayerNorm **re-centers and rescales each token on its own** so training does not depend on batch neighbors or absolute magnitude drift.

## How it works for a human (intuition, analogy)

Each token carries **32 dials** (features). LayerNorm reads only that token’s dials: find their average, measure spread, convert to “typical z-scores,” then **γ** stretches and **β** shifts each dial back to what the network finds useful. The token at position 7 does not change how position 3 is normalized—unlike batch norm across examples.

## Formal view (definitions, formulas, shapes, algorithms)

**Input:** matrix `x`, shape `T × d`, `d = LM_D_MODEL = 32`.

**Parameters:** vectors `gamma`, `beta` length `d`; scalar `eps = 1e-5`.

**Per row** \(x_t \in \mathbb{R}^d\):

\[
\mu_t = \frac{1}{d}\sum_{j=1}^{d} x_{t,j},\quad
\sigma_t^2 = \frac{1}{d}\sum_{j=1}^{d} (x_{t,j} - \mu_t)^2
\]

\[
\hat{x}_{t,j} = \frac{x_{t,j} - \mu_t}{\sqrt{\sigma_t^2 + \varepsilon}},\quad
y_{t,j} = \gamma_j \hat{x}_{t,j} + \beta_j
\]

**Output:** same shape as `x`; `layernorm_forward` overwrites in place.

**Pre-norm placement in block:**

\[
X' = X + F(\mathrm{LN}(X))
\]

Two separate `layernorm` instances per block: `ln1` before MHA, `ln2` before FFN ([transformer-blocks.md](transformer-blocks.md)).

**Algorithm (`layernorm_forward`).** For each row: compute μ, σ² in one pass; second pass write normalized scaled row.

## Complexity / numerical pitfalls / design choices (CS depth)

**Time / space.** **\(O(T \cdot d)\)** per call; no auxiliary `T×T` structure.

**ε.** Prevents division by zero when a row is constant (padding, dead ReLU); `1e-5` is Transformer-default territory.

**Pre-LN vs post-LN.** Pre-norm (this repo) applies LN **inside** the residual branch; often smoother optimization for deep stacks (Xiong et al., 2020). Post-norm: `LN(X + F(X))` as in the original diagram.

**RMSNorm alternative.** Skip mean centering, divide by RMS only—used in LLaMA; not in this C implementation.

**BatchNorm contrast.** BatchNorm mixes batch dimension—awkward for variable-length LM batches; LayerNorm is per-token standard in NLP.

**γ, β init.** γ = 1, β = 0 → start as pure normalize; network learns departures.

## How it shows up in this repo

- [`lang/layernorm.h`](../lang/layernorm.h) / [`lang/layernorm.c`](../lang/layernorm.c): `layernorm_create`, `layernorm_free`, `layernorm_forward`.
- [`lang/block.h`](../lang/block.h): `ln1`, `ln2` per block; `LM_N_BLOCKS = 2` → four LayerNorm modules total in the stack body.
- [`lang/lm_config.h`](../lang/lm_config.h): `LM_D_MODEL` sets vector width.

## Tiny worked example

Row \(x = [1, 2, 3]\), \(d = 3\): μ = 2, σ² = 2/3, normalized ≈ \([-1.22, 0, 1.22]\). With γ = 1, β = 0, output equals that. At full width \(d = 32\), the same formulas apply to each token row after embedding or residual add.

If γ₂ = 2, dimension 2’s output doubles after normalization—learned per-feature gain.

## What comes next

Feed normalized rows into [multi-head-attention.md](multi-head-attention.md) and [feed-forward-layers.md](feed-forward-layers.md); read [residual-connections.md](residual-connections.md) for where LN sits in the add path.
