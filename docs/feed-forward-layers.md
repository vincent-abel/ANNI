# Feed-forward layers

## Abstract

The **position-wise feed-forward network (FFN)** applies the same two-layer MLP with **ReLU** to **each token row independently**, expanding to `LM_D_FF` and projecting back to `LM_D_MODEL`. It follows attention in every [transformer block](transformer-blocks.md) under pre-norm + residual.

## What problem it solves (plain language)

Attention mixes information **across tokens** but is largely linear mixing plus softmax weights. The FFN adds **depth and nonlinearity per position**—a “local thinking step” so each token can recombine its features after listening to others. Without it, blocks would struggle to approximate rich per-token transforms.

## How it works for a human (intuition, analogy)

After a group meeting (attention), each person goes through the **same private two-room workshop**: first room has more knobs (`64` vs `32`) to try combinations; ReLU keeps only positive blends; second room compresses back to the usual `32` coordinates. Neighbors do not enter each other’s workshop—only the **same blueprint** (`W1`, `W2`, biases) is shared.

## Formal view (definitions, formulas, shapes, algorithms)

**Config:** `LM_D_MODEL = 32`, `LM_D_FF = 64`.

**Input / output:** `x` is `T × d_model`; `ffn_forward` writes MLP output back (in-place on the normalized buffer passed from the block).

**Parameters:**

| Tensor | Shape | Role |
|--------|-------|------|
| `W1` | `d_model × d_ff` | Up-project |
| `b1` | `d_ff` | Bias |
| `W2` | `d_ff × d_model` | Down-project |
| `b2` | `d_model` | Bias |

**Per row** \(x_t \in \mathbb{R}^{d_{\text{model}}}\):

\[
h_t = \operatorname{ReLU}(x_t W_1 + b_1) \in \mathbb{R}^{d_{\text{ff}}},\quad
y_t = h_t W_2 + b_2 \in \mathbb{R}^{d_{\text{model}}}
\]

**Batch algorithm (`ffn_forward`).**

1. `hidden = x @ W1` → `T × d_ff`; add `b1`; ReLU element-wise.
2. `out = hidden @ W2` → `T × d_model`; add `b2`; copy to `x`.

**In block:** \(Y = U + \mathrm{FFN}(\mathrm{LN}(U))\) after attention residual ([residual-connections.md](residual-connections.md)).

## Complexity / numerical pitfalls / design choices (CS depth)

**FLOPs.** Two GEMMs: **\(O(T \cdot d_{\text{model}} \cdot d_{\text{ff}})\)**—often most of a layer’s multiply count when \(d_{\text{ff}} \approx 2\text{–}4 \times d_{\text{model}}\).

**ReLU.** \(\max(0,z)\) can zero whole hidden units (“dead ReLU”); subsequent LayerNorm and depth usually recover expressivity in small models.

**Init.** Weights uniform in \([-0.02, 0.02]\); biases zero—matches attention/embed spirit.

**Variants (not implemented).** GELU (GPT-2/BERT), SwiGLU (LLaMA), MoE experts—increase quality or capacity at cost of code complexity.

**Position-wise.** No cross-\(T\) term; all sequence interaction remains in MHA.

## How it shows up in this repo

- [`lang/ffn.h`](../lang/ffn.h) / [`lang/ffn.c`](../lang/ffn.c): `ffn_create`, `ffn_free`, `ffn_forward`.
- [`lang/lm_config.h`](../lang/lm_config.h): `LM_D_MODEL`, `LM_D_FF`.
- [`lang/block.c`](../lang/block.c): second sub-layer after `ln2`.
- Linear algebra: [`num/mat.h`](../num/mat.h) `mat_mul`, [`num/vec.h`](../num/vec.h) biases.

## Tiny worked example

Toy: `T = 4`, `d_model = 32`, `d_ff = 64`.

| Stage | Shape |
|-------|-------|
| Input row batch `x` | `4 × 32` |
| After `x @ W1 + b1` | `4 × 64` |
| After ReLU | `4 × 64` |
| After `@ W2 + b2` | `4 × 32` |

One scalar row: if \(x_t W_1 + b_1\) has negative entries, ReLU zeros them before the down-project.

## What comes next

Stack with MHA inside [transformer-blocks.md](transformer-blocks.md); train end-to-end via [next-token-prediction.md](next-token-prediction.md).
