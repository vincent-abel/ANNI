# Transformer backpropagation

## Abstract

Training the toy decoder LM requires **reverse-mode automatic differentiation by hand**: a forward pass that **caches activations**, then a backward pass that turns **cross-entropy on the LM head** into gradients on every weight matrix—embeddings, Q/K/V/O projections, FFN layers, LayerNorm γ/β, and the head itself. This note walks the full chain for **causal multi-head attention** and **pre-norm residual blocks** as implemented in [`lang/block.c`](../lang/block.c), and specifies the planned `lm_forward_cache` / `lm_backward` API that wires those module backward functions together.

## What problem it solves (plain language)

Forward propagation only answers “what logits would we get?” Learning needs “which direction should each weight move to make the correct next byte more likely?” Without backward through the Transformer body, training can only nudge the **LM head** (and perhaps a crude embedding hack); attention and FFN weights stay random, so the model never learns **context**. Full backprop distributes the blame from the wrong predicted token back through causal attention (where each position only talks to the past), through FFN ReLU gates, through LayerNorm rescaling, and into the token embedding table.

## How it works for a human (intuition, analogy)

Imagine the forward pass as water flowing downhill through pipes (matmuls) and valves (softmax, ReLU, LayerNorm). The loss at the output is a **pressure reading** at the faucet. Backprop sends that pressure **upstream**: at each junction you ask, “if I wiggle this valve slightly, how much does the faucet pressure change?” You multiply local sensitivities along the path (the chain rule). **Caches** are snapshots of valve positions from the forward pass—you need them because ReLU “on/off” and softmax weights depend on what actually happened. **Causal attention** is a one-way street in time: position 5’s error can flow into keys/values at positions ≤ 5, but not into the future. **Pre-norm residuals** split the stream: part of the gradient goes through the sub-layer (MHA or FFN), part **skips** straight through the add (`resid_add`) unchanged.

## Formal view (definitions, formulas, shapes, algorithms)

**Notation.** Sequence length \(T\), model width \(d =\) `LM_D_MODEL` (32), vocab \(V =\) `LM_BYTE_VOCAB` (256), heads \(H =\) `LM_N_HEADS` (4), head dim \(d_h = d/H\) (8), FFN width \(d_{\text{ff}} =\) `LM_D_FF` (64), blocks \(L =\) `LM_N_BLOCKS` (2).

**Forward (with cache).** Input token ids \(x_0,\ldots,x_{T-1}\).

1. **Embed + pos enc:** \(X^{(0)} \in \mathbb{R}^{T \times d}\) — cache `X0` (post posenc).
2. For each block \(\ell = 0..L-1\):
   - **Pre-norm MHA branch:** cache residual input \(R_1 = X^{(\ell)}\); LN1 → \(\tilde{X}_1\); MHA → \(A_1\); \(X \leftarrow R_1 + A_1\). Cache \(\tilde{X}_1\), attention internals (Q,K,V, scores, probs), and \(A_1\).
   - **Pre-norm FFN branch:** cache \(R_2 = X\); LN2 → \(\tilde{X}_2\); FFN → \(F\); \(X \leftarrow R_2 + F\). Cache \(\tilde{X}_2\), pre-ReLU hidden \(H_{\text{ff}} \in \mathbb{R}^{T \times d_{\text{ff}}}\), and \(F\).
3. Final hidden \(H = X^{(L)} \in \mathbb{R}^{T \times d}\).

**Loss (teacher forcing).** For each \(t = 0..T-2\), logits \(\mathbf{z}_t = H_{t,:} W_{\text{head}} + \mathbf{b}\), target \(y_t = x_{t+1}\):

\[
\mathcal{L} = \frac{1}{T-1}\sum_{t=0}^{T-2} \mathrm{CE}(\mathbf{z}_t, y_t)
\]

**Head backward (softmax + CE).** Let \(\mathbf{p}_t = \mathrm{softmax}(\mathbf{z}_t)\). Gradient w.r.t. logits (standard):

\[
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_t} = \frac{1}{T-1}(\mathbf{p}_t - \mathbf{e}_{y_t})
\]

Call this \(\mathbf{g}_t \in \mathbb{R}^V\). Then:

\[
\frac{\partial \mathcal{L}}{\partial W_{\text{head}}} \mathrel{+}= \sum_t H_{t,:}^\top \mathbf{g}_t,\quad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} \mathrel{+}= \sum_t \mathbf{g}_t,\quad
\frac{\partial \mathcal{L}}{\partial H_{t,:}} \mathrel{+}= \mathbf{g}_t W_{\text{head}}^\top
\]

Accumulate into `d_w_head`, `d_b_head`, and row \(t\) of `d_hidden` (or a full `T×d` buffer).

**Residual backward.** If \(Y = X + \mathrm{Sub}(X)\), then \(\partial \mathcal{L}/\partial X = \partial \mathcal{L}/\partial Y + \partial \mathcal{L}/\partial X\big|_{\text{sub-branch}}\). Pre-norm uses \(Y = R + \mathrm{Sub}(\mathrm{LN}(R))\): gradient w.r.t. \(R\) is **sum of skip path** (same as \(\partial \mathcal{L}/\partial Y\)) **plus** LN-backward into the sub-layer.

**LayerNorm backward (per row).** Forward: \(\hat{x}_j = (x_j - \mu)/s\), \(y_j = \gamma_j \hat{x}_j + \beta_j\), \(s = \sqrt{\sigma^2 + \varepsilon}\). Given \(\partial \mathcal{L}/\partial y_j\), standard formulas accumulate `d_gamma`, `d_beta`, and `d_x` on the LN input row ([layer-normalization.md](layer-normalization.md)).

**FFN backward (per row).** \(h = \mathrm{ReLU}(\tilde{x} W_1 + b_1)\), \(f = h W_2 + b_2\). ReLU mask: \(\partial h / \partial (\tilde{x} W_1)\) zero where pre-activation ≤ 0. Matmuls reverse via transposes: `d_w2`, `d_b2`, `d_h`, then `d_w1`, `d_b1`, `d_x` for LN input.

**Causal self-attention backward.** Forward row \(i\): scores \(S_{ij} = (Q_i \cdot K_j)/\sqrt{d_h}\), masked \(\tilde{S}_{ij} = S_{ij}\) if \(j \le i\) else \(-10^9\), weights \(A_{ij} = \mathrm{softmax}_j(\tilde{S}_{i:})\), output \(O_i = \sum_j A_{ij} V_j\).

Given \(\partial \mathcal{L}/\partial O_i\):

1. **V path:** \(\partial \mathcal{L}/\partial V_j \mathrel{+}= A_{ij}\, \partial \mathcal{L}/\partial O_i\).
2. **A path:** \(\partial \mathcal{L}/\partial A_{ij} = (\partial \mathcal{L}/\partial O_i) \cdot V_j\); softmax backward on row \(i\) (only columns \(j \le i\) participate; masked entries have zero gradient w.r.t. scores).
3. **Q/K path:** \(\partial S_{ij}/\partial Q_i = K_j/\sqrt{d_h}\), etc., accumulate `d_q`, `d_k`.
4. **Projections:** \(Q = X W_q\) ⇒ `d_wq` += \(X^\top d_q\), `d_x` += \(d_q W_q^\top\) (and likewise Wk, Wv, Wo from MHA output projection).

**Multi-head:** split/merge is reshape-only; backward splits `d_out` per head and sums into shared `d_x`.

**Embedding backward.** \(X^{(0)}_{t,:} = E[x_t,:] + \mathrm{PE}_t\). Pos enc is fixed → \(\partial \mathcal{L}/\partial E[k,:] \mathrel{+}= \partial \mathcal{L}/\partial X^{(0)}_{t,:}\) for every position \(t\) where \(x_t = k\) (scatter-add into `d_embed` rows).

**Algorithm (`lm_backward`, planned).**

```text
lm_backward(model, cache, ids, n, d_loss_scale):
  zero all grad buffers on model
  for t in 0..n-2:
    g_z = softmax_ce_grad(logits_t, target ids[t+1]) * d_loss_scale
    accum head grads; dH[t] += g_z @ W_head^T
  d_stream = dH
  for block ell from L-1 down to 0:
    tblock_backward(&d_stream, block[ell], cache->block[ell])
  embed_backward(d_stream, ids, d_embed)
  (posenc: no param grads)
```

**Cache struct (`lm_fwd_cache`, planned).** Holds `X0`, and per-block `{ R1, ln1_out, q,k,v, scores, attn_out, R2, ln2_out, ff_hidden, ff_out }` as `mat`/`vec` views or owned buffers sized for `T ≤ LM_MAX_CTX`.

## Complexity / numerical pitfalls / design choices (CS depth)

**Memory.** Caching full \(T \times T\) attention score and prob matrices per head per layer is **\(O(L H T^2)\)** doubles—acceptable for \(T \le 32\), \(L=2\), \(H=4\) (≈ 32k floats), but not for long context. Production systems often checkpoint or use flash-attention-style recomputation.

**Time.** Backward roughly **2–3× forward** FLOPs for matmul-heavy blocks; attention backward is still \(O(T^2 d)\) in this dense implementation.

**Causal mask.** Treat masked logits as **constants** with zero gradient; do not backprop through the `-1e9` hack into keys at future positions. Softmax normalization runs only over \(j \le i\).

**Large negative mask.** If unmasked scores are also very negative, exp underflow can make softmax backward numerically stiff; keeping \(d_k^{-1/2}\) scaling and reasonable weight init helps.

**Pre-norm vs post-norm.** Pre-norm backward matches forward in [`tblock_forward`](../lang/block.c): LN sits **inside** the branch; residual skip carries full upstream gradient without passing through LN.

**ReLU.** Dead neurons (zero pre-activation) zero out FFN input gradients for that unit; many dead units slow learning—LayerNorm and small init mitigate.

**Accumulation over positions.** Sum CE gradients over \(t\) before one block backward, or backward per-\(t\) and add `d_hidden`—equivalent for mean loss if scaled by \(1/(T-1)\).

**Mixed precision / stability.** This repo uses `double`; softmax CE uses stable log-sum-exp in [`num/ops.c`](../num/ops.c). Backward should reuse the **same** logits or cached \(\mathbf{p}_t\) as forward to avoid drift.

## How it shows up in this repo

**Today (partial training).** [`lang/lm.c`](../lang/lm.c) `lm_train_steps` runs `lm_forward`, computes \(\mathbf{p}_t - \mathbf{e}_y\) on the head, updates `w_head` / `b_head`, and applies a **rough embed update** that treats the Transformer as identity—explicitly **not** full backprop.

**Forward path to differentiate.**

- [`lang/lm.h`](../lang/lm.h): `lm_forward` — embed, `posenc_sinusoidal_add`, `transformer_forward`.
- [`lang/block.c`](../lang/block.c): `tblock_forward` — pre-norm LN → `attn_mha(..., causal=1)` → `resid_add` → LN → `ffn_forward` → `resid_add`.
- [`lang/attn.c`](../lang/attn.c): `attn_mha`, `attn_self` with causal mask.
- [`lang/ffn.c`](../lang/ffn.c), [`lang/layernorm.c`](../lang/layernorm.c): sub-layer forwards.

**Planned module backward (new translation units).**

| Function | Role |
|----------|------|
| `layernorm_backward(mat *dx, ...)` | γ, β, input row grads |
| `ffn_backward(mat *dx, ...)` | w1,b1,w2,b2 |
| `attn_mha_backward(mat *dx, ...)` | wq,wk,wv,wo + causal softmax |
| `tblock_backward(mat *dx, tblock *, tblock_cache *)` | both branches + residuals |
| `transformer_backward(mat *dx, transformer *, tr_cache *)` | loop blocks reverse order |
| `embed_backward(const mat *dx, embed *, const int *ids, int n)` | scatter into table rows |

**Planned LM-level API.**

```c
/* lang/lm.h (planned) */
typedef struct lm_fwd_cache lm_fwd_cache;

int  lm_forward_cache(mat *hidden, lm_fwd_cache *cache,
    const lm *model, const int *ids, int n);
int  lm_backward(lm *model, lm_fwd_cache *cache,
    const int *ids, int n, double loss_scale);
void lm_zero_grads(lm *model);
```

Gradients live on **`mat`/`vec` shadow buffers** (e.g. `d_wq` parallel to `wq`) or a flat `lm_grads` struct; [`train-all-parameters.md`](train-all-parameters.md) describes consuming them in the optimizer.

**Tests (planned).** Finite-difference checks on one block with \(T=3\), comparing scalar loss perturbation to dot product of flat grad vector ([unit-tests.md](unit-tests.md) pattern from XOR net).

## Worked example

**Setup.** \(T=3\), ids `[104, 105, 33]` (`"hi!"`), one block, tiny hand-checked head only first—then one attention link.

**Forward snapshot at \(t=1\).** Context for predicting `'!'` (33) uses rows 0–1. Suppose after block, \(H_{1,:} = [0.1, -0.2, \ldots]\) (32 dims). Head gives logits \(\mathbf{z}_1\); softmax yields \(p_{33} = 0.05\) (low).

**Head step.** Target 33: \(\mathbf{g}_1 = (\mathbf{p}_1 - \mathbf{e}_{33}) / (T-1)\) with \(T-1=2\). Non-zero entries on every class; largest positive on index 33. `d_w_head[j,33] += H_{1,j} * g_1[33]`. `dH[1,j] += sum_v g_1[v] * W_head[j,v]`.

**One residual step (conceptual).** If only FFN branch received `dH[1]`, pre-norm block splits: `dR2[1] = dH[1]` (skip) plus FFN-backward into LN2 input; LN2-backward adds into `dR2[1]` from normalized path; `dR2[1]` becomes `dX[1]` entering MHA branch output; MHA-backward + skip from first residual yields `dX0[1]`; embed row 105 accumulates `dX0[1]` (token at position 1).

**Causal attention at row 2 (indexing 0..2).** When backpropping position 2, keys/values at positions 0,1,2 receive gradients; position >2 N/A. Position 0’s key never receives gradient from row 1’s output if... actually row 1 attends to j≤1 only; row 2’s backward touches K/V at 0,1,2. Row 0’s query only attends to key 0—gradient to K_0 from row 0 only.

This manual trace is what `attn_self_backward` automates for all rows in parallel.

## What next

- Implement **`lm_fwd_cache` allocation** in `lm_forward_cache` mirroring every `mat_create` temp in `tblock_forward`.
- Add **`lm_backward`** and wire **`lm_train_corpus`** to call forward+backward instead of head-only updates ([train-all-parameters.md](train-all-parameters.md)).
- Replace hand SGD with **`lm_adam_step`** over flat slots ([adam-lm-weights.md](adam-lm-weights.md)).
- Add gradient-norm logging and a **finite-diff** test for `tblock_backward` before scaling \(T\) or `LM_N_BLOCKS`.
