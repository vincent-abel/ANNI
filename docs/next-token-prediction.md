# Next-token prediction

## Abstract

Language modeling trains the Transformer to **predict the next byte (token) given all prior bytes** at each position. A **linear LM head** maps hidden vectors to vocabulary logits; **softmax + cross-entropy** scores the true next token. This note covers the objective and a tiny phrase-training scenario for the toy `./ann lm` path.

## What problem it solves (plain language)

The model must turn context into a **probability over what comes next**—the same problem autocomplete solves. Without a explicit head and loss, the stack would be an unlabeled representation; next-token prediction supplies the **supervision signal** that shapes attention and FFN weights into a usable generator.

## How it works for a human (intuition, analogy)

You read a phrase left to right and repeatedly ask, “What letter (byte) would I type next?” At each step you see only what’s already written ([causal attention](causal-attention.md)). Training shows the correct next character; the model is penalized when it assigns low probability to that character. Over many repeats on a short phrase, weights nudge toward copying the pattern.

## Formal view (definitions, formulas, shapes, algorithms)

**Vocabulary.** `LM_BYTE_VOCAB = 256` (raw bytes 0–255); token ids are `unsigned char` values.

**Hidden state.** After `LM_N_BLOCKS` blocks, \(H \in \mathbb{R}^{T \times d}\), \(d = 32\).

**LM head** (weight tying optional; default separate matrix):

\[
\mathbf{z}_t = H_{t,:}\, W_{\text{head}} + \mathbf{b}_{\text{head}},\quad W_{\text{head}} \in \mathbb{R}^{d \times V},\ V = 256
\]

Logits \(\mathbf{z}_t \in \mathbb{R}^V\) for position \(t\).

**Target.** For input sequence \(x_0,\ldots,x_{T-1}\), predict \(x_{t+1}\) at position \(t\) (shift-by-one). Positions without a next token are masked out of the loss.

**Probabilities:** \(\mathbf{p}_t = \mathrm{softmax}(\mathbf{z}_t)\) via `vec_softmax`.

**Loss (per position, cross-entropy):**

\[
\mathcal{L}_t = -\log p_t[x_{t+1}]
\]

Stable from logits: `vec_softmax_cross_entropy(z_t, x_{t+1})` ([cross-entropy.md](cross-entropy.md)).

**Batch loss:** mean over valid \(t\):

\[
\mathcal{L} = \frac{1}{T-1}\sum_{t=0}^{T-2} \mathcal{L}_t
\]

**Training step (`lm_train_step`).**

1. Tokenize phrase → ids length \(T\).
2. `lm_forward` → \(H\) (`T × 32`), causal mask on all blocks.
3. For each \(t\): logits = row \(t\) of \(H\) times `W_head`; CE vs target id \(x_{t+1}\).
4. Backprop (future) or finite-difference smoke test (toy); Adam on all parameters ([adam-optimizer.md](adam-optimizer.md)).

## Complexity / numerical pitfalls / design choices (CS depth)

**Head cost.** \(O(T d V)\) per forward; with \(V=256\), \(d=32\), comparable to one small GEMM per token—cheap vs attention at moderate \(T\), expensive vs \(d\) when \(V\) is 50k+ in large LMs.

**Weight tying.** Sharing embed matrix \(E \in \mathbb{R}^{V \times d}\) with \(W_{\text{head}}^\top\) cuts parameters and often improves generalization; toy code may keep untied for clarity.

**Label smoothing / regularization.** Not in baseline; raw CE can overfit tiny phrases quickly—which is fine for demo overfitting on “hello”.

**Softmax over 256 classes.** Stable max-subtraction is essential if logits drift large during naive training.

**Teacher forcing.** Training uses **ground-truth** prefix tokens as input at every position; generation uses model’s own predictions ([text-generation.md](text-generation.md))—exposure bias is acceptable at toy scale.

## How it shows up in this repo

- [`lang/lm.h`](../lang/lm.h): `lm_head_logits(logits, lm, h_row)`; `lm_forward(lm, token_ids, T)`; `lm_train(lm, text, steps, lr)`.
- [`lang/lm_head.h`](../lang/lm_head.h): `W` shape `32 × 256`, `b` length 256.
- [`lang/embed.h`](../lang/embed.h): byte lookup → `T × 32`; plus [`lang/posenc.h`](../lang/posenc.h).
- Loss: [`num/ops.h`](../num/ops.h) `vec_softmax`, `vec_softmax_cross_entropy`.
- CLI (planned): `./ann lm train --phrase "hello world"` runs short optimization loop.

## Tiny worked example

Phrase `"hi"` → bytes `[104, 105]`, \(T = 2\).

| Position \(t\) | Context (causal) | Predict | Target next byte |
|----------------|------------------|---------|------------------|
| 0 | `104` | logits \( \mathbf{z}_0 \) | `105` |
| 1 | `104,105` | \(\mathbf{z}_1\) | (none if end—often pad or stop) |

For a 3-byte phrase `"hey"` (`104,101,121`), valid targets at \(t=0,1\) are `101` and `121`. Suppose after one forward, \(\mathbf{z}_0\) gives max at wrong class; CE = \(-\log p(101)\). Repeating steps on the same phrase drives loss down as the 2-block, 32-d model memorizes.

Shapes: \(H\) is `3 × 32`; each \(\mathbf{z}_t\) is length `256`.

## What comes next

[Text generation](text-generation.md): greedy argmax or temperature sampling from \(\mathbf{z}_T\) at the final position; serialize weights ([model-serialization.md](model-serialization.md)) after training.
