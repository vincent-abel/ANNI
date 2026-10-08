# Causal (autoregressive) attention

## Abstract

Causal attention ensures token \(i\) **only attends to positions \(j \le i\)**—no peeking at future tokens. Implementation: before row-wise softmax, set scores with \(j > i\) to a large negative value (\(-10^9\)), so those weights become effectively zero.

## What problem it solves (plain language)

When training a **next-token predictor**, the model must not use words it has not “seen” yet at that position. If position 5 could read position 10, the task becomes trivial and generation at inference (where the future does not exist) would not match training. Causal masking enforces **left-to-right** information flow.

## How it works for a human (intuition, analogy)

Reading a sentence under a **blindfold that hides words to your right**: at each word, you may look back at everything already read, never forward. The attention table for “who can look at whom?” is lower-triangular—allowed cells on and below the diagonal, forbidden above.

## Formal view (definitions, formulas, shapes, algorithms)

**Score matrix** \(S \in \mathbb{R}^{T \times T}\) with entries \(S_{ij} = \frac{1}{\sqrt{d_k}} Q_i \cdot K_j\).

**Causal mask** \(M_{ij}\):

\[
M_{ij} = \begin{cases}
0 & j \le i \\
-\infty \text{ (approx } -10^9 \text{ in code)} & j > i
\end{cases}
\]

**Masked scores:** \(\tilde{S}_{ij} = S_{ij} + M_{ij}\) (only future entries changed).

**Weights:** \(A_{ij} = \mathrm{softmax}_j(\tilde{S}_{i:})\). For \(j > i\), \(\tilde{S}_{ij}\) is hugely negative → \(A_{ij} \approx 0\). For \(j \le i\), normalization is over **only past and current** keys.

**Output** unchanged in form: \(\mathrm{Out}_i = \sum_j A_{ij} V_j\).

**In `attn_self` (this repo).**

```text
for each row i:
  for each column j:
    score = dot(Q[i], K[j]) / sqrt(d)
    if (causal && j > i) score = -1e9
  softmax row of scores
  Out[i] = weighted sum of V rows
```

**Flag.** `attn_self(..., causal)` and `attn_mha(..., causal)` take `int causal`; non-zero enables masking. Training and generation for `./ann lm` use `causal = 1`.

**Context limit.** Sequence length \(T \le\) `LM_MAX_CTX` (32); mask is always \(T \times T\) dense in this educational implementation.

## Complexity / numerical pitfalls / design choices (CS depth)

**Same asymptotics** as unmasked attention: \(O(T^2 d_h)\) per head; half the softmax mass is structurally zero but loops still visit all \(j\).

**Mask constant.** \(-10^9\) is finite `double`; after subtracting row max in stable softmax, masked entries stay negligible. Using `-INFINITY` can interact badly with some softmax implementations; large negative logits are a safe toy choice.

**Bidirectional vs causal.** Encoder-only models (BERT) use no mask; decoder-only LMs (GPT-style) use causal; encoder–decoder uses causal in the decoder and often unmasked cross-attention to encoder memory.

**KV-cache (inference).** At generation time, only the new query row is computed; cached \(K,V\) from past tokens avoid rebuilding full \(T \times T\) scores—planned optimization for `./ann lm`, not required for correctness.

**Padding.** If batches include pad tokens, additional mask bits zero out pad keys; byte-level toy LM here often uses exact-length sequences.

## How it shows up in this repo

- [`lang/attn.c`](../lang/attn.c): mask applied inside `attn_self` when `causal` is true.
- [`lang/attn_mha`](../lang/attn.h): passes the same flag to every head’s `attn_self`.
- [`lang/lm.c`](../lang/lm.c) (planned): `lm_forward` / `lm_train` invoke `attn_mha(..., 1)` for all `LM_N_BLOCKS` layers.

See also [self-attention.md](self-attention.md) for the unmasked math and [text-generation.md](text-generation.md) for left-to-right decoding.

## Tiny worked example

\(T = 4\), causal mask on scores before softmax (× = masked, values schematic):

| \(i \setminus j\) | 0 | 1 | 2 | 3 |
|------------------|---|---|---|---|
| 0 | ✓ | × | × | × |
| 1 | ✓ | ✓ | × | × |
| 2 | ✓ | ✓ | ✓ | × |
| 3 | ✓ | ✓ | ✓ | ✓ |

Row 0’s weights are softmax over **one** key (only \(j=0\)) → always weight 1 on position 0. Row 2 normalizes over \(j \in \{0,1,2\}\). Output at position 2 cannot depend on \(V_3\).

With `LM_MAX_CTX = 32`, a 10-token prompt uses a \(10 \times 10\) active mask embedded in the upper-left of the conceptual max-size table.

## What comes next

[Next-token prediction](next-token-prediction.md) trains the stack with causal features; [text-generation.md](text-generation.md) appends one token at a time using the same mask at growing \(T\).
