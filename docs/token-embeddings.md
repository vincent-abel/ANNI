# Token embeddings

## Abstract

Token embeddings turn integer token ids into fixed-size real vectors. The model stores one learned vector per vocabulary entry in a table \(E \in \mathbb{R}^{V \times d}\) and gathers rows for each position in a sequence. This is the first numeric representation the decoder stack sees after tokenization.

## What problem it solves

Neural layers operate on continuous vectors, not on opaque integers. A vocabulary id says *which* symbol we mean; an embedding says *how that symbol is represented* in \(d_{\text{model}}\)-dimensional space so attention, feed-forward blocks, and the output head can mix and transform it. Without embeddings, there is no shared geometry between tokens: similar words cannot sit near each other, and gradients have nowhere to live for discrete symbols.

## How it works for a human

Imagine a thick book with one row per word in your dictionary. Each row is a list of \(d_{\text{model}}\) small numbers (here 32 for the toy LM). When you read a sentence, you look up each word’s row and stack those lists into a mini spreadsheet: one row per token, one column per feature. The network never sees the word “hello” as letters at this stage—it sees that row of numbers. Training nudges those numbers so useful patterns (syntax, co-occurrence, position-relative roles) can emerge in later layers.

## Formal view

Let \(V\) be vocabulary size and \(d = d_{\text{model}} = \texttt{LM\_D\_MODEL}\). The embedding matrix is

\[
E \in \mathbb{R}^{V \times d}, \quad E_{v,:} \in \mathbb{R}^d \text{ is the vector for token id } v.
\]

Given a length-\(n\) sequence of ids \(\text{ids} = (i_0, \ldots, i_{n-1})\) with each \(i_k \in \{0,\ldots,V-1\}\), the embedded sequence \(X \in \mathbb{R}^{n \times d}\) is row-wise lookup:

\[
X_{k,:} = E_{i_k,:} \quad \text{for } k = 0,\ldots,n-1.
\]

Equivalently, if \(O \in \{0,1\}^{n \times V}\) is a one-hot stack with \(O_{k,i_k} = 1\), then \(X = O E\). In code, `embed_lookup` materializes \(X\) directly without forming \(O\).

Initialization uses small random weights (approximately zero mean, scale 0.02) so early activations stay bounded before layer norm and residual paths stabilize training.

## Complexity / pitfalls / design choices

- **Time:** One lookup costs \(O(d)\) per token; a sequence of length \(n\) costs \(O(n d)\). No matmul with a full one-hot matrix is required.
- **Space:** The table alone is \(O(V d)\) doubles; for byte vocab \(V \approx 260\) and \(d=32\), this is tiny.
- **Bounds:** Invalid ids must be rejected; out-of-range indices would read wrong rows or crash if unchecked.
- **Alignment:** Output matrix rows must match sequence length \(n\); columns must equal `LM_D_MODEL` so downstream blocks can consume `out` without reshaping.
- **Shared table:** Input embeddings and the final logits layer often tie weights in full LMs; this repo keeps a single table module first; tying can come later.

## How it shows up in this repo

- [`lang/embed.h`](../lang/embed.h) / [`lang/embed.c`](../lang/embed.c): `embed_create`, `embed_free`, `embed_lookup`
- [`lang/lm_config.h`](../lang/lm_config.h): `LM_D_MODEL` sets embedding width
- [`lang/vocab.h`](../lang/vocab.h): `vocab_size` defines \(V\) when creating the table (e.g. 260 for byte vocab)
- [`num/mat.h`](../num/mat.h): `table` and lookup output are `mat` objects (row-major storage)

## Tiny worked example

Suppose \(V=5\), \(d=3\), and after training (numbers illustrative):

\[
E = \begin{bmatrix}
0.01 & -0.02 & 0.00 \\
0.05 & 0.01 & -0.01 \\
-0.01 & 0.03 & 0.02 \\
0.00 & 0.00 & 0.01 \\
0.02 & -0.01 & 0.04
\end{bmatrix}
\]

Ids `[1, 4, 2]` yield

\[
X = \begin{bmatrix}
E_{1,:} \\
E_{4,:} \\
E_{2,:}
\end{bmatrix}
=
\begin{bmatrix}
0.05 & 0.01 & -0.01 \\
0.02 & -0.01 & 0.04 \\
-0.01 & 0.03 & 0.02
\end{bmatrix}.
\]

In this project, `embed_create(vocab_size(v))` allocates `table` with `vocab_size` rows and `LM_D_MODEL` columns; `embed_lookup(out, e, ids, 3)` fills `out` with those three rows.

## What comes next

Positional encoding (or rotary embeddings) adds order information to \(X\). Stacked transformer blocks then map \(X\) through self-attention and feed-forward layers; the LM head maps final hidden states back to logits over \(V\) classes.
