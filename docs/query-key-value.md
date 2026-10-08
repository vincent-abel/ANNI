# Query / Key / Value

## Abstract

Self-attention compares positions using three linear views of the same input: queries, keys, and values. This module stores \(W_Q, W_K, W_V\) and projects \(X \mapsto (Q,K,V)\).

## What problem it solves

If every token only mixed through a fixed matrix, the model could not decide “who to listen to” depending on content. Q/K/V give each position a question (Q), a label (K), and a payload (V).

## How it works for a human

Imagine sticky notes on words. Each word writes a question, a filing label, and a message. Later, questions are matched against labels to decide whose messages to read.

## Formal view

For \(X \in \mathbb{R}^{T \times d}\):

\[
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
\]

with \(W_* \in \mathbb{R}^{d \times d}\). Here \(d =\) `LM_D_MODEL`.

Complexity: three GEMMs, \(O(T d^2)\) each.

## Complexity / pitfalls / design choices

- Full \(d \times d\) projections before splitting heads (common GPT-style layout).
- Small Gaussian-like init (\(\pm 0.02\)) to keep early attention soft.

## How it shows up in this repo

- [`lang/attn.h`](../lang/attn.h) / [`lang/attn.c`](../lang/attn.c): `attn_create`, `attn_qkv_project`

## Tiny worked example

\(T=2,d=32\): `attn_qkv_project` fills three `2×32` matrices from \(X\).

## What comes next

Scaled dot-product self-attention mixes V using scores from \(QK^\top\).
