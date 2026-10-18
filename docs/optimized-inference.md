# Optimized inference

## Abstract

**Autoregressive decode** must not rebuild the whole prefix Transformer on every new token. A **KV cache** stores each layer’s Keys and Values as the sequence grows; each step embeds only the new token, projects Q/K/V for that row, appends K/V, and attends the new query against the cache. `lm_generate` prefers `lm_generate_kv` and falls back to full reforward if needed.

## What problem it solves (plain language)

Old generate called a full forward of length \(n\), then \(n+1\), … — quadratic work in the number of new tokens. Caching past keys and values makes each new token roughly **linear** in current length for attention and **constant** for FFN/LN on the last row.

## How it works for a human (intuition, analogy)

Reading a book chapter by chapter, you **bookmark** what you already understood (K/V) and only read the next page carefully (new Q), instead of rereading the whole chapter every time.

## Formal view (definitions, formulas, shapes, algorithms)

**Prefill / step.**

```text
for pos in 0 .. prompt_len-1:
  x = embed(id) + posenc(pos)
  for each block:
    x = x + Attn(LN(x); append K,V to cache[block])
    x = x + FFN(LN(x))
for each new token:
  logits = Head(x); sample id
  decode one more pos into x using the same caches
```

**Attention step (one head).**

\[
s_j = \frac{q\cdot k_j}{\sqrt{d}},\quad
a=\mathrm{softmax}(s_{0:\mathrm{len}}),\quad
o=\sum_j a_j v_j
\]

## Complexity / numerical pitfalls / design choices

**Prefill** still walks the prompt one token at a time (simple, correct). A batched prefill GEMM remains future work.

**Fallback.** `lm_generate` goes through `lm_generate_stream`, which tries the device decoder (`lm_decoder_*`, one CUDA graph per token) first; on CPU or failure it uses the original full-prefix loop.

**Training** unchanged — KV path is inference-only.

## How it shows up in this repo

- [`lang/decode.h`](../lang/decode.h) / [`lang/decode.c`](../lang/decode.c): `lm_decoder`, prefill then device steps (`gpu_dec_*` in `num/gpu_cuda.cu`)
- [`lang/generate.c`](../lang/generate.c): prefers the device decoder
- [`lang/embed.c`](../lang/embed.c): `embed_lookup_row`
- [`lang/posenc.c`](../lang/posenc.c): `posenc_add_row`

## Worked example

`./ann lm generate --prompt hello --tokens 8` runs KV decode and prints a continuation. Bench of many tokens shows much less than \(O(T^2)\) growth versus naive reforward.

## What next

[GPU support](gpu-support.md) optionally moves large GEMMs to CUDA while CPU remains the default.


## E13 batch generate

`lm_generate_batch` runs KV/full generate per prompt for concurrent chats.
