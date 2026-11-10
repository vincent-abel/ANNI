# Packed batches: B windows in one forward

## Abstract

A microbatch packs `B` windows of `T` tokens into one `(B*T) x D` matrix. The embedding, LayerNorm, FFN and head run once over all `B*T` rows, as fat GEMMs. Only attention knows about sequence boundaries: it stays causal inside each window, and one fused launch covers every `(window, head)` pair.

## What problem it solves (plain language)

One window at a time means thousands of tiny GPU launches per step, each too small to fill the card. The GPU spends its time waiting for the host instead of multiplying.

## How it works for a human (intuition, analogy)

Sixteen pages laid side by side on one big table. Every operation that treats a token on its own (normalize, FFN, score the vocabulary) just sees a longer list of tokens. Attention is the only step that reads across tokens, so it is the only one that needs to know where a page ends.

## Formal view (definitions, formulas, shapes, algorithms)

- **Layout:** `ids` has `B*T` entries; row `b*T + t` of every activation is token `t` of window `b`.
- **Positions** restart at 0 for each window.
- **Per-token ops** (embed, LN, QKV projection, FFN, head): one launch over `B*T` rows.
- **Attention:** `B*H` independent causal problems of size `T x T`; row `i` of window `b` only sees rows `0..i` of the same window.
- **Loss:** rows `t = 0..T-2` of every window predict token `t+1`. The last row of each window has no target and gets a zero gradient.
- **Step:** gradients of `K` microbatches (`--grad-accum K`) are summed, then one Adam step scales by `1 / (windows * (T-1))`, so the update is a mean over predicted tokens.

## Complexity / numerical pitfalls / design choices

- **Memory grows with `B`, not with `K`.** Activations cost `O(B*T*D)` per block; accumulation reuses the same buffers. Big effective batch on a small card: raise `K`.
- **Attention is `O(B*H*T^2)`** in time. The fused FA2 path never stores the `T x T` scores, only one log-sum-exp per row.
- **No leak between windows.** A GEMM cannot leak (rows are independent); attention is the only place it could happen, and each launch is indexed by `(b, h)`.
- **Fallback chain.** The forward picks the best path that accepts the shape and records it in the cache; the backward follows the same choice.

## How it shows up in this repo

- [`lang/lm.c`](../lang/lm.c): `lm_cache_create_packed(B, T)` allocates `(B*T) x D` device buffers; `lm_forward_cache_packed` embeds all ids once and adds positions with `gpu_posenc_add_packed`. `B == 1` uses the plain `lm_forward_cache`.
- [`lang/attn.c`](../lang/attn.c): `attn_mha_forward_packed` tries, in order, FA2 tensor-core (`fused = 5`), strided tensor-core (`3`), tiled flash (`4`), fused causal with stored weights (`1`), then the host loop (`0`). `attn_mha_backward_packed` branches on the same value.
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_attn_fa_fwd`, `gpu_attn_flash_fwd`, `gpu_attn_causal_fwd` and their backward twins.
- [`lang/lm_train.c`](../lang/lm_train.c): the microbatch loop, `lm_train_set_grad_accum`, and `--tokens-per-step` which derives `K` from `B*T`.
- CLI: `--batch N|auto`, `--grad-accum K` (in [`main.c`](../main.c)). `auto` sizes `B` from free VRAM, see [Huge batch and VRAM](huge-batch-vram.md).

Related: [Mini-batch corpus training](mini-batch-corpus-training.md), [Variable-length packing](variable-length-packing.md), [FA2 attention](attention-fa2-mma.md), [GPU attention backward](gpu-attention-backward.md).

## Worked example

Default build (`D = 1024`, 16 heads, FF 4096) with `--batch 16 --window 256`: one microbatch is 4096 rows. Every FFN GEMM is `4096 x 1024` times `1024 x 4096`, attention is `16 * 16 = 256` causal problems of `256 x 256` in one launch, and the loss covers `16 * 255` predicted tokens. With `--grad-accum 4`, one Adam step averages over `64 * 255` of them at the memory cost of 16 windows.
