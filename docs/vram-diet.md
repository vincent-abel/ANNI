# VRAM diet: buffers that never needed to exist

## Abstract

The training step at the 185M shape (batch 16, window 256) peaked at 13.5 GB on a 16 GB card. A series of small changes brought it to about 9 GB with the same numbers coming out. None of them is a compression trick: each one removes a buffer that was allocated but never read, or kept twice, or cheap to recompute.

## What problem it solves (plain language)

VRAM is the one resource that fails hard: one allocation too many and the step dies. Sitting at 87% of the card left no room for the validation probe, a bigger model, a longer window, or anything else running on the machine. Memory that is written and never read also costs time, since writing it is traffic.

## How it works for a human (intuition, analogy)

Clearing a workshop. Some boxes were never opened (bought "in case"), some hold a second copy of a tool that hangs on the wall, and some hold things quicker to cut again than to store.

## Formal view (definitions, formulas, shapes, algorithms)

Sizes are per layer at `B*T = 4096`, `D = 1024`, `FF = 4096`: a `[4096][1024]` FP32 buffer is 16 MB, a `[4096][4096]` one is 64 MB.

| What went | How | Saved per layer |
|---|---|---|
| FFN pre-activation | allocated only when a fallback path asks | 64 MB |
| FFN hidden in FP32 | the GEMM writes BF16 only | 32 MB |
| `ffn_out`, `attn_out`, FP32 K and V, `dtmp` | lazy: headers without storage until a path really writes them | several x 16 MB |
| Copies of the attention and FFN inputs | pointer to the LayerNorm output that is already there | 2 x 16 MB |
| LayerNorm `x_hat` | recomputed in the backward from the input | 2 x 16 MB |
| FP32 `dK`, `dV` | the backward kernel writes only their BF16 twins | 2 x 16 MB of stores |
| Head gradient | BF16, in place over the logits | a `rows x vocab` buffer, once |

Peak VRAM of the same gate run, as recorded in [BF16 outputs](bf16-direct-outputs.md): 13485 MiB, then 12604 (candidate 12), 9934 (candidate 18), 9047 (candidate 20).

## Complexity / numerical pitfalls / design choices

- **Lazy means the fallback still works.** A path that needs a dropped buffer allocates it on first use. `ANN_VRAM_LAZY=0` allocates everything up front as before.
- **An alias needs an owner.** The attention cache points at the block's LayerNorm output; freeing the cache must not free it (`x_in_alias`).
- **A BF16-only tensor has no FP32 behind it.** Code that would read the FP32 hidden layer gets an error, not stale memory ([BF16 twins](bf16-twins.md)).
- **Recomputing must be exact.** `x_hat` is rebuilt with the forward's own expression so nothing downstream changes ([LayerNorm in one pass](layernorm-one-pass.md)).
- **Attention never held `T x T`.** The fused attention keeps one log-sum per row, which is what makes [context 2048](context-2048.md) fit at all.

## How it shows up in this repo

- [`lang/attn.c`](../lang/attn.c): `attn_vram_lazy`, `pcache_kv_create`, the aliased `x_in`.
- [`lang/ffn.c`](../lang/ffn.c): `ffn_pre_act` creates the pre-activation on demand.
- [`lang/block.c`](../lang/block.c): no `ffn_out`; `attn_out` only for single-sequence caches.
- [`lang/layernorm.c`](../lang/layernorm.c): `x_src` instead of a stored `x_hat`.
- Items **B9**, **B11**, **B12** and the lazy-buffer list in the [code audit](code-audit-dead-paths-and-compute.md).

## What next

- `--batch auto` sizes the batch from an estimate of these buffers ([Huge batch / VRAM fill](huge-batch-vram.md)); `lm_packed_bytes` still counts `x_hat`, the pre-activation and the other dropped buffers, so it is now pessimistic.
