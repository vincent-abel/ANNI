# Contiguous memory

## Abstract

A **bump / arena allocator** hands out consecutive bytes from one buffer. LM **forward caches** stay alive across calls when the sequence length is unchanged, and attention softmax / head scratch buffers live inside `attn_cache` instead of per-call `malloc`. Together these keep hot data contiguous and cut allocator noise so later SIMD and threading work on stable layouts.

## What problem it solves (plain language)

Calling `malloc` for every attention row or rebuilding the whole Transformer cache on every forward scatters memory and wastes time. Contiguous scratch and **persistent caches** mean the model reuses the same matrices for the next window or token batch.

## How it works for a human (intuition, analogy)

Think of a **notepad** instead of a stack of sticky notes: you write down the page once and erase with a wipe (`arena_reset`) rather than buying a new pad each sentence. The Transformer cache is the same notepad reused while the sentence length stays fixed.

## Formal view (definitions, formulas, shapes, algorithms)

**Arena.**

```text
arena_create(cap) -> buf[cap], used = 0
arena_alloc(n)    -> pointer at used (aligned), used += n
arena_reset()     -> used = 0   /* O(1), no free of individual objects */
```

**Persistent `lm_cache`.**

```text
lm_forward_cache(model, ids, n, c):
  if c.trc is null or c.trc.T != n:
    free old trc; c.trc = transformer_cache_create(n, model.tr)
  embed + posenc + transformer_forward_cache into c.hidden
```

**Attention scratch (owned by `attn_cache`).**

- `q_h`, `k_h`, `o_h`: `T × d_head`
- `softmax_tmp`: length `T` doubles for `vec_softmax` without per-row heap traffic

**Bench.**

```text
ms = lm_bench_forward_ms(model, n, iters)
./ann lm bench --tokens 32 --iters 20
```

Reports mean wall milliseconds per forward after one warmup (cache already allocated).

## Complexity / numerical pitfalls / design choices

**Arena lifetime.** Objects from an arena are invalid after `arena_reset` or `arena_free`. Do not store arena pointers in long-lived weight structs.

**Capacity.** `arena_alloc` fails if `used + n > cap`. Size scratch for worst-case `LM_MAX_CTX` when using the arena for temps.

**Cache identity.** Reuse is correct only when `T` matches. Changing window length reallocates `transformer_cache`.

**Correctness.** Persistent caches hold activations overwritten each forward; gradients still read the last forward’s values — same as recreate-every-time semantics.

## How it shows up in this repo

- [`num/arena.h`](../num/arena.h) / [`num/arena.c`](../num/arena.c): bump allocator
- [`lang/attn.h`](../lang/attn.h): `q_h` / `k_h` / `o_h` / `softmax_tmp` on `attn_cache`
- [`lang/lm.c`](../lang/lm.c): reuse `c->trc` when `T` matches
- [`lang/lm_bench.c`](../lang/lm_bench.c): `lm_bench_forward_ms`
- CLI: `./ann lm bench`

## Worked example

Create `arena` of 1 KiB, allocate two double arrays, reset, allocate again at offset 0. Run two `lm_forward_cache` calls of length 8; the second keeps the same `transformer_cache` pointer. `./ann lm bench --tokens 16 --iters 5` prints a positive `mean_ms`.

## What next

[Cache-friendly data structures](cache-friendly-data-structures.md) rewrite attention and logits to walk contiguous rows with GEMM instead of `mat_get` / `mat_set`.
