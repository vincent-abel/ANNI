# Activation / logits buffer reuse (S12)

## Abstract

Avoid per-microbatch `mat_create` for CE logits/delta and packed head gather scratch. Steady `(rows, V)` / `(rows, D)` shapes reuse persistent mats across steps.

## How it shows up in this repo

- [`lang/lm.c`](../lang/lm.c): `g_ce_delta`, `g_pack_hc`, `g_pack_dhc` via `reuse_mat`
- `ce_rows_backward` and `lm_loss_backward_packed_lens` no longer free those mats each call
- Complements E4 `pack_ctx` arena for caches / `dhidden`

## Pitfalls

- Shape change (tail B or different T) reallocates once
- Buffers live for process lifetime (tiny vs model weights)
