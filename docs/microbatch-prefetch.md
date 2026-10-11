# Microbatch prefetch (E3)

## Abstract

Stream training double-buffers packed ids: while BATCH GPU work from `pack_run` is still draining, the CPU fills `ids_next` before the microbatch sync.

## How it shows up in this repo

- [`lang/lm_train.c`](../lang/lm_train.c): `pack_ctx.ids_next`, stream loop swap/prefetch
