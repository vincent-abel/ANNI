# Adam m/v CPU offload (S13)

## Abstract

Optional `--adam-offload`: after each Adam step, copy moment buffers `m`/`v` to host and free device flats so VRAM can go to a larger B. Next step reallocates and H2D before the update.

## How it shows up in this repo

- CLI: `--adam-offload` → `lm_train_set_adam_offload` → `lm_adam_set_offload`
- [`lang/lm_adam.c`](../lang/lm_adam.c): `adam_offload_mv` after a successful GPU step; `gpu_buffers_init` re-allocates `d_m`/`d_v` when missing
- Default **off** (resident m/v is faster)

## Pitfalls

- Adds PCIe traffic every step; only a win when VRAM-bound and larger B improves ms/tok
- Checkpoint save still uses `lm_adam_sync_host`
