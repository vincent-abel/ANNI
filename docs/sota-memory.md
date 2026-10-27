# SOTA memory tactics for AnnI LM train (proposal)

**Status:** design-only — no train-loop changes in this doc. Targets packed GPU train in C/CUDA (`lang/lm_train.c`, `lang/lm.c`, `num/gpu_cuda.cu`).

## Goal

Fit larger **B** (microbatch) on one GPU without abandoning FP32 masters: shrink **activations**, **attention materialization**, and **Adam state** using ideas from ZeRO, checkpointing, low-precision optimizers, and paged KV — mapped to existing flags and plausible extensions.

## Current stack (shipped)

| Lever | CLI / API | Primary files | VRAM effect |
|-------|-----------|---------------|-------------|
| Auto batch | `--batch auto`, `--vram-frac F` | `main.c`, `lm_batch_auto`, `lm_packed_bytes` in `lang/lm.c` | Picks max B from estimate |
| Flash-style attn | `--flash-attn` | `attn.c`, `gpu_attn_flash_fwd` / `gpu_attn_tiled_bwd` in `gpu_cuda.cu` | Drops persistent `B·H·T·T` per layer |
| Block act checkpoint | `--act-checkpoint` | `block.c` (`block_cache_drop_device`), `lm_train_set_act_checkpoint` | ~2 live blocks vs `LM_N_BLOCKS` in estimate |
| Adam m/v offload | `--adam-offload` | `lm_adam.c` (`adam_offload_mv`, `offload_mv`) | Frees `d_m`/`d_v` between steps |
| Mixed GEMM prec | `--bf16` / `--fp16` | `gpu.h`, `gpu_set_compute_prec` | Lower GEMM footprint; **m/v still FP32** |
| Buffer reuse | (always on) | `lm.c` S12 mats; `pack_ctx` in `lm_train.c` | Less alloc churn; estimate uses 1.35× safety |

Recommended VRAM-bound combo (see `docs/speed-stack-bench.md`): `--flash-attn --act-checkpoint --batch auto`; add `--adam-offload` when params+m/v dominate after acts shrink.

---

## 1. ZeRO-style partitioning (single GPU analogues)

DeepSpeed ZeRO shards optimizer state, grads, and optionally weights across ranks. On **one** GPU the analogue is **time-multiplexing** and **host residency** — same math, no NCCL.

| ZeRO stage | AnnI today | Proposal |
|------------|------------|----------|
| **1 — optimizer states** | `--adam-offload`: m/v live on host between steps; `gpu_buffers_init` re-H2D each step | **S1a** Segment offload: offload m/v only for largest segments (FFN mats) via `lm_adam_seg.host_use` + flag `--adam-offload-ffn` in `lm_adam.c` |
| **2 — + gradients** | Grads stay on device in `dw_*` until Adam | **S1b** Ping-pong `d_grad`: apply Adam per `lm_adam_seg` then `cudaFree` segment grad slab (`gradutil.c` / `lm_adam_step_scaled`) — `--adam-grad-partition` |
| **3 — + weights** | Weights device-resident (`gpu-adam-residency.md`) | **S1c** CPU master + GPU compute tiles: stream one block’s weights H2D per layer forward/bwd (`block.c` / `tblock`) — `--weight-stream N` (inference quant path in `quant.c` is separate) |

**Estimate hook:** extend `lm_packed_bytes` with optional terms for resident vs offloaded `2P` (m/v) and `P` (weights), driven by flags read from `lm_train.h` setters (mirror `lm_train_set_adam_offload`).

---

## 2. Selective activation recompute

**Today:** `--act-checkpoint` is **all blocks, all recomputable caches** (LN, attn, FFN); keeps `x_in`; backward calls `tblock_forward` into scratch (`block.c`).

**Gap:** FFN hidden (`D_FF`) and attn Q/K/V dominate per-layer bytes; logits `(B·T·V)` dominate head. Uniform checkpoint over-pays recompute when only scores or FFN peaks matter.

| Proposal | Behavior | Files / flag sketch |
|----------|----------|---------------------|
| **S2a Attn-only checkpoint** | Drop `attnpc` device twins; keep FFN caches | `block.c`, `attn.c`; `--act-checkpoint attn` |
| **S2b FFN-only checkpoint** | Drop `ffnc` hidden/pre_act; keep attn concat path | `ffn.c`, `block.c`; `--act-checkpoint ffn` |
| **S2c Layer schedule** | Checkpoint every k-th block (min VRAM) vs none (max speed) | `block_set_act_checkpoint_mask` in `block.h`; `--act-checkpoint stride=k` |
| **S2d Head recompute** | Store hidden at `T-1` only; re-GEMM logits in bwd | `lm.c` CE path; ties to `docs/device-fused-ce.md` |

**Estimate:** teach `lm_packed_bytes` `live_blocks` and per-layer act coefficients from checkpoint mode (today: binary 2 vs `LM_N_BLOCKS` + flash score term).

---

## 3. Quantized optimizer states

**Today:** Adam slots are FP32 (`lm_adam_slot` in `lm_adam.h`); checkpoint serializes full m/v (`lm_serialize.c`). `quant.c` is **int8 inference weights**, not optimizer state.

| Proposal | Storage | Compute | Integration |
|----------|---------|---------|-------------|
| **S3a FP16 m/v** | Half device buffers | FP32 param update | `gpu_adam_step_dev` in `gpu_cuda.cu`; `--adam-fp16-mv` |
| **S3b Block FP8 / E4M3 m/v** | 1 byte/elem + per-segment scale | Dequant in kernel | New `lm_adam_quant_seg` beside `lm_adam_seg` |
| **S3c CPU quantized offload** | INT8/FP16 m/v on host | Same as S13 + smaller PCIe | `--adam-offload` + `--adam-quant-mv` |

**Checkpoint:** new sidecar or LMCF section for scales + quantized blobs; `lm_adam_sync_host` must dequant before save unless format stores quant natively.

**Risk:** small models + high lr — monitor with existing `grad_clip` / loss scaling (`docs/fp16-loss-scaling.md`).

---

## 4. Paged KV — train-time analogues

**Inference KV** (`lang/kv.c`, `docs/gpu-kv-generate.md`): fixed `[max_T × D]` per layer for decode. **Packed train** does not use `lm_kv`; K/V live in `attn_cache` for full sequence length **B×T** per layer.

Analogues (virtual memory for long **T** or wide **B**):

| Proposal | Idea | Maps to |
|----------|------|---------|
| **S4a Paged attn activations** | Fixed pool of `P` physical `(T_page × D)` tiles; logical K/V indexed by page table per (batch row, layer) | New `attn_page_table` in `attn.h`; kernels in `gpu_cuda.cu`; `--attn-pages P` |
| **S4b Windowed train KV** | Cap live context to `T_win < T`; recompute or ring-buffer older positions (cheap eval: `docs/cheaper-window-eval.md`) | `lm_train.c` windowing + `lm_packed_bytes(T_win)` |
| **S4c Flash + page peak** | `--flash-attn` already bounds score peak to one layer; pages bound **Q/K/V concat** footprint separately | Extend flash path when `T > 256` (current fused cap in `attn.c`) |

**Not proposing** literal vLLM-style paged KV for training backward without a fused paged bwd — S4a pairs with S2a/S11 recompute.

---

## Suggested evaluation matrix

1. Baseline: `--batch auto --vram-frac 0.85`
2. + `--flash-attn`
3. + `--act-checkpoint`
4. + `--adam-offload`
5. Future: S2c stride 2, S3a, S1a (measure ms/step and `cudaMemGetInfo` from `-d` stamps in `lm_train.c`)

Success: higher auto-**B** at same peak VRAM, or same **B** with lower peak — without breaking `make test` CPU stub paths (`gpu_stub.c`).

## Out of scope (this proposal)

Multi-GPU ZeRO (no process group in AnnI), training in INT8 weights (`quant.c` infer-only), editing `lm_train.c` loop structure, and changing default checkpoint format without version bump (`LM_MAGIC` in `lm_serialize.h`).

## Related docs

`docs/huge-batch-vram.md`, `docs/activation-checkpointing.md`, `docs/flash-attention.md`, `docs/adam-cpu-offload.md`, `docs/gpu-adam-residency.md`, `docs/packed-activation-arena.md`, `docs/mixed-precision-train.md`.
