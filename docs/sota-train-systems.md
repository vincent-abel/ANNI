# SOTA train systems (proposal)

Listed in [docs/README.md](README.md) under **GPU training**. Proposal only — no code changes implied until a wave ticket lands.

## Baseline (shipped)

AnnI is a from-scratch C/CUDA LM with **packed** `B×T` training (`lang/lm_train.c`, variable lengths via `pack_ctx` / E8). Attention: fused causal MHA **`T ≤ 256`**, optional **`--flash-attn`** (no persistent score tensor; bwd recomputes `w` once). Math: **cuBLAS GemmEx** FP16/BF16 with FP32 accumulate (`num/gpu_cuda.cu`). Launch reduction: **CUDA Graph** replay of steady `pack_run` (`--cuda-graphs`). Memory: **activation checkpointing** drops mid-block device activations (`lang/block.c`). See [flash-attention.md](flash-attention.md), [cuda-graphs-train.md](cuda-graphs-train.md), [activation-checkpointing.md](activation-checkpointing.md).

Goal of this note: map **FlashAttention-3**, **compiler-style fusion**, and **graph trees** to the next AnnI layers without PyTorch/Triton dependencies.

---

## 1. FlashAttention-3 / FA-class attention

**Ideas (literature / FA3 on Hopper):**

- **IO-aware tiling**: keep Q/K/V tiles in SRAM; never write full `T×T` scores to global memory (forward *and* backward).
- **Producer–consumer warps**: one warp group loads (TMA/async copy on sm90+), another computes MMA softmax blocks.
- **Low-precision inner attention**: FP8/BF16 tensor cores with FP32 softmax stats; optional descale vectors per block.
- **Split-K / sequence-parallel** for long `T` (future if `ATT_BLOCK` rises).

**Gap vs AnnI today:** flash path still **materializes full `B·H·T·T` in bwd** (the tiled backward); fwd kernel is monolithic, not multi-stage pipelined.

| Next step | Files | Notes |
|-----------|--------|--------|
| Tile fwd without global `S` | `num/gpu_cuda.cu` (`gpu_attn_flash_fwd`, new `attn_tile_*`) | Online softmax per row tile; match numerics to fused fwd in `gpu_attn_causal_fwd`. |
| Tiled bwd (no persistent `w_all`) | `num/gpu_cuda.cu`, `lang/attn.c` bwd branch | Recompute dQ/dK/dV in tiles; delete or shrink recompute buffer; extend `attn.h` cache flags. |
| Varlen / pad mask in kernel | `num/gpu_cuda.cu`, `lang/attn.c` | Use device `seqlens[B]` (E8 stub exists); skip pad rows in FA tiles. |
| Hopper-only fast path (optional) | `num/gpu_cuda.cu`, `Makefile` | Guard `__CUDA_ARCH__ >= 900` TMA/MMA path; CPU teaching path unchanged. |
| Bench / regression | `lang/lm_bench.c`, [speed-stack-bench.md](speed-stack-bench.md) | A/B vs `--flash-attn` on fixed `(B,T,H,dh)`. |

**Non-goals (this wave):** FP8 training end-to-end; `T > 256` until packing and CE masks are re-audited.

---

## 2. `torch.compile`-class fusion (static recipes)

**Ideas:** Treat `(B, T, d_model, flags)` as a **frozen graph**; fuse epilogues and small ops into GEMM/attn launches; specialize at **first steady step** instead of JIT IR.

**Gap vs AnnI:** S8 epilogues (bias+ReLU) and E7 LN are point fusions; block still alternates many launches (`lang/block.c` → attn → resid → LN → FFN).

| Next step | Files | Notes |
|-----------|--------|--------|
| Fusion registry / “recipe id” | `num/gpu.h`, new `num/fusion_recipe.c` (optional) | Enum: `FUSE_BLOCK0_FWD`, etc.; selected when `B,T` stable (mirror graph capture guards in `lm_train.c` *read-only* contract). |
| LN + residual + bias epilogue on GEMM | `num/gpu_cuda.cu`, `lang/ffn.c`, `lang/block.c` | Extend `gpu_bias_relu_forward` pattern to LN scale/shift after attn/FFN GEMMs where stats allow. |
| Fused QKV projection + split | `num/gpu_cuda.cu`, `lang/attn.c` | Single GEMM + device layout transform → fewer H2D syncs before FA kernel. |
| Device-fused CE + d_logits | `num/gpu_cuda.cu`, `lang/lm.c` / head bwd | Builds on [device-fused-ce.md](device-fused-ce.md); keep loss scalar on host for logging. |
| Profile matrix | `docs/compile-profile.md` | Document `OPT`, `ANN_CUBLAS`, BF16/TF32 combos per recipe — AnnI’s substitute for `-O` compile modes. |

**Principle:** one **manual** fused kernel per hot subgraph beats a generic compiler for this codebase size; recipes toggled by CLI like `--cuda-graphs`.

---

## 3. CUDA Graphs and graph trees

**Ideas:** **Graph trees** = root capture for whole step, **child graphs** per transformer block or per `(fwd,bwd)` half; **conditional nodes** for checkpoint recompute; **graph update** when only pointers change (arena bump).

**Gap vs AnnI:** single flat capture in `gpu_graph_begin/end/replay`; invalidates on any host op in region; no nested graphs or multi-stream trees ([multi-stream-train.md](multi-stream-train.md)).

| Next step | Files | Notes |
|-----------|--------|--------|
| Per-block child graphs | `num/gpu_cuda.cu`, `num/gpu.h` | `cudaGraphAddChildGraphNode` for `block_forward`/`block_backward` device-only subgraphs. |
| Graph set indexed by recipe | `num/gpu_cuda.cu`, `lang/lm_train.c` (integration spec only) | Array `exec[recipe][phase]`; replay selects after first successful capture. |
| Pointer update / arena | `lang/lm.c` cache arena, `num/gpu_cuda.cu` | `cudaGraphExecKernelNodeSetParams` when `d_data` base shifts but kernel args layout fixed. |
| Capture-friendly checkpoint | `lang/block.c` | Recompute branch as separate capturable subgraph vs full-cache branch (two child graphs). |
| Prefetch / second stream | `num/gpu_cuda.cu`, `lang/lm_train.c` | E3 prefetch stream **outside** captured graph; overlap H2D with replay of previous step’s graph. |

**Invariant:** keep epoch NLL / val probes on host ([cuda-graphs-train.md](cuda-graphs-train.md)); graphs cover only the numeric core.

---

## 4. Suggested wave order

1. **FA tiled bwd** (VRAM + bandwidth) — `gpu_cuda.cu` + `attn.c`.
2. **Block child graphs** (launch tax) — `gpu_cuda.cu` + graph API in `gpu.h`.
3. **QKV + LN epilogue fusion** (GEMM-bound steps) — `attn.c`, `ffn.c`, `block.c`.
4. **Recipe + bench matrix** — docs + `lm_bench.c`.

Cross-check memory proposals in `sota-memory.md` (activation arena, Adam offload) when FA3 tiles shrink residency.

## References (external)

- FlashAttention-2/3 papers and Hopper FA3 blog (tiling, warp specialization, FP8).
- PyTorch 2 `torch.compile` / Inductor: fusion + cudagraphs integration (conceptual analogue only).
- NVIDIA CUDA Graphs: child graphs, conditional nodes, exec update APIs.
