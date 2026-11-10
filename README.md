# AnnI

**AnnI** (`./anni`) is a neural network written from scratch in **C**. It started as a single neuron and grew into a **Transformer language model** you can chat with, without any machine-learning framework.

## Status

🟢 **It talks.** Stage 1 is done and Stage 2 is cooking: a 185M model chats today, and a 555M one is training right now.

Current implementation:

- Neurons
- Layers
- Inputs and weights
- Biases
- Connections
- Activation functions
- Forward propagation
- MSE loss
- Backpropagation
- Gradient descent
- Training loop
- XOR dataset
- Unit tests
- Memory management
- Vectors
- Matrices
- Tensors
- Optimized matrix multiplication
- Softmax
- Cross-entropy
- Mini-batches
- Adam optimizer
- Model serialization
- Tokenizer
- Vocabulary
- Token embeddings
- Positional encoding
- Query / Key / Value
- Self-attention
- Multi-head attention
- Feed-forward layers
- Layer normalization
- Residual connections
- Causal attention
- Transformer blocks
- Next-token prediction
- Text generation
- Transformer backpropagation
- Train all parameters
- Adam for LM weights
- Text dataset loading
- Mini-batch corpus training
- Generation from trained checkpoints
- Train / generate / load CLI
- Held-out loss / perplexity
- Checkpoint and resume
- LM weight serialization
- Per-epoch checkpoints, `--lr` / `--window` / `--seed`, `lm merge`
- FP32 (`ann_real`), generate sampling, `--out` resume confirm, BPE + streaming data path
- CUDA training with weights that stay on the GPU
- BF16 GEMMs, tiled flash attention
- Mid-epoch checkpoints, crash-safe resume
- Chat fine-tune (loss on the assistant's turns only)
- 2048-token context
- Weight tying, 100k BPE vocabulary
- HellaSwag, question bench, coding-task bench
- Live run dashboard, latent space viewer
- Weight-space edits

Trainable decoder-only LM in `lang/`: byte or BPE tokenizer, CUDA optional, BF16 GEMMs, checkpoint/resume, eval knobs, streaming splits, `lm merge`. Default config: `D=1024`, 16 heads, `FF=4096`, 12 blocks, ctx=2048 → ~**185M** params at 16k BPE. Sizes are compile-time (`lang/lm_config.h`), so a checkpoint needs a binary built with its sizes.

### Done vs open

**Done (Stage 1):** the plumbing, the speed stack, and a model that answers. **talk-170m** is 185M params, one epoch over 3.81B tokens of web text and code (~33 h on an RTX 5070 Ti, ~34k tok/s, 11.7 GB), then a chat fine-tune, then stretched from 256 to 2048 tokens of context. Try it: `./anni chat --model models/talk170m-chat.bin`.

**How good is it?** Fluent, often wrong: 5–7 of 20 general questions, 6–7 of 20 small coding questions (17 of 20 at least run), 0 of 20 multi-file coding tasks. See [`qbench/`](qbench/README.md) and [`taskbench/`](taskbench/README.md).

**Cooking (Stage 2):** **run555m**, 555M params (`D=1536`, 24 heads, `FF=6272`, 14 blocks, tied 100k BPE), one epoch over 15.6B tokens of web text, code and French. Started 2026-11-05; on 2026-11-10 it is 5.8B tokens in, train NLL 2.86, ~17k tok/s at 13 GB. HellaSwag so far: 29.7% → 31.4% → 33.9% (chance is 25%).

**Open:** runtime model config (**C1**), the modern block (**C6–C10**: RMSNorm, RoPE, SwiGLU, GQA), then Stage 3: chat template, data shards, multi-GPU, DPO, serve. See **Toward ChatGPT-class (code)**.

### Milestone map (code)

| Stage | What “done” means in this repo | Where in README | Status |
| --- | --- | --- | --- |
| **0 Teaching** | Neurons → Transformer train/generate in C | Neural Network … Performance | ✅ done |
| **1 Target small LLM** | ~10–50M, BPE, ≥100 MB, readable generate | Fill GPU R\* · Efficiency E\* · Autotrain · From toy → small LLM | ✅ done, passed by talk-170m |
| **2 Capable local** | ~100M–1B-class code path on 1–few GPUs | Talking model · 555M run · Capable local LLM (code) | 🚧 size is there (185M, 555M training); modern block open |
| **3 ChatGPT-class stack** | Modern arch + web-scale data path + distributed train + chat SFT/DPO + serve | Toward ChatGPT-class (code) | ⏳ first pieces: chat fine-tune, HellaSwag, 100k BPE |

## Roadmap

### Neural Network

- [x] Neurons
- [x] Layers
- [x] Inputs and weights
- [x] Biases
- [x] Connections
- [x] Activation functions
- [x] Stabilize current architecture
- [x] Forward propagation
- [x] Loss functions
- [x] Backpropagation
- [x] Gradient descent
- [x] Training loop
- [x] XOR / basic datasets
- [x] Unit tests
- [x] Memory management

### Numerical Engine

- [x] Vectors
- [x] Matrices
- [x] Tensors
- [x] Optimized matrix multiplication
- [x] Softmax
- [x] Cross-entropy
- [x] Mini-batches
- [x] Adam optimizer
- [x] Model serialization

### Language Model

- [x] Tokenizer
- [x] Vocabulary
- [x] Token embeddings
- [x] Positional encoding
- [x] Query / Key / Value
- [x] Self-attention
- [x] Multi-head attention
- [x] Feed-forward layers
- [x] Layer normalization
- [x] Residual connections
- [x] Causal attention
- [x] Transformer blocks
- [x] Next-token prediction
- [x] Text generation

### Language Model Training

- [x] Full backpropagation through the Transformer
- [x] Train all parameters
- [x] Adam for LM weights
- [x] Mini-batch corpus training
- [x] Text dataset loading
- [x] LM weight serialization
- [x] Checkpoint and resume
- [x] Held-out loss / perplexity
- [x] Generation from trained checkpoints
- [x] Train / generate / load CLI
- [x] Per-epoch checkpoint saves; `--lr` / `--window` / `--seed`; `lm merge` weight average
- [x] Subword BPE + stream/splits + packed batch
- [x] Hit Target size/quality — talk-170m, see **Target quality run (Q\*)**
- [x] Interactive chat CLI — see **CHAT1** / **CHAT2** / **CHAT3**
- [ ] ChatGPT-class code path — see **Toward ChatGPT-class (code)**

### Performance

- [x] Contiguous memory
- [x] Cache-friendly data structures
- [x] SIMD
- [x] Multithreading
- [x] AVX2 / AVX-512
- [x] Quantization
- [x] Optimized inference
- [x] GPU support

### GPU training / full device path

CUDA can run forward GEMM, backward GEMMs, Adam, KV decode, and optional cuBLAS; default `make` / `make test` stay on the CPU stub.

- [x] Device buffers for activations (keep H/Q/K/V on GPU across layers)
- [x] CUDA GEMM backward (`dA` / `dB` for `C = AB` and `C = ABᵀ`)
- [x] Attention backward on GPU (score / softmax / `AV` grads)
- [x] FFN + LayerNorm backward kernels (or fused epilogues)
- [x] LM head + cross-entropy backward on device
- [x] Adam step on GPU (flat param / grad / moment buffers)
- [x] KV-cache generate on GPU (append K/V, decode-step attention)
- [x] Optional cuBLAS / cuDNN (replace naive teaching kernels)
- [x] Host↔device sync only at batch boundaries (cut PCIe chatter)
- [x] `nvidia-smi` / bench proof that train sustains GPU util (not only `lm bench`)

### GPU residency / cut ms/tok

~6–7 ms/tok at D=256 is dominated by **PCIe + host glue**, not GEMM FLOPs. Goal: keep weights/activations/Adam on device and sync once per microbatch so train is well under 1 ms/tok on a 5070‑class GPU when the stack is resident.

Parallel tracks: **G** (runtime residency) ∥ **K** (elementwise kernels) → then **A** (Adam in-place) ∥ **H** (batched head/CE).

- [x] **G1** Persistent device twins for all weights/grads (`mat_try_pin` at create; no per-GEMM `cudaMalloc` of W); see [docs/gpu-residency-steps.md](docs/gpu-residency-steps.md)
- [x] **G2** `ensure_dev` / BATCH: H2D when host is authoritative; `mat_ensure_host` before host reads; drop stale `d_data` reuse bugs
- [x] **G3** One `cudaDeviceSynchronize` (or event) per microbatch — not after every GEMM/outer/LN/Adam launch
- [x] **K1** Softmax forward (+ bwd if still host) on GPU for attention rows
- [x] **K2** LayerNorm **forward** on GPU (bwd already exists; kill per-call alloc/copy)
- [x] **K3** FFN bias / ReLU (and mask) on GPU
- [x] **A1** Adam updates device-resident params/grads in place — no pack/H2D/D2H/unpack every step (D2H only on checkpoint)
- [x] **H1** Batched LM head + fused softmax-CE over all `T` tokens (replace per-token host loop in `lm_loss_backward`)
- [x] **H2** cuBLAS `OP_T` for `mat_mul_at` / bwd layouts (drop host weight transpose copies where possible); see [docs/transposed-gemm.md](docs/transposed-gemm.md)
- [x] Bench: `make cuda` train progress ~**3.0 ms/tok** at D=256 / window 64 (was ~6.8 ms/tok); `--bench-gpu` uses wall time (~3.2 ms/tok @ 64 tokens)

### Token-parallel train / fill the GPU

`--batch N` packs **N parallel windows** into one microbatch so linear layers see `(B·T)×D` GEMMs and VRAM holds a large activation working set (not just grad-accumulation of tiny sequential forwards). Use `--batch auto` to size `B` from free VRAM.

- [x] **P1** Pack `B` windows into one microbatch: ids layout `B·T`; embed / LN / FFN / LM-head GEMMs over all `B·T` tokens in one launch; see [docs/packed-batch-attention.md](docs/packed-batch-attention.md)
- [x] **P2** Attention stays causal **per sequence** (loop `b` over length-`T` slices, or block-diagonal mask) while QKV/FFN stay fat GEMMs
- [x] **P3** CLI: `--batch` = parallel sequences in one step; `--grad-accum K` for extra Adam accumulation without growing activations
- [x] **V1** Report peak activation estimate + `cudaMemGetInfo` free/total at train start (`-d`)
- [x] **V2** `--batch auto` (or `--vram-frac 0.85`): grow `B` until estimated/working alloc approaches budget; fall back on OOM
- [x] **M1** Optional fused batched attention (no per-`b` host loop) once P1–P2 land; see [docs/packed-batch-attention.md](docs/packed-batch-attention.md)
- [x] Bench (RTX 5070 Ti, `--window 64`, 1 epoch of 1 KB toy text, byte vocab): `--batch 1` **1.23 ms/tok** (sequential-equivalent) → `--batch 16` **0.12 ms/tok** (~10×); GPU util ~34% (was ~2% idle). Remaining time is host↔device staging of LN/FFN/residual activations (scales with tokens), not GEMM — next step is keeping those device-resident.

### Fill the GPU further / expand matrices

Observed headroom on a 16 GB card at current toy scale (`D=256`, `LM_BATCH_AUTO_MAX=64`): ~**30%** util, ~**2 GB** VRAM. Growing **D** makes GEMMs fatter (more capacity / util) but usually **raises** ms/tok; the speed stack at fixed `D` is **R1 → huge B → CUDA Graphs → mixed precision** (roughly compounding toward ~10–20×, not another 50× alone).

**Priority speed stack (fixed D)**

- [x] **R1** Keep LN / FFN / residual activations **device-resident** end-to-end (no per-op H↔D staging) — main lever for util ≫ 30% at fixed `D` (~2–5×); see [docs/gpu-activation-residency.md](docs/gpu-activation-residency.md)
- [x] **R3** Lift `LM_BATCH_AUTO_MAX` (64 → **1024**) and relax `lm_packed_bytes` so `--batch auto` can target most of free VRAM (**huge B**); see [docs/huge-batch-vram.md](docs/huge-batch-vram.md)
- [x] **R4** Large manual `--batch` (**128–1024+**) with OOM backoff (halve `B` on alloc fail); default/`-d` guidance for 16 GB cards
- [x] **R12** **CUDA Graphs**: wired into `pack_run` via `--cuda-graphs` (**S4**); capture fails if host ops remain
- [x] **R10** **Mixed precision**: TF32 on; **`--bf16`** GemmEx + FP32 masters (**S5a/S5b**, C13)

**Speed stack wave 1 (glue → graphs → MP)**

- [x] **S1** Device-fused CE (no full logits D2H); see [docs/device-fused-ce.md](docs/device-fused-ce.md)
- [x] **S2** GPU packed posenc; see [docs/gpu-posenc.md](docs/gpu-posenc.md)
- [x] **S3** GPU FFN bias column-sum grads; see [docs/gpu-bias-grad-reduce.md](docs/gpu-bias-grad-reduce.md)
- [x] **S4** Wire CUDA Graphs into packed train (finish R12); see [docs/cuda-graphs-train.md](docs/cuda-graphs-train.md)
- [x] **S5a** BF16 GEMMs + FP32 masters (`--bf16`); see [docs/mixed-precision-train.md](docs/mixed-precision-train.md)
- [x] **S5b** FP16 + dynamic loss scaling (`--fp16`); removed again (the loss was never scaled, so the unscale was wrong; BF16 needs no scaling)
- [x] **S6** Bench util / VRAM / ms/tok after S1–S5; see [docs/speed-stack-bench.md](docs/speed-stack-bench.md)

**W1 gate:** Packed CE keeps logits/delta on device (no full D2H). Graphs wire via `--cuda-graphs` (skip varlen; CE malloc/D2H scalar can still fail capture). Remaining bound: launch tax + materialized/`recompute` `B·H·T·T` attn — Wave 2.

### Speed stack wave 2 (Flash → epilogues → streams)

- [x] **S7** Flash-style attn (`--flash-attn`, finish **R11**); TC GemmEx QK/AV under the same flag; see [docs/flash-attention.md](docs/flash-attention.md)
- [x] **S8** Fused bias+ReLU epilogue; see [docs/fused-epilogues.md](docs/fused-epilogues.md)
- [x] **S9** Multi-stream overlap (copy stream + docs); see [docs/multi-stream-train.md](docs/multi-stream-train.md)
- [x] **S10** Wave 2 bench; see [docs/speed-stack-bench.md](docs/speed-stack-bench.md)

### Speed stack wave 3 (VRAM → larger B)

- [x] **S11** Real act-checkpoint (`--act-checkpoint`, finish **R9**); see [docs/activation-checkpointing.md](docs/activation-checkpointing.md)
- [x] **S12** Logits/delta / pack buffer reuse; see [docs/activation-buffer-reuse.md](docs/activation-buffer-reuse.md)
- [x] **S13** Optional `--adam-offload` async m/v; see [docs/adam-cpu-offload.md](docs/adam-cpu-offload.md)
- [x] **S14** Retune `lm_packed_bytes` / auto-B; see [docs/huge-batch-vram.md](docs/huge-batch-vram.md)
- [x] **S15** Final bench vs baseline; see [docs/speed-stack-bench.md](docs/speed-stack-bench.md)

### Reach ~40 TF on 16GB (scale + tiled FA)

Stretch goal on RTX 5070 Ti (~88 TF BF16 peak / ~40 TF realistic). Meter is \(6P\cdot(\mathrm{tok/s})/10^{12}\); D=512 alone cannot hit 40 TF at realistic tok/s — need **tiled FA (VRAM)** + **D=1024/L=12 (fat GEMMs)** + fusion/graphs. Gates: ~5 TF → ~15 TF → ~40 TF. See [docs/sota-train-systems.md](docs/sota-train-systems.md).

- [x] **T1** Tiled flash attn fwd+bwd (no `T×T` HBM, logsumexp only); see [docs/tiled-flash-attention.md](docs/tiled-flash-attention.md)
- [x] **T2** Scale config `D=1024`, heads=16, `FF=4096`, blocks=12; see [docs/scale-d1024-40tf.md](docs/scale-d1024-40tf.md)
- [x] **T3** Fused QKV + CE hot path / graphs-safe pack; see [docs/fused-qkv-ce-40tf.md](docs/fused-qkv-ce-40tf.md)
- [ ] **T4** Bench milestones 5 / 15 / 40 TF with `anni lm bench step|sweep` (packed path, measured GEMM ceiling, MFU); see [docs/bench-harness.md](docs/bench-harness.md). Note: the `lm train` TF meter overcounts by ~1.5× until it switches to the bench FLOP model
- [x] **T5** GPU training audit: stale caches fixed, BF16 numerics checked; see [docs/gpu-training-audit.md](docs/gpu-training-audit.md)
- [x] **T6** FFN ReLU backward in one pass; see [docs/ffn-relu-backward-fusion.md](docs/ffn-relu-backward-fusion.md)
- [x] **T7** Fused attention on `mma.sync`; see [docs/attention-fa2-mma.md](docs/attention-fa2-mma.md)
- [x] **T8** BF16 outputs straight from the GEMMs; see [docs/bf16-direct-outputs.md](docs/bf16-direct-outputs.md)
- [x] **T9** Code audit: dead paths and what compute is left; see [docs/code-audit-dead-paths-and-compute.md](docs/code-audit-dead-paths-and-compute.md)
- [ ] **T10** FP8 / 4-bit training that beats BF16 (today `--fp8` works but is slower); plan in [docs/low-precision-plan.md](docs/low-precision-plan.md)

**Other fill / scale**

- [x] **R2** GPU embed lookup + embed grad scatter; see [docs/gpu-embed-lookup.md](docs/gpu-embed-lookup.md)
- [x] **R5** Grow **D** (`LM_D_MODEL` 256 → 512 → 1024); set `LM_D_FF ≈ 4×D` — fattens every GEMM; **breaks old checkpoints** (retrain)
- [x] **R5b** Scale **attention heads** with width: keep `d_head` sensible (e.g. 32–64); today `LM_N_HEADS=8` at D=256 — at D=512 use **8 or 16** heads (`LM_N_HEADS`, Target 4–8+); must divide `D`
- [x] **R6** Grow context (`LM_MAX_CTX` / `--window` 128 → 256–512) for longer sequence matrices
- [x] **R7** Grow depth (`LM_N_BLOCKS` 4 → 6–8) toward Target 10–50M params
- [x] **R8** Async CUDA streams: overlap H2D of next microbatch with compute (only needed while any host glue remains)
- [x] **R9** Activation checkpointing — `--act-checkpoint` recompute (**S11**); see [docs/activation-checkpointing.md](docs/activation-checkpointing.md)
- [x] **R11** Flash-style attention (`--flash-attn`, **S7**); fused cap T≤256; see [docs/flash-attention.md](docs/flash-attention.md)
- [x] Bench: util %, `nvidia-smi` memory, and ms/tok after speed-stack waves — [docs/speed-stack-bench.md](docs/speed-stack-bench.md)

### Efficiency & systems

Broader throughput and wall-time wins beyond the GPU residency / pack / R-stack. Cross-links: LR schedule here is **E2** (Capacity bullet points here); activation checkpointing is **R9**; mixed precision is **R10**.

- [x] **E1** Default `--lr` safe for FP32 D≥256 (`0.001`); see [docs/safe-default-lr.md](docs/safe-default-lr.md)
- [x] **E2** LR warmup + cosine/linear decay CLI; optional grad clip / weight decay; see [docs/lr-schedule-clip-wd.md](docs/lr-schedule-clip-wd.md)
- [x] **E3** Prefetch next microbatch ids while BATCH GPU drains; see [docs/microbatch-prefetch.md](docs/microbatch-prefetch.md)
- [x] **E4** Persistent packed activation arena (one alloc per steady `B,T`); see [docs/packed-activation-arena.md](docs/packed-activation-arena.md)
- [x] **E5** Pin host staging buffers (`cudaHostAlloc`) for pack ids; see [docs/pinned-host-staging.md](docs/pinned-host-staging.md)
- [x] **E6** Multi-stream / multi-event dependency graph (beyond single BATCH sync)
- [x] **E7** cuDNN or hand-fused LN + softmax where beneficial after R1
- [x] **E8** Variable-length sequence packing with attention mask (less pad waste → more real tokens/sec)
- [x] **E9** Gradient accumulation by **token budget** (`--tokens-per-step`); see [docs/tokens-per-step.md](docs/tokens-per-step.md)
- [x] **E10** Throughput dashboard: tokens/sec, TFLOP/s estimate, VRAM used (`-d` / tower); see [docs/throughput-dashboard.md](docs/throughput-dashboard.md)
- [x] **E11** Repro JSON beside checkpoint (`OUT.autotrain.json`); see [docs/repro-json-sidecar.md](docs/repro-json-sidecar.md)
- [x] **E12** Compile profile docs: `-O3`, LTO, `ARCH=…`; clear `make` (CPU stub) vs `make cuda`
- [x] **E13** Inference efficiency: keep KV generate on GPU; batch generate; lean on existing int8 head path
- [x] **E14** Data pipeline: mmap for wiki-scale `--stream`; see [docs/streaming-dataloader.md](docs/streaming-dataloader.md)
- [x] **E15** Early-stop on val NLL plateau (saves wall time; used by Autotrain phase 2)
- [x] **E16** **Diverge ditch:** if train or val NLL rises sharply vs the best so far (e.g. +threshold or non-finite), abort the run/config, restore the last good checkpoint, and skip remaining epochs (catches LR blow-ups like the old `0.01` D=256 case)

### Autotrain (`./autotrain`)

Opinionated **second binary** (not more flags on `./anni`) so the teaching CLI stays explicit. `./anni lm train …` remains the manual path; `./autotrain` picks Target-oriented defaults and runs an end-to-end job.

```
./autotrain → probe GPU/VRAM → ensure data → BPE → pick window/batch/lr/epochs
           → packed train → smoke generate + report
```

**Build:** `make autotrain` → CUDA `./autotrain` (same as `make cuda-autotrain`). CPU stub only via `make autotrain-cpu`. Runtime refuses to train without a GPU unless `--cpu`.

**Phase 1 — one command (MVP)**

- [x] **AT1** Makefile target `autotrain` produces `./autotrain`
- [x] **AT2** Defaults: BPE, stream, val-frac 0.01, batch auto, window min(64,ctx), lr 0.001
- [x] **AT3** If `--data` missing: run `scripts/fetch_wiki.sh`, or exit offline
- [x] **AT4** Train/load vocab beside `--out`; resume checkpoint when present
- [x] **AT5** Print plan + write `OUT.autotrain.json` (E11)
- [x] **AT6** Smoke `generate` + final metric

**Phase 2 — light tune**

- [x] **AT7** Short sweep over `{lr}` (and optionally batch) for N steps; keep best val window NLL
- [x] **AT8** Continue best checkpoint for the full budget; early-stop via **E15**
- [x] **AT9** Apply **E16** diverge ditch during sweep and full train — rising/NaN NLL kills that trial and falls back to the best checkpoint (do not keep a blown-up epoch)

See [docs/autotrain.md](docs/autotrain.md).

### From toy LM → small LLM (not-toy)

What we had at `c7a9a88` was a **byte-level** educational model (`LM_BYTE_VOCAB` 256, short train windows, `double` mats, toy corpora). Low **window NLL** on `data/toy_*.txt` does **not** mean fluent generation: each token is one byte, so meaning is spread over long sequences and needs far more data (and usually subwords) than a wordpiece LM.

**Near-term steps (unblock speed and honest metrics)**

- [x] FP32 train/infer path (`ann_real` / `LMCF` / `cublasSgemm`) — mixed FP16/BF16 still later
- [x] Cheaper eval: `--eval-every` / `--eval-windows` (skip or subsample `lm_nll_windows`)
- [x] Train window closer to generate length (default `--window` 64)
- [x] Sampling controls for generate (`--temperature`, `--top-k`, `--top-p`)
- [x] `--out` without `--resume`: warn + `y` resumes, else overwrite (`--force`; non-TTY safe)
- [x] Keep byte tokenizer as a teaching path (`--tokenizer byte` default; see docs)

**Subword + data (main jump from toy → meaningful text)**

- [x] Subword BPE tokenizer with **10k–30k** vocab (default 16k; `--tokenizer bpe`)
- [x] Serialize vocab with checkpoints (`LMCV`); CLI `train` / `generate` / `load` use embedded BPE when present
- [x] Real corpora at scale: `make data-wiki` / `scripts/fetch_wiki.sh` (100 MB+ license-clean Wikipedia into `data/`)
- [x] Held-out split by **document/offset** (`--val-frac`, `--split doc|offset`)
- [x] Packing / streaming dataloader (`--stream`, `--stride`, `--shuffle-buf`)

**Capacity + training recipe (small LLM, still from-scratch C)**

- [x] Scale config toward Target: ~**10–50M** params — see **R5–R7** + **R5b** (`D`, heads **8/16**, ctx, blocks)
- [x] LR schedule (warmup + decay) + clip / weight decay — see **E2**
- [x] Batch by **tokens** via packed `--batch` / `--grad-accum` (further VRAM fill: R3–R4; token budget: **E9**)
- [x] Activation checkpointing when packed activations blow RAM — see **R9**
- [x] FP16/BF16 or TF32 GEMMs — see **R10**
- [x] Stronger generate: `--temperature` / `--top-k` / `--top-p` (length/EOS control still open)
- [ ] Eval suite: window NLL, long-context NLL, held-out PPL, and a few fixed prompts for qualitative checks (have: held-out NLL, HellaSwag, `qbench/`, `taskbench/`; long-context NLL still open)
- [ ] Multi-run: already have `--seed`, shuffle, `lm merge`—document as the cheap ensemble recipe; repro flags: **E11**
- [x] One-command Target path — see **Autotrain** (`./autotrain`, AT1–AT9)
- [ ] After Target: Stage 2 (**C1–C14**) then Stage 3 (**ARCH/TOK/DAT/DIST/ALN/EV/SRV/OPS**) — Stage 2 under way, see **555M run**

**Target quality run (Q\*)**

- [x] **Q1** Rebuild: `make clean && make cuda && make cuda-autotrain` after R5–R7 dims; old d=256 ckpts will not load
- [x] **Q2** Smoke train on `data/wiki_sample.txt` (~1 MB): BPE 16k + stream + val-frac → `wiki_bpe.bin` (use `--batch 64` if `--batch auto` OOMs near ~12 GB; not full `wiki_simple.txt`)
- [x] **Q3** Honest metrics resume: `ann lm train` with `--tokenizer bpe --vocab wiki_bpe.bin.vocab --data data/wiki_sample.txt --val-frac 0.05 --split doc --window 128 --batch 64 --resume wiki_bpe.bin`
- [x] **Q4** Gate: `lm info` shows ~tens of M params + BPE V≈16384; val window NLL ≪ ln(V)≈9.7; in-domain `lm generate` readable English (the 35.7M wiki smoke stayed at val≈8.6 and unreadable; talk-170m passed it: 185M, val≈3.35)
- [x] **Q5** Mark “Hit Target size/quality” [x] only after Q4 **and** a real ≥100 MB run — done on 15 GB of text (talk-170m), not on `data/wiki_simple.txt`

**Interactive chat (after Target quality)**

Stage 1 local turn loop on a pretrained checkpoint — not Stage 3 SFT/HTTP.

- [x] **CHAT1** `./anni chat --model models/talk170m-chat.bin`: User/Assistant history, `lm_generate`, context ≤ `LM_MAX_CTX` (stdin when no TTY)
- [x] **CHAT2** Docs + README example; exit on empty line / Ctrl-D (stdin) or a lone Esc (ncurses)
- [x] **CHAT3** Ncurses messenger UI on TTY (`libncurses-dev`); same flags as CHAT1; `./anni lm chat` is an alias

**Done when:** subword vocab, ≥100 MB train data, ~10M+ params, train window ≈ generate horizon, FP32 (or faster) on GPU, and generate produces readable multi-word text on held-out-style prompts—not just low NLL on `toy_tiny`. Then `./anni chat` for multi-turn local chat. ✅ Reached with talk-170m.

### Talking model (talk-170m)

The first model that answers back. 185M params on the **T2** config, one 16 GB GPU. Plan: [docs/talk-170m-plan.md](docs/talk-170m-plan.md).

- [x] **TALK1** Data: 3.81B tokens (3.21B FineWeb-Edu + 0.60B permissive code), 16k BPE retrained on the mix, held-out split, chat conversations; see [docs/talk-170m-data.md](docs/talk-170m-data.md)
- [x] **TALK2** Mid-epoch checkpoints (`--ckpt-minutes`), resume at the same window, and a tmux driver that survives reboots (`scripts/talk170m_run.sh`); see [docs/mid-epoch-checkpoints.md](docs/mid-epoch-checkpoints.md)
- [x] **TALK3** Pretrain: one epoch, batch 16 × 256 tokens × 32 accum, BF16 + tiled flash attention, ~34k tok/s, 11.7 GB, ~33 h; see [docs/talk-170m-pretrain.md](docs/talk-170m-pretrain.md)
- [x] **TALK4** Chat fine-tune: two epochs on 24M tokens, loss on the assistant's turns only; see [docs/talk-170m-chat.md](docs/talk-170m-chat.md)
- [x] **TALK5** Context 256 → 2048: 0.4B more tokens at window 2048, then the chat fine-tune again (`scripts/ctx2048_run.sh`); see [docs/context-2048.md](docs/context-2048.md)
- [x] **TALK6** Benches: [`qbench/`](qbench/README.md) (40 questions) and [`taskbench/`](taskbench/README.md) (20 coding tasks)
- [x] **TALK7** Weight-space edits: change a behaviour by editing FFN weights, no training; see [docs/weight-space-edits.md](docs/weight-space-edits.md)

### 555M run (run555m)

Same code, three times the parameters, four times the data.

- [x] **BIG1** Weight tying (`--tied`): one table for input and output, ~154M params saved at a 100k vocabulary; see [docs/weight-tying.md](docs/weight-tying.md)
- [x] **BIG2** Data: 15.6B tokens of web text, code and French, 100k BPE (`scripts/fetch_code.py`, `scripts/build_code.sh`)
- [x] **BIG3** Driver `scripts/run555m.sh`: builds its own 555M binary in `build/555m`; `--check` measures peak VRAM before the real run
- [x] **BIG4** Live dashboard (`scripts/dashboard.py`): progress, loss curve, GPU, log tail
- [x] **BIG5** HellaSwag during the run (`anni lm hellaswag`); see [docs/hellaswag.md](docs/hellaswag.md)
- [x] **BIG6** Latent space viewer (`anni-probe`, `anni-latent`); see [docs/latent-space-viewer.md](docs/latent-space-viewer.md)
- [ ] **BIG7** Finish the epoch (5.8B of 15.6B tokens on 2026-11-10)
- [ ] **BIG8** Chat fine-tune, then `qbench/` and `taskbench/` against talk-170m

### Capable local LLM (code) — Stage 2

Single-node / few-GPU path between Target and ChatGPT-class. Still from-scratch C; no chat product yet.

**Config & memory**

- [ ] **C1** Runtime model config (not only `lm_config.h` macros): load `D`, heads, blocks, ctx, vocab from checkpoint header / JSON
- [x] **C2** Grow to ~**100M–1B** params (example: D=768–2048, blocks=12–24, ctx=2048+) — 185M at ctx 2048, 555M training; still macros, so **C1** stays open
- [x] **C3** Weight tying option (embed ↔ LM head): `--tied` (**BIG1**); bias-free linear layers not done
- [x] **C4** Gradient checkpointing wired for packed train (`--act-checkpoint`, **S11**/R9)
- [ ] **C5** Sharded checkpoint I/O (multi-file or mmap) so save/load fits host RAM

**Modern decoder block (code in `lang/`)**

- [ ] **C6** **RMSNorm** (replace or sit beside LayerNorm)
- [ ] **C7** **RoPE** (replace absolute positional encoding for long ctx)
- [ ] **C8** **SwiGLU** (or GELU) FFN; set intermediate size in config
- [ ] **C9** **GQA / MQA** (grouped-query attention) for cheaper KV at generate
- [ ] **C10** Pre-norm vs post-norm selectable; document the default recipe

**Train loop hardening**

- [x] **C11** Grad clip + weight decay + warmup/cosine — **E1–E2**, used by every real run
- [ ] **C12** Token-budget steps (`--tokens-per-step`) — **E9**; epoch = token budget not file passes
- [x] **C13** BF16/FP16 train path with FP32 master weights — **R10** / **S5a/S5b**
- [ ] **C14** Data: multi-file shard reader + shuffle across shards (**E14** + beyond wiki dump)

**Done when:** one machine can train/resume a ~100M+ RoPE/RMSNorm model on multi‑GB text and generate coherent paragraphs (still not ChatGPT).

### Toward ChatGPT-class (code) — Stage 3

What ChatGPT-class systems need **in code**, ordered so each section can ship as its own track. This is not “train GPT-4 on a laptop”; it is the implementation surface required to *be* that kind of stack in this repo.

#### Architecture (beyond Stage 2 block)

- [ ] **ARCH1** Stable long context (4k → 32k+): RoPE scale / NTK / YaRN-style ctx extension in generate + train
- [ ] **ARCH2** Flash / fused attention kernels for train + decode — builds on **R11**
- [ ] **ARCH3** MoE optional path (router + expert FFNs + aux loss) — only after dense stack is solid
- [ ] **ARCH4** Speculative-decoding draft model hook (small draft + large verify) for serve

#### Tokenizer & specials

- [x] **TOK1** Production BPE/Unigram: 32k–100k+ vocab, Unicode-safe, explicit UNK/byte-fallback — 100k byte-level BPE in run555m
- [ ] **TOK2** Special tokens: `BOS`/`EOS`/`PAD`/`SEP`, plus chat roles (`system` / `user` / `assistant`)
- [ ] **TOK3** Chat template encode/decode (string → token ids with roles); shared by train SFT and serve
- [ ] **TOK4** Stop sequences / EOS-forced generate; max-new-tokens separate from context length

#### Pretrain data (web-scale path)

- [ ] **DAT1** On-disk shard format (length-prefixed token ids or zstd text + offset index)
- [ ] **DAT2** Multi-worker CPU tokenize → shard writer (offline job, not inside `./ann` hot loop)
- [ ] **DAT3** Document boundaries respected in packing; optional quality filter hooks (length, lang-id stub)
- [ ] **DAT4** Dedup interface (exact hash / MinHash stub) before shard finalize
- [ ] **DAT5** Train cursor: epoch-less infinite stream with deterministic skip by seed+step
- [ ] **DAT6** Mixture schedule (weights over corpora: web / code / wiki) expressed as config, not hardcode

#### Distributed training

- [ ] **DIST1** Multi-process data parallel (NCCL or CUDA IPC allreduce on grads) — first real multi-GPU
- [ ] **DIST2** Optimizer-state / param sharding (ZeRO-1/2 style) to fit larger models in aggregate VRAM
- [ ] **DIST3** Tensor-parallel linear + attention split for single-layer wider than one GPU
- [ ] **DIST4** Pipeline-parallel optional for very deep stacks
- [ ] **DIST5** Distributed checkpoint (each rank writes its shard; gather on resume)
- [ ] **DIST6** Global batch = `devices × B × T × accum`; CLI exposes tokens/sec **cluster-wide** (**E10**)

#### Instruction / chat alignment

- [ ] **ALN1** SFT dataset reader (JSONL: messages[] → TOK3 ids); pack multi-turn with loss mask on assistant tokens only (the mask exists, on plain `User:` / `Assistant:` text — **TALK4**)
- [x] **ALN2** Continue-train from a pretrain checkpoint on chat data: `lm train --resume` on masked ids (**TALK4**)
- [ ] **ALN3** Preference pairs JSONL (chosen / rejected); **DPO** loss + train loop in C
- [ ] **ALN4** Optional reward-model head + simple PPO/REINFORCE loop (only after DPO works)
- [ ] **ALN5** System-prompt + safety refusal samples in SFT mix (data + mask, not a separate product)

#### Eval & regression (code harness)

- [x] **EV1** Fixed prompt suite: `qbench/`, 40 chat questions scored by exact match or by running the code
- [ ] **EV2** Window NLL + bits-per-byte + long-ctx NLL CLI (`lm eval`)
- [x] **EV3** Multiple-choice harness (logprob pick): `anni lm hellaswag`, 10,042 questions; see [docs/hellaswag.md](docs/hellaswag.md)
- [ ] **EV4** Chat rubrics: format compliance (role tags, EOS), repetition / blank detectors
- [ ] **EV5** CI smoke: tiny model + tiny SFT step + EV1 must not regress

#### Inference & serve

- [ ] **SRV1** Paged / block KV cache so many concurrent chats share VRAM
- [ ] **SRV2** Continuous batching scheduler (add/remove sequences mid-decode)
- [ ] **SRV3** Quantized inference (int8/int4 weights; activate path already has int8 head — generalize)
- [ ] **SRV4** Multi-GPU generate (tensor-parallel decode using DIST3)
- [ ] **SRV5** Local HTTP API: chat completions request/response (OpenAI-shaped JSON) over `lm serve`
- [ ] **SRV6** Streaming tokens (SSE or chunked); cancel mid-stream
- [ ] **SRV7** Tool/function-call stub: JSON schema in messages → constrained decode hook (optional late)

#### Tooling & ops (still code in-repo)

- [ ] **OPS1** Training job manifest (JSON): model, data mix, LR, budget tokens, seed — written beside ckpt (**E11**)
- [x] **OPS2** Crash-safe resume: epoch + window index in the checkpoint (`--ckpt-minutes`, **TALK2**)
- [ ] **OPS3** Profiling hooks: NVTX ranges around fwd/bwd/Adam/data for nsys
- [ ] **OPS4** Model card generator from manifest + EV2/EV3 numbers

**Honest ceiling:** Stage 3 is the *code surface* of a ChatGPT-like system. Matching closed product quality also needs enormous compute and proprietary data — out of scope for checkboxes, but every checkbox above is implementable in this C stack.

## Target

**Stage 1** — the “small LLM, not toy” bar (✅ passed, with a bigger model than planned):

- Decoder-only Transformer
- ~10–50M parameters
- 10k–30k **subword** vocabulary (not byte-256)
- 256–512 token context
- 4–6 Transformer layers
- 4–16 attention heads (today 16 at D=1024)
- Training data from **100 MB** upward (then 1 GB+)

**Stage 2** is where we are: 100M–1B params on one GPU. 185M is done, 555M is training.

The goal is to implement and understand the complete stack from **neurons to text generation**, then grow that stack until generation is useful—not only until toy window NLL falls. ChatGPT-class is **Stage 3**, not a rename of Target.

## Training

Next-token prediction (next **subword** with BPE; next **byte** is still there as the teaching path).

    Input:  "The cat is"
    Target: "on"

Training loop:

1. Tokenize text
2. Predict the next token
3. Calculate the loss
4. Backpropagate gradients
5. Update parameters
6. Repeat

Practical CLI knobs: `--data`, `--epochs`, `--batch N|auto`, `--grad-accum K`, `--vram-frac F`, `--lr`, `--window`, `--seed`, `--resume`, `--out`, `--bf16`, `--flash-attn`, `--tied`, `--ckpt-minutes M`, `-d`; generate via `./anni lm generate --model … --prompt … --tokens …`; chat via `./anni chat --model models/talk170m-chat.bin` (see [docs/lm-chat.md](docs/lm-chat.md)). Ncurses tower (needs `libncurses-dev`): `./anni lm tower train`, `./anni lm tower inspect`; plain intel: `./anni lm info --model …` (see [docs/training-control-tower.md](docs/training-control-tower.md)).

A full run is one script: `scripts/talk170m_run.sh`, `scripts/ctx2048_run.sh`, `scripts/run555m.sh`. Each one skips the steps already done and resumes after a crash.

## Data

Progression toward non-toy:

    toy KB → 10 MB → 100 MB → 1 GB → 10 GB+   ← we are here

talk-170m read 15 GB of text (3.81B tokens); run555m is reading 15.6B tokens: FineWeb-Edu, permissively licensed GitHub code, and French from FineWeb-2. The fetch scripts download it all.

Byte LMs need **more raw text** than subword LMs for similar fluency. Prefer document-level held-out splits. Wikipedia and other freely licensed text are fine for experiments.

## Architecture

    Neuron
       ↓
    Layer
       ↓
    Network
       ↓
    Matrix / Tensor Engine
       ↓
    Transformer
       ↓
    Language Model
       ↓
    Text Generation

## Build

The project is written in C and aims to remain dependency-light.

    make            # CPU ./anni
    make test       # unit tests on a small D=64 / 2-block build
    make cuda       # CUDA ./anni (needs nvcc)
    make test-gpu   # GPU checks on the same small build
    make autotrain  # CUDA ./autotrain; autotrain-cpu for the CPU one
    make latent     # anni-probe + anni-latent (needs libglfw3-dev)

Want another model size? Override the macros: `make OPT="-O2 -DLM_D_MODEL=64 -DLM_N_HEADS=4 -DLM_D_FF=256 -DLM_N_BLOCKS=2"`.

## Philosophy

- Implement the fundamentals instead of hiding them behind frameworks.
- Keep the code understandable.
- Measure performance instead of guessing.
- Test numerical correctness.
- Optimize after correctness.
- Build incrementally.

## Long-Term Goal

Build a functional **text-generating Transformer in C from scratch**, then grow the *same* codebase through the milestone map:

**Stage 0** Teaching stack ✅ → **Stage 1** Target small LLM ✅ → **Stage 2** Capable local (modern block, 100M–1B path) 🚧 → **Stage 3** ChatGPT-class code (distributed pretrain, SFT/DPO, eval, serve)

Still: train and infer without ML frameworks; measure; keep the code readable.
