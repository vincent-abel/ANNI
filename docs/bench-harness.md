# Bench harness (`anni lm bench`)

## Abstract

`anni lm bench step|gemm|decode|sweep` times the code that training and generation actually run. It reports percentiles and variance, not a single mean. It records the commit, GPU and config with each number in one JSON line, so two runs can be diffed with `scripts/bench_compare.py`. The step mode drives the production packed path (`pack_run`: forward, fused CE, backward, then device Adam) at a chosen `B`, `T`, precision and memory levers. MFU is computed against a GEMM ceiling measured on the same card, not against a spec sheet.

## What problem it solves (plain language)

The old benches measured something else than training:

- `lm bench --train` and `lm train --bench-gpu` timed an **unpacked B=1, 32-token** step through `lm_forward_cache`. That code path is not `pack_run`, so CUDA Graphs, fused CE, packed attention and auto-B were never exercised.
- `lm train --bench-gpu` returns at the top of `run_lm_train_core`, **before** `--flash-attn`, `--bf16`, `--act-checkpoint` or `--adam-offload` are parsed. The Wave 2 and Wave 3 tables in [speed-stack-bench.md](speed-stack-bench.md) therefore compare runs with identical settings; the gaps are run-to-run noise (Wave 1 already saw 257–540 ms for one config).
- `scripts/bench_tf_gate.sh` scored the cumulative train meter (eval probes and warmup included) and ran `pkill -x anni`.
- The train meter's parameter count is `L·(12D² + 2·D·FF)`, which counts attention three times. The real block has `4D² + 2·D·FF` matmul weights, so the meter's P is 251.7M where the matmul P is 151.0M (+16.8M for the head). Every TF/s figure printed by `lm train` is about **1.5× too high**. The README's old "~280M params" had the same error: `lm_param_count` gives **184.7M** at V=16384.

## How it works for a human (intuition, analogy)

A lap timer on the real track instead of a stopwatch on the driveway. The car (model) is the same, but the bench puts it on the same road the race uses (packed batches, real precision path), times many laps, throws away the warm-up laps, and writes each lap time in a logbook with the date, car setup and track conditions. A second logbook from tomorrow can be laid next to it line by line.

## Formal view (definitions, formulas, shapes, algorithms)

**Step mode.** Build a model with `lm_create_vocab(V)` (byte tokenizer, `V`-row head, so the head GEMM matches a 16k BPE model without needing a trained vocab). Draw one fixed batch `ids[B·T]` from `seed`. Call `lm_train_bench_packed` (in `lang/lm_train.c`), which reuses `pack_ctx_init` (OOM backoff halves B) and runs `warmup + steps` iterations of:

```text
gpu_device_sync; t0
vec_zero(db_head); sync_mode = BATCH
pack_run(model, B, ce_ptr)         # fwd + fused CE + bwd, graphs if enabled
lm_adam_step_scaled(lr, 1/B)       # device Adam
gpu_device_sync; t1                # ms = t1 - t0
```

CE is fetched on the first, last and every 8th timed step only (the train meter samples every 8th too), so the scalar D2H does not dominate small steps. With `--profile`, `pack_run_body` stores fwd / loss / bwd times and the bench adds Adam; a replayed CUDA Graph skips the body, so profiled phases read `-1` under replay.

**Statistics.** Nearest-rank p10 / p50 / p90 over the timed steps, mean, and coefficient of variation `cv = sd / mean`. A cv above 5% prints a warning; nothing is dropped.

**FLOP model** (`lm_bench_flops_per_token`). Per trained token, forward + backward:

```latex
F_{model}(T, V) = 6\,\bigl(L\,(4D^2 + 2\,D\,D_{ff}) + D\,V\bigr) + 6\,L\,T\,D
```

The first term is every matmul weight (embedding lookup is a gather and costs nothing). The second is causal attention: `QKᵀ` and `AV` each cost `D·T` per token per layer on average under the causal mask, times 3 for forward + backward. With `--act-checkpoint`, recompute adds one block forward:

```latex
F_{recompute}(T) = 2\,L\,(4D^2 + 2\,D\,D_{ff}) + 2\,L\,T\,D
```

`tf_model = tok/s · F_model / 1e12`; `tf_hw` adds the recompute, so a checkpointed run is not credited for redoing its own work.

**Ceiling and MFU.** Before timing, `step` and `sweep` time the model's FFN-up GEMM `(M × D)·(D × D_ff)` with `M = 16384` (CUDA) at the run's precision and keep the best of 5. `MFU = tf_model / ceiling`. `gemm` mode prints the same probe for all four model shapes (attention projection, FFN up, FFN down, head) at `M = B·T`.

**Busy check.** Before any allocation: if `nvidia-smi --query-compute-apps` lists other processes holding more than 512 MB, or more than 1 GB of VRAM is already in use (WSL does not list compute apps, so this catches llama-server there), the bench refuses unless `--allow-busy`, and the JSON line carries `"busy":true`. The bench never kills processes.

**Loss check.** The batch is the same every step, so CE must be finite and fall from the first to the last sampled step. Adam can lift CE for the first few steps at D=1024, so runs with fewer than 16 timed steps only require finite CE. A fast step that fails the check exits non-zero.

**Decode mode.** Load a checkpoint, warm once, then `runs` times: greedy `lm_generate` of 1 token (prefill + first token) and of `N` tokens. Steady decode is `(N − 1) / (t_N − t_1)`. The FNV-1a hash of the greedy text must match across runs and is stored so `bench_compare.py` flags a changed output.

## Complexity / numerical pitfalls / design choices (CS depth)

- **Wall clock bracketed by device syncs, not CUDA events.** One step is one stream-ordered unit ending in a full sync anyway; events would add a `gpu.h` API for no extra accuracy at the 10–1000 ms scale.
- **Fresh model per sweep point.** Each point builds its own model and Adam state, so a precision or graph change cannot inherit the previous point's caches. Cost: model init (~1–2 s at D=1024) per point.
- **Attention path is "auto".** `--flash-attn` is set: the TC score path runs when `B·H·T·T` floats fit in 256 MiB, otherwise tiled flash. The once-per-process `[attn]` log line says which one ran. The tiled backward on master shares one logsumexp across layers (fixed on the local `phase2-flash-bwd` branch); step timings are valid either way, but trust training only on the fixed build.
- **`--prec fp8`** sets BF16 compute plus `lm_train_set_fp8(1)` (E4M3 forward, E5M2 dgrad, BF16 wgrad), matching what `lm train --fp8 --bf16` does.
- **The train meter is not changed here.** `lm train` still prints the old TF estimate; switching it to `lm_bench_flops_per_token` changes every logged number and the TF gate scripts, so it lands with the CLI trim.
- **GPU only.** Every mode refuses to run on the CPU stub: CPU timings say nothing about the CUDA path. `make test` still checks `lm_train_bench_packed` for correctness (timings recorded, phases filled, CE falls) and the FLOP formula, on a D=128, 2-block model, without reporting speed.

## How it shows up in this repo

- [`lang/lm_bench.c`](../lang/lm_bench.c), [`lang/lm_bench.h`](../lang/lm_bench.h): `lm_bench_cfg`, `lm_bench_step`, `lm_bench_gemm`, `lm_bench_decode`, `lm_bench_sweep`, FLOP model, busy check, JSON writer.
- [`lang/lm_train.c`](../lang/lm_train.c): `lm_train_bench_packed` (the production step on a fixed batch) and the `--profile` phase timers in `pack_run_body`.
- [`lang/lm.c`](../lang/lm.c) `lm_create_vocab`, [`lang/vocab.c`](../lang/vocab.c) `vocab_sized_create`: BPE-sized head without a trained tokenizer.
- [`main.c`](../main.c) `run_lm_bench_mode`: CLI.
- [`scripts/bench_compare.py`](../scripts/bench_compare.py): match rows by mode + GPU + model + config, flag a p50 regression beyond `--threshold` (default 3%), a failed loss check, or a changed greedy hash.
- [`scripts/bench_gate.sh`](../scripts/bench_gate.sh): standard matrix → `bench/run-<sha>-<time>.jsonl`, `--record` stores `bench/baseline-<gpu>.jsonl`, otherwise compares. `bench/` is gitignored.
- [`Makefile`](../Makefile): `LMCFG` overrides `lang/lm_config.h` sizes; `make test` builds the small model into `objects-test/`.

## Worked example

```bash
make cuda
# One point: D=1024 model, T=256, B=64, BF16, graphs on, per-phase times
./anni lm bench step --window 256 --batch 64 --prec bf16 --profile
# GEMM ceilings for the model's shapes
./anni lm bench gemm --window 256 --batch 64 --prec bf16
# A/B matrix (Cartesian product of the lists)
./anni lm bench sweep --windows 64,256 --batches 32,64,128 --precs fp32,bf16,fp8 --graphs 0,1
# Decode speed + greedy hash
./anni lm bench decode --model models/wiki_bpe.bin --prompt "April is the" --tokens 128
# Baseline, then gate later commits
scripts/bench_gate.sh --record
scripts/bench_gate.sh
python3 scripts/bench_compare.py bench/baseline-NVIDIA_GeForce_RTX_5070_Ti.jsonl bench/run-*.jsonl
```

Stdout row format:

```text
step bf16 B=64 T=256 graphs=1 ckpt=0 off=0  p50 <ms> (p10 <ms> p90 <ms> cv <%>)  <tok/s>  <TF>  MFU <%>  VRAM <MB>  loss <a>-><b> ok
     phases p50: fwd <ms>  loss <ms>  bwd <ms>  adam <ms>
```

The matching JSON line carries the same numbers plus `git`, `dirty`, `gpu`, `driver`, `vram_mb`, `busy`, model dims and the full config.

## What next

- Record the first RTX 5070 Ti baseline once the GPU is free, and replace the Wave 2/3 tables in [speed-stack-bench.md](speed-stack-bench.md) with `sweep` rows.
- Use the sweep to decide the CLI-trim default flips (BF16, graphs, auto attention), then delete `lm bench --train`, `lm train --bench-gpu` and `lm_bench_train_ms`.
- Switch the train meter to `lm_bench_flops_per_token`.
- Add an `attn` mode (TC vs tiled, fwd + bwd, max abs difference) once the per-layer logsumexp fix is on `master`.
