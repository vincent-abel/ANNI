# CUDA graphs of the training microbatch (R12 / S4)

## Abstract

One BF16 microbatch of the 170M model (B·T = 4096 tokens, 12 layers) is about 640 kernel launches with roughly 9 µs of idle GPU between them, about 7% of a 63 ms microbatch. `--cuda-graphs` now captures one steady microbatch (forward, cross-entropy, backward) into a CUDA graph and replays it for every later microbatch of the same shape: one `cudaGraphLaunch` instead of 640 launches. The arithmetic is unchanged: the replay runs the same kernels, with the same arguments, on the same buffers as an uncaptured microbatch, and `make test-gpu` trains with and without graphs and requires the same losses and gradients.

The first version of this flag never captured anything: it captured the legacy stream 0, which `cudaStreamBeginCapture` refuses. Had it captured, the replays would have been wrong: the token ids and targets were uploaded from host buffers inside the captured region, and the head bias gradient was skipped under capture.

## What had to change

A graph freezes device addresses and kernel arguments, and nothing on the host runs during a replay. Every microbatch therefore has to be device-only work on fixed buffers.

- **Compute stream.** Every kernel, cuBLAS / cuBLASLt call and async copy of the training path goes to `g_st`, a blocking stream (`cudaStreamCreate`): it orders itself with the legacy stream exactly as stream 0 did, so the synchronous copies keep their meaning, and unlike stream 0 it can be captured.
- **Token ids.** `gpu_ids_stage` copies the microbatch's ids into a two-slot page-locked ring and, on the compute stream, into one fixed device buffer, *before* the graph launch (outside the graph). The host only waits for the ring slot's copy from two microbatches back, never for the compute. The embedding lookup and scatter read that buffer.
- **Targets.** `gpu_ce_targets_staged` builds the next-token targets on the device from the staged ids (`next_token_targets_kernel`): no host target array at all. Loss-masked (chat) and varlen batches, whose masks are host data, never take the graph path.
- **Head bias gradient.** In graph mode `db_head` accumulates on the device in a `colsum_acc` like every other bias gradient, flushed by `gpu_device_sync` at the optimizer step. Each microbatch's column sums still land in their own zeroed buffer first and are then added to the accumulator element by element (`vec_acc_kernel`): the same FP32 additions in the same order as the host loop of the plain path. The mean CE is read from the pinned mirror after one stream sync, only when the training meter asks for it (every 8th microbatch).
- **Epilogue biases.** The W1 / projection GEMMs uploaded their host bias with a synchronous `cudaMemcpy` into one scratch buffer on every call. In graph mode they read the bias's persistent device copy (`bias_cache_get`), refreshed once per optimizer step like the LayerNorm vectors.
- **Allocation.** Each microbatch creates and frees its mat temporaries (`gpu_malloc` / `gpu_free`), and `cudaMalloc` / `cudaFree` are illegal inside a capture. In graph mode the two recycle exact-size blocks (LIFO per size): a steady microbatch asks for the same sizes in the same order, so after one uncaptured microbatch the capture finds every block in the pool and allocates nothing.
- **Capture-only branches.** The fused QKV GEMM, the BF16-only ReLU and LayerNorm outputs, the producer-written gradient twins and the fused ReLU backward all refused to run under capture, so the capture fell back to other kernels and the graph was not the microbatch it replaced. They now run unchanged; their slots exist from the uncaptured microbatch, and an allocation attempted inside the capture throws the capture away. The parallel-streams QKV fallback (a host wait per call) cannot be captured: a capture that would need it is dropped rather than run on another GEMM path.
- **Gradient-twin round robin.** It restarts every microbatch in graph mode, so each microbatch reserves the same slots and none has to grow inside a capture.

## When a graph runs

`lm_graph_step` (lang/lm.c) drives it; `pack_run` calls it when `--cuda-graphs` is on.

1. The first microbatch after every optimizer step runs **uncaptured**: it rebuilds the BF16 weight twins and re-uploads the biases (once per weight epoch, `gpu_param_epoch`), and leaves every buffer and pool block in place.
2. The next microbatch with the same B and T is **captured**. The capture only records, so the graph is then launched once for that microbatch.
3. Every later steady microbatch **replays** it; afterwards the host clears the gradient twins, its only per-microbatch host state.

With the live run's `--grad-accum 32`, 31 of 32 microbatches are replays.

A replay is refused, and the next steady microbatch captures again, whenever an address the graph holds may have changed owner (`gpu_graph_replayable`):

- any `cudaFree`, or the return to the pool of a block that is not a graph temporary, bumps `g_dev_free_gen`, and a graph from an older generation is never launched;
- a graph temporary lent to an uncaptured microbatch and not yet returned would be overwritten: the replay waits until `g_pool_graph_out` is zero.

A capture is thrown away (and the microbatch run for real) if anything inside it allocated, freed, synchronised or took an illegal path (`g_capture_bad`, `cudaStreamEndCapture` failing). After three such failures the run goes on without graphs; `ANN_GRAPH_DEBUG=1` prints why, with a backtrace of the refused call.

At every `gpu_device_sync` (the optimizer step) the pool gives back the idle blocks no graph uses (an evaluation's leftovers), so it never holds more than a microbatch needs.

## Results

`make test-gpu`, check 7 (small test model, B = 4, T = 64, three microbatches per Adam step, four steps): 8 replays out of 12 microbatches. Every per-microbatch loss and every gradient (weights, embedding, head bias, an FFN bias, a LayerNorm gain) matches the plain run within the plain run's own run-to-run noise. The plain run is not bitwise reproducible: float atomics in the CE sum, the bias column sums and the embedding scatter.

Five consecutive `make test-gpu` runs, worst values:

| Precision | graphs vs plain: max \|Δloss\| | max grad rel. diff | plain vs plain: max \|Δloss\| | max grad rel. diff |
|---|---|---|---|---|
| BF16 | 5.3e-6 | 9.7e-8 | 4.8e-6 | 1.0e-7 |
| FP32 | 4.8e-6 | 4.7e-7 | 4.4e-6 | 4.8e-7 |

A stale id, target or bias in a replay moves those numbers by 1e-3 and more; the test's bounds (2e-5 on a loss of ~5.5, 2e-6 relative on a gradient) sit just above the noise.

The `lm train` loop at the test dimensions (2 epochs, B = 8, T = 64, grad-accum 4, BF16, FA2) ends at the same train / validation NLL with and without `--cuda-graphs`: 3.5203 / 3.5789 against 3.5203 / 3.5790.

Not measured yet: the saving on the 170M run (expected up to ~4 ms of a 63 ms microbatch). Its VRAM rules out testing it at full size outside the training thread's pause-window gate.

## Pitfalls

- Summation order is part of "the same math". A first version summed each microbatch's head-bias column sums straight into the accumulator, which reassociates FP32 additions: still a valid gradient, but in about one run of three it moved one FFN pre-activation of the test model across zero by step 4 (a 2.7e-5 gradient gap, with or without any replay). Keeping the plain path's order removed it (0 of 14 runs).
- Never add a `stream_is_capturing()` branch that changes *what* is computed: a capture must record exactly the uncaptured microbatch. A path that cannot be captured refuses the capture (`capture_refuse`).
- Host data that changes between microbatches must not be read by a captured copy, pinned memory included (a replay reads it at replay time): stage it before the launch, as the ids are.
- `cudaPointerGetAttributes` is legal during a capture (probed on CUDA 13.3 in all three capture modes). The old shortcut that called every pointer "device" under capture would have handed a host pointer to a kernel.
- `gpu_malloc` / `gpu_free` run on OpenMP threads too (per-head attention in evaluation): the pool has its own mutex.

## How it shows up in this repo

- [`lang/lm.c`](../lang/lm.c): `lm_graph_step`, `lm_graph_enable`, device targets in `lm_loss_backward_packed_lens`
- [`lang/lm_train.c`](../lang/lm_train.c): `pack_run` → `lm_graph_step`; `lm_train_set_cuda_graphs`
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_graph_*`, the block pool (`pool_*`), `gpu_ids_stage`, `gpu_ce_targets_staged`, graph-mode CE
- [`tests/test_gpu.c`](../tests/test_gpu.c): `check_graphs_match`
- CLI: `./anni lm train --cuda-graphs …` (default off)
