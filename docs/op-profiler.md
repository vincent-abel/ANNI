# Where the step's time goes: the built-in profilers

## Abstract

Two switches make a training run report its own timing: `ANN_STEP_PROF=1` prints forward, loss and backward times for the first few microbatches, and `ANN_OP_PROF=1` prints a table of GPU time per kind of operation at every optimizer step. Every speed change in the GPU notes was chosen and checked with these.

## What problem it solves (plain language)

"The step takes 60 ms" says nothing about what to fix. An outside profiler was not an option on the development machine (it recorded no CUDA trace there), and wall-clock timing from the host lies on a GPU: a call returns as soon as the work is queued, long before it is done.

## How it works for a human (intuition, analogy)

A stopwatch that travels with the parcel. Timing when you hand a parcel to the courier tells you nothing; a stamp made at each depot it passes does.

## Formal view (definitions, formulas, shapes, algorithms)

**Per operation.** Each `gpu_*` entry point opens a scope with a category (`gemm_fwd`, `gemm_dgrad`, `gemm_wgrad`, `qkv_fused`, `attn_fwd`, `attn_bwd`, `ln_fwd`, `ln_bwd`, `ce`, `colsum`, `resid`, `embed`, `adam`, `convert`, ...). The scope records a CUDA event on the stream when it opens and another when it closes. Events are timestamps taken by the GPU when the queue reaches them, so the difference is the time the work really took. Pairs are resolved at the next device sync and summed per category.

**Per phase.** The host syncs the device after the forward, after the loss, and after the backward, and reads the clock each time.

## Complexity / numerical pitfalls / design choices

- **Off by default, and cheap when on.** Recording an event costs far less than a kernel. The phase profiler is different: its syncs stop the host from queueing ahead, so it only runs for the first 8 microbatches.
- **Nested scopes** are counted once, in the outer category, except a few marked sub-steps (attention packing, conversions) that get their own row. `convert` time is also inside the `gemm_*` rows, and the total says so.
- **Thread-safe.** Some paths call the GPU from several OpenMP threads; the slot table is under a mutex. An event error is reported once and cleared: the profiler can never fail a training step.
- **Per microbatch.** The table divides by the number of loss calls since the last dump, so the numbers stay comparable when gradient accumulation changes.
- It also prints gradient-twin hits and misses, which is how a [twin](bf16-twins.md) that silently stopped being used gets noticed.

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `OpScope`, `op_begin` / `op_end`, `gpu_op_prof_dump`.
- [`lang/lm_adam.c`](../lang/lm_adam.c) calls the dump at each optimizer step; [`lang/lm_train.c`](../lang/lm_train.c) holds the phase timers.
- [`scripts/prof_step.sh`](../scripts/prof_step.sh): `bench` (12 steps, tok/s and phase times) and `gate` (12 steps with validation probes and a checkpoint save while VRAM is sampled, pass or fail against 85% of the card).
- `anni lm bench step --profile` gives the same phase times on a fixed batch ([Bench harness](bench-harness.md)).

## Use

```text
ANN_OP_PROF=1 ./anni lm train ... 2> prof.log

[opprof] <n> microbatches, <ms> ms/microbatch of GPU op time (convert counted inside gemm rows); grad twin hits <n> misses <n>
[opprof]   gemm_fwd    <ms> ms/mb  <pct>%  <n> calls/mb
[opprof]   attn_bwd    <ms> ms/mb  <pct>%  <n> calls/mb
...

ANN_STEP_PROF=1 ./anni lm train ...

[stepprof] fwd=<ms>ms loss=<ms>ms bwd=<ms>ms total=<ms>ms
```
