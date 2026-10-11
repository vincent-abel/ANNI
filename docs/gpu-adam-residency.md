# GPU-resident Adam (Track C)

## Abstract

Today the Adam step packs flat param/grad/moment buffers and may copy them across PCIe each batch. Track C keeps **parameters, gradients, and Adam moments resident on the device** and does gradient scaling there too.

## What problem it solves

Host↔device traffic of the whole parameter set each step can cost more than the math. For a model with \(P\) parameters, moving weights, grads, \(m\) and \(v\) is \(4P\) elements per step each way.

## How it works for a human

Upload the weights once. Every step, backward writes gradients into device memory, a small kernel divides them by the batch size (and optionally clips), and another kernel applies Adam, all without leaving the card. Only the loss scalar and occasional checkpoints come back to the host.

## Formal view

Per element: \(g\leftarrow g/B\); \(m\leftarrow\beta_1 m+(1-\beta_1)g\); \(v\leftarrow\beta_2 v+(1-\beta_2)g^2\); \(\theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)\). All are elementwise, so one fused kernel is \(O(P)\) with a single pass over memory.

## Complexity / pitfalls / design choices

- **Memory:** weights + grads + \(m\) + \(v\) is \(4P\) floats (hence FP32 first).
- **Sync points:** download only for saving a checkpoint, eval on CPU, or logging.
- **Scale on device:** avoids a host pass over all grads before the step.
- **Fallback:** the CPU stub keeps the host loop so `make test` needs no CUDA.

## How it shows up in this repo

- [`lang/lm_adam.c`](../lang/lm_adam.c): `lm_adam_step_scaled(opt, lr, 1/B)` folds batch scaling into Adam. On GPU, persistent host/device flats are allocated once; `m`/`v` stay on the device across steps (uploaded on first step, synced back by `lm_adam_sync_host` before checkpoint save); each step uploads only packed param+grad and downloads param.
- [`num/gpu.h`](../num/gpu.h): `gpu_adam_step_dev` (device pointers, `grad_scale` argument) and the one-shot host wrapper `gpu_adam_step`.
- [`lang/lm_train.c`](../lang/lm_train.c): no host `scale_grads` pass and no batch-boundary `mat_ensure_host`.
- [Adam step on GPU](gpu-adam.md), [Host↔device sync](gpu-batch-sync.md)

## Tiny worked example

\(P=10\text{M}\) floats = 40 MB per buffer. Four buffers copied each way every step is about 320 MB of PCIe traffic; residency reduces that to a few bytes for the loss.

## What comes next

[FP32](fp32-train-infer.md) halves these buffers; FP16 with master weights could halve them again.
