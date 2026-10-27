# Adam step on GPU

## Abstract

[`lm_adam_step`](../lang/lm_adam.c) packs scattered param/grad/moment slots into contiguous host arrays, then called a one-shot `gpu_adam_step` (since removed; the device-resident path uses [`gpu_adam_step_dev_bf16`](../num/gpu.h)). The CUDA kernel updates \(m,v,\theta\) and zeros grads in one pass. Stubs return `-1` so the existing host Adam loop remains the default.

## What problem it solves (plain language)

After backward fills grads, Adam walks every weight. Doing that walk on the GPU keeps the train step on-device once activations and grads already live there.

## How it works for a human (intuition, analogy)

Think of packing all knobs into one tray, sliding the tray into the GPU oven for one Adam bake, then putting each knob back on its machine.

## Formal view (definitions, formulas, shapes, algorithms)

\[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g^2
\]

\[
\hat m=\frac{m_t}{1-\beta_1^t},\quad
\hat v=\frac{v_t}{1-\beta_2^t},\quad
\theta\leftarrow\theta-\eta\frac{\hat m}{\sqrt{\hat v}+\varepsilon}
\]

## Complexity / numerical pitfalls / design choices

**Pack/unpack.** Slots stay pointer-based for checkpoints; each GPU step copies \(4n\) doubles H2D/D2H. Teaching cost; batch-boundary sync later can pin flats.

**Fallback.** Any CUDA failure leaves the CPU loop unchanged. `make test` never needs nvcc.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h), [`gpu_stub.c`](../num/gpu_stub.c), [`gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_adam_step_dev_bf16` / `adam_kernel`
- [`lang/lm_adam.c`](../lang/lm_adam.c): pack → try GPU → unpack; else host loop

## Worked example

On CPU builds `gpu_adam_step_dev_bf16` returns `-1` and `lm_adam_step` uses the host loop. With `make cuda`, the same call runs the device kernel and matches host moments within double noise on tiny \(n\).

## What next

[KV-cache generate on GPU](gpu-kv-generate.md).
