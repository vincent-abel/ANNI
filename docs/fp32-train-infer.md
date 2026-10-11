# FP32 train and inference

## Abstract

The LM today stores every weight and activation as `double`. The plan is to move training and inference to single precision (`float`, FP32) behind one typedef, `ann_real`, and to use `cublasSgemm` on the GPU. This is a **migration to FP32**, not to pure FP16 yet.

## What problem it solves

Doubles cost twice the memory and half the GEMM throughput of floats on most GPUs. A model that is meant to grow past the toy scale hits both walls. FP32 is the standard training precision, so it halves cost without the loss-scaling and overflow care that FP16 demands.

## How it works for a human

Imagine every number in the model written with 15 digits when 7 are plenty. FP32 keeps 7. Weights, activations, gradients, and Adam moments all shrink by half. Reductions that need care (softmax sums, loss, LayerNorm statistics) may still accumulate in a wider type, then round once at the end.

## Formal view

Let \(\mathrm{fl}_{32}(x)\) round to binary32 with unit roundoff \(u_{32}=2^{-24}\approx 6\times10^{-8}\), versus \(u_{64}=2^{-53}\). A dot product of length \(n\) has error bounded roughly by \(n\,u\,\sum|a_i b_i|\), so \(n=256\) at FP32 still gives relative error near \(10^{-5}\), well under SGD noise.

## Complexity / pitfalls / design choices

- **Single typedef:** `ann_real` is `float` in the LM; the MLP code may stay `double` until it needs to change.
- **Checkpoints:** a new magic, `LMCF` (float), distinguishes float files from today's `LMCK` (double). Loaders reject a mismatched magic instead of reinterpreting bytes.
- **GPU:** `cublasDgemm` becomes `cublasSgemm`; the teaching CUDA kernels switch types too.
- **Tests:** finite-difference checks need looser epsilons at FP32.
- **Not FP16:** half precision needs loss scaling and master weights. That is deferred.

## How it shows up in this repo

- Types in [`lang/lm_config.h`](../lang/lm_config.h) and the matrix layer
- Checkpoint format in [`lang/lm_serialize.h`](../lang/lm_serialize.h) (`LM_MAGIC` today, `LMCF` planned)
- GPU path described in [Optional cuBLAS / cuDNN](cublas-cudnn.md)

## Tiny worked example

\(0.1\) in FP64 is `0.1000000000000000055…`; in FP32 it is `0.100000001490116…`. Over a 256-term dot product the difference stays near \(10^{-6}\), invisible next to a learning-rate step of \(10^{-3}\).

## What comes next

[Cheaper window eval](cheaper-window-eval.md) cuts eval cost; [GPU Adam residency](gpu-adam-residency.md) keeps FP32 state on device. FP16 with loss scaling is a later step.
