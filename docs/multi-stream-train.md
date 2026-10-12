# Multi-stream train overlap (S9)

## Abstract

Overlap host→device staging with compute where a second CUDA stream helps. R8 already provides a non-blocking copy stream for embed id H2D; S9 ensures that stream is created for packed train and documents the remaining no-ops.

## How it shows up in this repo

- `gpu_copy_stream()` / `gpu_stream_sync_copy()` in [`num/gpu.h`](../num/gpu.h)
- Packed train calls `gpu_copy_stream()` during `pack_ctx` init ([`lang/lm_train.c`](../lang/lm_train.c))
- Embed gather uses the copy stream for async H2D ([`num/gpu_cuda.cu`](../num/gpu_cuda.cu))
- Adam gather/scatter and fused attn stay on the default stream (mutex-serialized); dual-stream Adam would need graph-safe event waits — deferred

## When it is a no-op

If all activations stay device-resident and ids are already pinned, the copy stream has little work. The API remains so prefetch (E3) and future offload (S13) can enqueue without new stream plumbing.
