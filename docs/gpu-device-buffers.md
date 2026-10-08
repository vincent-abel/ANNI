# Device buffers for activations

## Abstract

Each `mat` may own a **device twin** (`d_data`) beside the host `data` buffer. Helpers allocate, free, and copy H2D/D2H through [`gpu.h`](../num/gpu.h). Attention and LM forward caches **try to pin** H/Q/K/V on the GPU when CUDA is linked so later kernels can keep activations on device across layers.

## What problem it solves (plain language)

Sending every matrix to the GPU for one multiply then copying it back wastes the bus. Keeping a live copy on the device means the next layer can reuse it without another upload.

## How it works for a human (intuition, analogy)

Two notebooks: one on your desk (CPU) and one in the lab (GPU). `mat_to_device` photocopies desk→lab; `mat_to_host` lab→desk. Pinning reserves the lab notebook so you stop photocopying every page.

## Formal view (definitions, formulas, shapes, algorithms)

```text
mat {
  data      // host doubles
  d_data    // device doubles or NULL
  on_device  // 1 if d_data is authoritative
}

mat_device_alloc(m) -> gpu_malloc(rows*cols*sizeof(double))
mat_to_device(m)    -> alloc + H2D; on_device=1
mat_to_host(m)      -> D2H; on_device=0
mat_try_pin(m)      -> alloc if gpu_available(); ignore failure
```

**Stub.** Default `gpu_stub` returns NULL / -1 so CPU builds never touch CUDA.

## Complexity / numerical pitfalls / design choices

**Lifetime of truth.** Host `data` remains valid for all existing CPU code; `on_device` is advisory until later commits keep compute on device end-to-end.

**Lifetime.** `mat_free` always calls `mat_device_free`.

**Doubles.** Device buffers match host `double` (no FP16 here).

## How it shows up in this repo

- [`num/mat.h`](../num/mat.h) / [`num/mat.c`](../num/mat.c): `d_data`, pin/sync helpers
- [`num/gpu.h`](../num/gpu.h): `gpu_malloc`, `gpu_memcpy_*`
- [`num/gpu_stub.c`](../num/gpu_stub.c) / [`num/gpu_cuda.cu`](../num/gpu_cuda.cu)
- [`lang/attn.c`](../lang/attn.c), [`lang/lm.c`](../lang/lm.c): `mat_try_pin` on caches

## Worked example

On a CPU build, `mat_try_pin` leaves `d_data == NULL`. After `make cuda` with a device, the same call allocates `d_data` and `gpu_mat_mul` can reuse it.

## What next

[CUDA GEMM backward](cuda-gemm-backward.md) writes `dA`/`dB` on device for training.
