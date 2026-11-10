# GPU residency, step by step: who owns the bytes

## Abstract

Every matrix has a host pointer, a device pointer and a small state saying which side holds the truth. Weights and gradients are pinned on the device once, activations are created there and never get a host copy unless someone asks, and the device is synchronized once per microbatch instead of after every launch. This is the plumbing under every other GPU note.

## What problem it solves (plain language)

The first GPU version copied operands to the card, ran one GEMM and copied the result back. The multiply was fast and the step was slow: most of the time went into allocations, copies over PCIe and waiting for each launch to finish.

## How it works for a human (intuition, analogy)

A workshop across town. Shipping every part there and back for each operation wastes the day on the road. Better: move the stock there once, keep the work in progress there, and only bring back what somebody on this side actually needs to read, when they ask for it.

## Formal view (definitions, formulas, shapes, algorithms)

A `mat` carries `data` (host), `d_data` (device) and `on_device`:

| State | Truth is | Typical use |
|-------|----------|-------------|
| `MAT_DEV_HOST` | on the host | freshly loaded weights, host-only builds |
| `MAT_DEV_DEVICE` | on the device | activations, gradients during a step |
| `MAT_DEV_SYNCED` | both, identical | weights between optimizer steps |
| `MAT_DEV_ALIASED` | on the device, as a view into the flat Adam buffer | weights and gradients while training |

Rules:

1. A GPU op calls `mat_compute_ptr` on its operands: it uploads only if the host is the authority, then marks its output `MAT_DEV_DEVICE`.
2. Host code calls `mat_ensure_host` before reading: it downloads only if the device is the authority.
3. Host code that writes calls `mat_host_dirty`, so the next GPU op uploads again.

## Complexity / numerical pitfalls / design choices

- **Stale copies are the classic bug.** Reading `data` without `mat_ensure_host` returns old numbers with no error. Most early residency bugs were exactly this.
- **Device-only buffers.** `mat_create_dev` allocates no host memory at all; the host side appears lazily on the first `mat_ensure_host`. At training sizes this saves gigabytes of RAM that would never be read.
- **One sync per microbatch.** In `GPU_SYNC_BATCH` launches are queued and the host waits once, at the loss. `GPU_SYNC_EAGER` waits after each op and is kept for debugging.
- **Aliased pointers must not be freed.** An aliased `d_data` belongs to the Adam flat buffer; `mat_device_free` leaves it alone.
- **Small ops had to move too.** One host-side softmax, LayerNorm or ReLU in the middle of the forward forces a download and an upload, so they all became kernels.

## How it shows up in this repo

- [`num/mat.h`](../num/mat.h) / [`num/mat.c`](../num/mat.c): the states and `mat_try_pin`, `mat_create_dev`, `mat_compute_ptr`, `mat_ensure_host`, `mat_host_dirty`, `mat_mark_device`, `mat_release_host`.
- [`num/gpu.h`](../num/gpu.h): `gpu_set_sync_mode`, `gpu_device_sync`, `gpu_stream_sync`.
- [`lang/lm_train.c`](../lang/lm_train.c) sets `GPU_SYNC_BATCH` around each optimizer step.

The README checklist items, in the order they landed:

| Item | What moved | Note |
|------|------------|------|
| **G1** | weights and gradients pinned at create, no `cudaMalloc` per GEMM | [GPU device buffers](gpu-device-buffers.md) |
| **G2** | upload only when the host is the authority, `mat_ensure_host` before host reads | this note |
| **G3** | one device sync per microbatch | [GPU batch sync](gpu-batch-sync.md) |
| **K1** | attention softmax rows | [GPU attention backward](gpu-attention-backward.md) |
| **K2** | LayerNorm forward | [One-pass LayerNorm](layernorm-one-pass.md) |
| **K3** | FFN bias, ReLU and its mask | [Fused epilogues](fused-epilogues.md) |

After these: [activation residency](gpu-activation-residency.md), [Adam on the device](gpu-adam-residency.md), then the BF16 work ([BF16 twins](bf16-twins.md), [BF16 Adam](bf16-adam.md)).
