# GPU activation residency (R1)

## Abstract

Packed train still pulls LN / FFN / residual activations to the host between ops. R1 keeps those buffers on the device end-to-end so BATCH GEMM and elementwise kernels stop paying PCIe each microbatch.

## What problem it solves

At D=256 with packed `--batch`, util sits near 30% because host `copy_mat`, `resid_add`, `mat_axpy`, and `mat_ensure_host` stage activations every layer. GEMM FLOPs are not the bottleneck; PCIe and host glue are.

## How it works for a human

Pin every block / FFN / LN cache matrix with a device twin. Copy and residual add become device-to-device. LayerNorm, bias, and ReLU read and write `d_data` when it is authoritative. Packed CE gathers hidden rows with D2D when possible. Sync once per microbatch before Adam, as before.

## Formal view

For activation tensor \(A\) of \(N=B\cdot T\cdot D\) floats, a host round-trip costs \(2N\) elements of PCIe. With \(L\) layers and several staging points per layer, traffic scales as \(O(L N)\). Device residency reduces that to \(O(1)\) scalar syncs per step.

## Complexity / pitfalls / design choices

- Unpinned GEMM outputs still D2H in BATCH mode (`finish_out`); pinning is mandatory.
- Embed / posenc remain host until R2; the first `mat_host_dirty(hidden)` still uploads once.
- CE one-hot edit is still host; logits GEMM / bias / softmax can stay on device until that edit.
- CPU stub returns `-1` from `gpu_mat_*` so `make test` stays host-correct.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h): `gpu_mat_copy`, `gpu_mat_add`, `gpu_mat_axpy`
- [`num/mat.h`](../num/mat.h): `mat_compute_ptr`, `mat_mark_device`
- [`lang/block.c`](../lang/block.c), [`lang/resid.c`](../lang/resid.c), [`lang/ffn.c`](../lang/ffn.c), [`lang/layernorm.c`](../lang/layernorm.c), [`lang/lm.c`](../lang/lm.c)

## Tiny worked example

\(B=64\), \(T=64\), \(D=256\) → \(N\approx 1\text{M}\) floats ≈ 4 MB per activation. Ten host round-trips per layer × 4 layers ≈ 320 MB PCIe/step; residency drops most of that.

## What comes next

[GPU embed lookup](gpu-embed-lookup.md) (R2); [CUDA Graphs](cuda-graphs-train.md) (R12) once the microbatch is device-only.
