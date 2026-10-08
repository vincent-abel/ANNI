# Host↔device sync at batch boundaries

## Abstract

[`gpu_set_sync_mode`](../num/gpu.h) switches GEMM helpers between **eager** D2H (default) and **batch** mode, which leaves pinned outputs on the device. [`lm_train_corpus`](../lang/lm_train.c) enables batch mode for each mini-batch, then `mat_ensure_host` on grads/activations before Adam and runs NLL only at epoch boundaries.

## What problem it solves (plain language)

Copying every GEMM result over PCIe after each multiply wastes bandwidth. Syncing once per optimizer step (and metrics once per epoch) cuts chatter so the GPU stays busy.

## How it works for a human (intuition, analogy)

Instead of mailing every page of homework home after each line, keep the notebook on the GPU desk for the whole problem set, then send the finished packet once.

## Formal view (definitions, formulas, shapes, algorithms)

\[
\texttt{GPU\_SYNC\_EAGER}:\ \forall\ \mathrm{op},\; C_{\mathrm{host}}\leftarrow C_{\mathrm{device}}
\]

\[
\texttt{GPU\_SYNC\_BATCH}:\ \text{if }C.d\_data,\; C.\mathrm{on\_device}\leftarrow 1\text{ (no D2H)}
\]

Train step: `BATCH` → forward/backward windows → `mat_ensure_host` → `EAGER` → Adam → … → epoch NLL on host.

## Complexity / numerical pitfalls / design choices

**Pinned only.** Batch mode skips D2H only when the output mat already has `d_data`. Temporary device buffers still download.

**Host still needed.** Softmax CE, Adam pack, and scale_grads read host grads today; boundary sync feeds them. Token ids stay on the host.

**Stub.** Sync mode is recorded on CPU builds; GEMMs never run on device.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h): `GPU_SYNC_*`, `gpu_set_sync_mode`
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): mat_mul / mul_bt / outer_add honor batch mode
- [`lang/lm_train.c`](../lang/lm_train.c): batch window around forward/backward

## Worked example

`make test` stays on the stub. With `make cuda`, train keeps GEMM results on device between windows in a mini-batch and only pulls them for Adam.

## What next

[Train util proof](gpu-train-util-proof.md).
