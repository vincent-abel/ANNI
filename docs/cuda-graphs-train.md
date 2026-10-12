# CUDA Graphs train (R12 / S4)

## Abstract

`gpu_graph_begin/end/replay` capture a fixed default-stream packed fwd+bwd for replay when `B` and `T` are steady. Capture fails if host ops remain in the region; after S1–S3 residency, enable with `--cuda-graphs`.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h): `gpu_graph_*`
- [`lang/lm_train.c`](../lang/lm_train.c): `pack_run` capture/replay; `lm_train_set_cuda_graphs`
- CLI: `./anni lm train --cuda-graphs …`
- Live train CE EMA is skipped on replayed steps (`mean_ce` unset); epoch-end / mid val probes unchanged
