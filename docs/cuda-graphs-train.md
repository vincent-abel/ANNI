# CUDA Graphs train (R12)

## Abstract

`gpu_graph_begin/end/replay` capture a fixed default-stream microbatch for replay. Capture fails while host ops remain; after full residency it cuts launch overhead.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h): graph API
