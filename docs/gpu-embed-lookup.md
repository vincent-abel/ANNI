# GPU embed lookup (R2)

## Abstract

Token embedding gather and grad scatter run on the device so the packed train stack no longer starts with a host table scan each microbatch.

## What problem it solves

Host `memcpy` of \(B\cdot T\) rows from a large vocab table and host scatter into `dtable` force PCIe and keep Adam’s embed segment on a host-use path.

## How it works for a human

Lookup launches a gather kernel from the pinned table using uploaded ids. Backward uses `atomicAdd` scatter into `dtable`. Adam treats embed like other mats (no host_use).

## Formal view

\(H_{t,:}=E_{id_t,:}\). \(\partial E_{id_t,:}\mathrel{+}= \partial H_{t,:}\). Both are \(O(n D)\) with \(n=B\cdot T\).

## Complexity / pitfalls / design choices

- Ids still upload each step (small); pinned host ids land in E5.
- Posenc remains host for now — one ensure_host after gather.
- Atomic scatter can serialize hot tokens; fine at Target scale.

## How it shows up in this repo

- [`num/gpu.h`](../num/gpu.h): `gpu_embed_lookup`, `gpu_embed_scatter`
- [`lang/embed.c`](../lang/embed.c), [`lang/lm_adam.c`](../lang/lm_adam.c)

## Tiny worked example

B=64, T=64, D=256 → 1M floats gathered once on device instead of host memcpy + H2D of hidden.

## What comes next

[Pinned host staging](pinned-host-staging.md) (E5) for ids; [CUDA Graphs](cuda-graphs-train.md) once posenc is device-side too.
