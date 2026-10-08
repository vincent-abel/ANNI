# LM head + cross-entropy backward

## Abstract

After softmax CE produces δ over the vocab, [`lm_loss_backward`](../lang/lm.c) updates `dw_head` with **`mat_outer_add(h_t, δ)`** (GPU outer when available) and accumulates `dhidden[t] += W δ`. Softmax itself stays on the host for the toy vocab.

## What problem it solves (plain language)

The LM head is a large dense map `D → V`. Its weight gradient is an outer product per time step. Routing that through `mat_outer_add` reuses the CUDA outer kernel already used by other modules.

## Formal view

\[
\delta = \mathrm{softmax}(z) - e_{y},\quad
dW \mathrel{+}= h^\top \delta,\quad
dh \mathrel{+}= W\delta
\]

## How it shows up in this repo

- [`lang/lm.c`](../lang/lm.c): `lm_loss_backward` — CE on host, `mat_outer_add` for `dw_head`, host GEMV for `dhidden`
- Relies on [`gpu_mat_outer_add`](../num/gpu.h) when `make cuda` and the device is up

## How to verify

```bash
make test
# with CUDA: make cuda && ./ann lm train ...  (dw_head path hits GPU outer)
```

## What next

[Adam step on GPU](gpu-adam.md).
