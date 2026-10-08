# Mini-batches

## What problem it solves

Updating weights after **every single example** is noisy: each sample pulls the model a different way. Updating once after seeing **a small group** (a mini-batch) averages those pulls, so training is steadier and often faster in wall-clock terms when you later vectorize.

## How it works (layman)

Pick a few training rows (here a chunk of the sample table). Run forward and backward on each, **adding** the weight gradients together. Divide by how many rows you used (mean gradient), then take one optimizer step. Next chunk, repeat. Epochs still mean “pass over the whole dataset,” just in bites.

## The math

For batch \(\mathcal{B}\) of size \(B\):

\[
g = \frac{1}{B}\sum_{x \in \mathcal{B}} \nabla_\theta L(x,\theta),
\quad
\theta \leftarrow \theta - \eta\, g
\]

Single-example SGD is the special case \(B = 1\).

## How it shows up in this repo

- [`layer.c`](../layer.c): `network_train_mb` zeros grads, calls `network_backward_accum` per sample in the chunk, `network_scale_grads(1/B)`, then `network_update`.
- `neuron_zero_delta` clears only deltas between samples; `neuron_zero_grad` clears `dw`/`dbias` at batch start.
- XOR can train with `batch_size` 2 or 4 on the four-row table.

## Tiny example

XOR with `batch_size=2`: pairs `(00,01)` then `(10,11)` each produce one averaged update per epoch half.
