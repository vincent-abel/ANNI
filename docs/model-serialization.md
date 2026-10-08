# Model serialization

## What problem it solves

Training is useless if you must start from scratch every run. **Serialization** writes the architecture and learned weights to a file so you can reload the same network later for inference or more training.

## How it works (layman)

The saver walks the graph in a fixed order: how many inputs, how many layers, neurons per layer, then each neuron’s bias and each incoming weight. The loader rebuilds an empty network with the same sizes, then pours those numbers back into the matching slots. Same walk order on both sides is what makes the round-trip line up.

## The math

No new learning math — only an ordered encoding of parameters \(\theta\) (all biases and weights) plus shape metadata \((n_{\text{in}}, n_{\text{layers}}, sizes[])\). Loading restores \(\theta\) exactly (within binary `double` precision).

## How it shows up in this repo

- [`serialize.h`](../serialize.h) / [`serialize.c`](../serialize.c): `network_save` / `network_load` use a small binary format with magic `ANN1`.
- Neuron and input lists are written head→`prev`, matching how the live graph is stored.
- Tests train XOR briefly, save, load, and compare outputs.

## Tiny example

Train XOR → `network_save("xor.ann")` → quit → `network_load("xor.ann")` → same predictions on the four rows.
