# Forward propagation

## In one page

Forward propagation runs the network from sample to guess. It writes the sample into the input buffer, then asks each layer, left to right, to recompute every neuron. At the end, the output layer's `output` fields are the prediction.

## The picture

A sample goes in the front door. Each room (layer) reads the notes pinned up by the previous room, does its arithmetic, and pins up new notes. The last room's notes are the answer. Nothing looks at the target during this pass. The target enters only when the loss and the backward pass run.

## The steps

For one sample, `network_train` does this, and you can do it by hand with `network_set_inputs` plus `network_forward`:

1. Copy the sample into `net->inputs`. The length must match `n_inputs`.
2. Walk layers from `net->layers` along `next`.
3. For each neuron in the layer, recompute `Tinput`, then `output`.

Step 3 in arithmetic, for every neuron:

```
total  = bias + Σ (value_on_wire * weight)
output = 1 / (1 + e^(-2 * total))
```

The first layer's wire values are the sample slots. The next layer's wire values are the `output` fields just written by the previous layer. That is why the walk has to be front to back. If the output layer ran first, it would multiply whatever leftover numbers were still sitting in the hidden neurons.

## A two-layer walk

Call the sample `x1, x2`. Hidden neuron `h` has bias `b_h` and weights `w_h1, w_h2`. The output neuron `y` has bias `b_y` and one weight `w_y` on `h`.

```
total_h = b_h + x1 * w_h1 + x2 * w_h2
h       = 1 / (1 + e^(-2 * total_h))

total_y = b_y + h * w_y
y       = 1 / (1 + e^(-2 * total_y))
```

The XOR network does the same thing with four hidden neurons and one output. Each hidden neuron has its own bias and its own two weights. The output neuron has its own bias and four weights, one per hidden output. Forward propagation is just that pair of formulas, repeated for every neuron.

## What comes out

Every output is between 0 and 1 because every neuron ends in the sigmoid. For XOR, a single output near 0 means "predict the bit 0", and a value near 1 means "predict the bit 1". The demo treats 0.5 as the cut: `output >= 0.5` counts as predicting 1.

The forward pass does not change weights or biases. It only refreshes `Tinput` and `output`. Running it twice on the same sample with the same weights gives the same guess.

## Where it lives

`network_set_inputs` copies the sample. `network_forward` calls `nlayer_forward`, which calls `neuron_output` on each layer. The weighted sum is `neuron_Tinput`.
