# Backpropagation

## In one page

Backpropagation assigns blame for the loss. It starts at the output, where the guess can be compared with the target, and walks toward the input. Each neuron ends up with a `delta`, and each wire ends up with a `dw`. Gradient descent applies those numbers. This pass only computes them.

## The picture

The output neuron notices it was too high or too low. It asks each incoming wire: "given the input you carried, how should your weight change?" It also tells each hidden neuron that fed it: "your output, multiplied by the weight on our wire, is how you affected me." Those hidden neurons scale that message by how sensitive their own S-curve currently is, then repeat the conversation with the layer behind them. The sample itself is never edited. Wires that read the sample adjust their weights and stop.

## Clearing the slate

`network_backward` first walks every layer and sets `delta`, `dbias`, and every `dw` to 0. One sample's blame must not leak into the next sample. The training loop calls backward once per sample, so the gradients always describe the sample just shown.

## At the output

For each output neuron paired with a target:

```
delta = (output - target) * output * (1 - output)
```

`(output - target)` is the gap from the squared-error loss, without the constant factors described in the loss note. `output * (1 - output)` is the slope of the sigmoid at the current output. Multiplying them answers: "how should the raw total change in order to reduce the loss?"

Then the neuron records local lessons and pushes blame one hop backward (`neuron_backward_accumulate`):

```
dbias        = delta
dw           = delta * input_value      for each wire
source_delta = source_delta + delta * weight
```

`dbias` equals `delta` because the bias is added with a factor of 1. `dw` multiplies by the input value because that value is what the weight was multiplied by. `source_delta` multiplies by the weight because that weight is what the source output was multiplied by. The `+=` on the source matters when several outputs depend on the same hidden neuron: each output adds its own share.

## In hidden layers

The walk follows `prev`, from the layer just behind the output back to the first layer. A hidden neuron's `delta` has already been filled by the layer in front, as a sum of `downstream_delta * weight` over the wires that leave it.

Before that neuron teaches anyone behind it, it scales by its own slope:

```
delta = delta * output * (1 - output)
```

Then it runs the same accumulate step: set its `dbias` and each `dw`, and add `delta * weight` into the neurons that feed it.

A one-output, one-hidden-neuron sketch makes the chain visible. `y` is the output, `h` is the hidden output, `w` is the weight from `h` to `y`, `t` is the target.

```
delta_y = (y - t) * y * (1 - y)
dw_y    = delta_y * h
dbias_y = delta_y

delta_h = (delta_y * w) * h * (1 - h)
```

If the hidden neuron also has a wire from sample value `x` with weight `w_x`:

```
dw_x    = delta_h * x
dbias_h = delta_h
```

There is no `src` on that sample wire, so nothing is added into a further neuron. The chain stops.

## Why the slope is in the middle

Skipping `output * (1 - output)` would treat every neuron as equally sensitive. A saturated neuron, stuck near 0 or 1, would still try to make large edits. The slope near those ends is almost 0, so `delta` collapses and the edits stay small. A neuron sitting near 0.5 has slope 0.25 and can still move.

## Where it lives

`network_backward` in `layer.c` zeros gradients, scores the output layer, then walks hidden layers backward. `neuron_backward_output`, `neuron_backward_hidden`, and `neuron_backward_accumulate` in `neuron.c` are the three pieces of arithmetic above.
