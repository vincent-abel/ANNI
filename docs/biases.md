# Biases

## In one page

A bias is a number added inside the neuron before the S-curve. It shifts the weighted sum up or down. Without it, a neuron whose inputs are all zero would be stuck at the same output every time, and the S-curve would always be centered on zero.

## The picture

Weights answer "how much does each input matter?" The bias answers "where is my starting line?"

Imagine a light sensor that should turn on when a room is bright. The weights measure the light. The bias is the threshold written as an offset: some sensors should already be halfway on in the dark, others should stay off until the light is strong. That preference is the bias.

In the formula it sits in the same sum as the weighted inputs:

```
total = bias + (input_1 * weight_1) + (input_2 * weight_2) + ...
```

You can also see the bias as a weight on a fake input that is always 1:

```
total = (1 * bias) + (input_1 * weight_1) + ...
```

Both views are the same arithmetic. The code uses the first one: `neuron_Tinput` starts `Tinput` at `bias`, then adds each `input * weight`.

## A numeric shift

Take weights that produce a weighted sum of 0. The S-curve of 0 is 0.5, the middle of the range.

```
weighted sum = 0
bias         = 0
total        = 0
output       ≈ 0.50
```

Add a bias of 1:

```
total  = 1
output = 1 / (1 + e^(-2 * 1)) ≈ 0.88
```

Same inputs, same weights, higher bias, higher output. A bias of -1 pulls that same neuron down to about 0.12. The bias slides the neuron along the S-curve without needing any particular input to be large.

## How a bias learns

The backward pass sets:

```
dbias = delta
```

`delta` is the error signal after it has been scaled by the slope of the S-curve. The bias gets that signal directly, with no extra multiplication, because the bias is added with a coefficient of 1. Changing the bias by a little changes `total` by that same little.

The update is the same hill-climbing step used for weights:

```
bias = bias - learning_rate * dbias
```

If the neuron's output was too high, `delta` is positive, and the bias moves down. If the output was too low, `delta` is negative, `learning_rate * dbias` is negative, and subtracting a negative number moves the bias up.

## How biases are born

`neuron_push` draws each bias from -0.5 to 0.5. That is a narrower range than the weights (-2 to 2). A new neuron starts near the middle of the S-curve, then training walks the bias to wherever the data needs it.

`dbias` starts at 0 and is cleared after each update, same as a weight's `dw`.

## Where it lives

`neuron.bias` and `neuron.dbias` are in `neuron.h`. The add happens in `neuron_Tinput`. The gradient is accumulated in `neuron_backward_accumulate`. The step is in `neuron_update`.
