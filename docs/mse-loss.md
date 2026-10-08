# MSE loss

## In one page

Mean squared error is the score for one guess. It takes each output, subtracts the target, squares that gap, adds the squares, and divides by how many outputs there were. Smaller is better. Zero means every output matched its target.

## The picture

The network says 0.8 and the teacher says 1. The miss is 0.2. Another output says 0.1 and the teacher says 0. The miss is 0.1. Squaring treats "too high" and "too low" as the same kind of mistake, because a negative gap times itself is positive. Squaring also makes a miss of 0.5 much worse than a miss of 0.1: 0.25 against 0.01. Big confident errors dominate the score.

## The formula

For outputs `y1 ... yn` and targets `t1 ... tn`:

```
loss = ((y1 - t1)^2 + (y2 - t2)^2 + ... + (yn - tn)^2) / n
```

That is `network_mse`. If the number of output neurons and `n_target` disagree, the function returns -1, which this code uses as "the score is not valid."

The unit test pins the formula down with one output:

```
guess  = 0.75
target = 0.25
gap    = 0.50
loss   = (0.50 * 0.50) / 1 = 0.25
```

Two outputs work the same way. Guess `[0.0, 1.0]` against target `[0.0, 0.0]`:

```
loss = (0^2 + 1^2) / 2 = 0.5
```

## A whole dataset

`network_mse_dataset` runs a forward pass on every sample, scores each sample with `network_mse`, and averages those scores:

```
dataset_loss = (loss_sample_1 + ... + loss_sample_N) / N
```

For XOR there are 4 samples and 1 output, so this is the average of four squared gaps. The training test expects that average to fall below 0.01 after 5000 epochs from a fixed random seed.

## What the backward pass uses from this

The derivative of `(y - t)^2` with respect to `y` is `2 * (y - t)`. Divide by `n` because of the average, and the slope of the loss against one output is:

```
(2 / n) * (y - t)
```

The backward pass in this program uses the gap itself:

```
(y - t)
```

It then multiplies by the sigmoid slope. The `2` and the `/ n` are constant scales. Dropping them does not flip the direction of a step. It changes the step length, which the learning rate already chooses. With a single XOR output, `n` is 1 anyway.

The sign is the part to keep straight. `y - t` is positive when the guess is too high. Gradient descent subtracts the gradient, so a positive gap pushes the output downward on the next update.

## Where it lives

`network_mse` and `network_mse_dataset` are in `layer.c`. The known-value check is `test_mse_known` in `tests/test_ann.c`.
