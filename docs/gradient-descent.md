# Gradient descent

## In one page

Gradient descent is the update step. Backpropagation has already stored, on every weight and bias, a number describing how the loss would change if that parameter grew. Descent subtracts a fraction of that number. Repeating this after each sample walks the parameters toward lower loss.

## The picture

Stand on a hillside in fog. The gradient is the direction of steepest uphill, which you can feel with your feet even though you cannot see the valley. To get lower, step the opposite way. A short step is safe and slow. A long step covers ground and can hop over the valley to the other slope.

Here the "hillside" has one dimension per weight and per bias. The XOR network has 12 weights and 5 biases, so the hill is 17-dimensional. Nobody visualizes that. The rule still applies one number at a time: each parameter only needs its own slope.

## The rule in this code

`neuron_update` applies, for every neuron and every wire:

```
bias   = bias   - learning_rate * dbias
weight = weight - learning_rate * dw
```

Then it sets `dbias`, `dw`, and `delta` back to 0 so the next sample starts clean.

`learning_rate` is the step length. Both `./ann xor` and the generic demo pass `0.5`. On one parameter, if `dw` is `0.2` and the learning rate is `0.5`:

```
weight moves by -0.5 * 0.2 = -0.1
```

If `dw` is `-0.2`, the weight moves by `+0.1`. Negative gradient means "increasing this weight would have lowered the loss," so the minus sign in the update increases it.

## What kind of descent this is

The training loop updates immediately after each sample. It does not add up gradients over the whole dataset and then step once. That per-example style is stochastic gradient descent with a batch size of one. On XOR the "dataset" is four rows, so each epoch takes four small steps, each step aimed at a different row's loss.

Those four directions disagree a little. The step for "0 xor 0 → 0" pulls parameters one way, and the step for "0 xor 1 → 1" pulls them another way. Over thousands of epochs the average of those pulls settles the weights into a region where all four rows are mostly right.

There is no momentum, no Adam, and no weight decay in the current code. The step is plain scaled gradient.

## Why the learning rate matters

The gradients coming out of backpropagation are already scaled by the sigmoid slope and by the size of the inputs. A learning rate of 0.5 is large for some networks and ordinary for this small one. Too large a rate makes the loss bounce: each step overshoots the bottom. Too small a rate means 5000 epochs are not enough to shape the hidden layer. The XOR test locks the pair `(5000 epochs, learning rate 0.5, seed 1)` and expects the dataset loss to drop below 0.01, which is evidence that this step size works for this problem.

## Where it lives

`neuron_update` in `neuron.c` is the subtraction. `network_update` in `layer.c` calls it for every layer. Callers pass the learning rate; nothing inside the neuron picks one.
