# Activation functions

## In one page

The activation function squashes a neuron's raw total into a predictable range. This program uses one function for every neuron: the logistic sigmoid, made steeper by doubling the total first. The squash is what lets stacked layers do more than one big weighted sum, and its slope is what backpropagation multiplies into the error signal.

## The picture

A raw total can be any real number: -40, 0, 3.5, 100. Later neurons and the loss expect something tame. The sigmoid is an S drawn on its side:

- far to the left, the output sits near 0
- at zero, the output is exactly 0.5
- far to the right, the output sits near 1
- the steep part is around the middle

Once the output is near 0 or near 1, a large extra push in the total barely moves it. The neuron is saturated. That is useful when a confident yes or no is wanted, and it is a problem for learning, because a flat curve means a tiny gradient.

## The formula in this code

`RESPONSE` is 0.5. Dividing by 0.5 is the same as multiplying by 2, so the forward pass is:

```
output = 1 / (1 + e^(-total / 0.5))
       = 1 / (1 + e^(-2 * total))
```

A few landmarks, using the doubled form:

```
total = -2   →   2*total = -4   →   output ≈ 0.018
total = -0.5 →   2*total = -1   →   output ≈ 0.269
total =  0   →   2*total =  0   →   output = 0.500
total =  0.5 →   2*total =  1   →   output ≈ 0.731
total =  2   →   2*total =  4   →   output ≈ 0.982
```

Compare that with the plain sigmoid `1 / (1 + e^(-total))`, which at `total = 0.5` would be about 0.62. Doubling makes the same total more decisive. `neuron_output` applies this to every neuron in the list.

## The slope, which learning needs

The backward pass asks a different question: if the total grew a tiny bit, how much would the output grow? That rate of change is the derivative.

For a plain sigmoid, if `y` is the output, the slope with respect to the value inside the exponential is:

```
slope = y * (1 - y)
```

Check the landmarks. At `y = 0.5`, slope = 0.25, the steepest this formula gets. At `y = 0.02`, slope = 0.0196, almost flat. At `y = 0.98`, slope = 0.0196 again. So a neuron that has already decided "almost 0" or "almost 1" sends a very small learning signal.

The code uses exactly that expression (`sigmoid_prime` in `neuron.c`):

```
slope = output * (1 - output)
```

The forward pass doubled the total, and the full derivative with respect to `total` would include that extra factor of 2:

```
d(output) / d(total) = 2 * output * (1 - output)
```

The backward pass multiplies by `output * (1 - output)` and leaves the 2 out. A constant factor like that scales every gradient the same way. The learning rate is itself a scale on every step, so leaving the 2 out makes steps gentler and does not change their direction.

## Why an S-curve at all

If the output were the raw total, stacking two layers would still be one linear function. A linear function of a linear function is another linear function. The hidden layer would be decorative. The S-curve bends each neuron's response, so the second layer sees a warped version of the first. That bend is what makes XOR learnable in this network.

Every neuron uses the same curve. There is no ReLU, tanh, or softmax in the current code. Outputs of the last layer are therefore also between 0 and 1, which matches targets such as the XOR bits 0 and 1.

## Where it lives

The forward curve is the one line in `neuron_output`. The slope is `sigmoid_prime`, used by `neuron_backward_output` and `neuron_backward_hidden`.
