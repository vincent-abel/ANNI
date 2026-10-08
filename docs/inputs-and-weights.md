# Inputs and weights

## In one page

An input is a number the neuron reads. A weight is the multiplier on that number. The neuron's raw total is mostly the sum of `input * weight` across every wire. Learning is, in large part, the slow editing of those weights.

## The picture

A weight is a volume knob on one incoming wire.

- A positive weight means "when this input goes up, pull my total up."
- A negative weight means "when this input goes up, pull my total down."
- A weight near zero means "I am hardly listening to this input."
- A large weight means "this input dominates me."

The input itself is not owned by the wire. The wire holds a pointer to some `double` that already lives elsewhere: either a slot in the sample buffer, or the `output` field of a neuron in the previous layer. When that other number changes, the next forward pass sees the new value through the same pointer. The weight stays on the wire until gradient descent changes it.

## The arithmetic

One wire contributes:

```
contribution = input_value * weight
```

The neuron adds every contribution, then adds the bias:

```
total = bias + contribution_1 + contribution_2 + ...
```

That is a dot product of the input vector with the weight vector, plus a bias. For two inputs:

```
total = bias + x1 * w1 + x2 * w2
```

If `x1 = 0.0` and `x2 = 1.0`, with `w1 = 1.5` and `w2 = -0.4`:

```
total = bias + 0 + (1.0 * -0.4) = bias - 0.4
```

The first weight did nothing on this sample because its input was zero. On a sample where `x1` is 1, that same weight would add 1.5. So a weight's influence is always "how big is the input right now?" times "how big is the knob?"

## How a weight learns

Backpropagation stores a suggested change on the wire as `dw`:

```
dw = delta * input_value
```

`delta` is the neuron's error signal (how the output should move). Multiplying by the input value is the chain rule in everyday form: if this input was large, this weight had a large effect on the total, so it deserves a large edit. If this input was zero, this weight did not affect the total on this sample, and `dw` is zero.

The later update is:

```
weight = weight - learning_rate * dw
```

## How weights are born

Each new wire draws a weight from a uniform random range of -2 to 2 (`randfrom(-2, 2)` in `input_push`'s callers). Random starting weights matter. If every weight in a layer started equal, every neuron in that layer would compute the same thing and receive the same lesson, and they would stay copies of each other. Different starting knobs let neurons specialize.

`dw` starts at 0. It is a scratch gradient, cleared at the beginning of each backward pass and again after the update.

## Where it lives

The wire is the `input` struct in `neuron.h`: `i` (pointer to the value), `w` (weight), `dw` (pending change), `src` (the neuron that owns the value, or NULL when the value is a raw sample slot). The multiply-add is the loop in `neuron_Tinput`.
