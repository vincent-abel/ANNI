# Neurons

## In one page

A neuron is the smallest unit that computes in this network. It keeps a few numbers, reads a list of incoming wires, and writes a single output. Layers, loss, and learning are all built by repeating that same small calculation.

## The picture

Think of a neuron as a person in a meeting who hears several colleagues, trusts some more than others, and then gives one opinion between "fully off" (0) and "fully on" (1).

Each colleague's comment is an input. How much this neuron trusts that colleague is a weight. The neuron also has a personal tendency to lean positive or negative before anyone speaks: that tendency is the bias. The opinion it finally gives is the output.

In memory, one neuron is a small struct:

- `Tinput` is the raw total, before the S-curve.
- `output` is the number after the S-curve, between 0 and 1.
- `bias` is the personal offset.
- `input` is the linked list of wires coming in.
- `delta`, `dbias` are scratch space used later, when the network learns.
- `prev` links this neuron to the next neuron in the same layer.

Neurons in a layer are stored as a linked list. Creating a neuron pushes it onto the front of that list, so the first neuron you visit is the last one that was created.

## The arithmetic

For one neuron the whole forward calculation is:

```
total   = bias + (input_1 * weight_1) + (input_2 * weight_2) + ...
output  = 1 / (1 + e^(-2 * total))
```

The first line is a weighted sum. Multiply, then add. The bias joins that sum as if it were an extra input that is always 1.

The second line is the activation. `e` is the exponential function. Whatever `total` was — largely negative, zero, or largely positive — the output lands between 0 and 1. Doubling `total` (the `/ 0.5` written in the source as `RESPONSE`) makes the jump from off to on steeper. The activation note explains that curve on its own.

A tiny numeric sketch:

```
bias = 0.1
input A = 1.0 with weight 0.5
input B = 0.0 with weight -1.0

total  = 0.1 + (1.0 * 0.5) + (0.0 * -1.0) = 0.6
output = 1 / (1 + e^(-1.2)) ≈ 0.77
```

Input B contributed nothing because it was zero. Input A pulled the total up by 0.5. The bias added another tenth. The S-curve turned 0.6 into about 0.77.

## What the neuron stores for learning

During the forward pass the neuron only needs the total and the output. Learning needs three more numbers, filled in later by backpropagation:

- `delta` is "how much should this output have been different?"
- `dbias` is "how much should the bias change to help?"
- each wire's `dw` is "how much should this weight change?"

They start at zero. After a useful step, gradient descent applies them and clears them. The neuron struct is the place those numbers live so the backward pass and the update pass can share them.

## Where it lives

`neuron` and `input` are declared in `neuron.h`. Creation, the weighted sum (`neuron_Tinput`), and the S-curve (`neuron_output`) are in `neuron.c`. A neuron does not know which layer it belongs to. It only knows its wires and its sibling through `prev`.
