# Documentation

## Table of contents

### Neural network

- [Neurons](neurons.md)
- [Layers](layers.md)
- [Inputs and weights](inputs-and-weights.md)
- [Biases](biases.md)
- [Connections](connections.md)
- [Activation functions](activation-functions.md)
- [Forward propagation](forward-propagation.md)
- [MSE loss](mse-loss.md)
- [Backpropagation](backpropagation.md)
- [Gradient descent](gradient-descent.md)
- [Training loop](training-loop.md)
- [XOR dataset](xor-dataset.md)
- [Unit tests](unit-tests.md)
- [Memory management](memory-management.md)

### Numerical engine

- [Vectors](vectors.md)
- [Matrices](matrices.md)
- [Tensors](tensors.md)
- [Optimized matrix multiplication](optimized-matrix-multiplication.md)
- [Softmax](softmax.md)
- [Cross-entropy](cross-entropy.md)
- [Mini-batches](mini-batches.md)
- [Adam optimizer](adam-optimizer.md)
- [Model serialization](model-serialization.md)

### Language model

- [Tokenizer](tokenizer.md)

---

# How this neural network works

Short notes on each piece that is already built. Each note explains the idea in ordinary language, then the arithmetic this C program actually runs.

## [Neurons](neurons.md)

A neuron is one small calculator in the network. It adds up its weighted inputs, adds a bias, and squashes the result into a number between 0 and 1. That number is what the next neurons read.

## [Layers](layers.md)

A layer is a row of neurons that all see the same previous row. The network is a chain of these rows, from the raw sample to the final guess. The last row is the output the loss compares with the target.

## [Inputs and weights](inputs-and-weights.md)

An input is one number coming in, and a weight is how strongly that number counts. The neuron multiplies each input by its weight and adds those products. A bigger weight means that input can push the neuron harder.

## [Biases](biases.md)

A bias is a number added on top of the weighted inputs, so the neuron can sit above or below zero even when every input is quiet. Each neuron has its own bias, chosen at random near zero when the network is created. Training nudges that bias up or down along with the weights.

## [Connections](connections.md)

A connection is a wire from a source number to a neuron, carrying a weight. The first layer's wires read the sample itself. Every later neuron is wired to every neuron in the previous layer, which is what fully connected means here.

## [Activation functions](activation-functions.md)

The activation is the S-shaped curve that turns an unlimited sum into a value between 0 and 1. This program uses the logistic sigmoid, and it doubles the sum before the curve, which makes the bend steeper. Near 0 and near 1 the curve flattens, so a saturated neuron barely moves during learning.

## [Forward propagation](forward-propagation.md)

Forward propagation is the left-to-right pass that turns a sample into a guess. Each layer reads the outputs the previous layer just wrote, then writes its own. When the last layer finishes, those outputs are the network's answer.

## [MSE loss](mse-loss.md)

Mean squared error measures how far the guess is from the target by squaring the gap and averaging. Squaring makes every miss count as a positive amount and punishes big misses more than small ones. A perfect guess has loss 0.

## [Backpropagation](backpropagation.md)

Backpropagation walks the error from the output back toward the inputs. Each neuron works out how much it contributed to the miss, then passes a share of that blame to the neurons behind it. The share on a wire is the error signal times the weight on that wire.

## [Gradient descent](gradient-descent.md)

Gradient descent changes each weight and bias by a small step in the direction that reduces the error. The gradient says which way increases the error, and the update steps the opposite way. The learning rate, 0.5 in the programs shipped here, sets the size of that step.

## [Training loop](training-loop.md)

The training loop repeats the same lesson: load a sample, guess, measure the miss, send the error backward, then update. One pass through every sample is an epoch. The XOR run does this for 5000 epochs.

## [XOR dataset](xor-dataset.md)

XOR is four tiny examples of "one or the other, but not both." A single weighted sum cannot draw that pattern, so the network needs a hidden layer. This repo stores the four rows and trains a 2-input, 4-hidden, 1-output network on them.

## [Unit tests](unit-tests.md)

The tests are small programs that check known numbers, so a change that breaks the math fails immediately. They check the loss formula, the 0-to-1 range of the S-curve, cleanup, rejected sizes, and that XOR actually learns. `make test` prints `ok` when all of them pass.

## [Memory management](memory-management.md)

Every neuron, wire, layer, and the input buffer is allocated on the heap and freed by walking those lists. `network_free` releases the whole graph and sets the caller's pointer to NULL so it cannot be used again. If creation fails partway, what was already allocated is freed before the function returns.

# Numerical Engine docs

Short guides for each Numerical Engine piece, written for humans first and math second.

### [Vectors](vectors.md)
A vector is a contiguous list of numbers the engine can scale, add, and combine. The core score between two lists is the dot product. This repo stores them in `vec` with helpers like `vec_dot` and `vec_axpy`.

### [Matrices](matrices.md)
A matrix is a row-major grid of weights that transforms vectors in one shot. Matrix–vector multiply mixes every input into each output; matrix–matrix multiply chains those transforms. `mat_vec_mul` and `mat_mul` implement those products here.

### [Tensors](tensors.md)
A tensor is contiguous storage plus a multi-axis shape for batches and higher ranks. Flat indexing maps coordinates into one array; rank-1/2 views expose the same memory as `vec` or `mat` without copying.

### [Optimized matrix multiplication](optimized-matrix-multiplication.md)
Blocked multiply does the same \(C=AB\) math as the naive loops but in cache-friendly tiles. `mat_mul` uses 32×32 blocks while `mat_mul_naive` stays as a correctness reference. Only the visit order changes, not the result.

### [Softmax](softmax.md)
Softmax turns logits into a probability distribution that sums to one. Subtracting the max logit before `exp` keeps the math stable. `vec_softmax` implements that path on contiguous vectors.

### [Cross-entropy](cross-entropy.md)
Cross-entropy scores how little probability the model put on the true class. `vec_cross_entropy` uses \(-\log p_y\); `vec_softmax_cross_entropy` does the same from logits with a stable log-sum-exp.

### [Mini-batches](mini-batches.md)
Mini-batches average gradients over a few examples before each weight update. That smooths the noisy single-sample steps. `network_train_mb` accumulates, scales by batch size, then updates once.

### [Adam optimizer](adam-optimizer.md)
Adam mixes momentum with per-weight adaptive step sizes from squared-gradient history. Bias correction warms up early averages. `adam_create` / `adam_step` drive training instead of plain SGD.

### [Model serialization](model-serialization.md)
Serialization writes architecture sizes plus every bias and weight in a fixed walk order. Loading rebuilds the graph and fills the same slots. `network_save` / `network_load` round-trip a trained net through a binary file.

# Language Model docs

Short guides for each Language Model piece, from plain intuition up to formal detail.

### [Tokenizer](tokenizer.md)
A tokenizer maps text to integer ids and back. This repo starts with one id per byte (0–255). Encode and decode are linear scans with a fixed 256-wide alphabet.
