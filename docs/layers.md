# Layers

## In one page

A layer is a row of neurons that share a job. The first row reads the sample. Middle rows mix what the previous row said. The last row's outputs are the network's guess. Stacking rows is what lets the network represent patterns that a single weighted sum cannot.

## The picture

A spreadsheet is a useful image. Each row is a layer. Each cell in the row is a neuron. Every cell in row 2 has a wire to every cell in row 1. The number written in a cell is that neuron's output.

This program stores the spreadsheet sideways, as a linked list of layers (`nlayer`). Each layer holds:

- `neuron`, the head of that row's neuron list
- `next`, the layer closer to the output
- `prev`, the layer closer to the input

The `network` struct keeps the raw sample in `inputs`, the first layer in `layers`, and the last layer in `out`. Keeping a direct pointer to the last layer means the loss and the backward pass can start at the answer without walking the whole chain first.

## Roles

`nlayer_show` names the rows from their position:

- one row only: that row is both the input side and the output side
- first of several: input layer (it reads the sample)
- last of several: output layer
- anything in between: hidden layer

"Input layer" here still means neurons. The raw sample values live in `network.inputs` and are not themselves neurons. The first layer of neurons reads those values.

Hidden layers are the mixing bowl. Each hidden neuron builds its own weighted view of the previous row, runs it through the S-curve, and hands that view forward. The XOR problem needs this bowl: the four examples cannot be separated by one straight cut, and a hidden row can bend the representation so that the output neuron's cut succeeds.

## How a network of layers is built

`network_create_sizes` takes the width of each row. For XOR that array is `{4, 1}`: four neurons, then one neuron, and the sample has 2 values. So the shape is 2 → 4 → 1.

`network_create` is the even case: every row has the same neuron count.

Building walks left to right. The first row connects to the sample buffer. Each following row connects to the neurons of the row just built. After the loop, `net->out` points at the tail.

## Forward and backward see the chain differently

Forward propagation follows `next`, from the sample side toward the answer. That order matters because a layer reads the outputs of the layer behind it, and those outputs must already be filled in.

Backpropagation follows `prev`, from the answer back toward the sample. The output row is scored against the target first. Each earlier row then receives blame that has already passed through the weights in front of it.

## Where it lives

`nlayer` and `network` are in `layer.h`. Appending a row, walking it forward, and creating a sized network are in `layer.c` (`nlayer_append`, `nlayer_forward`, `network_create_sizes`).
