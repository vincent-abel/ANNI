# Connections

## In one page

A connection ties one source number to one neuron and stores the weight for that pair. The first layer's connections read the sample buffer. Every later connection reads another neuron's output. The graph is fully connected: each neuron is wired to all neurons in the previous layer, and to no neurons further back.

## The picture

Draw the sample as two boxes, `x1` and `x2`. Draw a hidden layer of four neurons, then one output neuron. From each sample box, draw a line to each of the four hidden neurons. From each hidden neuron, draw a line to the output. That drawing is the XOR network this program builds: 2×4 wires into the hidden layer, plus 4 wires into the output, 12 connections in total, each with its own weight.

Nothing skips a layer. The output neuron does not read `x1` directly. It only reads the four hidden outputs. Any influence the sample has on the final guess is carried by those hidden neurons.

## What one connection stores

The `input` struct is one wire:

- `i` points at the number being read
- `w` is the weight
- `dw` is the pending weight change
- `src` points at the source neuron, or is NULL
- `prev` links to the other wires of the same destination neuron

`src` is the detail that makes backpropagation possible in this design. When a hidden neuron is the source, `i` points at `src->output` and `src` points at the hidden neuron itself. To send blame backward, the destination neuron does:

```
source_delta += my_delta * weight
```

It can do that only because the wire remembers who produced the number. Wires from the sample have `src == NULL`. Their values are data, fixed for that example, so the blame stops there. The program adjusts the weight on that wire and leaves the sample alone.

## How the wires are attached

`neuron_connect_values` is used for the first layer. For each slot in the sample array it pushes a wire whose pointer is `&values[i]` and whose source neuron is NULL.

`neuron_connect_layer` is used for every later layer. It walks the previous layer's neuron list and pushes a wire to each neuron's `output` field, remembering that neuron as `src`.

Both helpers are called once per destination neuron, from `layer_neurons_create`. So a layer of 4 neurons in front of 2 inputs creates 4 × 2 wires. A single output neuron in front of those 4 creates 4 wires.

Lists grow by pushing onto the front, so the walk order is the reverse of the order the neurons were created. For a layer of one neuron, as in the XOR output, order does not matter. For a wider output layer, target values are paired with neurons in that list order.

## Fully connected, and nothing else

"Fully connected" means every neuron in layer L receives every neuron in layer L−1. It does not mean every neuron receives every neuron in the whole network. There are no recurrent wires (a neuron feeding itself or a later layer feeding an earlier one) and no missing wires inside a layer pair.

That dense pattern is why a layer with `n` neurons followed by a layer with `m` neurons holds `n * m` weights, plus `m` biases in the destination layer.

## Where it lives

`neuron_connect_values` and `neuron_connect_layer` in `neuron.c` build the wires. `layer_neurons_create` in `layer.c` decides which helper to call: the sample for the first row, the previous neuron list for every row after that.
