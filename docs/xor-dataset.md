# XOR dataset

## In one page

XOR (exclusive or) is the small problem this network is trained to solve. There are four examples, two inputs each, and one target bit. The bit is 1 when the inputs differ, and 0 when they match. A single neuron cannot learn that table. A hidden layer can, and this program's 2 → 4 → 1 network does.

## The picture

Two light switches and a lamp that should glow only when the switches disagree. Both down: dark. Both up: dark. One up: lit. Every "more switches up means brighter" rule fails, because two ups is dark again. The pattern folds back on itself. That fold is why XOR is the classic first test for a hidden layer.

## The table

Stored in `dataset.c` as two flat arrays, row by row:

```
inputs      target
0    0        0
0    1        1
1    0        1
1    1        0
```

`XOR_SAMPLES` is 4, `XOR_INPUTS` is 2, `XOR_OUTPUTS` is 1. Sample `s` starts at index `s * 2` in `xor_inputs`. The matching target is `xor_targets[s]`.

## Why one neuron is not enough

One neuron computes:

```
output = sigmoid(bias + x1 * w1 + x2 * w2)
```

Inside the sigmoid, `bias + x1 * w1 + x2 * w2` is a flat plane. The line where that plane equals zero is a straight cut of the input square. The four XOR corners want opposite labels on opposite corners:

```
(0,0) = 0     (0,1) = 1
(1,0) = 1     (1,1) = 0
```

No straight cut puts both 0-corners on one side and both 1-corners on the other. You can check by trying. Any line that separates `(0,0)` from `(1,0)` and `(0,1)` leaves `(1,1)` on the same side as one of the 1-corners.

A hidden layer of several neurons makes several cuts. The output neuron then combines those bent signals with its own weights. With four hidden neurons there is spare capacity: XOR can be built with two hidden units, and four gives the small random initialization more chances to find a working shape.

## How this repo trains it

`./ann xor` builds that 2 → 4 → 1 network with `srand(1)`, so the random weights and biases are the same every run. It prints dataset mean squared error, trains for 5000 epochs at learning rate 0.5, prints the error again, then prints each row:

```
output >= 0.5   is read as a predicted 1
output <  0.5   is read as a predicted 0
```

Accuracy is how many of the four rows match. The program returns success when all four match. The unit test is stricter on the score as well: after the same training, dataset loss must be under 0.01, and accuracy must be 4/4.

## Where it lives

The numbers are `xor_inputs` and `xor_targets` in `dataset.c`, with the sizes in `dataset.h`. `run_xor` in `main.c` is the demo. `test_xor_trains` repeats the training under assertions.
