# Training loop

## In one page

The training loop is the repetition that turns a random network into a fitted one. For each epoch, and for each sample inside that epoch, it loads the sample, runs a forward guess, backpropagates the miss, and takes a gradient step. After enough laps, the same samples produce outputs close to their targets.

## The picture

A student with a four-card deck. Each epoch, they go through the deck in order. For each card they try an answer, check it, and adjust their method before the next card. Then they shuffle nothing — this loop keeps the sample order fixed — and deal the same four cards again. The method is the set of weights and biases. The cards are the dataset.

## The loop exactly

`network_train` is a pair of nested loops:

```
for epoch in 1 .. epochs:
    for sample in 1 .. n_samples:
        copy sample into the input buffer
        forward propagation
        backpropagation against that sample's target
        gradient descent with learning rate lr
```

One epoch is one pass over every sample. The XOR call uses 5000 epochs, 4 samples, 1 target, and learning rate 0.5, so the inner body runs 20,000 times.

The generic `./ann` demo uses the loop differently: one sample (the random inputs the network was born with), a target of all zeros, 50 epochs, learning rate 0.5. That path shows the loss moving. It is not a real dataset.

## What changes, and what does not

Inside one inner step the sample and the target are constant. The forward pass refreshes outputs. The backward pass fills `delta`, `dbias`, and `dw` for that sample only, because it zeros them first. The update edits weights and biases, then clears the scratch gradients.

Across steps, the weights and biases are the memory. Epoch 2 sees a different network than epoch 1, so the same sample produces a different guess and a different gradient. That is the whole point of repeating the deck.

The loop does not shuffle, does not split a validation set, and does not stop early when the loss looks good. It always runs the requested number of epochs.

## How you know it worked

After `network_train` returns, the weights are whatever the last update left. A fresh forward pass is required to see the new guess. `run_xor` measures dataset loss before training, trains, measures it again, then forwards each of the four rows and counts how many outputs land on the correct side of 0.5.

The order inside the inner step is fixed for a reason. Updating before the backward pass would apply last sample's gradient to this sample. Scoring the loss for learning without a forward pass would compare the target with a stale output. The four calls — set, forward, backward, update — are the lesson in the only order that matches the math.

## Where it lives

`network_train` in `layer.c`. `run_xor` and `main` in `main.c` choose the epoch count, the learning rate, and the samples.
