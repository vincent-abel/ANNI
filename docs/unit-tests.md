# Unit tests

## In one page

The tests are a second small program that checks the network against numbers you can compute by hand. They use C `assert`. If a check fails, the program aborts. If every check holds, it prints `ok`. They cover the loss, the shape of a forward pass, allocation failure, cleanup, and a full XOR training run.

## The picture

A unit test is a worked example with the answer written in the margin. The code runs the example and the assertion compares the margin with the result. Nothing here scans logs or looks at printed tables. A wrong formula fails the build of confidence immediately, at the exact check.

`make test` compiles `tests/test_ann.c` with the network objects and without `main.c`, then runs that program. The demo `main` and the tests are separate programs that share `layer.c`, `neuron.c`, and `dataset.c`.

## What each check is proving

**Known loss.** Build a one-input, one-neuron network, forward once, then overwrite the output with 0.75 and score it against target 0.25. The mean squared error must be 0.25, within a tiny tolerance of `1e-9` because the value is a `double`. Overwriting the output tests the loss alone, independent of the random weights.

**Forward shape.** A network with sizes `{3, 2}` and 2 inputs must report an output width of 2. After a forward pass both output neurons must lie in `[0, 1]`, which is the range of the sigmoid. This catches a broken activation or a layer wired in the wrong order badly enough to produce nonsense magnitudes.

**Create, use, free.** Twenty times, create a 2 → 3 → 2 → 1 network, forward once, and free it. After `network_free` the caller's pointer must be NULL. Repeating the cycle is a simple way to notice a leak or a double-free that a single call might not trip over, especially under a memory checker.

**Rejected shapes.** A size of 0 in the layer list, a layer count of 0, or an input count of 0 must make `network_create_sizes` return NULL. Those are the guards at the top of creation, checked without looking at weights.

**XOR learns.** With seed 1, the 2 → 4 → 1 network trained for 5000 epochs at learning rate 0.5 must reach a dataset loss below 0.01 and classify all four rows on the correct side of 0.5. This is the end-to-end check: connections, activation, loss, backpropagation, gradient descent, and the training loop together.

## What they do not measure

The tests do not time anything, do not check every derivative by finite differences, and do not try a second dataset. Passing means the promises above hold on this machine for this seed. `test_xor_trains` is the slow one, because it really trains.

## Where it lives

All five functions are in `tests/test_ann.c`. The `test` target in the `Makefile` builds and runs them.
