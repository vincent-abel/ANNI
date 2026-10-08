# Memory management

## In one page

The network is a tree of heap blocks: one `network`, one input buffer, one block per layer, one block per neuron, and one block per wire. Creating the graph allocates those blocks from the inside out. Freeing it walks the same links and releases them from the inside out, then sets the caller's network pointer to NULL.

## The picture

Each struct is a box you rented from `malloc`. The box is useless until its fields point at the smaller boxes it owns. If you throw away a layer box without first throwing away its neurons, those neuron boxes are still rented and you have lost the only map to them. Cleanup is therefore "free the contents, then free the container, and null the pointer you entered through."

## Who owns what

- `network_create_sizes` allocates the `network` and the `inputs` array of `n_inputs` doubles.
- Each layer is an `nlayer` block from `nlayer_append`.
- Each neuron is a `neuron` block from `neuron_push`. The layer's `neuron` pointer owns that list.
- Each wire is an `input` block from `input_push`. The neuron's `input` pointer owns that list.
- The doubles inside a sample are not always owned here. During training, `network_set_inputs` copies values into the network's own buffer. The XOR tables in `dataset.c` stay where they are.

Wires do not own the `double` they point at. A first-layer wire points into `net->inputs`. A later wire points at some neuron's `output` field. Freeing a wire frees the wire block only. Freeing a neuron frees its wires first, then the neuron. The output `double` dies with the neuron that contains it.

## The free walk

`network_free` takes a `network **`, a pointer to the caller's pointer.

1. `nlayer_clear` walks `next`. For each layer it calls `neuron_clear`.
2. `neuron_clear` walks `prev`. For each neuron it calls `input_clear`, then frees the neuron.
3. `input_clear` walks `prev` and frees every wire.
4. Back in `network_free`, `out` is set to NULL, the input buffer is freed and set to NULL, the `network` block is freed, and `*net` is set to NULL.

After that, the variable in the caller is NULL. `test_create_free_cycle` asserts exactly that. A second `network_free` on the same variable is safe, because the function returns immediately when the pointer is already NULL.

## When creation fails halfway

`network_create_sizes` can fail if a size is invalid or if a `malloc` returns NULL. On those paths it calls `network_free` on whatever prefix exists: layers already linked, neurons already hanging off them, and the input buffer if it was allocated. The function then returns NULL and the caller does not receive a half-built graph.

`layer_neurons_create` does the same at a smaller scale. If the fifth neuron of a row fails to allocate or to connect, it clears the neurons built for that row and returns NULL, and the network builder then frees the whole partial network.

The demo `main` has a second cleanup for its own arrays. If `samples`, `targets`, or `target` fails to allocate, it frees whichever of the three exist and frees the network before exiting. Those three arrays are not part of the `network` struct. The network does not know about them.

`network_create` (the equal-width helper) treats failure as fatal and exits the process. `network_create_sizes` returns NULL instead. The tests call `network_create_sizes`.

## Where it lives

`network_free` and `nlayer_clear` are in `layer.c`. `neuron_clear` and `input_clear` are in `neuron.c`. The NULL-after-free assertion is `test_create_free_cycle` in `tests/test_ann.c`.
