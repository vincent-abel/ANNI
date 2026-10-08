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
