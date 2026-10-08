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
- [Vocabulary](vocabulary.md)
- [Token embeddings](token-embeddings.md)
- [Positional encoding](positional-encoding.md)
- [Query / Key / Value](query-key-value.md)
- [Self-attention](self-attention.md)
- [Multi-head attention](multi-head-attention.md)
- [Feed-forward layers](feed-forward-layers.md)
- [Layer normalization](layer-normalization.md)
- [Residual connections](residual-connections.md)
- [Causal attention](causal-attention.md)
- [Transformer blocks](transformer-blocks.md)
- [Next-token prediction](next-token-prediction.md)
- [Text generation](text-generation.md)

### Language model training

- [Transformer backpropagation](transformer-backpropagation.md)
- [Train all parameters](train-all-parameters.md)
- [Adam for LM weights](adam-lm-weights.md)
- [Text dataset loading](text-dataset-loading.md)
- [Mini-batch corpus training](mini-batch-corpus-training.md)
- [LM weight serialization](lm-weight-serialization.md)
- [Checkpoint and resume](checkpoint-and-resume.md)
- [Held-out loss / perplexity](held-out-loss-perplexity.md)
- [Generation from trained checkpoints](generation-from-checkpoints.md)
- [Scale toward target size](scale-toward-target-size.md)
- [Train / generate / load CLI](lm-cli.md)

### Performance

- [Contiguous memory](contiguous-memory.md)
- [Cache-friendly data structures](cache-friendly-data-structures.md)
- [SIMD](simd.md)
- [Multithreading](multithreading.md)
- [AVX2 / AVX-512](avx2-avx-512.md)
- [Quantization](quantization.md)
- [Optimized inference](optimized-inference.md)
- [GPU support](gpu-support.md)

### GPU training

- [Device buffers for activations](gpu-device-buffers.md)
- [CUDA GEMM backward](cuda-gemm-backward.md)
- [Attention backward on GPU](gpu-attention-backward.md)
- [FFN + LayerNorm backward](gpu-ffn-layernorm-backward.md)
- [LM head + cross-entropy backward](gpu-head-cross-entropy.md)
- [Adam step on GPU](gpu-adam.md)
- [KV-cache generate on GPU](gpu-kv-generate.md)
- [Optional cuBLAS / cuDNN](cublas-cudnn.md)
- [Host↔device sync at batch boundaries](gpu-batch-sync.md)

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

### [Vocabulary](vocabulary.md)
A vocabulary names every id the model can emit, including pad/bos/eos/unk. Byte tokens sit after four specials, for size 260. Lookups give strings for logging and generation.

### [Token embeddings](token-embeddings.md)
Token embeddings store a learned vector per vocab id in a V×d table. Looking up a sequence of ids stacks those rows into a T×d matrix. That matrix is the model's continuous view of the text.

### [Positional encoding](positional-encoding.md)
Sinusoidal positional encoding adds a fixed pattern so order matters. Even dimensions get sine, odd get cosine, with wavelengths growing across channels. The values are added onto the token embedding matrix in place.

### [Query / Key / Value](query-key-value.md)
Queries, keys, and values are three linear views of the same token matrix. Matching questions to labels decides whose payloads mix. This repo stores Wq/Wk/Wv and projects X into Q, K, and V.

### [Self-attention](self-attention.md)
Self-attention scores every position against every other with scaled dot products of Q and K. Softmax turns those scores into weights that mix the value rows. This repo implements that path in attn_self on T×d matrices.

### [Multi-head attention](multi-head-attention.md)
Multi-head attention runs several smaller attentions in parallel subspaces. Heads are concatenated and mixed by Wo. attn_mha projects X, splits heads, attends, and projects back.

### [Feed-forward layers](feed-forward-layers.md)
The feed-forward block is a position-wise two-layer MLP after attention. This repo expands to d_ff, applies ReLU, then projects back to d_model. Same weights run on every token row independently.

### [Layer normalization](layer-normalization.md)
Layer norm recenters and rescales each token vector. Learnable gamma and beta restore scale after the unit Gaussian step. This repo normalizes rows of the T×d sequence matrix in place.

### [Residual connections](residual-connections.md)
Residuals add the sub-layer output back onto the stream so depth stays trainable. This stack commits to pre-norm: normalize, transform, then add. resid_add and resid_copy are the small helpers blocks use.

### [Causal attention](causal-attention.md)
Causal masking blocks future positions so token t only attends to ≤ t. Scores for j>i are driven to a large negative before softmax. Decoder training and generation share that left-context rule.

### [Transformer blocks](transformer-blocks.md)
A decoder block is pre-norm causal MHA plus residual, then pre-norm FFN plus residual. Stacking LM_N_BLOCKS repeats that pattern. transformer_forward runs the toy two-block body.

### [Next-token prediction](next-token-prediction.md)
Next-token prediction trains the model to score the following id at each position. Softmax cross-entropy on the LM head is the loss. A tiny loop on a fixed phrase updates the head and embeddings until NLL drops.

### [Text generation](text-generation.md)
Generation samples the next id from the final-position logits and appends it. Greedy uses argmax; temperature>0 samples the softened distribution. ./ann lm trains a phrase then prints a short continuation.

# Language Model Training docs

### [Transformer backpropagation](transformer-backpropagation.md)
Manual backward walks CE through the head, blocks, and embeddings with activation caches. Grad mats sit beside every learnable weight. Finite-difference checks on the head confirm the chain rule plumbing.

### [Train all parameters](train-all-parameters.md)
Full-stack SGD replaces the old head-and-embed shortcut. Every attention, FFN, LayerNorm, embedding, and head weight receives gradients. lm_train_steps now zeros grads, runs cached forward/backward, then steps all params.

### [Adam for LM weights](adam-lm-weights.md)
LM Adam registers every param/grad scalar into flat slots. Bias-corrected moments update weights like the MLP optimizer, without touching network*. lm_train_steps uses Adam by default after full backward.

### [Text dataset loading](text-dataset-loading.md)
Text datasets load a UTF-8 file into byte token ids. Sliding windows of length window feed next-token pairs. data/toy.txt is the shipped fixture for tests.

### [Mini-batch corpus training](mini-batch-corpus-training.md)
Corpus training walks sliding windows in batches. Gradients average over the batch before one Adam step. lm_train_corpus runs epochs over a text_dataset.

### [LM weight serialization](lm-weight-serialization.md)
LMCK binary files store config dims and every weight tensor in a fixed walk order. lm_save and lm_load round-trip a model without the MLP ANN1 format. Matching NLL after reload verifies the bytes.

### [Checkpoint and resume](checkpoint-and-resume.md)
Checkpoints store Adam moments and step counters with the weights. Resume continues optimization without resetting momentum. lm_save_checkpoint and lm_load_checkpoint implement that path.

### [Held-out loss / perplexity](held-out-loss-perplexity.md)
Held-out NLL scores generalization on unseen windows. Perplexity is exp of that mean NLL and stays at least one. data/toy_val.txt exercises the metric in tests.

### [Generation from trained checkpoints](generation-from-checkpoints.md)
Loaded checkpoints feed lm_generate without reinitializing random weights. Greedy decode from a trained file is the smoke path. Tests save a checkpoint then generate a short continuation.

### [Scale toward target size](scale-toward-target-size.md)
Config bumps d_model to 64, heads to 8, ff to 256, context to 128, and blocks to 4. Byte vocab stays. README Target 10–50M remains aspirational beyond this toy scale.

### [Train / generate / load CLI](lm-cli.md)
The CLI exposes lm train, lm generate, and lm load over model.bin checkpoints. Flags select data, prompt, and output paths. A bare lm [phrase] demo remains for quick smoke runs.

# Performance docs

### [Contiguous memory](contiguous-memory.md)
A bump arena hands out consecutive scratch bytes without per-object free. LM transformer caches persist across forwards of the same length, and attention keeps head/softmax temps in the cache. ./ann lm bench reports mean forward milliseconds for later speedups.

### [Cache-friendly data structures](cache-friendly-data-structures.md)
Attention scores and outputs use mat_mul_bt and mat_mul on contiguous rows. Embeddings memcpy token rows; logits are a blocked 1xD GEMM against the head. Hot paths drop mat_get/mat_set loops.

### [SIMD](simd.md)
ann_dot and friends use restrict on contiguous doubles so -O2 can auto-vectorize. vec_dot and vec_axpy call those kernels. Explicit AVX intrinsics come later.

### [Multithreading](multithreading.md)
OpenMP splits mat_mul tiles and attention heads across cores. Each head keeps private scratch so softmax and writes do not race. OMP_NUM_THREADS controls width.

### [AVX2 / AVX-512](avx2-avx-512.md)
ann_dot picks AVX-512, AVX2, or scalar at compile time. Pass ARCH=-mavx2 when the CPU supports it. ann_simd_backend reports which path linked.

### [Quantization](quantization.md)
Int8 qmat stores weights with one scale for cheaper inference GEMVs. lm_quantize_head packs the LM head; lm_logits_at uses it when present. Training remains full precision.

### [Optimized inference](optimized-inference.md)
A KV cache stores per-layer Keys and Values so generate attends only the new query. lm_generate_kv walks the prompt once then appends tokens. lm_generate prefers that path with a full-reforward fallback.

### [GPU support](gpu-support.md)
Default builds link a CPU GPU stub so make test needs no CUDA. make cuda swaps in a naive CUDA mat_mul when nvcc is available. mat_mul falls back to the CPU GEMM on any GPU miss.

# GPU training docs

### [Device buffers for activations](gpu-device-buffers.md)
Mats may own a CUDA d_data twin with H2D/D2H helpers. Attention and LM caches try to pin H/Q/K/V when a GPU is present. The CPU stub leaves pinning as a no-op.

### [CUDA GEMM backward](cuda-gemm-backward.md)
gpu_mat_mul_backward forms dA and dB for C=AB on device. mat_mul_bt and mat_outer_add try GPU kernels then fall back to CPU. Stubs keep make test green without nvcc.

### [Attention backward on GPU](gpu-attention-backward.md)
MHA backward uses mat_mul, mat_mul_bt, mat_mul_at, and outer_add so CUDA GEMMs apply. Softmax Jacobian rows stay on the host.

### [FFN + LayerNorm backward](gpu-ffn-layernorm-backward.md)
FFN backward is GEMM-shaped with a ReLU mask. LayerNorm tries a CUDA row kernel then falls back to the CPU loop.

### [LM head + cross-entropy backward](gpu-head-cross-entropy.md)
CE delta updates dw_head via mat_outer_add so CUDA outer_add can run. dhidden accumulates W times delta on the host for the toy vocab.

### [Adam step on GPU](gpu-adam.md)
lm_adam_step packs flat param/grad/m/v buffers and tries gpu_adam_step. CUDA Adam updates moments and weights then zeros grads; stubs keep the host loop.

### [KV-cache generate on GPU](gpu-kv-generate.md)
Decode-step attention tries gpu_kv_attn_decode for QKV, K/V append, scores, and Wo. Softmax of the short score row stays on the host; sampling stays on the host.

### [Optional cuBLAS / cuDNN](cublas-cudnn.md)
make cuda defaults to ANN_CUBLAS=1 so gpu_mat_mul prefers cublasDgemm then falls back to the teaching kernel. cuDNN is reserved; LN/softmax stay educational.

### [Host↔device sync at batch boundaries](gpu-batch-sync.md)
GPU_SYNC_BATCH skips per-GEMM D2H when outputs are pinned. The train loop syncs at Adam and epoch NLL boundaries instead.
