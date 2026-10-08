# ANN

A neural network implemented from scratch in **C**, with the long-term goal of building a small **Transformer-based language model** without machine-learning frameworks.

## Status

🚧 **Early development**

Current implementation:

- Neurons
- Layers
- Inputs and weights
- Biases
- Connections
- Activation functions
- Forward propagation
- MSE loss
- Backpropagation
- Gradient descent
- Training loop
- XOR dataset
- Unit tests
- Memory management
- Vectors
- Matrices
- Tensors
- Optimized matrix multiplication
- Softmax
- Cross-entropy
- Mini-batches
- Adam optimizer
- Model serialization
- Tokenizer
- Vocabulary
- Token embeddings
- Positional encoding
- Query / Key / Value
- Self-attention
- Multi-head attention
- Feed-forward layers
- Layer normalization
- Residual connections
- Causal attention
- Transformer blocks
- Next-token prediction
- Text generation
- Transformer backpropagation
- Train all parameters
- Adam for LM weights
- Text dataset loading
- Mini-batch corpus training
- Generation from trained checkpoints
- Scale toward target size
- Train / generate / load CLI
- Held-out loss / perplexity
- Checkpoint and resume
- LM weight serialization

Trainable toy decoder-only LM in `lang/` with Performance work: arenas, GEMM attention, SIMD/OpenMP/AVX paths, int8 head quant, KV-cache generate, optional `make cuda`.

## Roadmap

### Neural Network

- [x] Neurons
- [x] Layers
- [x] Inputs and weights
- [x] Biases
- [x] Connections
- [x] Activation functions
- [x] Stabilize current architecture
- [x] Forward propagation
- [x] Loss functions
- [x] Backpropagation
- [x] Gradient descent
- [x] Training loop
- [x] XOR / basic datasets
- [x] Unit tests
- [x] Memory management

### Numerical Engine

- [x] Vectors
- [x] Matrices
- [x] Tensors
- [x] Optimized matrix multiplication
- [x] Softmax
- [x] Cross-entropy
- [x] Mini-batches
- [x] Adam optimizer
- [x] Model serialization

### Language Model

- [x] Tokenizer
- [x] Vocabulary
- [x] Token embeddings
- [x] Positional encoding
- [x] Query / Key / Value
- [x] Self-attention
- [x] Multi-head attention
- [x] Feed-forward layers
- [x] Layer normalization
- [x] Residual connections
- [x] Causal attention
- [x] Transformer blocks
- [x] Next-token prediction
- [x] Text generation

### Language Model Training

- [x] Full backpropagation through the Transformer
- [x] Train all parameters
- [x] Adam for LM weights
- [x] Mini-batch corpus training
- [x] Text dataset loading
- [x] LM weight serialization
- [x] Checkpoint and resume
- [x] Held-out loss / perplexity
- [x] Generation from trained checkpoints
- [x] Scale toward target size
- [x] Train / generate / load CLI

### Performance

- [x] Contiguous memory
- [x] Cache-friendly data structures
- [x] SIMD
- [x] Multithreading
- [x] AVX2 / AVX-512
- [x] Quantization
- [x] Optimized inference
- [x] GPU support

### GPU training / full device path

Today only forward `mat_mul` can use CUDA. These items move the rest of training and decode onto the GPU where it pays off (skip toy-only CPU glue and things that stay host-side by design).

- [x] Device buffers for activations (keep H/Q/K/V on GPU across layers)
- [ ] CUDA GEMM backward (`dA` / `dB` for `C = AB` and `C = ABᵀ`)
- [ ] Attention backward on GPU (score / softmax / `AV` grads)
- [ ] FFN + LayerNorm backward kernels (or fused epilogues)
- [ ] LM head + cross-entropy backward on device
- [ ] Adam step on GPU (flat param / grad / moment buffers)
- [ ] KV-cache generate on GPU (append K/V, decode-step attention)
- [ ] Optional cuBLAS / cuDNN (replace naive teaching kernels)
- [ ] Host↔device sync only at batch boundaries (cut PCIe chatter)
- [ ] `nvidia-smi` / bench proof that train sustains GPU util (not only `lm bench`)

## Target

Initial language model target:

- Decoder-only Transformer
- ~10–50M parameters
- 10k–30k vocabulary
- 256–512 token context
- 4–6 Transformer layers
- 4–8 attention heads

The goal is to implement and understand the complete stack from **neurons to text generation**.

## Training

The model will use next-token prediction.

    Input:  "The cat is"
    Target: "on"

Training loop:

1. Tokenize text
2. Predict the next token
3. Calculate the loss
4. Backpropagate gradients
5. Update parameters
6. Repeat

## Data

Training data will progressively increase:

    10 MB → 100 MB → 1 GB → 10 GB+

Wikipedia and other freely available text datasets can be used for experimentation.

## Architecture

    Neuron
       ↓
    Layer
       ↓
    Network
       ↓
    Matrix / Tensor Engine
       ↓
    Transformer
       ↓
    Language Model
       ↓
    Text Generation

## Build

The project is written in C and aims to remain dependency-light.

    make

## Philosophy

- Implement the fundamentals instead of hiding them behind frameworks.
- Keep the code understandable.
- Measure performance instead of guessing.
- Test numerical correctness.
- Optimize after correctness.
- Build incrementally.

## Long-Term Goal

Build a functional **text-generating Transformer in C from scratch**:

**Training → Inference → Tokenization → Transformer → Text Generation**
