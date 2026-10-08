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

Toy decoder-only LM in `lang/` (byte vocab, pre-norm blocks, next-token train, `./ann lm`).

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

### Performance

- [ ] Contiguous memory
- [ ] Cache-friendly data structures
- [ ] SIMD
- [ ] Multithreading
- [ ] AVX2 / AVX-512
- [ ] Quantization
- [ ] Optimized inference
- [ ] GPU support

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
