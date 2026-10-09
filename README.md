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
- Train / generate / load CLI
- Held-out loss / perplexity
- Checkpoint and resume
- LM weight serialization
- Per-epoch checkpoints, `--lr` / `--window` / `--seed`, `lm merge`
- (Scale to Target 10–50M / subword — still open; see Roadmap)

Trainable decoder-only LM in `lang/` (byte vocab, educational C stack): arenas, GEMM attention, SIMD/OpenMP/AVX, int8 head quant, KV-cache generate, optional `make cuda`, checkpoint/resume, `--lr` / `--window` / `--seed`, per-epoch saves, `lm merge` weight averaging. Still a **toy** relative to the Target below—usable for learning and local experiments, not a small production LLM.

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
- [x] Train / generate / load CLI
- [x] Per-epoch checkpoint saves; `--lr` / `--window` / `--seed`; `lm merge` weight average
- [ ] Scale toward target size (10–50M / subword) — current D=256 byte LM is still toy; see **From toy LM → small LLM**

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

CUDA can run forward GEMM, backward GEMMs, Adam, KV decode, and optional cuBLAS; default `make` / `make test` stay on the CPU stub.

- [x] Device buffers for activations (keep H/Q/K/V on GPU across layers)
- [x] CUDA GEMM backward (`dA` / `dB` for `C = AB` and `C = ABᵀ`)
- [x] Attention backward on GPU (score / softmax / `AV` grads)
- [x] FFN + LayerNorm backward kernels (or fused epilogues)
- [x] LM head + cross-entropy backward on device
- [x] Adam step on GPU (flat param / grad / moment buffers)
- [x] KV-cache generate on GPU (append K/V, decode-step attention)
- [x] Optional cuBLAS / cuDNN (replace naive teaching kernels)
- [x] Host↔device sync only at batch boundaries (cut PCIe chatter)
- [x] `nvidia-smi` / bench proof that train sustains GPU util (not only `lm bench`)

### From toy LM → small LLM (not-toy)

What we have at `f3a44b3` is a **byte-level** educational model (`LM_BYTE_VOCAB` 256, short train windows, `double` mats, toy corpora). Low **window NLL** on `data/toy_*.txt` does **not** mean fluent generation: each token is one byte, so meaning is spread over long sequences and needs far more data (and usually subwords) than a wordpiece LM.

**Near-term steps (unblock speed and honest metrics)**

- [ ] FP32 (or mixed) train/infer path — consumer GPUs are weak at FP64; D≥256 in `double` is too slow
- [ ] Cheaper eval: skip or subsample `lm_nll_windows` (full corpus forward each epoch dominates wall time)
- [ ] Train window / context closer to generate length (long “NLL” vs window NLL diverge when you only train on 8–16 bytes)
- [ ] Sampling controls for generate (`--temperature`, top-k / top-p) — greedy byte decode looks worse than the model’s distribution
- [ ] Default or warn: `--out` without `--resume` must not silently re-init over a trained checkpoint
- [ ] Keep byte tokenizer as a teaching path; do not treat it as the final vocab

**Subword + data (main jump from toy → meaningful text)**

- [ ] Subword tokenizer (BPE / WordPiece / Unigram) with **10k–30k** vocab (Target)
- [ ] Serialize vocab with checkpoints; migrate CLI `train` / `generate` / `load` off raw bytes
- [ ] Real corpora at scale: **100 MB → 1 GB+** (Wikipedia dumps, books, code—license-clean)
- [ ] Held-out split by **document/offset**, not overlapping byte windows
- [ ] Packing / streaming dataloader (don’t require full corpus `lm_nll_windows` every epoch)

**Capacity + training recipe (small LLM, still from-scratch C)**

- [ ] Scale config toward Target: ~**10–50M** params, **4–6** blocks, **256–512** context, FF ≈ 4× `d_model`
- [ ] LR schedule (e.g. decay stages or cosine); clip grads; optional dropout / weight decay
- [ ] Batch by **tokens**, not only window count; gradient accumulation for large effective batch
- [ ] Activation checkpointing or block-wise backwards if activation RAM blows up
- [ ] FP16/BF16 or TF32 GEMMs once FP32 path is solid (optional; keep FP32 reference)
- [ ] Stronger generate: KV already present—add temperature/top-p and length/EOS control
- [ ] Eval suite: window NLL, long-context NLL, held-out PPL, and a few fixed prompts for qualitative checks
- [ ] Multi-run: already have `--seed`, shuffle, `lm merge`—document as the cheap ensemble recipe

**Done when:** subword vocab, ≥100 MB train data, ~10M+ params, train window ≈ generate horizon, FP32 (or faster) on GPU, and generate produces readable multi-word text on held-out-style prompts—not just low NLL on `toy_tiny`.

## Target

Initial language model target (the “small LLM, not toy” bar):

- Decoder-only Transformer
- ~10–50M parameters
- 10k–30k **subword** vocabulary (not byte-256)
- 256–512 token context
- 4–6 Transformer layers
- 4–8 attention heads
- Training data from **100 MB** upward (then 1 GB+)

The goal is to implement and understand the complete stack from **neurons to text generation**, then grow that stack until generation is useful—not only until toy window NLL falls.

## Training

Next-token prediction (today: next **byte**; Target: next **subword**).

    Input:  "The cat is"
    Target: "on"

Training loop:

1. Tokenize text
2. Predict the next token
3. Calculate the loss
4. Backpropagate gradients
5. Update parameters
6. Repeat

Practical CLI knobs already: `--data`, `--epochs`, `--batch`, `--lr`, `--window`, `--seed`, `--resume`, `--out`, `-d`; generate via `lm generate --model … --prompt … --tokens …`.

## Data

Progression toward non-toy:

    toy KB → 10 MB → 100 MB → 1 GB → 10 GB+

Byte LMs need **more raw text** than subword LMs for similar fluency. Prefer document-level held-out splits. Wikipedia and other freely licensed text are fine for experiments.

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
