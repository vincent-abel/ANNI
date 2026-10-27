# Plan: proper 8-bit and 4-bit training on the RTX 5070 Ti

## Abstract

The current `--fp8` path (E4M3 forward, E5M2 dgrad, BF16 wgrad, per-tensor delayed scaling) is numerically sound since the audit ([GPU training audit](gpu-training-audit.md)) but *slower* than BF16 at D=1024: 42.8 vs 57.9 TFLOP/s on the meter, because every FP8 GEMM is preceded by one or two full passes over its FP32 operands (quantize, amax, transpose) that cost more than the tensor-core time the narrower GEMM saves. A proper 8-bit path produces FP8 tensors where they are born (LayerNorm, ReLU, attention and GEMM epilogues emit FP8 and their amax directly), keeps the transposed FP8 weights cached between optimizer steps, and uses block-scaled MXFP8 through cuBLASLt on sm_120 so the scale is no longer a per-tensor compromise. 4-bit (NVFP4 / MXFP4, E2M1 with 16- or 32-element block scales) is realistic for inference weights now and for the forward GEMMs of training later, with gradients kept in 8 bits. The rule for every step is the one set for this run: the loss curve must match the BF16 reference on the A/B harness within noise, or the step does not ship. Targets: 70 TFLOP/s with fused MXFP8 forward and dgrad (BF16 wgrad), 80 to 85 with MXFP8 wgrad, and 4-bit weights only where they cost nothing measurable.

## Where the time goes today

Per 4096-token microbatch in BF16 (bench of 2026-10-25, `scripts/prof_step.sh bench`): forward 34 ms, head and loss 7.4 ms, backward 71 ms, 112 ms total, 58 TFLOP/s. The GEMM work is 6.2 TFLOP per microbatch; at the 5070 Ti's dense BF16 tensor peak (about 176 TFLOP/s) that is 35 ms, so about 65% of the step is spent outside tensor cores: operand conversions (every GEMM converts its FP32 activations to BF16 in a separate pass, the weight-gradient GEMMs convert both operands), the FP32 elementwise and normalisation kernels, launch gaps between about 400 kernels per microbatch, and host-side work that the stream hides only partially. The per-operation table from `ANN_OP_PROF=1` (added with this plan) replaces this estimate with measurements at the next pause.

FP8 today adds to that overhead instead of removing it. For a forward GEMM `C = A·W` the path reads the FP32 activation `A` once to quantize it (a fused delayed-scale kernel that also reduces the amax), and the FP8 GEMM then reads 2 bytes less per element than BF16; the net is a saving of about one byte per activation element against an extra kernel launch and an extra pass. For the dgrad GEMM the same holds for `dY`. For the weight gradient the path transposes and quantizes both operands (two extra passes, both strided), which is why it stays off by default. The FP8 weights are cached transposed and re-quantized once per optimizer step, which is right. And a per-tensor scale, even per tensor and delayed as it is now, wastes most of E4M3's 3 mantissa bits on tensors whose rows differ by orders of magnitude (post-ReLU hidden, attention output), which is where the 0.004-nat gap of the A/B comes from.

## 8-bit: what a proper path looks like

### Produce FP8 where the tensor is produced

Every FP8 consumer in the block has a producer kernel that already touches the tensor:

| FP8 consumer (GEMM) | Producer that can emit FP8 and amax |
|---|---|
| QKV projection, FFN W1 | LayerNorm forward (one-pass kernel already supports an FP8 side output: `fp8_hot_prepare`) |
| FFN W2 | the W1 epilogue (ReLU + bias) |
| attention output projection Wo | the attention `P·V` epilogue (or the unpack-heads kernel) |
| head GEMM | the final LayerNorm (to be added) |
| every dgrad GEMM | the kernel that produces `dY`: the dgrad GEMM epilogue of the layer above, the ReLU backward, the LayerNorm backward |

The rule: an FP8 tensor is written once, by its producer, with the amax reduced in the same kernel; no standalone quantize pass remains in the forward or the dgrad path. The scale used by the producer is the delayed per-tensor scale (or the block scale, below); the amax it reduces feeds the next microbatch. This is what the "hot" LayerNorm path does for two of the eleven FP8 GEMMs per block today; the plan extends it to all of them, which means teaching the cuBLASLt epilogues to write a second, FP8 output (`CUBLASLT_MATMUL_DESC_D_OUT_SCALE`-style auxiliary output on CUDA 12.8+ / 13), and adding the FP8 side output to the ReLU-backward, attention and LayerNorm-backward kernels.

### Cache the transposed FP8 weights, once per step

Already done: `fp8_wt_get_ex` keeps `[N, K]` E4M3 copies for the forward and `[K, N]` copies for dgrad, invalidated by the device Adam step. The missing piece is producing them *from the Adam kernel*: the BF16 twin is already written in the Adam kernel (`adam_kernel_bf16`), and the same kernel can write the E4M3 twin plus a per-block amax, so the re-quantization pass after each step disappears too. Transposed layouts come from a single transpose kernel per weight per step (12 blocks × 6 weights plus the head: 73 transposes of at most 16 MB, about 2 ms per step, negligible against a 3.5 s step).

### Block scaling (MXFP8) instead of per-tensor scales

sm_120 tensor cores and cuBLASLt in CUDA 12.8+ / 13 support block-scaled FP8: E4M3 data with one E8M0 scale per 32 consecutive elements along K (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`). The scale is a power of two chosen from the block's own amax, so a row of small values next to a row of large values no longer loses its mantissa, and there is no delayed-scale history to keep. Cost: the producer kernel reduces 32-element amax groups instead of one amax (cheap, in registers), and the scale tensor is 1/32 of the data. Expected quality: the TransformerEngine and DeepSeek-V3 results put MXFP8/FP8-block training within noise of BF16 on GPT-style models of this size; the A/B harness will confirm or reject it here.

### What stays in high precision

- Master weights and Adam moments: FP32 (as now).
- Accumulation: FP32 in every GEMM (cuBLASLt `COMPUTE_32F`; no `FAST_ACCUM` for the weight gradients).
- Embedding table and its gradient scatter, the output head's `softmax` and cross-entropy, LayerNorm statistics, residual stream, optimizer: FP32.
- The head GEMM input may be FP8 (it is a normal GEMM), its logits are FP32.
- Weight gradients: BF16 operands first (phase 2), FP8 only when phase 3 shows no loss delta; the wgrad GEMM is the most sensitive to quantization because it sums over 4096 tokens.

### Why this gets faster

With producer-side FP8 the only extra traffic over BF16 is the amax reduction in kernels that already stream the data, and the GEMMs read half the bytes. On a card where the microbatch is 65% non-tensor-core time, the FP8 GEMMs themselves save about 10 to 12 ms of the 35 ms of GEMM time (the 5070 Ti's FP8 dense peak is about 2× its BF16 peak but these GEMMs are small, K=1024 or 4096, and do not reach either peak), and the removal of every quantize/convert pass saves another 8 to 12 ms, which is the part BF16 cannot get. That is the 70+ TFLOP/s target for phase 2 (about 85 ms per microbatch).

## 4-bit: where it is realistic

Blackwell tensor cores take FP4 (E2M1) with block scales: NVFP4 (16-element blocks, E4M3 scales plus a per-tensor FP32 scale) and MXFP4 (32-element blocks, E8M0 scales). On the 5070 Ti the dense FP4 rate is about 2× FP8. cuBLASLt exposes FP4 GEMMs on sm_120 in CUDA 13 with the same block-scale API as MXFP8.

1. **Inference weights (realistic now).** The `gpu_dec` decode path reads weights once per token; NVFP4 weights with a BF16 activation stream halve the weight bytes again versus FP8 and need no training change. Quality gate: perplexity on the validation set within 1% of the BF16 model (the usual result for 4-bit weight-only quantization of models this size with 16-element blocks is well under that). (The old int8 head quantization, `lm_quantize_head`, was removed as unused.)
2. **Forward GEMMs in training (possible, later).** Weights in NVFP4 for the forward pass (the FP32 master stays), activations in FP8 or FP4 with 16-element blocks, gradients in FP8. Published recipes (NVIDIA's NVFP4 training work, 2025) need three things to match BF16 loss on GPT-style models: a random Hadamard rotation of the gradient operands before quantization (spreads outliers across the block), stochastic rounding for the gradient tensors, and keeping the first and last blocks plus the head in higher precision. Each of these is a kernel in this codebase's style, none is exotic, but the loss-curve risk is real and the A/B harness decides.
3. **Weight gradients in 4-bit: not planned.** The sum over 4096 tokens per step makes the wgrad GEMM the first place quantization noise shows up in the loss; FP8 with block scales is the floor there.

Expected speedup from 4-bit on top of a fused MXFP8 path: the forward is about 30% of the microbatch; halving its GEMM bytes again buys at most 4 to 6 ms (5%), so 4-bit training is a phase-5 item justified mainly by what it teaches, while 4-bit *inference* is a clear win for the chat model (memory and tokens per second on the same card).

## Risks

- **cuBLASLt coverage on sm_120.** The FP8 path already had to live with the `(OP_T, OP_N)` and no-E5M2×E5M2 restrictions; MXFP8 and FP4 add layout requirements (K multiple of 32, scale tensors in a tiled layout) that must be checked kernel by kernel with the algorithm-heuristic query, with a BF16 fallback per GEMM shape as today.
- **Epilogue side outputs.** Writing an FP8 copy from a cuBLASLt epilogue is only available on recent toolkits; where it is not, a fused elementwise kernel (bias + activation + FP8 out + amax) replaces the epilogue, which is still one pass instead of two.
- **Determinism and the twin verifier.** Every FP8 tensor gets a producer; the `ANN_TWIN_CHECK` verifier must learn to re-quantize and compare FP8 twins, or stale-twin bugs of the kind the audit found would return.
- **Quality drift that short A/Bs miss.** 150-step A/Bs catch gross errors; the gate for shipping a phase into the long run is a 1000-step A/B (one hour) plus the mid-epoch validation probes of the run itself.

## Phases and targets

| Phase | Change | Gate (A/B, same seed and slice) | Target |
|---|---|---|---|
| 0 | Measure with `ANN_OP_PROF=1`; remove the BF16 conversion passes (BF16-resident activations produced by LN / epilogues, cached BF16 dY) | loss within 1e-3 of BF16 reference | 62 to 65 TFLOP/s |
| 1 | FP8 produced in LN, ReLU, attention and dgrad epilogues; cached FP8 weights from the Adam kernel; wgrad stays BF16 | within 3e-3 of BF16 at 150 steps, within 2e-3 at 1000 steps | 70 TFLOP/s |
| 2 | MXFP8 block scaling (32 × E8M0) replaces per-tensor delayed scales | gap to BF16 at or below phase 1's | same speed, better margin |
| 3 | MXFP8 weight gradients with FP32 accumulation | within 2e-3 at 1000 steps | 80 to 85 TFLOP/s |
| 4 | NVFP4 weights for inference (`gpu_dec`), chat model | val perplexity within 1% | 2× decode tokens/s |
| 5 | NVFP4 forward weights in training with Hadamard + stochastic rounding | within 2e-3 at 1000 steps, or dropped | +5% at best |

Phase 0 is the groundwork for everything after it and is the next thing the pause cycle measures; phases 1 to 3 are the 8-bit plan; 4 is cheap and useful; 5 is research.

## References

- [GPU training audit](gpu-training-audit.md): the FP8 defects fixed and the A/B that measured FP8 at 42.8 TFLOP/s.
- [Mixed precision train (R10)](mixed-precision-train.md), [Fused epilogues (S8)](fused-epilogues.md): the BF16 and epilogue machinery this builds on.
- NVIDIA cuBLASLt documentation, CUDA 13: block-scaled FP8/FP4 matmul (`cublasLtMatmulMatrixScale_t`), sm_120 support matrix.
- OCP Microscaling Formats (MX) specification v1.0: MXFP8 and MXFP4 element and scale formats.
- DeepSeek-V3 technical report (2024): fine-grained FP8 training recipe (block scaling, FP32 accumulation).
- NVIDIA, "Pretraining Large Language Models with NVFP4" (2025): Hadamard rotation, stochastic rounding, high-precision first and last layers.
