# GPU training audit: stale device caches and the BF16 divergence

## Abstract

The talk-170m pretraining diverged in BF16 at a learning rate where FP32 was stable, so the GPU training path (`num/gpu_cuda.cu`, `lang/lm_adam.c`, `lang/lm.c`) was audited end to end against a CPU reference on the small test model. One bug was BF16-specific; the four most damaging ones affected every precision and had been active since the FP8/residency rewrite. The worst: the cross-entropy targets were cached on the device by host address and never refreshed, so after the first optimizer step every microbatch was scored and back-propagated against the targets of one fixed microbatch. Twelve defects were fixed, each verified by a GPU regression test (`make test-gpu`) that compares the BF16 and FP32 gradients tensor by tensor from the first step and checks that a forward pass is a pure function of its inputs.

## Method

`tests/test_gpu.c` builds the 64-wide / 2-block test LM against the real CUDA backend and trains it for a few Adam steps in FP32 and in BF16 on the same synthetic batches. Each step records every weight gradient on the host before the update. The CPU stub build of the same program gives the reference. The invariants that caught the bugs:

- the loss of a forward pass depends only on the weights and the batch (same batch twice, a different batch, a fresh model);
- GPU FP32 equals the CPU reference exactly at step 1 and follows the same trajectory;
- BF16 follows FP32: loss within \(2\times10^{-3}\), every weight gradient within 20% relative error and within 10% in norm, from step 1;
- after one Adam step every parameter class has moved (matrices by more than weight decay, FFN biases, LayerNorm vectors) and the next forward uses the new vectors;
- freeing a trained model leaves no pending CUDA error.

An env-gated verifier (`ANN_TWIN_CHECK=1`) re-converts every cached BF16 twin and every cached parameter vector before serving it and reports any byte that differs from a fresh conversion. It found none once the fixes were in.

## Defects, ranked by damage

1. **CE targets and LayerNorm `inv_std` served stale (all precisions).** `ew_acquire` routed every upload-only host buffer of at most 64 KB through the persistent "bias cache", keyed on the host address and refreshed only when the weight epoch bumped, which happened once (Adam init). The targets are a `malloc` per call that lands at the same address, so from the second optimizer step on the loss and the gradient used the targets of one fixed microbatch; the backward's `inv_std` (one float per row) was stale the same way. The model still "learns" (position-wise memorisation of a fixed target sequence drives the reported NLL down), which is why the runs looked healthy for hundreds of steps. Fix: the cache is opt-in through `ew_acquire_param`, used only for bias and LayerNorm vectors; everything else is staged per call.
2. **Parameter vectors never refreshed after Adam (all precisions).** Nothing bumped the cache epoch after a step, so LayerNorm gamma/beta and the head bias stayed at their step-1 values on the device while the host copies kept drifting (weight decay alone shrinks gamma by 57% over the planned run). The saved checkpoint therefore did not describe the network that produced the loss curve. Fix: `lm_adam_step_scaled` calls `gpu_half_wt_invalidate(NULL)` after every step (host and device paths), as do `lm_weights_to_device` and `lm_weights_host_dirty`.
3. **FFN bias gradients never reached Adam in batch mode (all precisions).** `gpu_mat_col_sum_add` accumulates db1/db2 on the device and only `gpu_device_sync` flushes them to the host, which the training loop never called before the optimizer gathered the host vectors. Fix: one device sync at the top of the Adam step (once per 32 microbatches).
4. **cuBLASLt timing loop multiplied accumulate GEMMs (BF16/FP16).** Picking the fastest heuristic runs each candidate once with the caller's arguments. For `dW += Aᵀ dY` (beta = 1) that added the product up to 16 times on the first call of each shape: the test showed weight gradients inflated up to 7x at step 1. Fix: when beta is non-zero the candidates write out of place into a scratch D (cuBLASLt allows D ≠ C); the measured winner then runs once for real.
5. **First-step weight gradients dropped (all precisions).** `gpu_buffers_init` aliased every dW into the Adam flat and zeroed the slice, discarding the gradient already accumulated by the first step's backward passes. Step 1 was a pure weight-decay step. Fix: copy the existing twin (or host grad) into the slice.
6. **Aliased grads lost `MAT_DEV_ALIASED` (all precisions).** `finish_out` and `mat_mark_device` turned the aliased dW into plain device mats, so `mat_free` later ran `cudaFree` on a slice of the flat. The sticky "invalid argument" made the next `cudaGetLastError` check fail and silently pushed that op (the embedding lookup of a new model) onto the CPU fallback. Fix: both preserve the alias; `release_gpu` detaches the views and pulls the weights to host before freeing the flats.
7. **Residency marked wrong after host fallbacks (all precisions).** The pattern `p = mat_compute_ptr(m); gpu_op(p); mat_mark_device(m)` marked the device authoritative even when `p` was the host pointer, so the following ops read the stale twin. Fix: `mat_mark_written(m, p)` marks whichever side the kernel wrote; used at every such site in `lm.c`, `ffn.c`, `layernorm.c`.
8. **Twin caches outliving their buffers (BF16, latent).** Half-precision weight twins and the two activation twins are keyed on the FP32 device address; `cudaMalloc` reuses addresses, and `half_wt_get` also accepted activations (per-head K/V in the fallback attention path) as if they were weights. Fix: `gpu_free` drops any twin of the freed buffer, `vec_free` drops the device copy and column-sum accumulator of a freed host vector, and once the Adam flat is registered only pointers inside it get a persistent twin.

9. **FP8 delayed scaling shared one scale across all tensors (FP8).** The E4M3/E5M2 path kept a single device scale for every forward activation, another for every dgrad operand and one pair for wgrad, and the LayerNorm one-pass quantizer one scale for every LN output. Delayed scaling means a tensor is quantized with the amax recorded by the *previous* call, which here was a different tensor of a different layer: whenever the magnitude rose the values saturated at 448 (E4M3 `SATFINITE`), and when it fell the tensor lost its mantissa bits. Fix: scale slots keyed on the FP32 source address (`fp8_act_scale`), one per tensor and format, so each tensor sees its own previous amax, as in TransformerEngine. The FP8 cuBLASLt wrapper had the same algorithm-timing accumulate bug as defect 4 and got the same out-of-place fix. The FP8 weight cache itself was sound: 320 slots cover the 146 (weight, layout) pairs of the 12-block model, and the device Adam step invalidates it.
10. **Pinned host vectors never unregistered (all precisions, multi-model processes).** The Adam scatter pins each bias/LN vector with `cudaHostRegister` and nothing unregistered it at `free`, so the seventh model created in one process (tests, merge, a fine-tune after a pretrain in the same process) found its vectors "already pinned" at overlapping addresses and its device-to-host copies failed with "invalid argument". Fix: `gpu_host_forget` unregisters on `vec_free`.
11. **Eager-mode GEMM outputs on device-only mats read as zeros (all precisions, evaluation paths).** Outside the training loop the backend runs in eager mode, where `finish_out` mirrored each GEMM result to the host and marked the host authoritative; for the device-only activation mats (no host pages) that left consumers a NULL host pointer and the CPU fallbacks read zeros. Symptoms: the "start NLL" of a resumed run was 9.2 while training continued at 4.9 and the next validation probe read 5.7; `lm info`, generation and the chat path evaluate a loaded model the same way. The packed forward in eager mode crashed in the attention host fallback for the same reason. Fix: device-only mats stay device-authoritative; host-backed mats keep the old rule (host code edits them in place without marking, as the causal mask does). A GPU round-trip test (train, save, load, evaluate on both paths) now covers it.
12. **`gpu_set_fp8(0)` left the FP8 dgrad flag on (FP8, same-process runs).** After an FP8 run, a BF16 run in the same process kept quantizing its data gradients to E5M2 (every GEMM whose dimensions are multiples of 16). It made the test's Tensor-Core BF16 check measure FP8 gradients; production runs one precision per process. Fix: off clears both flags.

Not fixed, noted: the cuBLASLt algorithm cache evicts into slot 0 when full (thrash, performance only); CUDA-graph replay re-reads host targets only through the captured memcpy nodes, so `--cuda-graphs` still needs pinned, stable target buffers; there is no final LayerNorm before the output head (quality, not correctness).

## Results on the test model

Before the fixes, two identical runs in one process gave different step-1 losses (5.572 vs 5.554 against a CPU reference of 5.582) and BF16 gradients 1.2x to 7.3x larger than FP32. After the fixes GPU FP32 matches the CPU reference to all printed digits over three Adam steps (per-parameter checksums identical), two runs are bitwise repeatable, and BF16 stays within the tolerances above at every step; the largest BF16 deviation is dW1 at about 12% relative error with the same norm, from ReLU-mask flips near zero.

With the per-tensor scales, FP8 training on the test model stays within 1e-2 of the FP32 loss over four steps and every weight gradient within 23% relative error and 3% in norm on a clean process, up to 43% on dW1 after other GPU work has shuffled the allocations (BF16: 13% and 1%). `make test-gpu` checks BF16 at 20%/10% and FP8 at 60%/30%.

## Precision A/B at the real scale

`scripts/ab_precision.sh 150`: the first 19.7M tokens of the mix, the pretraining flags (batch 16, 32 accumulated microbatches, LR 3e-4 with 1000 warmup steps), seed 1, 150 Adam steps per precision on the RTX 5070 Ti.

| GEMMs | Train NLL | Val NLL (256 windows) | Steady tok/s | Meter TFLOP/s | VRAM |
|---|---|---|---|---|---|
| FP32 | 7.5893 | 7.5048 | 19.9k | 30.0 | 11.25 GB |
| BF16 | 7.5896 | 7.5047 | 38.3k | 57.9 | 11.52 GB |
| FP8 (E4M3 fwd, E5M2 dgrad, BF16 wgrad) | 7.5916 | 7.5092 | 28.3k | 42.8 | 11.70 GB |

BF16 and FP32 agree to four digits at every 5% validation probe; FP8 trails by 0.004 nats at the end. FP8 is slower than BF16 here: the extra quantization and transpose passes (and the E4M3 weight re-quantization after each step) cost more than the FP8 tensor-core GEMMs save at D=1024 on sm_120. The run restarts in BF16.

## Consequence for the talk-170m run

Every checkpoint written before this note trained on fixed targets with frozen LayerNorm vectors and untrained FFN biases, in FP32 as well as BF16; the FP32-vs-BF16 comparison that motivated the FP32 restart measured the bugs, not the precision. The run has to restart from scratch with the fixed binary. BF16 is believed correct again, but a short FP32-vs-BF16 A/B on the fixed code is the cheap way to confirm it at the real scale before committing 30 hours.

## References

- [Adam step on GPU](gpu-adam.md), [GPU-resident Adam](gpu-adam-residency.md): the flat alias scheme that defects 5 and 6 live in.
- [Device-fused CE (S1)](device-fused-ce.md), [GPU bias grad reduce (S3)](gpu-bias-grad-reduce.md): the target upload and column-sum paths of defects 1 and 3.
- [Mixed precision train (R10)](mixed-precision-train.md), [Fused epilogues (S8)](fused-epilogues.md): the BF16 twin and cuBLASLt paths of defects 4 and 8.
- [Talking model pretraining run](talk-170m-pretrain.md): the run this audit restarts.
