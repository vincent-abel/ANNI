# Code audit: dead paths and remaining compute (talk-170m, candidate 16)

Read-only audit of the tree at df483bb ("candidate 16"), 2026-10-26 night, done by two reviewers over `num/gpu_cuda.cu` (9.7k lines), `num/attn_fa2.cuh`, `lang/*.c` and the tests, against a fresh nsys profile of candidate 16 on the idle RTX 5070 Ti (129 microbatches, `logs/prof/nsys___bf16___flash_attn.sqlite`). Nothing was changed. The production configuration assumed throughout: `./anni lm train --bf16 --flash-attn`, B=16, T=256, D=1024, 16 heads of 64, FF=4096, vocab 16384, batch sync mode in training, eager mode for the probes, every `ANN_*` switch at its default, no `--cuda-graphs`, `--fp8`, `--fp16` or `--act-checkpoint`. Everything proposed below keeps the model, the checkpoint format and the Adam state usable as they are.

## Summary

- **Dead code:** about 1,100 lines reach nothing in any configuration (a dead attention branch, the FP16 path, no-op FP8 leftovers, two unreachable FFN fallbacks, unused twin machinery, uncalled entries, pre-CUDA-11 `#else` bodies) and can go; about 3,700 lines are reachable only through an env switch, a failure, the CPU build or the probes and should stay as fallbacks unless the owner decides to drop a whole family (the WMMA attention set, 575 lines; the TC + causal + tiled attention set, 1,400 lines, of which TC is the FP32 reference of `make test-gpu`).
- **Three real defects found on the way:** `--fp16` unscales gradients that were never scaled (loss scaling half-implemented); about 1 GiB of VRAM is allocated per run and never touched, plus about 1.5 GiB of FP32 buffers that serve only as cache keys; 24 blocking pageable `cudaMemcpy` bias uploads per microbatch inside batch mode.
- **Compute:** the microbatch is 60.1 ms of kernels + 1.67 ms of gaps = 61.5 ms (plus 0.5 ms of amortised Adam step); GEMMs are 46.8 ms (78%) and already near the card. The measured gaps are 1.67 ms, not the 5.5 ms estimated earlier (that figure counted start-up algorithm timing and the probes), so CUDA-graph replay can recover at most about 1.7 ms (2.7%) and about 1.25 ms of that is recoverable without graphs. The high-confidence, bit-identical items below add up to about 3.4 ms per microbatch (62.2 to about 58.8 ms, +5.8%, about 69.5k tok/s); the validation probes (40 s each, 21 per epoch) and the checkpoint save (hundreds of millions of 4-byte `fwrite` calls) are worth another 1.1 ms-equivalent of wall time.

---

## Part 1. Dead and unreached code

Where the current configuration actually goes:
- Training (`pack_run` -> `lm_*_packed`, batch mode): FA2 `mma.sync` attention on the fused BF16 QKV rows, `gpu_mat_mul_bt_twin` for dO, BF16-only LayerNorm and ReLU outputs, the warp-per-row LayerNorm kernels, `ce_softmax_delta_bf16_kernel`, the Adam BF16 flat buffer.
- Validation probes (`lm_nll` -> `lm_forward`, B=1 caches, eager, BF16): `attn_mha_forward_cache`, i.e. per-head OpenMP host attention with small GPU GEMMs, `gpu_softmax_rows`, and `gpu_attn_qkv_fused` without a hint (the 3-stream `qkv_par` path).
- `lm info` runs FP32 (Sgemm, `w1_finish`, `bias_row_add`). Generation uses `gpu_dec` after a B=1 FP32 prefill (`generate.c:299` switches to BATCH mode).
- Tests: `test_ann.c` is CPU-only (GPU stubs plus the B=1 host backward). `test_gpu.c` (B=4, T=64, dh=16) covers FP32 (TC attention, since `w_all` fits), BF16 (FA2), FP8 (FA2 + `ln_fwd_kernel` + `w1_finish` + `gpu_relu_backward`) and `bf16-tc` (TC + `qkv_par`); it sets none of the `ANN_*` fallback switches. `tests/bench_attn.cu` and `tests/bench_ln.cu` are not in the Makefile; they include `gpu_cuda.cu` and use the WMMA kernels, the head-major `fa2_*_launch` and `ln_fwd_kernel` / `ln_bwd_kernel` as references.

### (a) Attention paths

DELETE (about 154 lines)
- `fused == 2` branch, `lang/attn.c:977-990`: nothing sets `c->fused = 2` (values are 0, 1, 3, 4, 5 at 836-883). No test.
- `gpu_attn_recompute_w`, `gpu_cuda.cu:5154-5195`, and `attn_lse_kernel` 4595-4654: only caller is that branch. Also the stub `gpu_stub.c:443-454` and the `gpu.h` declaration.
- `gpu_attn_flash_bwd_ms` 4796-4799, stub `gpu_stub.c:371-374`, `gpu.h:258`: no callers. Optional: the `fa_time_begin` / `fa_time_end` timing (4745-4794) exists only for it.
- `tc_fp8_softmax_buf` 5681-5703 plus the `w8` parameter of `attn_tc_softmax_kernel` (5711, 5722, 5759-5761): the E4M3 copy of P is written under `--fp8` and never read.

KEEP AS FALLBACK
- Tiled flash (`fused = 4`): `attn_fwd_kernel` 4508-4593 (shared), `gpu_attn_flash_fwd` 4879-4951, `attn_tiled_bwd_kernel` 4953-5069, `gpu_attn_tiled_bwd` 5071-5152; `attn.c` 867-879, 964-969 (about 360 lines). Reached when the FA forward fails, or with FP32/FP16 `--flash-attn` when the score buffer exceeds 256 MiB (which the production shape does). Not tested.
- Score-buffer causal kernels (`fused = 1`): `attn_bwd_rows_kernel` / `attn_bwd_cols_kernel` 4656-4740, `gpu_attn_causal_fwd` 4809-4877, `gpu_attn_causal_bwd` 5513-5587; `attn.c` 881-885, 991-996 (about 233 lines). Reached only when `w_all` exists and TC fails (an `ANN_CUBLAS=0` GPU build without `--flash-attn`). No test.
- TC strided GemmEx (`fused = 3`): 5589-6070 and 7066-7400 (about 795 lines); `attn.c` 662-682, 854-865, 970-976. Reached without `--flash-attn` or in non-BF16 runs. Tested by `test_gpu` (FP32 and `bf16-tc`).
- WMMA fused attention and the head-packed FA fallback (about 575 lines): `attn_fa_wmma_fwd_kernel` 6248-6397, `attn_fa_wmma_bwd_kernel` 6658-6851, `fa_pack` slots 6411-6471, the packing branch of `gpu_attn_fa_fwd` 6542-6575, the packed branches of `gpu_attn_fa_bwd` 6983-6990 and 7003-7049, `attn_fa_dvec_kernel` 6597-6617, `attn_pack_heads_bf16_from16_kernel` 6641-6656, and the head-major `fa2_fwd_launch` / `fa2_bwd_launch` / `fa2_lay_headmajor` in `attn_fa2.cuh` (35-43, 523-532, 553-560). Reached with `ANN_FA2=0`, `ANN_TWIN_EMIT=0`, or when the fused QKV GEMM fails. Only `bench_attn.cu` runs them. Caveat: production still calls `cudaFuncSetAttribute` on the two WMMA kernels and gates on `fa_wmma_ok`; deleting WMMA means moving those two pieces.
- `gpu_softmax_bwd_rows` 4418-4465 and `softmax_bwd_kernel` 3880-3895: only B=1 training (`attn_mha_backward`, `attn.c:461-592`) and the host packed backward. No GPU test; `test_ann` covers the host side.
- Host packed attention (`attn.c:886-917`, `997-1100`, `pcache_ensure_seq` 698-712): CPU build, covered by `test_ann`.

KEEP (live): FA2 DH=64 (production) and DH=16 (`test_gpu`); DH=32/48 are unused but cheap template cases. `gpu_attn_qkv_fused` including the `qkv_par` 3-stream path (152-191, 7616-7675), which the probes use. `attn_mha_forward_cache` (probes). The serial `cublas_gemm_rm_ha` QKV fallback (7676-7682) and `s_x_half`.

Cosmetic: `attn.c:846-852` still logs "[attn] fused tensor-core attention (WMMA)" when FA2 runs; the comments at `attn.c:829-833` and `gpu_cuda.cu:7441-7444` contradict the current dispatch.

### (b) FP16 and FP8

FP16 (about 111 lines): DELETE recommended; untested and numerically wrong. Code: `f32_to_f16_kernel` 1388-1394; the F16 tail of `half_wt_convert` 1708-1711; `cublas_gemm_ex_f16` 1999-2016; F16 branches in `rm_ha` (2140-2142), `rm_bt_ex` (2393-2402), `rm_at` (2525-2537), `tc_gemm_qk_half` (5888-5893), `tc_gemm_qk_batched` (5941-5946); the `FAST_16F` selection in `cublas_lt_gemm_half`; `--fp16` in `main.c:234-245`; loss scaling in `lm_train.c:87-88, 516-518, 538-563`. The bug: `train_grad_unscale` divides the gradients by `g_loss_scale` (1024 growing to 65536) but nothing ever multiplies the loss or the delta by it; `lm_train_loss_scale()` (538-541) has no caller. Adam's scale invariance hides most of it; it shows up as `--grad-clip` being 1024-65536 times looser and `eps` relatively larger. If FP16 is kept, fix the scaling.

FP8 (about 1,300 lines): 483-1386, the GEMM branches 2068-2108, 2310-2339, 2434-2480, the QKV branch 7497-7562, the LayerNorm hot buffer 7812-7833, `ln_fwd_kernel`'s `y8` parts, Adam 8394-8528 / 8555-8586.
- KEEP (about 960 lines): `--fp8` / `--fp8-dgrad` work and `test_gpu` tests them.
- DELETE (about 342 lines, no-ops even under `--fp8`): `fp8_wt_get` 1152-1156 (never called); ten scale globals allocated, set to 1 and never read (`g_fp8_scale_A/B/A_next/B_next/WA/WB/WA_next/WB_next`, `g_fp8_hot_quant`, `g_fp8_hot_buf_scale`, 499-506, 511-512, and their setup in `fp8_scales_ensure`, about 45 lines; the per-tensor `fp8_act_slot` scales replaced them); `ANN_FP8_ADAM` (about 171 lines: globals 515-522, kernels and `fp8_adam_ensure` 8394-8528, dispatch 8559-8586; default off, the code itself says it "explodes"); `ANN_FP8_WGRAD` (about 111 lines: 2434-2480, `fp8_quant_transpose_tensor` 1043-1060, `f32_transpose_to_fp8_delayed_kernel` 688-731, `g_fp8_B`; default off, "costs more than the GEMM saves", not tested).

### (c) CPU / host fallbacks duplicated on the GPU

DELETE (about 147 lines)
- Two dead fallback branches in `lang/ffn.c:203-227`: the `gpu_bias_relu_forward` and `gpu_relu_forward` branches can run only if `gpu_ffn_w1_finish` failed, and on the CPU build their GPU calls are stubs returning -1, so control always reaches the host loop at 228. With them go `gpu_bias_relu_forward` (5197-5244), `gpu_relu_forward` / `_locked` (8086-8112), `relu_fwd_kernel` (4351-4357) and the stubs (`gpu_stub.c:456-464, 591-597`). No test.
- Per-sequence `gpu_memcpy_d2d` gather/scatter fallbacks, `lang/lm.c:767-780` and `800-809`: reached only if the `pack_seq_rows` launch fails, never on CPU (gated on `gpu_available()`).

KEEP (CPU build, `test_ann`): the host loops in `attn.c`, `ffn.c`, `layernorm.c`, `lm.c` (softmax and CE at 484-545), `gradutil.c`; host Adam (`lm_adam.c:558-609`).
KEEP AS FALLBACK: the naive teaching GEMM kernels (`gemm_kernel` 2541-2555, `gemm_bt_kernel` 3158-3173, `outer_add_kernel` 3175-3190, `gemm_at_kernel` 3245-3262; `ANN_CUBLAS=0` builds or a cuBLAS failure); `lm_adam.c` non-aliased gather/scatter (384-394, 424-440), defensive.

### (d) cuBLASLt epilogue / aux leftovers

No AUX, DRELU or BGRAD code remains; the epilogues used are DEFAULT, BIAS and RELU_BIAS. `cublas_gemm_rm_bias` / `_bias_relu` / `_epi` (2252-2271) are live (eager BF16 probes, `--fp8`, `ANN_HID_BF16=0`). Not dead but costly: `lt_bias_upload` (2154-2172) does a blocking pageable `cudaMemcpy` on every bias-epilogue GEMM (W1 and the W2 residual GEMM: 24 stalls per microbatch inside batch mode), and `g_lt_bias16` is re-converted on each of those calls; `bias_cache_get` (keyed on the weight epoch) would avoid both (see B1).

### (e) CUDA-graph code (`--cuda-graphs` only, untested): KEEP AS FALLBACK, about 200 lines

`gpu_graph_begin` / `end` / `replay` / `destroy` 9500-9550, globals 40-42, `gpu_ce_fetch_mean` 9391-9399, `stream_is_capturing` 2772-2779 plus about 25 guard sites, the `ew_is_dev` shortcut 3518-3520; `lm_train.c` 836-889, 500-508 and the invalidation calls at 486, 519, 527, 534; `main.c:230-231`; stubs `gpu_stub.c:769-787`. Every twin-emitting fast path opts out under capture, so `--cuda-graphs` today runs the pre-candidate-8 conversion paths (and, per the graphs stream, capture on the legacy stream is refused anyway). The decode graph (`ANN_DECODE_GRAPH`, 10007-10024) is live for generation: KEEP.

### (f) Twin and cache machinery no longer hit

DELETE (about 55 lines): the `GRAD_TWIN_BIG` slot and `ANN_GRAD_TWIN_BIG` (1484-1486, 1500-1520; the only tensor over 16 M elements is the head delta, and with `ANN_CE_BF16=1` `grad_twin_peek` finds the in-place twin first); `gpu_half_wt_refresh_all` (1901-1916, stub `gpu_stub.c:495-497`, `gpu.h:71`; no caller); the `ln_bf16_publish` / `ln_bf16_consume` compat wrappers (1889-1899; inline them).
KEEP AS FALLBACK: the `fa_pack` slots (`ANN_FA2=0`); the per-matrix `half_wt` cache (runs before the first Adam step).

Leftovers that waste work or VRAM (to confirm with a measurement):
- The FA2 backward allocates `g_tc_half_do` (16 MB) even on the strided path (line 6958) where it is never written.
- `attn_fa2_bwd_kernel` writes FP32 dK and dV that nothing reads: both GEMMs consume the BF16 twins (32 MB of writes per layer per microbatch; see B9).
- Eagerly `cudaMalloc`'d FP32 buffers never touched in packed training, 16 MiB each per layer (about 960 MiB over 12 layers): `block_cache.ffn_out` (`block.c:141`, unused in every configuration), `block_cache.attn_out` (B=1 caches only), `attn_pcache.k` and `.v` (never written, not keys), `attn_pcache.dtmp` (only `mat_mul_wt_add`'s fallback).
- FP32 buffers that serve only as twin keys: q, `ln1_out`, `ln2_out`, `dconcat` (16 MiB each) and the FFN hidden (64 MiB): about 1.5 GiB over 12 layers. Freeing them needs the slots to be keyed on the `mat` struct instead of the device pointer.
- `half_scratch_warm` preallocates 64 + 32 MB of scratch; `g_half_B` is never needed in training (weights come from the flat) but the probes' per-head GEMMs rely on no-realloc scratch.

### (g) CLI flags and env switches that change nothing today

- `ANN_GRAD_TWIN_BIG=1`: no effect while `ANN_CE_BF16=1`. `ANN_FP8_ADAM`, `ANN_FP8_WGRAD`: no effect without `--fp8`. `ANN_TWIN_CHECK=1` now skips BF16-only activation twins and every producer-written gradient twin, so it checks only weight, flat and bias twins. `--fp16`: wrong (see b). `--cuda-graphs`: silently falls back to the older paths (see e).
- Unused getters, DELETE: `gpu_compute_prec()` (51-54), `gpu_fp8_dgrad()` (74-77), `lm_train_loss_scale()` (`lm_train.c:538-541`).
- `lm_adam_seg.host_use` is always 0 for matrices (every `add_mat` passes 0): the `host_use` branches in `seg_gather` / `seg_scatter` (`lm_adam.c:392, 417-421, 425-427, 436-439`) are dead in every configuration; the comment at `lm_adam.c:115` is stale. DELETE.
- Test or debug only: `ANN_LT_HEURISTIC`, `ANN_OP_PROF`, `ANN_STEP_PROF`, `ANN_GNORM_EVERY`, `ANN_LN_RPW`.
- Real fallback switches, none exercised by `make test-gpu`: `ANN_FA2=0`, `ANN_LN_FAST=0` (re-enables `ln_bwd_kernel`, whose `dgamma`/`dbeta` parameters are `(void)` at 3979-3980, plus the `col_sum_tiled_kernel` pair at 7988-7995), `ANN_LN_BF16=0`, `ANN_HID_BF16=0`, `ANN_TWIN_EMIT=0`, `ANN_CE_BF16=0`. `ln_fwd_kernel` stays live for FP8 (`y8`).

### (h) Anything else

DELETE (about 300 lines): `col_sum_kernel` 8977-8990 (never launched); `gpu_layernorm_forward` 7904-7910 plus stub and declaration (no caller); `gpu_stream_sync_copy` 8945-8949 plus stub and declaration (no caller); `gpu_mat_mul_backward` 8646-8660, `gpu_adam_step` 8599-8644, `gpu_adam_step_dev` 8531-8537 (called only by `test_ann`'s stub assertions at `test_ann.c:1419, 1423-1426`, which go with them); the quantized head (`lm_quantize_head`, `lm.c:173-181`, has no caller, so `w_head_q` is always NULL: `lm.c:358-361`, `432-441`, the guard at `decode.c:27`; `lang/quant.c` / `.h` (97 lines) then serve only `test_qmat_roundtrip`; about 120 lines); the pre-CUDA-11 `#else` bodies (about 26 lines at 1705-1707, 2011-2015, 2055-2058, 2211-2213, 2344-2346, 5708-5710, 5768-5770, 5778-5780, 5824-5826, 5894-5901, 5947-5951, 7564-7566, 8292-8295, 9243-9245, 9361-9365, plus their directives; the Makefile targets CUDA 12.8/13 only).

### Totals

Deletable with nothing reaching them: about 1,100 lines ((a) 154, (b) 453, (c) 147, (f) 55, (h) 300, plus about 40 matching `gpu_stub.c` / `gpu.h` lines); the FP16 removal drops `--fp16`, and the `test_ann.c` stub asserts at 1419 and 1423-1426 must go with it. Keep as fallback: about 3,700 lines (old attention paths 2,350; core FP8 960; CUDA graphs 200; teaching GEMMs 66; the `LN_FAST=0`, `CE_BF16=0`, `HID_BF16=0` paths). Decisions that need the owner: the WMMA set (575 lines; `bench_attn.cu` and two production references depend on it) and the TC + causal + tiled set (1,400 lines; TC is the only path `test_gpu`'s FP32 reference uses).

---

## Part 2. Remaining compute optimizations

All figures are ms per microbatch out of about 62.2 (60.1 kernels + 1.67 gaps + 0.5 amortised Adam), measured from the candidate-16 nsys export over the 32 microbatches between Adam steps 2 and 3 (456 GPU events per microbatch). Cross-check: the run's 65.8k tok/s is 62.2 ms per 4096 tokens.

Where the 1.67 ms of gaps goes (gap before each event pair, per microbatch): b1 upload (16 KB pageable H2D) then its BF16 conversion, 12 x 44 us = 0.53 ms; b2 upload (4 KB) then the W2 GEMM, 12 x 33 us = 0.40 ms; `embed_scatter` to the next microbatch's ids upload (the `cudaFree` syncs and the host loop), 0.19 ms; `f32_to_bf16` to GEMM launch gaps, 38 x 3 us = 0.12 ms; ids to `embed_lookup` and `embed_lookup` to `posenc` (`cudaFree` of `d_ids`), 0.07 ms; `bias_add` to the CE targets upload, 0.03 ms.

| # | What | Where | Gain (ms/mb) | Effort | Numerics / risk | Depends on |
|---|---|---|---|---|---|---|
| B1 | Device-cached W1/W2 epilogue biases plus a per-epoch BF16 copy of b1: removes 24 pageable `cudaMemcpy` syncs and 12 conversion kernels | `gpu_cuda.cu` `lt_bias_upload` 2154-2171, `cublas_gemm_rm_epi_res` 2181-2250 | ~1.0 (gaps measured 0.93) | 2-3 h | Bit-identical; risk = a stale bias if the epoch invalidation is missed | `BIAS_CACHE_SLOTS` 64 -> 128 (line 3529) |
| H1 | Batched validation probe: 16 windows per forward, GPU head + CE | `lm_train.c` `lm_nll_windows` 943; `lm.c` `lm_nll` 921, `lm_logits_at` 340 | ~0.9 equivalent (about 834 s per epoch, 1.4%) | 4-6 h | Training untouched; the probe value moves by ~1e-5 (FP32 on GPU instead of FP64 on host) | Fix the resume double probe too (~40 s per restart) |
| B2 | Add the head bias inside `ce_softmax_delta_bf16_kernel`, drop `bias_add_kernel` (834 us) | `lm.c` `ce_rows_backward` 443-446; `gpu_cuda.cu` 9033, `ce_delta_run` 9169 | 0.83 | 1-2 h | Bit-identical (same FP32 add) | Keep `bias_add` on the non-fused paths |
| B3 | One QKV dgrad GEMM (K=3072 with the cached packed `[Wq|Wk|Wv]`) and one strided-batched QKV wgrad instead of 3 + 3 GEMMs | `attn.c` proj 1101-1112; FA2 backward twin stores | 0.5-1.5 (uncertain) | 8-12 h | Not bit-identical (FP32 summation order) | dK/dV twins and a BF16 dQ in one `[N][3D]` buffer; dWq/dWk/dWv equally spaced (to verify) |
| B4 | Remove the 38 residual 16 MB device-to-device copies per microbatch | `block.c` `copy_mat` 232, 317, 324, 548, 565 | 0.51 | 4-6 h | Bit-identical if the "old + new" add order is kept | none |
| B5 | dvec: (a) coalesced / vectorised kernel, or (b) computed inside the FA2 backward prologue | `attn_fa_dvec16_kernel` 6620-6639 | (a) 0.45; (b) 0.6-0.8 | (a) 1 h; (b) 4-6 h | Bit-identical if the sequential `acc +=` order over c=0..63 is kept | none |
| B6 | W2 forward: avoid the split-K reduction + bias + residual pass (`globalKernel`, 66 us x 12) | `cublas_lt_gemm_half` 210-480 (16 heuristics at 365, timing loop) | 0-0.4 | 2-4 h | Different algorithm, different FP32 summation order | none |
| B7 | Checkpoint: one bulk write instead of 370 M 4-byte `fwrite` calls (format identical), then optional snapshot + writer thread | `lm_serialize.c` 252-285 (loop 271-275) | ~0.2-0.28 equivalent | 2 h bulk / +6 h async | None (same bytes on disk) | 2.2 GB pinned host RAM for the async version |
| B8 | Remove the microbatch-boundary syncs: `cudaMalloc`/`cudaFree` of the ids in embed lookup and scatter, `dx` created / copied / freed in `lm_backward`, CE targets as pageable H2D | `gpu_embed_lookup` 8836-8873, `gpu_embed_scatter` 8877-8915, `gpu_free` 2638, `lm.c` `lm_backward` 606-626, `ew_acquire` 3723 | ~0.3 | 2-3 h | Bit-identical | Prerequisite for graphs |
| A1 | Adam: rewrite `sumsq` (grid-stride loops, one double atomic per block) | `sumsq_kernel` 8307, `gpu_grad_sumsq` | 0.14 (4.4 ms per step) | 1 h | Norm only (order already nondeterministic) | none |
| A2 | Adam: keep bias/LN vectors and column sums on the device (no host gather, scatter or flush) | `lm_adam.c` `seg_gather` 360, `seg_scatter` 398; `colsum_flush_all` 2800 | 0.05 (1.5 ms per step) | 4-6 h | Bit-identical | Shares the db_head work with the graphs stream |
| A3 | Adam: float4 vectorisation | `adam_kernel_bf16` 8368 | 0.03 (~0.9 ms per step) | 1 h | Bit-identical | none |
| B9 | Do not store FP32 dK/dV when the BF16 twins are written | `attn_fa2.cuh` 485-487 | 0.1-0.3 | 1 h | Bit-identical | Check nothing reads FP32 dk/dv |
| B10 | Fold the b2 column sum into the next LN1 backward | `col_sum_tiled` 12 x 21.7 us | 0.2-0.25 | 3-4 h | Atomic order (already nondeterministic) | none |
| B11 | FA2 forward writes the BF16 twin of `concat` itself (instead of `act_bf16_publish`) | 6580-6583, `act_bf16_publish` 1782 | 0.13-0.15 | 1-2 h | Bit-identical (same RN rounding) | Keep FP32 O for dvec |
| B12 | LayerNorm: recompute `x_hat` from x_in/x_mid instead of storing it | LN forward and backward | 0-0.2 (+384 MB VRAM freed) | 3-4 h | Bit-identical (sub then mul, no FMA contraction) | none |
| G | CUDA-graph replay (separate stream) | | <= 0.35 after B1 and B8 (<= 1.7 in total) | 3-5 h+ | Replay correctness | B1, B8 |
| - | dQ accumulated without atomics; CE kernel occupancy; seq pack/unpack; data loading | | ~0 to 0.1 | | | not recommended |

The high-confidence bit-identical items (B1, B2, B4, B5a, B8, B11, A1-A3) total about 3.4 ms per microbatch: 62.2 to about 58.8 ms, about +5.8% (~69.5k tok/s). H1 and B7 add about 1.1 ms-equivalent of wall time.

### Answers to the specific questions

1. **Head GEMM and bias.** A plain GEMM (`mat_mul(delta, hv, w_head)`, `lm.c:443`) then a separate `bias_add_kernel` (`gpu_cuda.cu:4342`, one `i % cols` per element) at 834 us. Adding the bias inside the CE kernel (B2) is bit-identical; a cuBLASLt BIAS epilogue would change which algorithm runs.
2. **The W1 RELU_BIAS epilogue is already fused; the separate `globalKernel` pass belongs to W2.** W1 runs `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64` (BF16 output) with no epilogue kernel after it. The kernel before every `globalKernel<8,32,float,float,float,1,1,1>` is the `cutlass 64x64_32x6_nn` W2 forward with split-K 2 (369 us), and the kernel after it is the next layer's `ln_fwd` (or `pack_seq_rows`): the pass is the split-K reduction + bias + residual of `gpu_mat_mul_res` (5488), already chosen by the timing loop as the fastest of 16 candidates including the extra pass. Options: time more candidates, or enumerate algorithms with `CUBLASLT_ALGO_CONFIG_SPLITK_NUM=1`; at most ~0.4 ms and not bit-identical.
3. **Small launches** (456 events per microbatch in steady state): 195 GEMMs (Q, K, V still have separate dgrad and wgrad GEMMs), 38 kernels under 20 us (`f32_to_bf16`: 12 x b1 bias at 0.8 us, 12 x concat and 12 x dQ at 11.2 us, 2 x head hidden), 38 D2D copies (13.4 us each), about 37 H2D and 6 D2H copies, about 17 memsets (12 are the 16 MB dQ zeroings). Removable or mergeable: the 24 bias H2D + 12 conversions (B1), 38 D2D (B4), 12 concat conversions (B11), 12 dvec (B5b), 12 colsum (B10), 4 QKV-backward GEMMs per layer (B3). `qkv_wt_pack` runs once per Adam step (0.38 per microbatch): not a concern.
4. **Attention backward.** dvec costs 69.8 us per layer because one thread per row reads with a 4 KB stride; a coalesced kernel (B5a) saves about 0.45 ms, inside the FA2 prologue (B5b) about 0.6-0.8 ms (O tile loads are 2.5x redundant across key-tile blocks). dQ without atomics needs either per-(b,h) blocks with 64 KB of shared dQ (256 blocks for 70 SMs), a separate dQ pass recomputing S and dP (+40% attention-backward FLOPs), or partials plus a reduction (about 40 MB of extra traffic, cancelling the memset, atomics and conversion it saves): 0 to 0.3 ms for high effort, not recommended. The cheap gain is B9.
5. **Validation probe.** One window at a time: each window creates and frees a full `transformer_cache` (`cudaMalloc`/`cudaFree`, which sync), then for each of 255 rows a gemv into a stack view, a 64 KB synchronous D2H, a `cudaFree` and an FP64 softmax over 16384 logits on the host: about 156 ms per window, 40 s per probe, 19 mid-epoch probes plus start and end. On resume, `next_mid_pct = 5` (`lm_train.c:1102`) fires a probe immediately after the start probe (the log shows 22:10:13, 22:10:53, then 40% at 22:13:12). A batched version (B=16 packed forward ~17 ms, one head GEMM, a per-row log-sum-exp kernel) takes about 0.3 s per probe: about 834 s saved per epoch.
6. **Checkpoint save.** Synchronous with the GPU idle, an estimated 7-10 s every 30 min (about 0.5%): `lm_save_checkpoint` writes m and v interleaved with one 4-byte `fwrite` per value, 370 M calls (`lm_serialize.c:271-275`), after copying 185 M values through `slots[]` and a pageable D2H. Bulk-writing the interleaved data in chunks keeps the format and should take it to ~1.5-2 s; overlap needs a pinned snapshot on the copy stream (~0.1 s) plus a writer thread (a device-side snapshot is not possible at 13.6 GB of 16 GB used).
7. **Adam step** (16.2 ms at the step boundary): `adam_kernel_bf16` 8.7 ms moving 34 bytes per parameter, about 6.3 GB in total, ~720 GB/s = 80% of the card's 896 GB/s (float4 could save ~0.9 ms); `sumsq` 5.28 ms where ~0.9 ms would do (723k blocks each doing one double `atomicAdd` on a single address); about 1.8 ms idle across ~370 small copies (120 H2D, 120 D2H, 60 memsets for the bias/LN vectors and the column-sum flush).
8. **Host syncs per microbatch in training:** 24 pageable H2D bias copies (`lt_bias_upload`); `cudaFree` x 3 (embed lookup `d_ids`, embed scatter `d_ids`, `dx` in `lm_backward`) plus their `cudaMalloc`; the CE targets as pageable H2D (`ew_acquire`) and the synchronous 64 KB `db_head` D2H with the host add (`ce_delta_run` 9169). Per step: `gpu_device_sync` + `colsum_flush_all`, the `sumsq` D2H and `gpu_stream_sync`. About 1.3 ms per microbatch plus about 1.8 ms per step. Data loading is negligible (`[loop] win=0.3ms` per 32 microbatches; the 15 GB id file sits in the 52 GB page cache).

### Ranked plan

1. B1 + B2: about 1.8 ms, 3-5 h, bit-identical.
2. B8 + B4: about 0.8 ms, bit-identical; also clears the path for graphs.
3. A1 + A3 + B5a + B11: about 0.7 ms, each about 1 h.
4. H1 with the resume-probe fix, and the bulk-write part of B7: wall time only (about 15 min per epoch).
5. B3, then B6: not bit-identical; only through the 12-step loss gate.
6. G and A2 with the graphs stream (now worth at most ~1.7 ms, 0.35 after B1 and B8).
7. B9, B10, B12 if a few tenths of a millisecond still matter.

Caveats: the GEMM-to-layer mapping for B3 is inferred from kernel counts and timings; the checkpoint duration is an estimate; the B3 layout assumption (equally spaced dWq/dWk/dWv) is unchecked.
