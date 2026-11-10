# 2048-token context

## Abstract

The model trains and runs with windows of up to 2048 tokens instead of 256. Nothing in the weights depends on the length, so talk-170m was stretched from 256 to 2048 by training a little more on long windows (`scripts/ctx2048_run.sh`), and the 555M run uses 2048 from the first step.

## What problem it solves (plain language)

256 tokens is a paragraph. A chat with a few turns, or a source file, does not fit, and whatever falls out of the window is forgotten.

## How it works for a human (intuition, analogy)

A reader who only ever saw single paragraphs can read a page, but has to practise: the far-away words are new territory. The model's position signal is a formula, not a learned table, so position 1500 already has a value. What is missing is practice at using it.

## Formal view (definitions, formulas, shapes, algorithms)

- `LM_MAX_CTX` in [`lang/lm_config.h`](../lang/lm_config.h) is the largest window of the build, now 2048.
- Positions are sinusoidal ([Positional encoding](positional-encoding.md)): no parameter has a `T` in its shape. A checkpoint stores the context it was built with and loads in any build whose `LM_MAX_CTX` is at least that.
- Attention costs `O(T^2)` per window. The fused attention ([Attention on tensor cores](attention-fa2-mma.md)) never stores the `T x T` scores, only one log-sum per row (`B * H * T` values), so memory grows like `T`, not `T^2`.

## Complexity / numerical pitfalls / design choices

- **Same tokens per step, fewer windows.** A batch of 16 windows of 256 becomes 2 windows of 2048: the GEMM shapes stay the same, attention does 8 times more work per token.
- **The old attention kernels stop at 256.** They use one thread per key. Past 256 the fused path is required (`--flash-attn --bf16`); the host fallback works but is far too slow to train.
- **One long probe can eat the VRAM.** The one-sequence NLL probe does keep `T x T` rows per head and per block: about 6 GB at 2048 on the 555M sizes. It now runs at the training window (`ctx_probe_len` in [`lang/lm_train.c`](../lang/lm_train.c)).
- **Continued training needs its own schedule.** See [LR schedule for continued training](lr-schedule-continue.md).

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `fa2_args_ok` checks the fused attention arguments without the `T <= 256` limit.
- [`lang/lm_serialize.c`](../lang/lm_serialize.c): a shorter stored context loads; a longer one is refused.
- [`lang/attn.c`](../lang/attn.c): the packed host fallback returns real outputs and gradients past 256 tokens.
- [`lang/decode.c`](../lang/decode.c): long prompts are prefilled on the GPU ([GPU prefill](gpu-prefill.md)).
- Tests: `test_lm_load_shorter_ctx` in [`tests/test_ann.c`](../tests/test_ann.c); check 10 of [`tests/test_gpu.c`](../tests/test_gpu.c) runs a microbatch at `T = 512` and compares BF16 fused attention with the FP32 host path.

## Worked example

Stretching talk-170m (`scripts/ctx2048_run.sh`):

```text
data     0.4B tokens: 340M of web text, then 60M of code
windows  2048 tokens, 2 per microbatch, 32 microbatches per Adam step
steps    400M / 2048 / 64 = 3052
LR       1e-4, 100 warmup steps, cosine, counted from step 29084
then     the chat fine-tune again, at 2048
```

## What next

- `LM_MAX_CTX` is still a compile-time macro (**C1** in the README).
