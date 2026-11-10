# GPU prefill of long prompts

## Abstract

Before the model can write its first token it has to read the prompt and fill the key/value cache of every block. `lm_decoder_prefill` does that in one of two ways: a short prompt goes through one full forward, a prompt longer than 256 tokens is fed to the device decode step one token after the other. Either way, attention never runs on the CPU.

## What problem it solves (plain language)

With a 2048-token context, a chat history or a source file is a prompt of a thousand tokens or more. The full forward used for prefill only has a GPU attention kernel up to 256 tokens; past that it fell back to a loop on the host, and the first token took a very long time to come.

## How it works for a human (intuition, analogy)

You cannot answer a letter before reading it. A short letter you take in at a glance. A long one you read line by line, taking notes as you go, and the notes are exactly what you need to keep writing afterwards.

## Formal view (definitions, formulas, shapes, algorithms)

The decoder keeps, per block, the keys and values of the tokens seen so far: `K, V` of shape `len x D` ([Optimized inference](optimized-inference.md)).

**Short prompt, `n <= 256`:**

1. `lm_forward_cache` runs the whole prompt at once.
2. The logits of position `n - 1` are computed.
3. Each block's `K` and `V` are copied into the decoder (`gpu_dec_kv_load`).

**Long prompt, `n > 256`:**

```text
for i in 0 .. n-1:
    gpu_dec_step(dec, ids[i], i, logits)
```

Step `i` computes one row: it projects the token, appends its key and value, and attends over the `i + 1` rows of the cache. The logits of the last step are the ones generation starts from.

## Complexity / numerical pitfalls / design choices

- Token by token, attention does `1 + 2 + ... + n` rows of work, the same total as the full forward, but as `n` small launches instead of a few big ones. It is slower per token than a batched forward and much faster than the host loop.
- No `T x T` buffer is needed, so a 2048-token prompt costs no more VRAM than generation itself.
- The step-by-step path is the same code as generation, so the cache is in the right state by construction.
- `ANN_PREFILL_STEPS=1` forces the step path for any length. `ANN_GPU_ONLY=1` makes generation fail rather than fall back to the CPU forward, which is what a bench wants.

## How it shows up in this repo

- [`lang/decode.c`](../lang/decode.c): `lm_decoder_prefill(d, ids, n, logits)`, the `PREFILL_FULL_MAX` limit, `prefill_steps`.
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_dec_step`, `gpu_dec_kv_load`.
- [`lang/generate.c`](../lang/generate.c): `lm_generate_stream` uses the decoder first.
- Check 11 of [`tests/test_gpu.c`](../tests/test_gpu.c): a 300-token device prefill and the next step match the full forward.

## What next

- A batched prefill through the fused attention would read a long prompt in a few launches.
