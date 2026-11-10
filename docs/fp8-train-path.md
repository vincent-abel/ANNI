# The FP8 training path (`--fp8`)

## Abstract

`--fp8` runs the forward products and the data-gradient products on 8-bit floats, keeps the weight-gradient products in BF16, and keeps the weights themselves in FP32. It trains correctly. On this card and at this model width it is slower than BF16, so the runs use BF16; the path stays as a working base for the [low-precision plan](low-precision-plan.md).

## What problem it solves (plain language)

A matrix product on 8-bit values moves half the bytes of a 16-bit one and the Tensor Cores go through them faster. The question was whether that pays for the work of squeezing every tensor into 8 bits first.

## How it works for a human (intuition, analogy)

Writing down measurements with only three significant digits. It works if you first choose the unit so the numbers fill the scale: metres for a room, kilometres for a road. Choose it wrong and everything reads "999+" or "0.00".

## Formal view (definitions, formulas, shapes, algorithms)

Two 8-bit formats, both with a sign bit:

| Format | Exponent / mantissa bits | Largest value | Used for |
|---|---|---|---|
| E4M3 | 4 / 3 | 448 | weights and forward activations (more precision) |
| E5M2 | 5 / 2 | 57344 | gradients (more range) |

**Per-tensor scale.** A tensor `x` is stored as `q = fp8(x / s)` with `s = amax / max_fp8`, where `amax` is the largest magnitude of `x`. The product of two quantized tensors is multiplied by `s_a * s_b` to get real units back.

**Delayed scaling.** Finding `amax` first would need one more pass. Instead the quantizing kernel uses the `amax` this same tensor had on the previous step, and records the new one on the way.

**Which product uses what:**

```text
forward     y  = x W          E4M3 x E4M3
data grad   dx = dy W^T       E5M2 x E4M3
weight grad dW += x^T dy      BF16 x BF16, accumulated in FP32
update      Adam on FP32 weights, then weights re-quantized to E4M3
```

## Complexity / numerical pitfalls / design choices

- **One scale per tensor, not one for all.** The first version shared a single scale across every activation of every layer, so the scale applied was another tensor's: values saturated at 448 when it was too small and lost their mantissa when it was too large. Scales are now kept in slots keyed on the tensor's address (defect 9 in the [GPU training audit](gpu-training-audit.md)).
- **The library accepts one layout.** cuBLASLt's FP8 kernels on this GPU generation take only (transposed, plain) operands and refuse E5M2 x E5M2. So weights are cached quantized *and* transposed, and the weight gradient cannot be FP8 at all. Compare with [Transposed GEMMs](transposed-gemm.md), where BF16 needs no copy.
- **Sizes must be multiples of 16**; other products fall back to BF16.
- **Fusions are off.** The FFN bias+ReLU epilogue and the [fused residual](residual-fused-gemm.md) are skipped under `--fp8`.
- **Turning it off clears both flags.** `gpu_set_fp8(0)` once left the data-gradient flag on, and a BF16 run in the same process kept quantizing its gradients (defect 12).
- **Measured** (150 steps, 185M shape, same seed; table in the audit note): FP32 19.9k tok/s, BF16 38.3k, FP8 28.3k. FP8 ends 0.004 nats behind on validation. The quantize and transpose passes, and re-quantizing every weight after each step, cost more than the 8-bit products save at `D = 1024`.

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_set_fp8`, `gpu_set_fp8_dgrad`, `fp8_act_scale` (scale slots), `fp8_wt_get_ex` (quantized weight cache), `fp8_quant_tensor`, and the LayerNorm one-pass kernel that writes E4M3 directly for the next product.
- `anni lm train --fp8` in [`main.c`](../main.c); `lm_train_set_fp8` in [`lang/lm_train.c`](../lang/lm_train.c).
- [`scripts/ab_precision.sh`](../scripts/ab_precision.sh): the FP32 / BF16 / FP8 comparison on one data slice.
- [`tests/test_gpu.c`](../tests/test_gpu.c) holds FP8 gradients to looser bounds than BF16 (60% per tensor, 30% in norm).

## What next

- FP8 pays when the quantization is free: producers writing FP8 directly, as they now write BF16 ([BF16 twins](bf16-twins.md)). That is the first step of the [low-precision plan](low-precision-plan.md).
