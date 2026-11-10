# BF16 twins: every tensor the GEMMs read has a half-size copy

## Abstract

The model's values are FP32, but the matrix products read BF16 (16 bits). A **twin** is the BF16 copy of an FP32 buffer, found by that buffer's address. There are three kinds (weights, activations, gradients), each with its own lifetime rule. The point of all the bookkeeping is that a twin is written once, by whoever produces the value, and never rebuilt by a separate conversion pass.

## What problem it solves (plain language)

Converting FP32 to BF16 just before each product means reading the whole tensor once more. With four or five products per block, forward and backward, those extra reads were a large part of the step. And the FP32 version of many tensors is never looked at again, so writing it was wasted too.

## How it works for a human (intuition, analogy)

A pocket edition of a reference book. You carry the small one to work; the full one stays on the shelf for the rare case that needs every decimal. The one rule that matters: when the book is revised, the pocket edition must be reprinted or thrown away, never left lying around.

## Formal view (definitions, formulas, shapes, algorithms)

| Kind | Who writes it | Lives until | Lookup |
|---|---|---|---|
| **Weight** | the Adam kernel, in the same pass as the FP32 update | the next update (always fresh) | `half_wt_get`: an offset into one flat BF16 buffer that mirrors the flat FP32 parameters |
| **Activation** | the producer: a GEMM with BF16 output, the LayerNorm kernel, the attention forward | the buffer is written again | `act_bf16_get`, 64 slots keyed on the FP32 address |
| **Gradient** | the producer: attention backward, LayerNorm backward, fused ReLU backward, the loss kernel | consumed once, and at most the end of the microbatch | `grad_twin_peek` (weight gradient reads it first), `grad_twin_take` (data gradient reads and clears it) |

Some activations are **BF16 only**: the GEMM writes the twin and the FP32 buffer is never filled (the FFN hidden layer, K and V). A reader that really needs FP32 asks `gpu_act_bf16_only` first.

## Complexity / numerical pitfalls / design choices

- **Stale twins are the whole risk.** The key is an address, and buffers are reused. A twin left behind by a producer nobody consumed would be served for whatever the next microbatch writes at that address. So: a new twin of a buffer drops the older ones, and `gpu_grad_twins_clear` wipes every gradient twin when the backward ends.
- **A path that skips the twin must forget it.** FP8 data gradients quantize from FP32 and do not consume the BF16 twin; they clear it anyway.
- **The head's delta is too big for a slot** (rows x vocab). The loss kernel writes it in BF16 over the first half of the logits' own storage instead ([Fused head and loss](fused-head-ce.md)).
- **Checking.** `ANN_TWIN_CHECK=1` reconverts each served twin and prints `[twin] STALE` on any difference. Slow, debug only.
- **Switches.** `ANN_TWIN_EMIT=0` makes the kernels stop writing gradient twins (the GEMMs convert as before); `ANN_HID_BF16=0` and `ANN_LN_BF16=0` do the same for the hidden layer and LayerNorm.

## How it shows up in this repo

- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `half_wt_get`, `act_bf16_reserve` / `act_bf16_publish` / `act_bf16_get`, `grad_twin_put` / `grad_twin_reserve` / `grad_twin_peek` / `grad_twin_take`, `gpu_grad_twins_clear`.
- [`lang/lm.c`](../lang/lm.c): the packed backward clears the gradient twins at its end.
- [`lang/lm_adam.c`](../lang/lm_adam.c): registers the flat BF16 weight buffer ([BF16 Adam](bf16-adam.md)).
- The step-by-step history, with timings, is in [BF16 outputs](bf16-direct-outputs.md) (candidates 9 to 14).

## What next

- An intermittent 2% gradient gap was first blamed on stale twins. It was the order of float additions: [Deterministic gradients](deterministic-gradients.md).
