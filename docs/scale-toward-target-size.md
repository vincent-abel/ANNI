# Scale toward target size

## Abstract

Default [`lm_config.h`](../lang/lm_config.h) uses **`LM_D_MODEL=32`**, **`LM_N_BLOCKS=2`**, **`LM_D_FF=64`**, **`LM_MAX_CTX=32`**—enough to **debug backprop and training loops** on CPU in milliseconds. Scaling toward a **target profile** (e.g. **`d_model=64`**, 4 blocks, context 128) increases capacity and FLOPs while staying **tiny versus production LMs** (millions–billions of params). This note records **what to change**, **cost growth**, and **checkpoint compatibility** when dimensions move.

## What problem it solves (plain language)

Toy dims prove correctness; slightly larger dims reveal whether **training infrastructure** (Adam, batches, checkpoints, PPL) still behaves before you invest in big data or GPU kernels. Scaling is deliberate knob-turning, not jumping to GPT scale.

## How it works for a human (intuition, analogy)

A model’s width and depth are **engine displacement**: 32-wide is a moped for learning to ride; 64-wide is still a scooter, not a truck—cheap, but you feel whether the transmission (training code) slips under load.

## Formal view (definitions, formulas, shapes, algorithms)

**Primary knobs** (all compile-time `#define` today):

| Symbol | Current | Example target | Effect |
|--------|---------|----------------|--------|
| `LM_D_MODEL` | 32 | 64 | Embed dim, all `d×d` mats scale |
| `LM_N_HEADS` | 4 | 4 or 8 | Must divide `d_model` |
| `LM_D_FF` | 64 | 128–256 | FFN inner width |
| `LM_N_BLOCKS` | 2 | 4 | Depth |
| `LM_MAX_CTX` | 32 | 64–128 | Attention \(T^2\) memory |

**Parameter scaling (order).** Per block: ~\(4d^2 + 2d\cdot d_{\text{ff}} + d_{\text{ff}} d + 4d\). Total ~ \(L \cdot 4d^2 + \cdots + Vd\) embed + \(Vd\) head.

**Example: d=64, L=4, d_ff=256, V=256.**

- Attention per block: 4×4096 ≈ 16k weights × 4 blocks ≈ 64k (vs ~8k at d=32,L=2).
- Still **<500k** parameters—toy by industry standards.

**FLOPs forward (rough).** Attention \(O(L T^2 d)\); FFN \(O(L T d d_{\text{ff}})\). Doubling `T` quadruples attn cost in dense implementation.

**Migration checklist.**

1. Edit `lm_config.h`; rebuild all `lang/` objects.
2. Re-run unit tests + finite-diff backward at new shapes.
3. **New checkpoints**—old LMCK headers with d=32 **must not** load into d=64 build.
4. Retune lr (often ↓ when widening).
5. Corpus: toy.txt still valid; may need more epochs to use extra capacity.

**Still toy even at d=64.** GPT-2 small uses d=768, L=12, V=50257—three orders of magnitude more weights and data.

## Complexity / numerical pitfalls / design choices (CS depth)

**Head divisibility.** `LM_D_MODEL % LM_N_HEADS == 0` enforced in `attn_create`.

**Cache RAM.** `lm_fwd_cache` stores multiple `T×T` mats per layer—scales with `L H T^2`; T=128, L=4 may hurt on embedded targets.

**Numerical stability.** Wider models with same init std may need **smaller init** or **warmup**—watch NaN in first Adam steps.

**GUI / tests.** Hard-coded 32 in assertions must update ([tests/test_ann.c](../tests/test_ann.c)).

**Future: runtime config.** Binary header already carries dims—long-term, single binary could allocate from header without recompile; not required for first scale step.

## How it shows up in this repo

- Single source: [`lang/lm_config.h`](../lang/lm_config.h).
- All `mat_create(LM_D_MODEL, ...)` in attn, ffn, embed, lm head.
- Docs referencing 32: update cross-links when scaling (these training notes mention both baseline and target).

## Worked example

Move `LM_D_MODEL` 32→64, keep L=2, d_ff=128, T=32.

- Embed table: 256×64 vs 256×32 (2×).
- Each `wq`: 64×64 vs 32×32 (4×).
- Expect ~3–4× slower step and ~4× checkpoint bytes—still seconds per epoch on toy.txt.

## What next

- Pick target row (e.g. d=64, L=4) as “phase 2” milestone after full backprop lands.
- Benchmark tokens/sec before/after scale on same corpus.
- Document chosen profile in CLI `--help` once configurable flags exist.
