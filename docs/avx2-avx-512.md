# AVX2 / AVX-512

## Abstract

**AVX2** processes 4 doubles per 256-bit register; **AVX-512** processes 8. `ann_dot` selects an intrinsics path at compile time (`__AVX512F__`, else `__AVX2__`, else scalar). Build with `ARCH=-mavx2` or `-mavx512f` when the CPU supports it; default builds stay portable scalar/auto-vec.

## What problem it solves (plain language)

Auto-vectorization is best-effort. Explicit intrinsics guarantee wide FMA loops for the hottest dot products when you opt in with the right `-m` flags.

## How it works for a human (intuition, analogy)

Scalar code uses a teaspoon; AVX2 a soup ladle (4); AVX-512 a pitcher (8). You only grab the big ladle if the kitchen (CPU) has one — otherwise the teaspoon still works.

## Formal view (definitions, formulas, shapes, algorithms)

```text
if __AVX512F__:
  for i in 0..n step 8: vac += load8(a)*load8(b)   /* fmadd */
elif __AVX2__:
  for i in 0..n step 4: vac += load4(a)*load4(b)
else:
  scalar sum
tail: scalar for remainder
```

**Backend query.** `ann_simd_backend()` returns `"avx512"`, `"avx2"`, or `"scalar"`.

**Build examples.**

```bash
make ARCH=-mavx2
make ARCH=-mavx512f
make ARCH=          # portable default
```

## Complexity / numerical pitfalls / design choices

**Compile-time, not CPUID.** Wrong `-mavx512f` on an old CPU may illegal-instruction at runtime — leave `ARCH` empty for CI.

**FMA.** Intrinsics use `_mm256_fmadd_pd` / `_mm512_fmadd_pd` when available.

**Pedantic.** `<immintrin.h>` is gated so `-ansi` builds without ARCH still compile.

## How it shows up in this repo

- [`num/simd.c`](../num/simd.c): `ann_dot_avx2` / `ann_dot_avx512` / scalar
- [`num/simd.h`](../num/simd.h): `ann_simd_backend`
- [`Makefile`](../Makefile): `ARCH ?=`

## Worked example

`ann_dot` on length-16 vectors matches scalar within `1e-12`. Printing `ann_simd_backend()` shows which path linked.

## What next

[Quantization](quantization.md) shrinks inference weights to int8 for bandwidth.
