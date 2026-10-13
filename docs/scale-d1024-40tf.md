# Scale to D=1024 / 12 blocks (40TF Wave 2)

## Abstract

Compile-time Target shape for the ~40 TF stretch: \(D=1024\), \(H=16\) (\(d_h=64\)), \(D_{FF}=4096\), \(L=12\), ctx \(256\). Matmul meter mass \(P \approx 252\,\mathrm{M}\) so **~26.5k tok/s** reports ~40 TF — vs ~212k tok/s at D=512.

## What problem it solves

Small GEMMs stay memory-bound on a 5070 Ti even with tiled FA. Widening \(D\) and depth fattens every projection and FFN matmul so BF16 Tensor Cores have enough work per token.

## Formal view

\[
P_{\mathrm{meter}} = L\bigl(12 D^2 + 2 D\cdot D_{FF}\bigr)
= 12\bigl(12\cdot 1024^2 + 2\cdot 1024\cdot 4096\bigr) \approx 2.52\times 10^8.
\]

Train TFLOP/s \(\approx 6 P \cdot (\mathrm{tok/s}) / 10^{12}\).

Rough total params @ 16k BPE (embed+head): \(\sim 280\,\mathrm{M}\). FP32 weights+Adam \(\sim 3\)–\(4\,\mathrm{GB}\); activations dominate — pair with `--flash-attn --act-checkpoint --adam-offload`.

## How it shows up in this repo

- [`lang/lm_config.h`](../lang/lm_config.h): macros above
- **Breaks** older D=512 checkpoints (`LMCF` shape mismatch) — new `--out`
- README checklist **T2**

## Smoke

```bash
make cuda -j
./anni lm train --data /tmp/toy_tiny.txt --epochs 1 --batch auto --window 64 \
  --flash-attn --bf16 --act-checkpoint --adam-offload \
  --force --out /tmp/d1024.bin --eval-every 0 -d
```

Expect auto-B ≫ 1 and a `[attn] tiled flash` log line. Wiki resume of D=512 bins will fail — retrain.

## Pitfalls

- First alloc / Adam state is heavy; OOM → lower `--vram-frac` or rely on act-checkpoint.
- Live NLL EMA is train CE; ctx NLL is stamped separately at epoch end.
- Prefer `--lr 3e-4` (or lower) at D=1024; weight init scales as \(0.02\sqrt{512/D}\).
