# Flash / TC / tiled attention (R11 / S7 / 40TF)

## Abstract

`--flash-attn` prefers **tiled flash** (`fused==4`): forward writes logsumexp only; backward recomputes \(P\) in KV tiles with **no** \(B\cdot H\cdot T\cdot T\) HBM. Fallback: Tensor-Core strided GemmEx (`fused==3`) when a score buffer fits, then fused causal (`fused==1`). Cap **T ≤ 256**.

## How it shows up in this repo

- CLI: `--flash-attn` → tiled first; log line `[attn] tiled flash (logsumexp, no T×T HBM)`
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu): `gpu_attn_flash_fwd` (+ `g_fa_L`), `gpu_attn_tiled_bwd`, optional `gpu_attn_tc_*`
- Paper: [tiled-flash-attention.md](tiled-flash-attention.md)
- `lm_packed_bytes`: flash counts \(B\cdot H\cdot T\) Lstat, not full scores

## Smoke (avoid “does nothing”)

```bash
./anni lm train --data data/toy_micro.txt --epochs 1 --batch 128 --window 32 \
  --flash-attn --bf16 --force --out /tmp/fa.bin --eval-every 0 -d
```

Use `--force --out ...`. Default `--eval-windows` is 32 (`0` = all windows).

## Bench snapshot (5070 Ti, D=512 era)

See [speed-stack-bench.md](speed-stack-bench.md) for 40TF-wave numbers after D=1024 scale.

## Status vs FlashAttention-2/3

Tiled bwd avoids full score HBM (FA-2 idea). Not yet FA-3 warp-specialized / TMA. Next: fusion + wider model ([scale-toward-target-size.md](scale-toward-target-size.md)).
