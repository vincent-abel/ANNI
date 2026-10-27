# Speed stack bench (S6 / S10 / S15)

## Abstract

Record util / VRAM / ms/step (and packed ms/tok) after each speed-stack wave.

## Wave 1 (S6)

| Mode | Result |
|------|--------|
| `./anni lm train --bench-gpu --tokens 32 --steps 15` | mean **~257–540 ms/step** (B=1 synthetic; variance by load) |
| Packed wiki train (pre-stack, B=128, T=128) | ~**0.23–0.36 ms/tok** live progress |
| Flags to A/B | `--cuda-graphs`, `--bf16` |

## Wave 2 (S10) — this machine (RTX 5070 Ti)

| Mode | mean_ms / step (`--bench-gpu --tokens 32 --steps 12`) |
|------|--------|
| baseline FP32+TF32 | **537.5** |
| `--flash-attn` | **283.3** (~1.9×) |
| `--flash-attn --bf16` | **250.8** (~2.1×) |

`nvidia-smi` during idle after runs: util low (bench ends), ~329 MiB reported at sample. Packed wiki ms/tok remains the production metric.

## Wave 3 (S15) — this machine

| Mode | mean_ms / step (`--bench-gpu --tokens 32 --steps 10`) |
|------|--------|
| `--flash-attn --act-checkpoint` | **256.9** |
| `--flash-attn --adam-offload` | **252.0** |

Checkpoint/offload are VRAM levers; ms/step can match flash-alone when synthetic B=1 is not VRAM-bound. Prefer packed train + `nvidia-smi` memory for the real tradeoff.

```bash
./anni lm train --data data/toy_micro.txt --epochs 1 --batch 4 --window 64 \
  --flash-attn --bf16
```

## W2 gate (VRAM feed for W3)

Dominators at D=512 / T=256: head `rows×V` CE buffer, Adam flats, then pack activations. Flash removes persistent scores; bwd still peaks one `wtmp`. Wave 3: S11 ckpt, S12 reuse, S13 offload, S14 auto-B retune.
