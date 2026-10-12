# Speed stack bench (S6)

## Abstract

After Wave 1 glue + graphs + mixed precision, record util / VRAM / ms/tok for regression.

## Snapshot (this machine)

| Mode | Result |
|------|--------|
| `./anni lm train --bench-gpu --tokens 32 --steps 15` | mean **257 ms/step** (~8 ms/tok @ B=1 synthetic) backend=cuda+cublas |
| Packed wiki train (pre-stack, B=128, T=128) | ~**0.23–0.36 ms/tok** live progress |
| Flags to A/B | `--cuda-graphs`, `--bf16`, `--fp16` |

Packed throughput (not `--bench-gpu`) is the metric that matters for wiki; re-run after enabling graphs/MP and note util via `nvidia-smi`.
