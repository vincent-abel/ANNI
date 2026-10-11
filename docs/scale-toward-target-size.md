# Scale toward Target size (R5 / R6 / R7)

## Abstract

Compile-time config is now Target-shaped: D=512, FF=2048, ctx=256, blocks=6 (~10M+ non-embed params class). Old checkpoints with other dims will not load.

## Current knobs

See [`lang/lm_config.h`](../lang/lm_config.h). Heads scale is [attention-heads-scale.md](attention-heads-scale.md) (R5b).

## What comes next

Runtime config (C1) so dims come from the checkpoint header without a rebuild.
