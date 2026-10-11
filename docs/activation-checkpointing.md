# Activation checkpointing (R9)

## Abstract

When packed `B·T·D` exceeds VRAM, recompute block activations in backward instead of storing them. Hook: prefer smaller B via OOM backoff today; full block recompute lands next to R9 enablement in `tblock_backward`.

## Status

Infrastructure: huge-B OOM halve + residency. Explicit recompute path is the next code increment when Target D OOMs at desired B.
