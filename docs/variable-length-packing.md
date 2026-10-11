# Variable-length packing (E8)

## Abstract

Less pad waste means more real tokens/sec. Today packs use fixed windows; the next step is per-sequence lengths plus an attention mask in fused attn.

## Status

Fixed-T packing ships; mask-aware fused attn is the follow-on to raise effective tokens/step under the same VRAM.
