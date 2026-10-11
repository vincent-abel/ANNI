# Variable-length packing (E8)

## Abstract

Fixed-`T` windows waste the short EOF tail of a stream and treat pad tokens as real next-token targets. E8 emits a padded short final window, records per-sequence lengths, zeros CE grads on pads, and masks fused attention beyond each real length.

## What problem it solves

At the end of an epoch the stream often has `2 ≤ remain < T` tokens. Dropping them throws away data; keeping them as a full `T` with silent pads trains on pad→pad noise. Length-aware packing raises effective tokens/sec under the same VRAM.

## How it works for a human

Enable with `./ann lm train … --varlen` (on by default in `./autotrain`). The stream pads the last short window with `<eos>`, stores `lens[b]`, and the packed CE / fused attn ignore positions `≥ lens[b]`.

## Formal view

For sequence \(b\) of real length \(L_b \le T\), ids are \(x_{b,0},\ldots,x_{b,L_b-1}\) then pad. Loss sums only over \(t < L_b-1\). Attention scores use keys \(j < L_b\) and queries \(i < L_b\).

## Complexity / pitfalls / design choices

- Shuffle buffer still stores full-`T` windows; short tails bypass the buffer.
- Host (non-fused) attention path does not yet truncate per-`b` mats; fused GPU path applies `seqlens`.
- Pads are `VOCAB_EOS`, not a separate pad id.

## How it shows up in this repo

- [`lang/dataset_stream.c`](../lang/dataset_stream.c) — `text_stream_set_varlen`, short EOF emit
- [`lang/lm_train.c`](../lang/lm_train.c) — `lm_train_set_varlen`, pack `lens[]`
- [`lang/lm.c`](../lang/lm.c) — `lm_loss_backward_packed_lens`
- [`num/gpu_cuda.cu`](../num/gpu_cuda.cu) — `gpu_attn_set_seqlens` + fused mask
- CLI: `--varlen`; autotrain enables it

## Tiny worked example

```text
window=64, 20 tokens left at EOF
→ emit ids[0..19] + EOS pad, lens=20
→ CE only on 19 predict rows; attn ignores j≥20
```

## What comes next

Document-boundary packing (never cross `<eos>` inside a window) and host-path length truncate for the non-fused fallback.
