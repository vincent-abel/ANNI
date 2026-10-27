# Talking model pretraining run (talk-170m)

## Abstract

One epoch over the 3.81B-token mix ([data note](talk-170m-data.md)) on the RTX 5070 Ti: batch 16 sequences of 256 tokens, 32 accumulated microbatches (131k tokens per optimizer step, about 29k steps), BF16 with tiled flash attention, peak learning rate \(3\times10^{-4}\) with 500 warmup steps and cosine decay. Measured: 11.7 GB of VRAM at peak, about 33.7k tok/s (about 51 TF on the matmul meter), about 33.5 h end to end. Checkpoints are written every 30 minutes and resume at the window they reached.

## What problem it solves

A day-and-a-half run on a workstation will be interrupted: this machine's WSL VM went down three times while the data was built. The trainer used to save only at epoch ends and to reshuffle on resume, so losing the VM 20 hours into a single epoch would have lost all of it. Batch size also had to be chosen by measurement: `--batch auto` sizes from free VRAM, and the 16k vocabulary had already run out of memory at batch 120 in earlier runs.

## Batch probe

Each probe trained about 30 steps at `--window 256 --stride 256 --bf16 --flash-attn` on a slice of the train ids while `nvidia-smi` was sampled every 0.5 s. The budget was 85% of 16,303 MiB, about 13.9 GB.

| Batch | Extra flag | Peak VRAM | Steady tok/s | Train-process RSS |
|---|---|---|---|---|
| 4 | | 6.5 GB | 19k | 7.9 GB |
| 8 | | 8.0 GB | 27k | 8.0 GB |
| **16** | | **11.7 GB (72%)** | **31k** | 7.9 GB |
| 32 | `--act-checkpoint` | 15.3 GB (94%) | 19k | 7.9 GB |

From 8 to 16 the peak grows by about 0.46 GB per sequence, so batch 24 would reach about 15.4 GB. Activation checkpointing fits batch 32 only at 94% and is slower. Batch 16 is the largest safe size, and gradient accumulation supplies the rest of the 131k tokens per step.

## Formal view

\[
\text{steps} = \frac{3.81\times10^{9}}{16 \cdot 256 \cdot 32} \approx 29{,}100,
\qquad
\eta(t) = \begin{cases}
\eta_{\max}\,\frac{t+1}{500} & t < 500\\[2pt]
\tfrac12\,\eta_{\max}\bigl(1+\cos\pi\,\tfrac{t-500}{28600}\bigr) & \text{otherwise}
\end{cases}
\]

with \(t\) the Adam step count. That count is stored in the checkpoint, so the schedule continues across a resume.

## Checkpoint and resume

- `--ckpt-minutes 30` saves `models/talk170m.bin` (weights and Adam moments) mid-epoch, with the epoch and the window index it reached.
- Each epoch's window order comes from its own xorshift stream seeded by `--seed` and the epoch number (`lm_train_epoch_order`), so `--resume` rebuilds the same order and skips the windows already trained.
- Saves go to `PATH.tmp` and are renamed, so a crash during a save keeps the previous checkpoint.
- Held-out loss on `data/mix_val_ids.bin` (256 windows) is logged at every 5% of the epoch and at the end.

## How to run it

```bash
scripts/talk170m_run.sh          # tmux session talk170m; resumes if models/talk170m.bin exists
tail -f logs/talk170m_pretrain.log
tmux attach -t talk170m          # Ctrl-b d to detach, Ctrl-c to stop
```

The driver stops `janni-llama-server` with `systemctl --user stop` first, waits until the GPU holds under 2 GB and RAM has over 30 GB free, skips finished steps by marker file, and runs:

```bash
./anni lm train --resume models/talk170m.bin --out models/talk170m.bin \
  --data data/mix_train_ids.bin --val-data data/mix_val_ids.bin \
  --tokenizer bpe --vocab data/mix.vocab \
  --window 256 --stride 256 --batch 16 --grad-accum 32 --epochs 1 \
  --lr 3e-4 --lr-warmup 500 --lr-decay cosine --lr-decay-steps 28600 \
  --grad-clip 1.0 --weight-decay 0.1 --flash-attn --bf16 \
  --eval-every 1 --eval-windows 256 --ckpt-minutes 30 --seed 1 -d
```

(`--force` instead of `--resume` on the first start.) For a restart after a WSL crash, add `@reboot …/scripts/talk170m_run.sh --boot` with `crontab -e`: it waits 5 minutes, then resumes.

## First steps

| Step | Train NLL | tok/s |
|---|---|---|
| 0 | 9.725 (uniform over 16k is 9.70) | |
| 50 | 8.221 | 33.7k |

## Pitfalls

- The CUDA build needs `make cuda NVCC="…/nvcc -Xcompiler -U_GNU_SOURCE"` here: glibc 2.43's `rsqrt` declaration clashes with CUDA 13.0's headers.
- With 256 windows the held-out estimate is noisy at about ±0.05 nats; compare trends, not single points.
- Each held-out pass takes about 2 minutes, because `lm_nll_windows` scores one window at a time.
- Stopping between saves loses up to 30 minutes of training.
